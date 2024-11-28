import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import numpy as np
import pandas as pd
from torch import nn

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from model.dataset import EmotionDataset
import model.hyperparameters as hp
from model.classifier_utils import save_model, save_train_process

# File paths
BEST_MODEL_PATH = 'best_model/mbert_best_model.pt'
CHECKPOINT_PATH = 'checkpoint/mbert_checkpoint.pt'
TRAIN_DATA_PATH = 'checkpoint/mbert_train_data.txt'

# Load dataset
data = pd.read_csv('data/merged.csv')

# Split data into train and test sets
train_ratio = 0.85
train_df = data.sample(frac=train_ratio)
test_df = data.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

# Define label columns
label_columns = train_df.columns[2:]  # Skip 'id' and 'text'

# Extract text and labels for training and testing
train_texts, train_labels = train_df['text'].tolist(), train_df[label_columns].values
test_texts, test_labels = test_df['text'].tolist(), test_df[label_columns].values
label_num = len(data.columns)-2

# Load hyperparameters
hyperparams = hp.Hyperparameters()

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create Dataset and DataLoader
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, hyperparams.max_len)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, hyperparams.max_len)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=label_num).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
criterion = nn.BCEWithLogitsLoss()

# Training parameters
best_loss = float('inf')
early_stopping_counter = 0
label_names = label_columns.tolist()

# Training loop
for epoch in range(hyperparams.num_epochs):
    print(f"Epoch {epoch + 1}/{hyperparams.num_epochs}")
    model.train()
    total_train_loss = 0

    # Training phase
    for batch in train_dataloader:
        optimizer.zero_grad()  # Zero gradients
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Create mask for ignoring gradients where labels == -1
        mask = (labels != -1).float()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss for non-missing labels
        loss = criterion(logits * mask, labels * mask)  # Apply mask

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Evaluation phase
    model.eval()
    print("Evaluating...")
    total_val_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels.float())

            # Apply mask for ignoring labels == -1
            mask = (labels != -1).float()
            loss = (loss * mask).sum() / mask.sum()  # Only compute loss for non-missing labels
            total_val_loss += loss.item()
            
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_dataloader)
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)

    # Calculate metrics
    mask = np.array(all_labels) != -1  # Mask to exclude -1 labels

    # Filter out samples with -1 labels
    filtered_labels = np.array(all_labels)[mask]
    filtered_predictions = binary_predictions[mask]
    
    # Compute weighted F1 and overall accuracy
    val_f1_weighted = f1_score(filtered_labels, filtered_predictions, average='weighted')
    val_accuracy = (filtered_labels == filtered_predictions).mean()

    label_metrics = {}
    for i, label_name in enumerate(label_names):
        true_labels = np.array(all_labels)[:, i]
        pred_labels = binary_predictions[:, i]
        
        # Mask out -1 labels
        mask = true_labels != -1

        # Calculate metrics only for valid labels
        label_metrics[label_name] = {
            'accuracy': accuracy_score(true_labels[mask], pred_labels[mask]),
            'precision': precision_score(true_labels[mask], pred_labels[mask], average='binary', zero_division=0),
            'recall': recall_score(true_labels[mask], pred_labels[mask], average='binary', zero_division=0),
            'f1': f1_score(true_labels[mask], pred_labels[mask], average='binary', zero_division=0),
            'jaccard': jaccard_score(true_labels[mask], pred_labels[mask], average='binary')
        }

    # Print metrics
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Overall Accuracy: {val_accuracy:.4f}, Weighted F1: {val_f1_weighted:.4f}")
    for label, metric in label_metrics.items():
        print(f"{label}: {metric}")

    # Save training logs
    save_train_process(avg_train_loss, avg_val_loss, val_accuracy, val_f1_weighted, epoch, TRAIN_DATA_PATH, label_metrics)

    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        save_model(model, save_path=CHECKPOINT_PATH)
        print(f"Saved best model at epoch {epoch + 1}")
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter}")

    if early_stopping_counter >= hyperparams.early_stopping_patience:
        print("Early stopping triggered")
        break
