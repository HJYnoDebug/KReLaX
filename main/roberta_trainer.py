import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import numpy as np
from dataset import EmotionDataset
import pandas as pd
import hyperparameters as hp
from torch import nn
from classifier_utils import save_model, save_train_process

# Paths
best_model_path = 'best_model/roberta_best_model.pt'
checkpoint_path = 'checkpoint/roberta_checkpoint.pt'
train_data_path = 'checkpoint/roberta_train_data.txt'

# Load data
data = pd.read_csv('data/eng.csv')

# Split data into training and testing sets
train_size = 0.85
train_df = data.sample(frac=train_size)
test_df = data.drop(train_df.index).reset_index(drop=True)
train_data = train_df.reset_index(drop=True)

# Prepare training and testing texts and labels
train_texts = train_df['text'].tolist()
train_labels = train_df[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()

# Create hyperparameters instance
hyperparams = hp.Hyperparameters()

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create dataset instances for training and testing
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, hyperparams.max_len)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, hyperparams.max_len)

# Create DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model instance and move to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
criterion = nn.BCEWithLogitsLoss()

# Training loop
best_loss = float('inf')  # Initialize best loss for early stopping
early_stopping_counter = 0  # Initialize early stopping counter
label_names = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']  # Label names for reporting

for epoch in range(hyperparams.num_epochs):
    print("Training...")
    model.train()  # Set model to training mode
    total_loss = 0

    # Iterate over batches in the training DataLoader
    for batch in train_dataloader:
        optimizer.zero_grad()  # Zero the gradients
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)  # The output is a SequenceClassifierOutput

        # Extract logits from the output
        logits = outputs.logits  # logits should be used for loss calculation and predictions

        # Calculate loss
        loss = criterion(logits, labels)  # Use logits for BCEWithLogitsLoss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate loss

    print(f'Epoch {epoch + 1}/{hyperparams.num_epochs}, Loss: {total_loss / len(train_dataloader)}')

    # Evaluate model on the validation set
    print("Evaluating...")
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Extract logits
            logits = outputs.logits

            # Calculate loss
            loss = criterion(logits, labels)  # Use logits for BCEWithLogitsLoss
            total_val_loss += loss.item()  # Accumulate validation loss

            # Apply sigmoid to get probabilities, then convert to binary predictions
            predictions = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())  # Collect labels

    # Convert predictions to binary
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)

    # Calculate per-label metrics
    metrics = {}
    for i, label in enumerate(label_names):
        label_true = np.array(all_labels)[:, i]
        label_pred = binary_predictions[:, i]
        metrics[label] = {
            'accuracy': accuracy_score(label_true, label_pred),
            'precision': precision_score(label_true, label_pred),
            'recall': recall_score(label_true, label_pred),
            'f1': f1_score(label_true, label_pred),
            'jaccard': jaccard_score(label_true, label_pred)
        }

    # Calculate average losses and overall metrics
    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(test_dataloader)
    val_f1_weighted = f1_score(np.array(all_labels), binary_predictions, average='weighted')
    val_accuracy_overall = (np.array(all_labels) == binary_predictions).mean()

    # Print validation metrics
    print(f'Val Loss: {avg_val_loss}')
    print(f'Train Loss: {avg_train_loss}')
    print(f'Overall Val Accuracy: {val_accuracy_overall}')
    print(f'Weighted Val F1 Score: {val_f1_weighted}')

    # Print per-label metrics
    for label, metric in metrics.items():
        print(f"{label} - Accuracy: {metric['accuracy']:.4f}, Precision: {metric['precision']:.4f}, "
              f"Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}, Jaccard: {metric['jaccard']:.4f}")

    # Save training process metrics
    save_train_process(avg_train_loss, avg_val_loss, val_accuracy_overall, val_f1_weighted, epoch, train_data_path, label_metrics=metrics)

    # Early stopping and save checkpoint model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        save_model(model, save_path=checkpoint_path)  # Save model checkpoint
        print("Saved checkpoint in epoch:", epoch + 1)
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter}")

    if early_stopping_counter >= hyperparams.early_stopping_patience:
        print("Early stopping")
        break
