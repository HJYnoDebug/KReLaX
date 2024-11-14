import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import numpy as np
from dataset import EmotionDataset
import pandas as pd
import hyperparameters as hp
from torch import nn
from classifier_utils import save_model, save_train_process
from selfatten import RobertaWithAttention

# Paths
best_model_path = 'best_model/roberta_best_model.pt'
checkpoint_path = 'checkpoint/selfatten_checkpoint.pt'
train_data_path = 'checkpoint/selfatten_train_data.txt'

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
label_names = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Create hyperparameters instance
hyperparams = hp.Hyperparameters()

# Initialize tokenizer and load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model
model = RobertaWithAttention(roberta, num_labels=5, num_heads=4, hidden_dim=128).to(device)
optimizer = AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
criterion = nn.BCEWithLogitsLoss()

# Create DataLoaders
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, hyperparams.max_len)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, hyperparams.max_len)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(hyperparams.num_epochs):
    print("Training...")
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{hyperparams.num_epochs}, Loss: {avg_train_loss}')

    # Evaluate model on the validation set
    print("Evaluating...")
    model.eval()
    total_val_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            predictions = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_dataloader)
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)

    # Calculate overall metrics
    val_f1_weighted = f1_score(np.array(all_labels), binary_predictions, average='weighted')
    val_accuracy_overall = (np.array(all_labels) == binary_predictions).mean()
    val_precision_weighted = precision_score(np.array(all_labels), binary_predictions, average='weighted')
    val_recall_weighted = recall_score(np.array(all_labels), binary_predictions, average='weighted')

    # Calculate per-label metrics
    label_metrics = {}
    for i, label_name in enumerate(label_names):
        label_metrics[label_name] = {
            'accuracy': accuracy_score(np.array(all_labels)[:, i], binary_predictions[:, i]),
            'precision': precision_score(np.array(all_labels)[:, i], binary_predictions[:, i]),
            'recall': recall_score(np.array(all_labels)[:, i], binary_predictions[:, i]),
            'f1': f1_score(np.array(all_labels)[:, i], binary_predictions[:, i]),
            'jaccard': jaccard_score(np.array(all_labels)[:, i], binary_predictions[:, i])
        }

    # Print validation metrics
    print(f'Val Loss: {avg_val_loss}')
    print(f'Overall Val Accuracy: {val_accuracy_overall}')
    print(f'Overall Val F1 Score (weighted): {val_f1_weighted}')
    for label, metric in label_metrics.items():
        print(f"{label} - Accuracy: {metric['accuracy']:.4f}, Precision: {metric['precision']:.4f}, "
          f"Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}, Jaccard: {metric['jaccard']:.4f}")


    # Save training process metrics
    save_train_process(avg_train_loss, avg_val_loss, val_accuracy_overall, val_f1_weighted, epoch, train_data_path, label_metrics=label_metrics)

    # Early stopping and save checkpoint model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        save_model(model, save_path=checkpoint_path)
        print("Saved checkpoint in epoch:", epoch + 1)
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter}")

    if early_stopping_counter >= hyperparams.early_stopping_patience:
        print("Early stopping")
        break
