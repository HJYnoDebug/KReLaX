from transformers import AutoTokenizer
import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast

# Add project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from model.dataset import EmotionDataset
import model.hyperparams as hp
from model.classifier_utils import save_train_process
from model.model_head import MultiLabelMultiClassModel  # Change: Import your custom model

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
data = pd.read_csv('data/mergedB.csv')

# Define label columns
label_columns = data.columns[2:-1]
texts = data['text'].tolist()
labels = data[label_columns].values
langs = data['lang'].tolist()
label_num = len(label_columns)

# Load hyperparameters
hyperparams = hp.Hyperparameters()

# Model configurations
model_configs = [
    {'name': 'google/rembert', 'tokenizer': 'google/rembert', 'save_dir': 'rembert'},
    {'name': 'xlm-roberta-base', 'tokenizer': 'xlm-roberta-base', 'save_dir': 'xlm-roberta'},
    {'name': 'sentence-transformers/LaBSE', 'tokenizer': 'sentence-transformers/LaBSE', 'save_dir': 'labse'}
]

# Calculate positive sample weights based on frequency for each label
epsilon = 1e-8
total_samples = len(labels)
num_classes = 4  # Assumes each label has 4 classes (0, 1, 2, 3)

# Initialize weight tensor for each class
label_frequencies = torch.zeros(num_classes, dtype=torch.float)

# Compute class frequencies for all labels
for i in range(label_num):
    for j in range(num_classes):
        label_frequencies[j] += (labels[:, i] == j).sum()

# Normalize frequencies to get the weights
label_frequencies /= total_samples
weights_per_class = 1.0 / (label_frequencies + epsilon)

# Clamp weights to avoid extreme values
weights_per_class = torch.clip(weights_per_class, min=1, max=8)  # Optionally adjust min/max if needed

# Ensure that weights are on the correct device
weights_per_class = weights_per_class.to(device)

# Set up CrossEntropyLoss with class weights
criterion = nn.CrossEntropyLoss(weight=weights_per_class, reduction='none')

# K-fold cross-validation
print("Processing K-fold cross-validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True)

# Group data by 'lang'
grouped_data = data.groupby('lang')

# Main training loop
for model_config in model_configs:
    print(f"Training model: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, langs)):
        print(f"Fold {fold + 1}/{kfold.get_n_splits()} for {model_config['name']}")
        
        if fold == 3:
            print("only train 3 folds, quit")
            break
        
        # Initialize model and optimizer
        model = MultiLabelMultiClassModel(num_labels=label_num, model_name=model_config['name']).to(device)  # Change: Initialize custom model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams.learning_rate,
            weight_decay=hyperparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
        
        # Prepare data for this fold
        train_texts, train_labels, val_texts, val_labels = [], [], [], []
        for lang_value, group in grouped_data:
            lang_texts = group['text'].tolist()
            lang_labels = group[label_columns].values
            lang_idx = group.index

            lang_train_idx, lang_val_idx = next(kfold.split(lang_texts, [langs[i] for i in lang_idx]))
            train_texts.extend([lang_texts[i] for i in lang_train_idx])
            train_labels.extend([lang_labels[i] for i in lang_train_idx])
            val_texts.extend([lang_texts[i] for i in lang_val_idx])
            val_labels.extend([lang_labels[i] for i in lang_val_idx])

        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, hyperparams.max_len)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, hyperparams.max_len)

        train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False)

        # Training loop
        best_loss = float('inf')
        fold_best_f1 = 0.01
        early_stop_counter = 0
        fold_best_model = None

        for epoch in range(hyperparams.num_epochs):
            print(f"Epoch {epoch + 1}/{hyperparams.num_epochs}")
            model.train()
            train_loss = 0.0
        
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)  # Shape: (batch_size, num_labels, num_classes)
                
                with autocast():
                    logits = model(input_ids, attention_mask=attention_mask)  # Shape: (batch_size, num_labels, num_classes)
                    # Compute loss for each label independently
                    loss = 0
                    for i in range(label_num):
                        label_loss = criterion(logits[:, i, :], labels[:, i])  # (batch_size,)
                        loss += label_loss.sum()
                        
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            avg_train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
        
            # Validation Phase
            model.eval()
            total_val_loss = 0.0
            all_labels, all_predictions = [], []
        
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
        
                    logits = model(input_ids, attention_mask=attention_mask)
        
                    # Compute loss for each label independently
                    for i in range(label_num):
                        label_loss = criterion(logits[:, i, :], labels[:, i])  # (batch_size,)
                        total_val_loss += label_loss.sum().item()  # Sum the loss for this label

        
                    # Predictions for evaluation
                    predictions = torch.argmax(logits, dim=2)  
                    # Shape: (batch_size, num_labels)
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())  # Flatten labels

        
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
            # Flatten predictions and labels for metrics calculation
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            print(f"Shape of all_predictions: {all_predictions.shape}")
            print(f"Shape of all_labels: {all_labels.shape}")
            
            label_metrics = {}
            avg_f1, total_f1 = 0.0, 0.0
            
            for i, label_name in enumerate(label_columns):
                # Ensure the true labels are correctly indexed for each label
                true_labels = all_labels[:, i]  # Get true labels for the i-th label
                pred_labels = all_predictions[:, i]  # Get predicted labels for the i-th label
                
                #print(f"True labels (first 10): {true_labels[:10]}")
                #print(f"Predicted labels (first 10): {pred_labels[:10]}")
                
                # Compute F1 score for each label
                label_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                total_f1 += label_f1
            
                label_metrics[label_name] = {
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'precision': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
                    'recall': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
                    'f1': label_f1,
                    'jaccard': jaccard_score(true_labels, pred_labels, average='macro', zero_division=0)
                }
            
            avg_f1 = total_f1 / len(label_columns)

            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Overall F1: {avg_f1:.4f}")
            for label, metric in label_metrics.items():
                print(f"{label}: {metric}")

            # Replace the call to save_train_process with direct file saving
            fold_checkpoint_dir = f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}"
            os.makedirs(fold_checkpoint_dir, exist_ok=True)
            
            # Ensure that the paths are strings, and save the log files properly
            log_file_path = os.path.join(fold_checkpoint_dir, "train_log.txt")
            
            # Open the file to write logs
            with open(log_file_path, 'a') as f:
                f.write(f"fold: {fold}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Train Loss: {avg_train_loss:.4f}\n")
                f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
                f.write(f"Validation Accuracy: {avg_f1:.4f}\n")
                
                # Write per-label metrics if available
                if label_metrics:
                    f.write("Per-Label Metrics:\n")
                    for label, metrics in label_metrics.items():
                        f.write(f"  {label}:\n")
                        f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                        f.write(f"    Precision: {metrics['precision']:.4f}\n")
                        f.write(f"    Recall: {metrics['recall']:.4f}\n")
                        f.write(f"    F1 Score: {metrics['f1']:.4f}\n")
                        f.write(f"    Jaccard Score: {metrics['jaccard']:.4f}\n")
                
                f.write("\n")
                f.write(f"macro f1: {avg_f1:.4f}\n")
                f.write("\n")
            
            # Save the avg_f1 score to a separate text file for each fold
            avg_f1_file_path = os.path.join(fold_checkpoint_dir, "avg_f1.txt")
            with open(avg_f1_file_path, "w") as f:
                f.write(f"{avg_f1}")
            print(f"Saved avg_f1 for fold {fold + 1} to {avg_f1_file_path}")
            
            # Early stopping and model saving logic
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                fold_best_model = model
                if avg_f1 > fold_best_f1:
                    fold_best_f1 = avg_f1
                print("Best model updated based on loss")
                early_stop_counter = 0 
            elif avg_f1 > fold_best_f1:  # If F1 score improves
                fold_best_f1 = avg_f1
                fold_best_model = model
                print(f"Saved model with improved average F1 at epoch {epoch + 1}")
                if early_stop_counter > 0:
                    early_stop_counter -= 1
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}")
            
            if early_stop_counter >= hyperparams.early_stopping_patience:
                print("Early stopping triggered")
                break
            
            scheduler.step(avg_f1)
            
        # After training for this fold, save the best model
        if fold_best_model is not None:
            fold_best_model_path = os.path.join(f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}", "best_model.pt")
            os.makedirs(os.path.dirname(fold_best_model_path), exist_ok=True)
            torch.save(fold_best_model, fold_best_model_path)
            print(f"Best model for fold {fold + 1} saved to {fold_best_model_path}")
            
            # Save the avg_f1 for this fold
            with open(f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}/avg_f1.txt", "w") as f:
                f.write(f"{avg_f1}")
            print(f"Saved avg_f1 for fold {fold + 1} to {fold_best_model_path.replace('best_model.pt', 'avg_f1.txt')}")
