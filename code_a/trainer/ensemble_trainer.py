from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
from model.dataset_w import EmotionDataset
import model.hyperparameters as hp
from model.classifier_utils import save_train_process

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
data = pd.read_csv('data/aug_mergedA.csv')

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

# Calculate positive sample weights
epsilon = 1e-8
total_samples = len(labels)
label_sums = labels.sum(axis=0)
label_frequencies = total_samples / (len(label_columns) * label_sums + epsilon)
label_frequencies = torch.tensor(label_frequencies, dtype=torch.float).to(device)

# Adjust positive sample weights
adjusted_label_weights = torch.clip(
    label_frequencies, 
    min=label_frequencies * 0.8, 
    max=label_frequencies * 2.5
)

# Set up BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=adjusted_label_weights, reduction='none')

# K-fold cross-validation
print("Processing K-fold cross-validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group data by 'lang'
grouped_data = data.groupby('lang')

# Main training loop
for model_config in model_configs:
    print(f"Training model: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, langs)):
        print(f"Fold {fold + 1}/{kfold.get_n_splits()} for {model_config['name']}")
        
        # Initialize model and optimizer
        model = AutoModelForSequenceClassification.from_pretrained(model_config['name'], num_labels=label_num).to(device)
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
                labels = batch['labels'].to(device)
                weights = batch['weight'].to(device)

                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    raw_loss = criterion(logits, labels.float())
                    
                    if (raw_loss < 0).any():
                        print("error! loss<0")
                        return

                loss = raw_loss.mean()  # Normalize over batch
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            
            # After evaluation phase, where you compute all_preds and all_labels
            model.eval()
            total_val_loss = 0
            all_labels, all_predictions = [], []
            
            with torch.no_grad():
                for batch in val_loader:  # Fixed val_loader here
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
            
                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    raw_loss = criterion(logits, labels.float())
            
                    # Simply take the mean of raw loss (no weighting)
                    loss = raw_loss.mean()  
                    total_val_loss += loss.item()
            
                    # Collect predictions and labels for metrics
                    predictions = torch.sigmoid(logits).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate and print average validation loss
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
            # Metrics calculation
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
        
            val_macro_f1 = f1_score(all_labels, all_predictions, average='macro')
            val_f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
            val_accuracy = accuracy_score(all_labels, all_predictions)
        
            label_metrics = {}
            total_f1 = 0
        
            for i, label_name in enumerate(label_columns):
                true_labels = all_labels[:, i]
                pred_labels = all_predictions[:, i]
                
                label_f1 = f1_score(true_labels, pred_labels, zero_division=0)
                total_f1 += label_f1
        
                label_metrics[label_name] = {
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'precision': precision_score(true_labels, pred_labels, zero_division=0),
                    'recall': recall_score(true_labels, pred_labels, zero_division=0),
                    'f1': label_f1,
                    'jaccard': jaccard_score(true_labels, pred_labels)
                }
        
            avg_f1 = total_f1 / len(label_columns)
        
            # Print metrics
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Overall Accuracy: {val_accuracy:.4f}, Weighted F1: {val_f1_weighted:.4f}, macro F1: {val_macro_f1:.4f}")
            print(f"Average F1 (across all labels): {avg_f1:.4f}")
            for label, metric in label_metrics.items():
                print(f"{label}: {metric}")

            # Save training logs
            fold_checkpoint_dir = f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}"
            os.makedirs(fold_checkpoint_dir, exist_ok=True)
            
            # Save the training logs after each fold
            save_train_process(
                avg_train_loss, 
                avg_val_loss, 
                val_accuracy, 
                val_f1_weighted, 
                epoch, 
                os.path.join(fold_checkpoint_dir, "train_log.txt"), 
                label_metrics, 
                avg_f1, 
                fold
            )
            
            # Save the avg_f1 score to a separate text file for each fold
            with open(os.path.join(fold_checkpoint_dir, "avg_f1.txt"), "w") as f:
                f.write(f"{avg_f1}")
            print(f"Saved avg_f1 for fold {fold + 1} to {os.path.join(fold_checkpoint_dir, 'avg_f1.txt')}")

            # Early stopping and model saving logic
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                fold_best_model = model.state_dict()
                print("Best model updated based on loss")
                early_stop_counter = 0 
            elif avg_f1 > fold_best_f1:  # If F1 score improves
                fold_best_f1 = avg_f1
                fold_best_model = model.state_dict()
                print(f"Saved model with improved average F1 at epoch {epoch + 1}")
                if early_stop_counter > 1 
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
            fold_best_model_path = f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}/best_model.pt"
            os.makedirs(os.path.dirname(fold_best_model_path), exist_ok=True)
            torch.save(fold_best_model, fold_best_model_path)
            print(f"Best model for fold {fold + 1} saved to {fold_best_model_path}")
        
            # Save the avg_f1 for this fold
            with open(f"checkpoint_kfold/{model_config['save_dir']}/fold_{fold + 1}/avg_f1.txt", "w") as f:
                f.write(f"{avg_f1}")
            print(f"Saved avg_f1 for fold {fold + 1} to {fold_best_model_path.replace('best_model.pt', 'avg_f1.txt')}")
