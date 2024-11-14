import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer ,RobertaModel
from textcnn import TextCNN  # 需要根据您的项目导入实际的 TextCNN 类
from dataset import EmotionDataset  # 需要根据您的项目导入实际的 EmotionDataset 类
import hyperparameters as hp
from selfatten import RobertaWithAttention

# Path to model checkpoints
textcnn_model_path = 'checkpoint/textcnn_checkpoint.pt'
selfattention_model_path = 'checkpoint/selfatten_checkpoint.pt'
roberta_model_path = 'checkpoint/roberta_checkpoint.pt'
output_path = 'data/dev data/eng_dev_pred.csv'
data_path = 'data/dev data/eng_dev.csv'

# Hyperparameters
hyperparams = hp.Hyperparameters()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')

# Load test data
test_data = pd.read_csv(data_path)
test_texts = test_data['text'].tolist()
test_labels = test_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()

# Prepare dataset and dataloaders
# Set test_labels to None because it's for inference (no labels available)
test_dataset = EmotionDataset(test_texts, labels=test_labels, tokenizer=tokenizer, max_len=hyperparams.max_len)
test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. TextCNN model loading
textcnn_model = TextCNN(roberta_model_name='roberta-base', num_classes=5).to(device)
textcnn_model.load_state_dict(torch.load(textcnn_model_path, map_location=device))
textcnn_model.eval()

# 2. RobertaWithAttention model loading
attnmodel = RobertaWithAttention(roberta,num_labels=5, num_heads=4, hidden_dim=128).to(device)
attnmodel.load_state_dict(torch.load(selfattention_model_path, map_location=device))
attnmodel.eval()

# 3. Roberta model loading
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
roberta_model.load_state_dict(torch.load(roberta_model_path, map_location=device))
roberta_model.to(device)
roberta_model.eval()

# Predicting with each model
def predict_with_model(model, dataloader):
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Perform prediction
            outputs = model(input_ids, attention_mask=attention_mask)

            # Check if the model has a 'logits' attribute (for RobertaForSequenceClassification)
            if hasattr(outputs, 'logits'):
                predictions = torch.sigmoid(outputs.logits)  # Sigmoid for multi-label classification
            else:
                predictions = torch.sigmoid(outputs)  # For TextCNN and other models without 'logits'

            binary_predictions = (predictions > 0.5).int().cpu().tolist()
            all_predictions.extend(binary_predictions)
    return all_predictions

# Get predictions from each model
roberta_predictions = predict_with_model(roberta_model, test_loader)
textcnn_predictions = predict_with_model(textcnn_model, test_loader)
attnmodel_predictions = predict_with_model(attnmodel, test_loader)

# Majority voting mechanism for combining predictions
def majority_vote(predictions_list):
    predictions = torch.stack([torch.tensor(pred) for pred in predictions_list])
    final_predictions = predictions.sum(dim=0) > (len(predictions_list) / 2)  # Majority vote
    return final_predictions.int().tolist()

# Get final predictions via majority vote
final_predictions = majority_vote([roberta_predictions, textcnn_predictions, attnmodel_predictions])

# Prepare results and save to CSV
predictions_df = pd.DataFrame(final_predictions, columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
id_df = pd.DataFrame(test_data['id'].tolist(), columns=['id'])
result_df = pd.concat([id_df, predictions_df], axis=1)

result_df.to_csv(output_path, index=False)
print(f'Predictions saved to {output_path}')