import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os,sys
import concurrent.futures

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # 修正这里的 '..' 指向项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)

    
from model.model_head import MultiLabelMultiClassModel
# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model paths and their corresponding tokenizers
model_paths = {
    "checkpoint_kfold/xlm-roberta/fold_1/best_model.pt": "xlm-roberta-base",
    "checkpoint_kfold/xlm-roberta/fold_2/best_model.pt": "xlm-roberta-base",
    "checkpoint_kfold/xlm-roberta/fold_3/best_model.pt": "xlm-roberta-base",
    "checkpoint_kfold/labse/fold_1/best_model.pt": "sentence-transformers/LaBSE",
    "checkpoint_kfold/labse/fold_2/best_model.pt": "sentence-transformers/LaBSE",
    "checkpoint_kfold/labse/fold_3/best_model.pt": "sentence-transformers/LaBSE",
    "checkpoint_kfold/rembert/fold_1/best_model.pt": "google/rembert",
    "checkpoint_kfold/rembert/fold_2/best_model.pt": "google/rembert",
    "checkpoint_kfold/rembert/fold_3/best_model.pt": "google/rembert",
}

# Define a universal set of labels for classification
universal_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Function to read the avg_f1 score from a text file
def read_avgf1(model_path):
    avgf1_file = os.path.join(os.path.dirname(model_path), 'avg_f1.txt')
    if os.path.exists(avgf1_file):
        with open(avgf1_file, 'r') as f:
            avgf1_value = float(f.read().strip())
        return avgf1_value
    else:
        raise FileNotFoundError(f"Error: {avgf1_file} not found.")

# Function to predict for a single model
def predict_for_model(model, tokenizer, test_texts, device):
    model_pred = []
    for text in test_texts:
        encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask']).cpu().numpy()  # Custom forward pass
            probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # Apply softmax for probabilities
            model_pred.append(probabilities)
    return np.squeeze(model_pred)

# Main prediction function
def predict(model_paths, input_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    # Load datasets and prepare placeholders for predictions
    datasets = {}
    for input_file in input_files:
        data = pd.read_csv(input_file)
        datasets[input_file] = data
        for label in universal_labels:
            if label not in data.columns:
                data[label] = np.nan  # Add missing label columns if not present

    # Initialize containers for predictions, models, and weights
    all_predictions = {input_file: [] for input_file in input_files}
    models = {}
    tokenizers = {}
    model_weights = {}

    # Load models, tokenizers, and corresponding weights
    for model_path, tokenizer_name in model_paths.items():
        print(f"Loading model and tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Initialize custom multi-label multi-class model
        model=torch.load(model_path, map_location=device).to(device)
        model.eval()

        models[model_path] = model
        tokenizers[model_path] = tokenizer

        # Read the average F1 score to use as weight
        try:
            avgf1_score = read_avgf1(model_path)
            model_weights[model_path] = avgf1_score
        except FileNotFoundError as e:
            print("Error loading avg_f1 file!")
            print(e)
            continue

    if not model_weights:
        raise ValueError("No valid model weights found. Please check your model files.")

    # Calculate the average weight for normalization
    weights = list(model_weights.values())
    w_avg = sum(weights) / len(weights)

    # Process the input files
    for input_file, data in datasets.items():
        test_texts = data['text'].tolist()
        ids = data['id'].tolist()

        # Use concurrent futures to parallelize predictions
        model_predictions = {model_path: [] for model_path in model_paths}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for model_path, model in models.items():
                tokenizer = tokenizers[model_path]
                futures.append(executor.submit(predict_for_model, model, tokenizer, test_texts, device))

            # Collect all predictions from parallel threads
            for i, future in enumerate(futures):
                model_predictions[list(models.keys())[i]] = future.result()

        # Calculate weighted predictions across all models
        total_preds = None
        for model_path, model_pred in model_predictions.items():
            weight = model_weights[model_path]
            if total_preds is None:
                total_preds = model_pred * weight
            else:
                total_preds += model_pred * weight

        # Normalize the final probabilities
        final_probs = total_preds / (w_avg * len(model_weights))
        final_predictions = np.argmax(final_probs, axis=-1)

        # Ensure the final_predictions are in the same order as universal_labels
        pred_df = pd.DataFrame(final_predictions, columns=universal_labels)
        pred_df.insert(0, 'id', ids)

        output_file = os.path.join(output_dir, f"pred_{os.path.basename(input_file)}")
        print(f"Saving predictions to {output_file}")
        pred_df.to_csv(output_file, index=False)

    # Clean up resources to free memory
    for model_path in models:
        del models[model_path]
        del tokenizers[model_path]
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Get the list of input files
    all_files = [f for f in os.listdir("test_data") if f.endswith(".csv")]
    input_files = [os.path.join("test_data", f) for f in all_files]

    # Specify the output directory for predictions
    output_dir = os.path.abspath("test_data/track_b")
    print("Input files:", input_files)

    # Run the prediction function
    predict(model_paths, input_files, output_dir)
