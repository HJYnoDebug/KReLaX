import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径与对应分词器
model_paths = {
    "../checkpoint_kfold/xlm-roberta-test/fold_1/xlm-r1.pt": "xlm-roberta-base",
    "../checkpoint_kfold/labse-test/fold_1/labse_fold_1.pt": "sentence-transformers/LaBSE",
    "../checkpoint_kfold/labse-test/fold_2/labse_fold_2.pt": "sentence-transformers/LaBSE",
    "../checkpoint_kfold/labse-test/fold_3/labse_fold_3.pt": "sentence-transformers/LaBSE",
    "../checkpoint_kfold/rembert-test/fold_1/rembert_fold_1.pt": "google/rembert",
    "../checkpoint_kfold/rembert-test/fold_2/rembert_fold_2.pt": "google/rembert",
    "../checkpoint_kfold/rembert-test/fold_3/rembert_fold_3.pt": "google/rembert",
}

# 定义统一标签
universal_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# 读取avgf1值
def read_avgf1(model_path):
    avgf1_file = os.path.join(os.path.dirname(model_path), 'avg_f1.txt')
    if os.path.exists(avgf1_file):
        with open(avgf1_file, 'r') as f:
            avgf1_value = float(f.read().strip())
        return avgf1_value**2
    else:
        raise FileNotFoundError(f"Error: {avgf1_file} not found.")

# 预测函数
def predict(model_paths, input_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    datasets = {}
    for input_file in input_files:
        data = pd.read_csv(input_file)
        datasets[input_file] = data
        for label in universal_labels:
            if label not in data.columns:
                data[label] = np.nan

    all_predictions = {input_file: [] for input_file in input_files}
    models = {}
    tokenizers = {}
    model_weights = {}

    for model_path, tokenizer_name in model_paths.items():
        print(f"Loading model and tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=len(universal_labels)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        models[model_path] = model
        tokenizers[model_path] = tokenizer

        try:
            avgf1_score = read_avgf1(model_path)
            model_weights[model_path] = avgf1_score
        except FileNotFoundError as e:
            print("error!")
            print(e)
            continue

    if not model_weights:
        raise ValueError("No valid model weights found. Please check your model files.")

    # 计算平均权重
    weights = list(model_weights.values())
    w_avg = sum(weights) / len(weights)

    for input_file, data in datasets.items():
        test_texts = data['text'].tolist()
        model_predictions = {model_path: [] for model_path in model_paths}
        ids = data['id'].tolist()

        for model_path, model in models.items():
            tokenizer = tokenizers[model_path]
            model_pred = []

            for text in test_texts:
                encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**encoding).logits
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                    model_pred.append(probabilities)

            model_predictions[model_path] = np.squeeze(model_pred)

        # 计算加权预测
        total_preds = None
        for model_path, model_pred in model_predictions.items():
            weight = model_weights[model_path]
            if total_preds is None:
                total_preds = model_pred * weight
            else:
                total_preds += model_pred * weight

        final_probs = total_preds / (w_avg * len(model_weights))  # 归一化公式
        print(f"Final probabilities (before rounding): {final_probs}")  # Debug 输出

        final_predictions = (final_probs > 0.5).astype(int)

        pred_df = pd.DataFrame(final_predictions, columns=universal_labels)
        pred_df.insert(0, 'id', ids)

        output_file = os.path.join(output_dir, f"pred_{os.path.basename(input_file)}")
        print(f"Saving predictions to {output_file}")
        pred_df.to_csv(output_file, index=False)

    for model_path in models:
        del models[model_path]
        del tokenizers[model_path]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    all_files = [f for f in os.listdir("../test_data/track_c") if f.endswith(".csv")]
    input_files = [os.path.join("../test_data/track_c", f) for f in all_files]

    output_dir = os.path.abspath("../test_data/c_pred")
    print("Input files:", input_files)
    predict(model_paths, input_files, output_dir)
