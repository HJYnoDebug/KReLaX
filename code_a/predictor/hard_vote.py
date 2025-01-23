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

# 硬投票函数
def hard_majority_vote(predictions):
    # 对于每个类别，选择投票最多的类别
    return np.round(np.mean(predictions, axis=0)).astype(int)

# 读取avgf1值
def read_avgf1(model_path):
    # 获取对应的avg_f1.txt路径
    avgf1_file = os.path.join(os.path.dirname(model_path), 'avg_f1.txt')
    
    # 检查文件是否存在
    if os.path.exists(avgf1_file):
        with open(avgf1_file, 'r') as f:
            avgf1_value = float(f.read().strip())
        return avgf1_value
    else:
        raise FileNotFoundError(f"Error: {avgf1_file} not found.")

# 预测函数
def predict(model_paths, input_files, output_dir):
    # 确保输出目录存在
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

    # 加载所有模型和分词器
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

        # 读取模型的 avgf1 权重
        avgf1_score = read_avgf1(model_path)
        model_weights[model_path] = avgf1_score ** 3  # 使用avgf1的平方作为权重

    # 预测所有文件
    for input_file, data in datasets.items():
        test_texts = data['text'].tolist()
        model_predictions = {model_path: [] for model_path in model_paths}
        true_labels = data[universal_labels].values
        ids = data['id'].tolist()  # 获取id列

        # 轮流对所有模型进行预测
        for model_path, model in models.items():
            tokenizer = tokenizers[model_path]
            model_pred = []

            for text in test_texts:
                encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**encoding).logits
                    # 对logits进行sigmoid转换，得到概率
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                    model_pred.append(probabilities)

            model_predictions[model_path] = np.squeeze(model_pred)

        # 进行硬投票
        all_model_preds = []
        for model_path, model_pred in model_predictions.items():
            all_model_preds.append(model_pred)

        # 硬投票并生成最终预测
        final_predictions = hard_majority_vote(np.array(all_model_preds))
        print(f"Final predictions (hard vote): {final_predictions}")  # Debug 输出

        # 生成最终的预测标签
        pred_df = pd.DataFrame(final_predictions, columns=universal_labels)
        pred_df.insert(0, 'id', ids)  # 保留原来的id列

        # 不输出text列
        pred_df.drop(columns=['id', 'text'], inplace=True, errors='ignore')

        # 保存预测结果
        output_file = os.path.join(output_dir, f"pred_{os.path.basename(input_file)}")
        print(f"Saving predictions to {output_file}")
        pred_df.to_csv(output_file, index=False)

    # 清理内存
    for model_path in models:
        del models[model_path]
        del tokenizers[model_path]
    torch.cuda.empty_cache()


# 主函数
if __name__ == "__main__":
    all_files = [f for f in os.listdir("../test_data/track_a") if f.endswith(".csv")]
    input_files = [os.path.join("../test_data/track_a", f) for f in all_files]  # 拼接完整路径

    output_dir = os.path.abspath("../test_data/a_pred")  # 确保输出目录的绝对路径
    print("Input files:", input_files)
    predict(model_paths, input_files, output_dir)
