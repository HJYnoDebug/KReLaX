import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
import random
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianTokenizer, MarianMTModel
import torch

rp_model_name = "t5-base"  # 可以使用更大的模型，如 "t5-base" 或 "t5-large"
rp_model = T5ForConditionalGeneration.from_pretrained(rp_model_name)
rp_tokenizer = T5Tokenizer.from_pretrained(rp_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rp_model = rp_model.to(device)

import pandas as pd

def del_dup(data_path, output_path):
    """
    Clean the dataset by performing the following:
    1. Remove rows with missing values (NaN) in any column.
    2. Remove rows where the 'text' column has duplicate values.
    3. Remove rows where the 'text' column contains fewer than two words.

    Args:
        data_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.

    Returns:
        None
    """
    # Load data
    data_df = pd.read_csv(data_path)
    print("Original data shape:")
    print(data_df.shape)

    # 1. Remove rows with any missing values (NaN)
    data_df = data_df.dropna()

    # 2. Remove rows where 'text' column has duplicate values
    data_df = data_df.drop_duplicates(subset=['text'])

    # 3. Remove rows where 'text' column has fewer than two words
    data_df = data_df[data_df['text'].apply(lambda x: len(str(x).split()) >= 2)]

    # Save the cleaned data to the specified output path
    data_df.to_csv(output_path, index=False)

    print("Cleaned data shape:")
    print(data_df.shape)
    print("Cleaned data saved to:", output_path)



def vis(data_path, label_list):
    """
    Visualize the distribution of emotion labels and combinations in the data.

    Args:
        data_path (str): Path to the input CSV file.
        label_list (list): List of emotion label column names.

    Returns:
        None
    """
    # Load data
    data = pd.read_csv(data_path)

    # Calculate the distribution of each emotion label
    emotion_distribution = data[label_list].sum() / len(data)
    # Combine labels to represent each unique combination in the dataset
    data['Combination'] = data[label_list].astype(str).agg(''.join, axis=1)

    # Count occurrences of each unique label combination
    combination_counts = data['Combination'].value_counts()
    # Calculate the ratio of each combination
    combination_ratios = combination_counts / len(data)

    # Print emotion distribution and combination count
    print("Emotion distribution:")
    print(emotion_distribution)
    print("Number of combinations:")
    print(combination_counts)

    # Plot bar chart for emotion label distribution
    plt.figure(figsize=(8, 6))
    emotion_distribution.plot(kind='bar')
    plt.xlabel('Labels')
    plt.ylabel('Sample Count')
    plt.title('Emotion Label Distribution')
    plt.show()

    # Plot pie chart for label combination ratios
    plt.figure(figsize=(8, 8))
    # Create explodes for small ratios to highlight them
    explodes = [0.1 if ratio < 0.05 else 0 for ratio in combination_ratios]

    plt.pie(combination_ratios,
            labels=combination_ratios.index,
            autopct='%1.2f%%',  # Display percentage
            startangle=90,  # Start pie chart at 90 degrees
            explode=explodes,  # Slightly separate small ratio sections
            pctdistance=0.85,  # Position of percentage labels
            wedgeprops={'edgecolor': 'black'})  # Add edge color for clarity

    plt.title('Proportion of Label Combinations', pad=20)
    plt.axis('equal')  # Ensure pie chart is circular
    plt.show()


def chi_square_test(df, labels_list):
    """
    Perform chi-square test between each pair of labels in the list.

    Args:
        df (pd.DataFrame): Input data containing the labels.
        labels_list (list): List of label column names.

    Returns:
        dict: A dictionary with label pairs as keys and chi-square results as values.
    """
    results = {}
    # Iterate over each unique pair of labels
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            label1 = labels_list[i]
            label2 = labels_list[j]

            # Create contingency table for the pair of labels
            contingency_table = pd.crosstab(df[label1], df[label2])

            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Store chi-square statistic and p-value in results
            results[(label1, label2)] = {'chi2': chi2, 'p-value': p}

    return results


def mutual_information_score(df, labels_list):
    """
    Compute mutual information score between each pair of labels in the list.

    Args:
        df (pd.DataFrame): Input data containing the labels.
        labels_list (list): List of label column names.

    Returns:
        dict: A dictionary with label pairs as keys and mutual information scores as values.
    """
    results = {}
    # Iterate over each unique pair of labels
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            label1 = labels_list[i]
            label2 = labels_list[j]

            # Calculate mutual information score for the pair of labels
            mi = mutual_info_score(df[label1], df[label2])

            # Store mutual information score in results
            results[(label1, label2)] = mi

    return results


def rephrase_with_t5(text):
    """
    Rephrase the input text using the T5 model.

    Parameters:
    - text (str): Input text.

    Returns:
    - str: Rephrased text, or None if an error occurs.
    """
    input_text = f"paraphrase: {text}"  # Instruction for paraphrasing
    input_ids = rp_tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = rp_model.generate(input_ids,max_length=512,num_beams=5,early_stopping=True)
    rephrased_text = rp_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rephrased_text


from transformers import MarianMTModel, MarianTokenizer

# 加载英中和中英翻译模型及其tokenizer
en_to_zh_model_name = 'Helsinki-NLP/opus-mt-en-zh'  # 英语到中文
zh_to_en_model_name = 'Helsinki-NLP/opus-mt-zh-en'  # 中文到英语

en_to_zh_tokenizer = MarianTokenizer.from_pretrained(en_to_zh_model_name)
en_to_zh_model = MarianMTModel.from_pretrained(en_to_zh_model_name)

zh_to_en_tokenizer = MarianTokenizer.from_pretrained(zh_to_en_model_name)
zh_to_en_model = MarianMTModel.from_pretrained(zh_to_en_model_name)


def translate_back(text, src_lang='en', intermediate_lang='zh'):
    """
    Perform back-translation to generate an augmented version of the input text.

    Parameters:
    - text (str): Input text.
    - src_lang (str): Original language of the input text (default is 'en').
    - intermediate_lang (str): Language to translate to before translating back (default is 'zh').

    Returns:
    - str: Back-translated text, or None if an error occurs.
    """
    # 第一步：将原文翻译成中介语言（如中文）
    if src_lang == 'en' and intermediate_lang == 'zh':
        translated = en_to_zh_tokenizer.encode(text, return_tensors="pt", truncation=True, padding=True)
        translated_text = en_to_zh_model.generate(translated, max_length=512, num_beams=5, early_stopping=True)
        intermediate_text = en_to_zh_tokenizer.decode(translated_text[0], skip_special_tokens=True)
    else:
        raise ValueError("Unsupported language pair for translation.")

    # 第二步：将中介语言翻译回源语言（如从中文翻译回英语）
    if intermediate_lang == 'zh' and src_lang == 'en':
        back_translated = zh_to_en_tokenizer.encode(intermediate_text, return_tensors="pt", truncation=True,
                                                    padding=True)
        back_translated_text = zh_to_en_model.generate(back_translated, max_length=512, num_beams=5,
                                                       early_stopping=True)
        back_translated = zh_to_en_tokenizer.decode(back_translated_text[0], skip_special_tokens=True)

    return back_translated


def augment_text(df, num_augments=2):
    """
    Generate augmented versions of the input text using rephrasing and back-translation while retaining labels.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'text' and labels.
    - num_augments (int): Number of augmented texts to generate.

    Returns:
    - pd.DataFrame: DataFrame of augmented text samples with labels.
    """
    augmented_samples = []  # List to hold augmented samples

    methods = [rephrase_with_t5, translate_back]
    #methods = [rephrase_with_t5]
    for _ in range(num_augments):
        for idx, row in df.iterrows():  # Iterate through each row in the DataFrame
            text = row['text']
            labels = row[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].tolist()  # Get the label values as list
            method = random.choice(methods)

            # Ensure the text is within the size limit
            if len(text) > 5000:
                text_chunks = [text[i:i + 5000] for i in range(0, len(text), 5000)]  # Split text into chunks
                augmented_text = ' '.join(
                    [method(chunk) for chunk in text_chunks if method(chunk)])  # Combine augmented chunks
            else:
                # Use method directly on text
                augmented_text = method(text)

            # Only add the augmented sample if it's not None (i.e., no error occurred)
            if augmented_text:
                augmented_samples.append([augmented_text] + labels)  # Add augmented text and labels

    # Create DataFrame from augmented samples
    augmented_df = pd.DataFrame(augmented_samples, columns=['text', 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
    return augmented_df
