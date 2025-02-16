import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    """
    A custom PyTorch dataset for multi-label multi-class classification.

    Args:
        texts (list of str): List of input text samples.
        labels (list of list of int): Two-dimensional list of labels, each label being a list of integer values (0, 1, 2, 3).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        max_len (int): Maximum length of tokenized text sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # List of label lists (integers for multi-class classification)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a single sample of tokenized text and its corresponding labels.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels tensors for the model.
        """
        # Get text and label for the specified index
        text = self.texts[idx]
        label = self.labels[idx]  # List of integers for each label

        # Tokenize the text using the tokenizer and set configuration
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Include special tokens (e.g., [CLS] and [SEP])
            max_length=self.max_len,  # Truncate or pad to max_len
            return_token_type_ids=False,  # Token type ids not needed for this task
            padding='max_length',  # Pad sequences to max_len
            return_attention_mask=True,  # Include attention mask
            return_tensors='pt',  # Return tensors in PyTorch format
            truncation=True,  # Truncate if the text exceeds max_len
        )

        # Ensure label is a tensor of the correct shape
        label_tensor = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor of integers
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove the extra dimension
            'attention_mask': encoding['attention_mask'].squeeze(),  # Remove the extra dimension
            'labels': label_tensor  # Convert label to a tensor of type long
        }
