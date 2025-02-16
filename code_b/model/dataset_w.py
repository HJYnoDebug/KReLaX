import torch
from torch.utils.data import Dataset
import numpy as np  # 使用 NumPy 提升效率


class EmotionDataset(Dataset):
    """
    A custom PyTorch dataset for multi-label multi-class classification with weight adjustments.

    Args:
        texts (list of str): List of input text samples.
        labels (list of list of int): Two-dimensional list of labels, each label being a list of integer values (0, 1, 2, 3).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        max_len (int): Maximum length of tokenized text sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = np.array(labels)  # Convert labels to NumPy array for vectorized operations
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Precompute weights based on labels (vectorized operation)
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """
        Computes the weight for each sample in the dataset using a vectorized approach.

        Returns:
            np.ndarray: Array of weights for each sample.
        """
        # Check if each row in the label matrix is all zeros
        all_zero_mask = np.all(self.labels == 0, axis=1)  # Vectorized operation
        weights = np.where(all_zero_mask, 0.1, 1.0)  # Assign weights: 0.1 for all-zero rows, 1.0 otherwise
        return weights

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a single sample of tokenized text, its corresponding labels, and the weight.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, labels, and weight tensors for the model.
        """
        # Get text, label, and weight for the specified index
        text = self.texts[idx]
        label = self.labels[idx]  # NumPy array for the specific label
        weight = self.weights[idx]

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

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove the extra dimension
            'attention_mask': encoding['attention_mask'].squeeze(),  # Remove the extra dimension
            'labels': torch.tensor(label, dtype=torch.long),  # Convert label to a tensor of type long
            'weight': torch.tensor(weight, dtype=torch.float)  # Add the weight as a float tensor
        }
