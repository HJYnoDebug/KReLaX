import torch
import numpy as np
'''
class EmotionDataset(Dataset):
    """
    A custom PyTorch dataset for handling emotion classification data.

    Args:
        texts (list of str): List of input text samples.
        labels (list of list of int): Two-dimensional list of labels, each label being a list of binary values.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        max_len (int): Maximum length of tokenized text sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # List of label lists (binary arrays for multi-label classification)
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
            dict: A dictionary containing input_ids, attention_mask, labels tensors, and weight.
        """
        # Get text and label for the specified index
        text = self.texts[idx]
        label = self.labels[idx]  # Single label as a binary array

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

        # Check if the label is all zeroes (i.e., label == [0, 0, 0, ..., 0])
        weight = 1.0 if torch.sum(torch.tensor(label)) > 0 else 0.1  # Low weight for all-zero labels
        #weight = 1.0 if torch.sum(label) > 0 else 0.1  # Low weight for all-zero labels


        return {
            'input_ids': encoding['input_ids'].flatten(),  # Flatten to a 1D tensor of input ids
            'attention_mask': encoding['attention_mask'].flatten(),  # Flatten to a 1D tensor of attention mask
            'labels': torch.tensor(label, dtype=torch.float),  # Convert label to a tensor of type float
            'weight': torch.tensor(weight, dtype=torch.float)  # Return weight as a tensor
        }'''

from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    """
    Optimized PyTorch dataset for emotion classification.

    Args:
        texts (list of str): List of input text samples.
        labels (list of list of int): Two-dimensional list of labels, each label being a list of binary values.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        max_len (int): Maximum length of tokenized text sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.max_len = max_len

        # Pre-tokenize texts to save time during training
        self.encodings = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        labels = np.array(labels)

        # Now, convert the NumPy array to a tensor
        self.labels = torch.tensor(labels, dtype=torch.float)
        # Convert labels to tensors
        #self.labels = torch.tensor(labels, dtype=torch.float)

        # Pre-compute weights for each sample
        self.weights = torch.where(self.labels.sum(dim=1) > 0, torch.tensor(1.0), torch.tensor(0.1))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a single sample of pre-encoded text and its corresponding labels.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, labels tensors, and weight.
        """
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
            'weight': self.weights[idx],
        }

