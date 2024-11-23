import torch
from torch import nn
from transformers import RobertaModel


class TextCNN(nn.Module):
    """
    TextCNN model that combines RoBERTa embeddings with convolutional layers for multi-class classification.

    Args:
        roberta_model_name (str): Name of the pre-trained RoBERTa model to use.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(self, roberta_model_name, num_classes):
        super(TextCNN, self).__init__()

        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)

        # Define convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, self.roberta.config.hidden_size)) for k in [3, 4, 5]
        ])
        
        # Dropout layer
        self.drop_rate = 0.5
        self.dropout = nn.Dropout(self.drop_rate)

        # Fully connected layer for classification
        self.fc = nn.Linear(300, num_classes)  # 3 different kernel sizes with 100 output channels each

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs.
            attention_mask (torch.Tensor): Tensor containing attention masks.

        Returns:
            torch.Tensor: Output logits for each class.
        """
        # Get outputs from RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Add a channel dimension for convolutional layers
        x = outputs.last_hidden_state.unsqueeze(1)  # Shape: (batch_size, 1, seq_length, hidden_size)

        # Convolution and pooling operations
        conv_outs = [torch.relu(conv(x)).squeeze(3) for conv in
                     self.convs]  # Apply conv layers and remove last dimension
        pooled_outs = [torch.max(out, dim=2)[0] for out in conv_outs]  # Perform max pooling over the sequence dimension

        # Concatenate pooled outputs from different kernel sizes
        out = torch.cat(pooled_outs, dim=1)  # Shape: (batch_size, 300)

         #dropout
        out = self.dropout(out)

        # Pass through the fully connected layer to get class logits
        return self.fc(out)
