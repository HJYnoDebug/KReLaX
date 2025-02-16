import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class MultiLabelMultiClassModel(nn.Module):
    def __init__(self, num_labels, num_classes=4, model_name='sentence-transformers/LaBSE'):
        """
        :param num_labels: Number of labels (tasks) for multi-label classification.
        :param num_classes: Number of classes for each label.
        :param model_name: Huggingface pretrained model name (e.g., 'google/rembert', 'xlm-roberta-base', 'sentence-transformers/LaBSE')
        """
        super(MultiLabelMultiClassModel, self).__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes

        # Load pretrained model and tokenizer dynamically based on model_name
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Custom classification heads for each label
        # You can also add more layers in each head for more complex transformations
        self.classifiers = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, num_classes)
            for _ in range(num_labels)
        ])

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        :param input_ids: Tensor of shape (batch_size, seq_len).
        :param attention_mask: Tensor of shape (batch_size, seq_len).
        :return: Logits of shape (batch_size, num_labels, num_classes).
        """
        # Pass inputs through the pre-trained model (e.g., LaBSE, XLM-R, Rembert)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract pooled output (e.g., [CLS] token embedding)
        # Here we assume we use the [CLS] token output for classification, but you can modify this
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token (batch_size, hidden_size)

        # Apply each classification head for each label (task)
        logits = torch.stack(
            [classifier(pooled_output) for classifier in self.classifiers], dim=1
        )  # Shape: (batch_size, num_labels, num_classes)

        return logits
