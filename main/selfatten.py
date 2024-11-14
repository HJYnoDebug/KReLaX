import torch
from torch import nn

class RobertaWithAttention(nn.Module):
    def __init__(self, roberta, num_labels, num_heads, hidden_dim):
        super(RobertaWithAttention, self).__init__()
        self.roberta = roberta
        self.attention = nn.MultiheadAttention(embed_dim=roberta.config.hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(roberta.config.hidden_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = roberta_output.last_hidden_state.transpose(0, 1)  # Switch to [seq_len, batch_size, hidden_dim]
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = attn_output.transpose(0, 1).mean(dim=1)  # Average over sequence dimension
        x = torch.relu(self.fc(attn_output))
        logits = self.out(x)
        return logits