import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Dataset
class ATISDataset(Dataset):
    def __init__(self, X, y, attention_mask):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx],
            "attention_mask": self.attention_mask[idx],
        }


# BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        pad_id: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, T)
        attention_mask: (B, T)  1 = valid token, 0 = pad
        """
        embeds = self.embedding(input_ids)  # (B, T, E)
        lstm_out, _ = self.lstm(embeds)    # (B, T, 2H)

        # use attention_mask to get the last not PAD hidden state
        lengths = attention_mask.sum(dim=1)         # (B,)
        lengths = torch.clamp(lengths, min=1)       # avoid 0

        B = input_ids.size(0)
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, lstm_out.size(2))
        last_hidden = lstm_out.gather(1, idx).squeeze(1)  # (B, 2H)

        out = self.dropout(last_hidden)
        logits = self.fc(out)  # (B, C)
        return logits

