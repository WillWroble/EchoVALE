"""EchoFocus mean-pool for CHD classification."""

import torch.nn as nn


class EchoFocus(nn.Module):
    def __init__(self, input_dim=768, n_heads=12, ff_dim=768, dropout=0.1, n_targets=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, n_targets)

    def forward(self, x):
        h = self.encoder(x)
        h = h.mean(dim=1)
        return self.head(self.norm(h))
