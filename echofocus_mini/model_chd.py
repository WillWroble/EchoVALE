"""EchoFocus mean-pool for CHD classification."""

import torch.nn as nn


class EchoFocus(nn.Module):
    def __init__(self, input_dim=1024, n_heads=16, ff_dim=1024, dropout=0.1, n_targets=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, n_targets)
    def forward(self, x, mask=None):
        h = self.encoder(x, src_key_padding_mask=mask)
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(self.norm(h))
