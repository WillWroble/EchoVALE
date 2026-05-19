"""Text probe: transformer over line embeddings → measurement regression."""

import torch
import torch.nn as nn


class TextProbe(nn.Module):
    def __init__(self, dim=768, num_heads=12, num_targets=33):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=3072,
            batch_first=True, norm_first=True,dropout=0.1,
        )
        self.head = nn.Linear(dim, num_targets)

    def forward(self, x, pad_mask):
        # x: (B, N, D), pad_mask: (B, N) True=valid
        x = self.encoder(x, src_key_padding_mask=~pad_mask)
        w = pad_mask.unsqueeze(-1).float()
        x = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
        return self.head(x)
