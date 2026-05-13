"""LineEncoder + CrossAttentionPool for video-attended skip-gram training."""

import torch
import torch.nn as nn
from transformers import AutoModel


class LineEncoder(nn.Module):

    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.encoder.layer[-2].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)


class CrossAttentionPool(nn.Module):
    """Per-line cross-attention over a study's videos"""

    def __init__(self, dim=768):
        super().__init__()
        self.W_Q = nn.Linear(768, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
        self.proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        

    def forward(self, lines, videos, video_mask):
        """
        lines:      (B, L, D)
        videos:     (B, V, D)
        video_mask: (B, V) — 1 for real, 0 for pad
        returns:    (B, L, D) attended pool per line, in raw video space
        """
        videos = self.proj(videos)
        Q = self.W_Q(lines)
        K = self.W_K(videos)
        #V = videos
        V = self.W_V(videos)
        scores = torch.einsum("bld,bvd->blv", Q, K) * self.scale
        mask = video_mask.unsqueeze(1) == 0
        scores = scores.masked_fill(mask, -1e9)
        weights = scores.softmax(dim=-1)
        out = torch.einsum("blv,bvd->bld", weights, V)
        return out
        #return self.proj(out)
