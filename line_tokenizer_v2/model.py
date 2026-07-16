"""LineEncoder + CrossAttentionPool for video-attended skip-gram training."""

import torch
import torch.nn as nn
from transformers import AutoModel


class LineEncoder(nn.Module):

    #"emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, model_name="michiyasunaga/BioLinkBERT-large", dim=1024):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.encoder.layer[-2].parameters():
            param.requires_grad = True
        #for param in self.bert.encoder.layer[-3].parameters():
            #param.requires_grad = True
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )


    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)


class QuerySAPool(nn.Module):
    """Prepend query token to clips, run SA, classify query output."""

    def __init__(self, dim=1024, num_heads=16, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=2*dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        #self.query = nn.Parameter(torch.zeros(1, 1, dim))


    def forward(self, lines, videos, video_mask):
        B, L, D = lines.shape
        V = videos.shape[1]

        # (B*L, 1+V, D)
        queries = lines.reshape(B * L, 1, D)
        #queries = self.query.expand(B * L, 1, D)
        videos_exp = videos.unsqueeze(1).expand(B, L, V, D).reshape(B * L, V, D)
        seq = torch.cat([queries, videos_exp], dim=1)

        # mask: query always valid, clips follow video_mask
        clip_mask = video_mask.unsqueeze(1).expand(B, L, V).reshape(B * L, V)
        pad_mask = torch.cat([clip_mask.new_zeros(B * L, 1), 1 - clip_mask], dim=1).bool()

        out = self.encoder(seq, src_key_padding_mask=pad_mask)
        #return out[:, 0].view(B, L, D)
        return self.head(out[:, 0]).view(B, L, 1)

class CrossAttentionPool(nn.Module):
    def __init__(self, dim=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # clip self-attention
        
        #self.clip_sa = nn.TransformerEncoderLayer(
        #    d_model=dim, nhead=num_heads, dim_feedforward=4*dim,
        #    batch_first=True, norm_first=True,
        #)
        
        # cross-attention
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim)

        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1),
        )
        #self.V_proj = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, 4*dim),
        #    nn.GELU(),
        #    nn.Linear(4*dim, dim),
        #)

    def forward(self, lines, videos, video_mask):
        B, L, _ = lines.shape

        
        V = videos.shape[1]
        h, d = self.num_heads, self.head_dim
        
        # self-attend over clips
        #videos = self.clip_sa(videos, src_key_padding_mask=(video_mask == 0))
        
        #videos = self.V_proj(videos)
        
        # cross-attention
        Q = self.W_Q(lines).view(B, L, h, d).transpose(1, 2)   # (B, h, L, d)
        K = self.W_K(videos).view(B, V, h, d).transpose(1, 2)   # (B, h, V, d)
        Vs = self.W_V(videos).view(B, V, h, d).transpose(1, 2)  # (B, h, V, d)

        scores = torch.einsum("bhld,bhvd->bhlv", Q, K) * self.scale
        mask = video_mask[:, None, None, :] == 0                 # (B, 1, 1, V)
        scores = scores.masked_fill(mask, -1e9)
        weights = scores.softmax(dim=-1)
        
        #mask = video_mask[:, None, :, None] == 1                  # (B, 1, V, 1)
        #Vs = Vs*mask

        out = torch.einsum("bhlv,bhvd->bhld", weights, Vs)
        
        #out = Vs.sum(2) / video_mask.sum(1).view(B, 1, 1) #(B, h, d)
        #out = out.unsqueeze(2)
        #out = out.expand(-1, -1, L, -1) #(B,h,L,d)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)   # (B, L, dim)

        return self.proj(out)
