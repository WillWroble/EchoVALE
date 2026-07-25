"""Clip encoder utilities for video_pretraining_v3 UMAP probes.

Loads ViT-L + trained AttentivePooler and provides encode_video
matching the API of JEPA_probes/encoder.py.
"""

import sys
sys.path.insert(0, '/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes')

import torch
import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentivePooler
from encoder import load_and_sample_clips, _clean_backbone_key


def load_clip_encoder(encoder_ckpt, v3_ckpt, device='cuda',
                      model_name='vit_large', embed_dim=1024,
                      num_heads=16, depth=4):
    encoder_fn = getattr(vit, model_name)
    encoder = encoder_fn(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('target_encoder') or ckpt.get('encoder')
    state = _clean_backbone_key(state)
    msg = encoder.load_state_dict(state, strict=False)
    print(f"  encoder: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}", flush=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    pooler = AttentivePooler(
        num_queries=1, embed_dim=embed_dim,
        num_heads=num_heads, depth=depth,
    )
    v3 = torch.load(v3_ckpt, map_location='cpu', weights_only=False)
    pooler_state = {k.replace('module.', ''): v for k, v in v3['pooler'].items()}
    pooler.load_state_dict(pooler_state)
    pooler.to(device).eval()
    for p in pooler.parameters():
        p.requires_grad_(False)

    print(f"  pooler: {sum(p.numel() for p in pooler.parameters()) / 1e6:.1f}M params",
          flush=True)
    return encoder, pooler


@torch.no_grad()
def encode_video(encoder, pooler, avi_path, device='cuda'):
    clips = load_and_sample_clips(avi_path)
    if clips is None:
        return None
    clips = clips.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        patches = encoder(clips)
        clip_embs = pooler(patches).squeeze(1)
        emb = clip_embs.mean(dim=0)
    return emb.float().cpu().numpy()
