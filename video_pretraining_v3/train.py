"""Clip-line alignment pretraining (v3).

Train AttentivePooler on frozen JEPA patch tokens jointly with
BioLinkBERT-large line encoder. BCE loss with shared negatives,
MIL-style positive assignment from study-level report lines.

Launch:
    torchrun --standalone --nproc_per_node=4 train.py \
        --encoder_checkpoint .../echojepa_vitl_mimic_bch_cooldown/latest.pt \
        --avi_manifest .../avi_manifest.csv \
        --train_manifest .../manifests/train_modern.txt \
        --val_manifest .../manifests/val_modern.txt \
        --h5_dir .../Echo_Labels \
        --output_dir results/v1
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

# Echo_JEPA (on PYTHONPATH via sbatch)
from src.datasets.video_dataset import VideoDataset
from src.hub.backbones import _clean_backbone_key
from src.models import vision_transformer as vit
from src.models.attentive_pooler import AttentivePooler

# line_tokenizer_v2
sys.path.insert(0, '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2')
from model_SA import LineEncoder


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ---------------------------------------------------------------------------
# DDP
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

class ClipTransform:
    """(T, H, W, 3) uint8 -> (3, T, H, W) float, ImageNet-normalized."""
    def __init__(self, resolution=224):
        self.resolution = resolution

    def __call__(self, buffer):
        x = torch.from_numpy(buffer).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        x = x.permute(1, 0, 2, 3)
        return (x - MEAN) / STD


def clip_collate(batch):
    clips = torch.stack([b[0][0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return clips, labels


def ensure_pretrain_csv(avi_manifest, study_manifest, output_csv):
    output_csv = Path(output_csv)
    keep = set(Path(study_manifest).read_text().split())
    sid_to_int = {}
    lines = []
    with open(avi_manifest) as f:
        for line in f:
            path = line.split(maxsplit=1)[0]
            sid = path.split("/")[-2]
            if sid.endswith("_trim"):
                sid = sid[:-5]
            if sid not in keep:
                continue
            if sid not in sid_to_int:
                sid_to_int[sid] = len(sid_to_int)
            lines.append(f"{path} {sid_to_int[sid]}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n")
    with open(output_csv.with_suffix(".sid_map.txt"), "w") as f:
        for sid, i in sorted(sid_to_int.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{sid}\n")
    print(f"wrote {output_csv}: {len(lines):,} videos, {len(sid_to_int):,} studies", flush=True)


def load_sid_map(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            i, sid = line.rstrip("\n").split("\t")
            mapping[int(i)] = sid
    return mapping


# ---------------------------------------------------------------------------
# Line data
# ---------------------------------------------------------------------------

def merge_soft_wraps(lines):
    if not lines:
        return []
    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if line[0].islower() or prev.endswith('-'):
            sep = '' if prev.endswith('-') else ' '
            merged[-1] = prev.rstrip('-') + sep + line
        else:
            merged.append(line)
    return merged


class LineBank:
    """Study lines, tokenization, and sampling."""

    def __init__(self, h5_dir, study_ids, field='study_findings',
                 subsample_t=1e-3, line_filters=None, K=2):
        self.K = K

        if line_filters:
            patterns = [re.compile(l.strip(), re.IGNORECASE)
                        for l in open(line_filters) if l.strip() and not l.startswith("#")]
        else:
            patterns = []
        keep = (lambda line: not any(p.search(line) for p in patterns)) if patterns else (lambda _: True)

        sid_set = set(study_ids)
        self.study_lines = {}
        with h5py.File(f"{h5_dir}/{field}.h5", "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid not in sid_set:
                    continue
                lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                         for x in f[sid_raw][()]]
                lines = merge_soft_wraps(lines)
                lines = [l.lstrip("\u2022 ").strip() for l in lines]
                lines = [l for l in lines if keep(l)]
                if len(lines) >= 1:
                    self.study_lines[sid] = lines

        print(f"LineBank[{field}]: {len(self.study_lines):,} studies", flush=True)

        # Frequency stats
        counter = Counter()
        for ll in self.study_lines.values():
            counter.update(ll)
        total = sum(counter.values())
        self.line_keep_prob = {
            line: min(1.0, np.sqrt(subsample_t / (count / total)))
            for line, count in counter.items()
        }
        self.all_lines = list(counter.keys())
        freqs = np.array([counter[l] for l in self.all_lines], dtype=np.float64)
        freqs = freqs ** 0.75
        self.neg_probs = freqs / freqs.sum()
        print(f"  {len(self.all_lines):,} unique lines", flush=True)

        # Pre-tokenize
        tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
        enc = tokenizer(self.all_lines, padding="max_length", truncation=True,
                        max_length=128, return_tensors="np")
        self.token_ids = {l: enc["input_ids"][i] for i, l in enumerate(self.all_lines)}
        self.token_masks = {l: enc["attention_mask"][i] for i, l in enumerate(self.all_lines)}

    """
    def sample_positives(self, sid):
        lines = self.study_lines.get(sid)
        if lines is None:
            return None, None
        kept = [l for l in lines if np.random.rand() < self.line_keep_prob.get(l, 1.0)]
        if len(kept) < self.K:
            kept = lines
        sel = np.random.choice(len(kept), self.K, replace=False)
        chosen = [kept[i] for i in sel]
        ids = np.stack([self.token_ids[l] for l in chosen])
        masks = np.stack([self.token_masks[l] for l in chosen])
        return ids, masks
    """
    def sample_positives(self, sid):
        lines = self.study_lines.get(sid)
        if lines is None:
            return None, None
        kept = [l for l in lines if np.random.rand() < self.line_keep_prob.get(l, 1.0)]
        if len(kept) < self.K:
            kept = lines
        n = min(len(kept), self.K)
        sel = np.random.choice(len(kept), size=n, replace=False)
        positives = [kept[i] for i in sel]
        ids = np.zeros((self.K, 128), dtype=np.int64)
        masks = np.zeros((self.K, 128), dtype=np.int64)
        for i, l in enumerate(positives):
            ids[i] = self.token_ids[l]
            masks[i] = self.token_masks[l]
        return ids, masks

    def sample_negatives(self, N):
        idxs = np.random.choice(len(self.all_lines), N, replace=False, p=self.neg_probs)
        chosen = [self.all_lines[i] for i in idxs]
        ids = np.stack([self.token_ids[l] for l in chosen])
        masks = np.stack([self.token_masks[l] for l in chosen])
        return ids, masks


# ---------------------------------------------------------------------------
# Frozen encoder
# ---------------------------------------------------------------------------

def load_frozen_encoder(checkpoint_path, model_name, device):
    encoder_fn = getattr(vit, model_name)
    model = encoder_fn(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("target_encoder") or ckpt.get("encoder")
    state = _clean_backbone_key(state)
    msg = model.load_state_dict(state, strict=False)
    print(f"  encoder: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
"""
Multi-positive InfoNCE. logits: [B, K + N_neg], first K are positives.

    -log( sum_pos softmax ) via logsumexp -- no underflow, no epsilon.
    Once positives hold all the mass the gradient vanishes regardless of how
    it is distributed among them, so specialization is not penalized.
"""    
"""
def infonce_loss(logits, K):
    logits = logits.float()
    pos_lse = torch.logsumexp(logits[:, :K], dim=1)
    all_lse = torch.logsumexp(logits, dim=1)
    return -(pos_lse - all_lse).mean()

"""
def infonce_loss(logits, K):
    """L_out: mean of per-positive log-probs. Equalizes across positives."""
    logits = logits.float()
    all_lse = torch.logsumexp(logits, dim=1, keepdim=True)   # [B, 1]
    per_pos = logits[:, :K] - all_lse                         # [B, K]
    return -per_pos.mean()
# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--avi_manifest", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--field", default="study_findings")
    p.add_argument("--line_filters", default=None)

    # encoder
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--model_name", default="vit_large")

    # architecture
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)

    # training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--N_neg", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ipe", type=int, default=300)
    p.add_argument("--val_ipe", type=int, default=100)
    p.add_argument("--frames_per_clip", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # checkpoints
    p.add_argument("--line_checkpoint", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--log_every", type=int, default=10)

    args = p.parse_args()

    rank, world, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    out = Path(args.output_dir)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ---- CSV generation (rank 0 only) ------------------------------------
    train_csv = out / "train_videos.csv"
    val_csv = out / "val_videos.csv"
    if is_main:
        ensure_pretrain_csv(args.avi_manifest, args.train_manifest, train_csv)
        ensure_pretrain_csv(args.avi_manifest, args.val_manifest, val_csv)
    dist.barrier()

    train_sid_map = load_sid_map(train_csv.with_suffix(".sid_map.txt"))
    val_sid_map = load_sid_map(val_csv.with_suffix(".sid_map.txt"))

    # ---- Line data -------------------------------------------------------
    all_sids = list(set(train_sid_map.values()) | set(val_sid_map.values()))
    line_bank = LineBank(args.h5_dir, all_sids, field=args.field,
                         line_filters=args.line_filters, K=args.K)

    # ---- Datasets --------------------------------------------------------
    transform = ClipTransform(args.resolution)

    train_dataset = VideoDataset(
        [str(train_csv)], transform=transform,
        frames_per_clip=args.frames_per_clip, frame_step=args.frame_step,
        num_clips=1, random_clip_sampling=True,
    )
    val_dataset = VideoDataset(
        [str(val_csv)], transform=transform,
        frames_per_clip=args.frames_per_clip, frame_step=args.frame_step,
        num_clips=1, random_clip_sampling=True,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, prefetch_factor=1, collate_fn=clip_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, prefetch_factor=1, collate_fn=clip_collate,
    )

    # ---- Models ----------------------------------------------------------
    if is_main:
        print("loading frozen encoder...", flush=True)
    encoder = load_frozen_encoder(args.encoder_checkpoint, args.model_name, device)

    pooler = AttentivePooler(
        num_queries=1, embed_dim=args.embed_dim,
        num_heads=args.num_heads, depth=args.depth,
    ).to(device)

    line_encoder = LineEncoder().to(device)
    if args.line_checkpoint:
        ckpt = torch.load(args.line_checkpoint, map_location="cpu", weights_only=True)
        line_encoder.load_state_dict(ckpt.get("encoder", ckpt))
        if is_main:
            print(f"loaded line encoder: {args.line_checkpoint}", flush=True)

    pooler = DDP(pooler, device_ids=[local_rank])

    video_proj = nn.Sequential(
        nn.LayerNorm(args.embed_dim),
        nn.Linear(args.embed_dim, 4 * args.embed_dim),
        nn.GELU(),
        nn.Linear(4 * args.embed_dim, args.embed_dim),
    ).to(device)
    video_proj = DDP(video_proj, device_ids=[local_rank])

    line_encoder = DDP(line_encoder, device_ids=[local_rank], find_unused_parameters=True)

    # CLIP-style learnable temperature, init 1/0.07
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    #logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device), requires_grad=False)

    # ---- Optimizer -------------------------------------------------------
    params = (list(pooler.parameters()) + list(video_proj.parameters())
              + list(line_encoder.parameters()) + [logit_scale])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.ipe

    if is_main:
        n_pooler = sum(p.numel() for p in pooler.parameters() if p.requires_grad)
        n_line = sum(p.numel() for p in line_encoder.parameters() if p.requires_grad)
        print(f"pooler: {n_pooler / 1e6:.1f}M  line_encoder: {n_line / 1e6:.1f}M  "
              f"total steps: {total_steps}", flush=True)

    # ---- Resume ----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        pooler.module.load_state_dict(ckpt["pooler"])
        line_encoder.module.load_state_dict(ckpt["encoder"])
        video_proj.module.load_state_dict(ckpt["v_proj"])
        with torch.no_grad():
            logit_scale.copy_(ckpt["logit_scale"].to(device))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        if is_main:
            print(f"resumed from epoch {start_epoch} step {global_step}", flush=True)

    # ---- Training --------------------------------------------------------
    log_path = out / "log.jsonl"

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        pooler.train()
        line_encoder.train()
        video_proj.train()

        epoch_loss, epoch_n = 0.0, 0
        t0 = time.time()

        for step in range(args.ipe):
            try:
                clips, int_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                clips, int_labels = next(train_iter)

            # Map to SIDs, filter clips without lines
            sids = [train_sid_map[l.item()] for l in int_labels]
            valid = [i for i, s in enumerate(sids) if s in line_bank.study_lines]
            if len(valid) < 2:
                continue
            clips = clips[valid]
            sids = [sids[i] for i in valid]
            B = len(sids)
            clips = clips.to(device, non_blocking=True)

            # Sample lines
            pos_ids, pos_masks = [], []
            for sid in sids:
                ids, masks = line_bank.sample_positives(sid)
                pos_ids.append(ids)
                pos_masks.append(masks)
            pos_ids = torch.from_numpy(np.stack(pos_ids)).to(device)    # [B, K, 128]
            pos_masks = torch.from_numpy(np.stack(pos_masks)).to(device)

            #neg_ids, neg_masks = line_bank.sample_negatives(args.N_neg)
            #neg_ids = torch.from_numpy(neg_ids).to(device)              # [N_neg, 128]
            #neg_masks = torch.from_numpy(neg_masks).to(device)

            # LR
            lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Forward — frozen encoder
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    patches = encoder(clips)                            # [B, 1568, 1024]

            # Forward — trainable
            with torch.autocast("cuda", dtype=torch.bfloat16):
                """
                clip_embs = video_proj(pooler(patches).squeeze(1))      # [B, 1024]
                clip_embs = F.normalize(clip_embs.float(), dim=-1)

                all_ids = torch.cat([pos_ids.view(-1, 128), neg_ids])
                all_masks = torch.cat([pos_masks.view(-1, 128), neg_masks])
                all_line_embs = line_encoder(all_ids, all_masks)        # [B*K + N_neg, 1024]
                all_line_embs = F.normalize(all_line_embs.float(), dim=-1)

                p_embs = all_line_embs[:B * args.K].view(B, args.K, -1)
                n_embs = all_line_embs[B * args.K:]

                scale = logit_scale.exp().clamp(max=100.0)
                pos_logits = (clip_embs.unsqueeze(1) * p_embs).sum(-1) * scale  # [B, K]
                neg_logits = (clip_embs @ n_embs.T) * scale                     # [B, N_neg]
                logits = torch.cat([pos_logits, neg_logits], dim=1)
                """
                clip_embs = video_proj(pooler(patches).squeeze(1))
                clip_embs = F.normalize(clip_embs.float(), dim=-1)
                line_embs = line_encoder(pos_ids.view(-1, 128), pos_masks.view(-1, 128))
                line_embs = line_embs.view(B, args.K, -1)
                line_mask = (pos_masks.view(B, args.K, 128).sum(-1) > 0).float().to(device)
                text_embs = (line_embs * line_mask.unsqueeze(-1)).sum(1) / line_mask.sum(1, keepdim=True)
                text_embs = F.normalize(text_embs.float(), dim=-1)
                scale = logit_scale.exp().clamp(max=100.0)
                logits = (clip_embs @ text_embs.T) * scale
                labels = torch.arange(B, device=device)

            #loss = infonce_loss(logits, args.K)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item() * B
            epoch_n += B
            global_step += 1

            if is_main and (step + 1) % args.log_every == 0:
                print(f"  epoch {epoch} step {step + 1}/{args.ipe}  "
                      f"loss={loss.item():.4f}  lr={lr:.1e}", flush=True)

        # ---- Val ---------------------------------------------------------
        pooler.eval()
        line_encoder.eval()
        video_proj.eval()
        val_loss_sum, val_n = 0.0, 0
        val_iter = iter(val_loader)

        with torch.no_grad():
            for _ in range(args.val_ipe):
                try:
                    clips, int_labels = next(val_iter)
                except StopIteration:
                    break

                sids = [val_sid_map[l.item()] for l in int_labels]
                valid = [i for i, s in enumerate(sids) if s in line_bank.study_lines]
                if len(valid) < 2:
                    continue
                clips = clips[valid].to(device, non_blocking=True)
                sids = [sids[i] for i in valid]
                B = len(sids)

                pos_ids, pos_masks = [], []
                for sid in sids:
                    ids, masks = line_bank.sample_positives(sid)
                    pos_ids.append(ids)
                    pos_masks.append(masks)
                pos_ids = torch.from_numpy(np.stack(pos_ids)).to(device)
                pos_masks = torch.from_numpy(np.stack(pos_masks)).to(device)
                #neg_ids, neg_masks = line_bank.sample_negatives(args.N_neg)
                #neg_ids = torch.from_numpy(neg_ids).to(device)
                #neg_masks = torch.from_numpy(neg_masks).to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    patches = encoder(clips)
                    """
                    clip_embs = video_proj(pooler(patches).squeeze(1))
                    clip_embs = F.normalize(clip_embs.float(), dim=-1)
                    all_ids = torch.cat([pos_ids.view(-1, 128), neg_ids])
                    all_masks = torch.cat([pos_masks.view(-1, 128), neg_masks])
                    all_line_embs = line_encoder(all_ids, all_masks)
                    all_line_embs = F.normalize(all_line_embs.float(), dim=-1)
                    p_embs = all_line_embs[:B * args.K].view(B, args.K, -1)
                    n_embs = all_line_embs[B * args.K:]
                    scale = logit_scale.exp().clamp(max=100.0)
                    pos_logits = (clip_embs.unsqueeze(1) * p_embs).sum(-1) * scale
                    neg_logits = (clip_embs @ n_embs.T) * scale
                    logits = torch.cat([pos_logits, neg_logits], dim=1)
                    """
                    clip_embs = video_proj(pooler(patches).squeeze(1))
                    clip_embs = F.normalize(clip_embs.float(), dim=-1)
                    line_embs = line_encoder(pos_ids.view(-1, 128), pos_masks.view(-1, 128))
                    line_embs = line_embs.view(B, args.K, -1)
                    line_mask = (pos_masks.view(B, args.K, 128).sum(-1) > 0).float().to(device)
                    text_embs = (line_embs * line_mask.unsqueeze(-1)).sum(1) / line_mask.sum(1, keepdim=True)
                    text_embs = F.normalize(text_embs.float(), dim=-1)
                    scale = logit_scale.exp().clamp(max=100.0)
                    logits = (clip_embs @ text_embs.T) * scale
                    labels = torch.arange(B, device=device)

                #val_loss = infonce_loss(logits, args.K)
                val_loss = F.cross_entropy(logits, labels)
                val_loss_sum += val_loss.item() * B
                val_n += B

        train_loss = epoch_loss / max(epoch_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0

        if is_main:
            row = dict(epoch=epoch, train_loss=round(train_loss, 6),
                       val_loss=round(val_loss, 6),
                       temp=round(1.0 / logit_scale.exp().clamp(max=100.0).item(), 5),
                       lr=round(lr, 8), time=round(elapsed, 1))
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"epoch {epoch}  train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"temp={1.0 / logit_scale.exp().clamp(max=100.0).item():.4f}  "
                  f"lr={lr:.1e}  {elapsed:.0f}s", flush=True)

            ckpt = {
                "pooler": pooler.module.state_dict(),
                "encoder": line_encoder.module.state_dict(),
                "v_proj": video_proj.module.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": global_step,
                "best_val": min(best_val, val_loss),
                "args": vars(args),
            }
            torch.save(ckpt, out / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, out / "best.pt")
                print(f"  -> new best: {val_loss:.6f}", flush=True)

    if is_main:
        print(f"\nBest val loss: {best_val:.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
