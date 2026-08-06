"""End-to-end VALE with unfrozen clip encoder.

Frozen ViT-L + unfrozen AttentivePooler + LineEncoder + study aggregator,
trained jointly on study-level skip-gram BCE. DDP on 4 GPUs with gradient
accumulation.

Launch:
    torchrun --standalone --nproc_per_node=4 train.py \
        --encoder_checkpoint .../echojepa_vitl_mimic_bch_cooldown/latest.pt \
        --data_dir .../Echo_Internal_30k \
        --h5_dir .../line_tokenizer/data \
        --train_manifest .../manifests/platon_train.txt \
        --val_manifest .../manifests/platon_val.txt \
        --fields study_findings summary \
        --K 2 --M 16 \
        --clip_dropout 0.8 --max_clips 32 \
        --output_dir results/v1
"""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Echo_JEPA (on PYTHONPATH via sbatch)
from src.models import vision_transformer as vit
from src.models.attentive_pooler import AttentivePooler
from src.hub.backbones import _clean_backbone_key
from dataset import StudyAVIDataset, collate_fn

# VALE study-level models
import sys
sys.path.insert(0, '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2')
from model import LineEncoder, CrossAttentionPool, QuerySAPool
#from model import LineEncoder, CrossAttentionPool  # cross encoder



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
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Frozen encoder
# ---------------------------------------------------------------------------

def load_frozen_encoder(checkpoint_path, device):
    encoder = vit.vit_large(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('target_encoder') or ckpt.get('encoder')
    state = _clean_backbone_key(state)
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    # data (no defaults — set in sbatch)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--fields", nargs="+", required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--clip_dropout", type=float, required=True)
    p.add_argument("--max_clips", type=int, required=True)
    p.add_argument("--num_clips_per_video", type=int, default=2)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--subsample_t", type=float, default=1e-3)

    # model
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--encoder_warmstart", default=None,
                   help="v3 clip pretraining checkpoint to init pooler")

    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    # output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default=None)

    args = p.parse_args()

    rank, world, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out = Path(args.output_dir)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ---- Data ----------------------------------------------------------------
    train_ids = [str(int(float(x)))
                 for x in Path(args.train_manifest).read_text().split()]
    val_ids = [str(int(float(x)))
               for x in Path(args.val_manifest).read_text().split()]

    train_ds = StudyAVIDataset(
        args.h5_dir, train_ids, args.data_dir, args.fields,
        args.K, args.M, subsample_t=args.subsample_t,
        clip_dropout=args.clip_dropout, max_clips=args.max_clips,
        num_clips_per_video=args.num_clips_per_video,
        line_filters=args.line_filters)

    val_ds = StudyAVIDataset(
        args.h5_dir, val_ids, args.data_dir, args.fields,
        args.K, args.M, subsample_t=args.subsample_t,
        clip_dropout=args.clip_dropout, max_clips=args.max_clips,
        num_clips_per_video=args.num_clips_per_video,
        line_filters=args.line_filters)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True)

    if is_main:
        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    # ---- Models --------------------------------------------------------------
    encoder = load_frozen_encoder(args.encoder_checkpoint, device)

    pooler = AttentivePooler(
        num_queries=1, embed_dim=1024, num_heads=16, depth=4,
    ).to(device)

    if args.encoder_warmstart:
        ckpt = torch.load(args.encoder_warmstart, map_location='cpu', weights_only=False)
        state = {k.replace('module.', ''): v for k, v in ckpt['pooler'].items()}
        pooler.load_state_dict(state)
        if is_main:
            print(f"Warm-started pooler from {args.encoder_warmstart}", flush=True)

    pooler = DDP(pooler, device_ids=[local_rank])

    line_encoder = LineEncoder().to(device)
    line_encoder = DDP(line_encoder, device_ids=[local_rank], find_unused_parameters=True)

    #attn_pool = CrossAttentionPool().to(device)
    attn_pool = QuerySAPool().to(device)
    attn_pool = DDP(attn_pool, device_ids=[local_rank])

    params = (list(pooler.parameters()) + list(line_encoder.parameters())
              + list(attn_pool.parameters()))

    n_pooler = sum(p.numel() for p in pooler.parameters() if p.requires_grad)
    n_line = sum(p.numel() for p in line_encoder.parameters() if p.requires_grad)
    n_attn = sum(p.numel() for p in attn_pool.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable: pooler={n_pooler / 1e6:.1f}M  line_encoder={n_line / 1e6:.1f}M  "
              f"attn_pool={n_attn / 1e6:.1f}M  "
              f"total={(n_pooler + n_line + n_attn) / 1e6:.1f}M", flush=True)

    # ---- Optimizer -----------------------------------------------------------
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_val = float('inf')
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        pooler.module.load_state_dict(ckpt['pooler'])
        line_encoder.module.load_state_dict(ckpt['encoder'])
        attn_pool.module.load_state_dict(ckpt['attn_pool'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['step']
        best_val = ckpt['best_val']
        if is_main:
            print(f"Resumed from epoch {start_epoch}", flush=True)

    if is_main:
        print(f"Steps/epoch: {steps_per_epoch}  total_opt_steps: {total_steps}  "
              f"warmup: {warmup_steps}", flush=True)

    log_path = out / 'log.jsonl'

    # ---- Training ------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        pooler.train()
        line_encoder.train()
        attn_pool.train()

        epoch_loss, epoch_n = 0.0, 0
        optimizer.zero_grad()

        for step, (clips, pos_ids, pos_masks, n_clips) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            pos_ids = pos_ids.to(device, non_blocking=True)
            pos_masks = pos_masks.to(device, non_blocking=True)
            B = len(n_clips)

            # Frozen ViT-L
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    patches = encoder(clips)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Unfrozen pooler → per-clip CLS
                cls_embs = pooler(patches).squeeze(1)

                # Reshape to per-study [B, max_clips, 1024] + mask
                clip_list = torch.split(cls_embs, n_clips)
                max_n = max(n_clips)
                videos = cls_embs.new_zeros(B, max_n, 1024)
                video_mask = cls_embs.new_zeros(B, max_n)
                for i, emb in enumerate(clip_list):
                    videos[i, :len(emb)] = emb
                    video_mask[i, :len(emb)] = 1.0

                # Positive line embeddings
                pos_embs = line_encoder(pos_ids, pos_masks)
                pos_embs = pos_embs.view(B, args.K, -1)

                # Shared negatives
                neg_ids, neg_masks = train_ds.sample_negatives(args.M)
                neg_ids = neg_ids.to(device)
                neg_masks = neg_masks.to(device)
                neg_embs = line_encoder(neg_ids, neg_masks)
                neg_embs = neg_embs.unsqueeze(0).expand(B, -1, -1)

                # Concat lines
                line_embs = torch.cat([pos_embs, neg_embs], dim=1)

                # Study aggregator
                attended = attn_pool(line_embs, videos, video_mask)

                # Logits
                #logits = (line_embs * attended).sum(dim=-1)     # CrossAttentionPool
                logits = attended.sum(dim=-1)                   # QuerySAPool

                # Labels
                labels = torch.cat([
                    torch.ones(B, args.K, device=device),
                    torch.zeros(B, args.M, device=device),
                ], dim=1)

                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1))

            (loss / args.accum_steps).backward()
            epoch_loss += loss.item() * B
            epoch_n += B

            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg = epoch_loss / max(epoch_n, 1)
                    print(f"  epoch {epoch} step {global_step}  "
                          f"loss={loss.item():.4f}  avg={avg:.4f}  "
                          f"lr={lr:.1e}", flush=True)

        # Flush remaining gradients
        if (step + 1) % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        train_loss = epoch_loss / max(epoch_n, 1)

        # ---- Validation ------------------------------------------------------
        pooler.eval()
        line_encoder.eval()
        attn_pool.eval()
        val_loss_sum, val_n = 0.0, 0

        with torch.no_grad():
            for clips, pos_ids, pos_masks, n_clips in val_loader:
                clips = clips.to(device, non_blocking=True)
                pos_ids = pos_ids.to(device, non_blocking=True)
                pos_masks = pos_masks.to(device, non_blocking=True)
                B = len(n_clips)

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    patches = encoder(clips)
                    cls_embs = pooler(patches).squeeze(1)

                    clip_list = torch.split(cls_embs, n_clips)
                    max_n = max(n_clips)
                    videos = cls_embs.new_zeros(B, max_n, 1024)
                    video_mask = cls_embs.new_zeros(B, max_n)
                    for i, emb in enumerate(clip_list):
                        videos[i, :len(emb)] = emb
                        video_mask[i, :len(emb)] = 1.0

                    pos_embs = line_encoder(pos_ids, pos_masks)
                    pos_embs = pos_embs.view(B, args.K, -1)

                    neg_ids, neg_masks = val_ds.sample_negatives(args.M)
                    neg_ids = neg_ids.to(device)
                    neg_masks = neg_masks.to(device)
                    neg_embs = line_encoder(neg_ids, neg_masks)
                    neg_embs = neg_embs.unsqueeze(0).expand(B, -1, -1)

                    line_embs = torch.cat([pos_embs, neg_embs], dim=1)
                    attended = attn_pool(line_embs, videos, video_mask)

                    #logits = (line_embs * attended).sum(dim=-1)
                    logits = attended.sum(dim=-1)

                    labels = torch.cat([
                        torch.ones(B, args.K, device=device),
                        torch.zeros(B, args.M, device=device),
                    ], dim=1)

                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1), labels.view(-1))

                val_loss_sum += loss.item() * B
                val_n += B

        val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        if is_main:
            row = dict(epoch=epoch, train_loss=round(train_loss, 6),
                       val_loss=round(val_loss, 6),
                       lr=round(lr, 8), time=round(elapsed, 1))
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"epoch {epoch}  train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr:.1e}  {elapsed:.0f}s", flush=True)

            ckpt = {
                "pooler": pooler.module.state_dict(),
                "encoder": line_encoder.module.state_dict(),
                "attn_pool": attn_pool.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
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
