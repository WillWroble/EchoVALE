"""Train LineEncoder with video-attended skip-gram BCE objective."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_SA import LineEncoder, CrossAttentionPool#, QuerySAPool
from dataset import SkipGramDataset, collate_fn, load_videos_by_study, FIELD_CONFIG


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_step(encoder, attn_pool, batch, device, L, dataset=None, tau=1.5):
    input_ids, attn_mask, videos, video_mask, labels = batch
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    videos = videos.to(device)
    video_mask = video_mask.to(device)
    labels = labels.to(device)

    line_embs = encoder(input_ids, attn_mask)
    B = videos.shape[0]
    line_embs = line_embs.view(B, L, -1)
    
    if dataset is not None:
        neg_ids, neg_mask = dataset.sample_negatives(32)
        neg_ids, neg_mask = neg_ids.to(device), neg_mask.to(device)
        neg_embs = encoder(neg_ids, neg_mask)
        neg_embs = neg_embs.unsqueeze(0).expand(B, -1, -1)
        line_embs = torch.cat([line_embs, neg_embs], dim=1)
        neg_labels = torch.zeros(B, 32, device=device)
        labels = labels.to(device).view(B, dataset.K)
        labels = torch.cat([labels, neg_labels], dim=1)

    attended = attn_pool(line_embs, videos, video_mask)
    logits = (line_embs * attended).sum(dim=-1)
    #logits = (attended).sum(dim=-1)
    loss = F.binary_cross_entropy_with_logits(logits.view(-1) / tau, labels.view(-1))
    return loss, labels.size(0)


def run_val(encoder, attn_pool, loaders, field_L, device):
    encoder.eval()
    attn_pool.eval()
    field_losses = {}
    with torch.no_grad():
        for field, loader in loaders.items():
            L = field_L[field]
            total_loss, total_n = 0.0, 0
            for batch in loader:
                loss, n = run_step(encoder, attn_pool, batch, device, L, dataset=loader.dataset)
                total_loss += loss.item() * n
                total_n += n
            field_losses[field] = total_loss / total_n
    return field_losses


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--video_embeddings", required=True, help="infonce_768_all.npz")
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)

    # sampling
    p.add_argument("--fields", nargs="+", default=["study_findings", "summary", "history"])
    p.add_argument("--subsample_t", type=float, default=1e-3)
    p.add_argument("--max_videos", type=int, default=64)
    p.add_argument("--line_filters", default=None)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)

    # output
    p.add_argument("--output_dir", required=True)

    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- data ----
    videos_by_study = load_videos_by_study(args.video_embeddings)

    train_manifest = set(str(int(float(x)))
                         for x in Path(args.train_manifest).read_text().strip().splitlines())
    val_manifest = set(str(int(float(x)))
                       for x in Path(args.val_manifest).read_text().strip().splitlines())
    train_ids = [s for s in train_manifest if s in videos_by_study]
    val_ids = [s for s in val_manifest if s in videos_by_study]
    print(f"Train: {len(train_ids):,}  Val: {len(val_ids):,}", flush=True)

    # Create per-field loaders
    train_loaders = {}
    val_loaders = {}
    field_L = {}

    for field in args.fields:
        cfg = FIELD_CONFIG[field]
        field_L[field] = cfg['K'] + cfg['M']

        train_ds = SkipGramDataset(args.h5_dir, train_ids, videos_by_study, field=field,
                                    subsample_t=args.subsample_t, max_videos=args.max_videos,
                                    line_filters=args.line_filters)
        val_ds = SkipGramDataset(args.h5_dir, val_ids, videos_by_study, field=field,
                                  subsample_t=args.subsample_t, max_videos=args.max_videos,
                                  line_filters=args.line_filters, clip_dropout=0.8)

        train_loaders[field] = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True, num_workers=4, collate_fn=collate_fn)
        val_loaders[field] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=4, collate_fn=collate_fn)

    # ---- model ----
    encoder = LineEncoder().to(device)
    attn_pool = CrossAttentionPool().to(device)
    #attn_pool = QuerySAPool().to(device)


    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        if "attn_pool" in ckpt:
            attn_pool.load_state_dict(ckpt["attn_pool"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    n_total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in attn_pool.parameters())
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
              sum(p.numel() for p in attn_pool.parameters() if p.requires_grad)
    print(f"Parameters: {n_train:,} trainable / {n_total:,} total", flush=True)

    total_steps = sum(len(train_loaders[f]) for f in args.fields) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(attn_pool.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    log_path = Path(args.output_dir) / "log.jsonl"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        encoder.train()
        attn_pool.train()

        # Build shuffled schedule
        schedule = []
        for field in args.fields:
            schedule.extend([field] * len(train_loaders[field]))
        np.random.shuffle(schedule)

        iters = {f: iter(train_loaders[f]) for f in args.fields}
        train_field_losses = {f: [] for f in args.fields}

        for field in schedule:
            batch = next(iters[field])
            loss, n = run_step(encoder, attn_pool, batch, device, field_L[field], dataset=train_loaders[field].dataset)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(attn_pool.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            train_field_losses[field].append(loss.item())

        val_field_losses = run_val(encoder, attn_pool, val_loaders, field_L, device)
        train_loss = np.mean([l for losses in train_field_losses.values() for l in losses])
        val_loss = np.mean(list(val_field_losses.values()))
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
                   lr=round(lr, 6), time=round(elapsed, 1))
        for field in args.fields:
            if train_field_losses[field]:
                row[f"train_{field}"] = round(np.mean(train_field_losses[field]), 6)
            row[f"val_{field}"] = round(val_field_losses[field], 6)

        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={lr:.1e}  {elapsed:.0f}s", flush=True)

        ckpt = {"encoder": encoder.state_dict(), "attn_pool": attn_pool.state_dict()}
        torch.save(ckpt, Path(args.output_dir) / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, Path(args.output_dir) / "best.pt")
            print(f"  -> new best: {val_loss:.6f}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.6f}", flush=True)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
