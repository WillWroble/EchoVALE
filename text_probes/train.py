"""Train text probe: line embeddings → echo measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (TextProbeDataset, collate_fn, load_line_lookup,
                     CONTINUOUS_COLS, BINARY_COLS, ALL_COLS, N_CONT)
from model import TextProbe


def masked_l1(pred, target, mask):
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1)


def masked_bce(pred, target, mask):
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device, cont_mean, cont_std):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    for embs, pad_mask, targets, target_masks in loader:
        pred = model(embs.to(device), pad_mask.to(device))
        all_preds.append(pred.cpu())
        all_targets.append(targets)
        all_masks.append(target_masks)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    masks = torch.cat(all_masks)

    results = {}
    for i, col in enumerate(CONTINUOUS_COLS):
        m = masks[:, i].bool()
        if m.sum() == 0:
            continue
        # denormalize
        p = preds[m, i] * cont_std[i] + cont_mean[i]
        t = targets[m, i] * cont_std[i] + cont_mean[i]
        results[col] = {"mae": round((p - t).abs().mean().item(), 4),
                        "n": int(m.sum())}

    for i, col in enumerate(BINARY_COLS):
        j = N_CONT + i
        m = masks[:, j].bool()
        if m.sum() == 0:
            continue
        acc = ((preds[m, j].sigmoid() > 0.5).float() == targets[m, j]).float().mean()
        results[col] = {"acc": round(acc.item(), 4), "n": int(m.sum())}

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--field", required=True)
    p.add_argument("--line_embeddings", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--targets_csv", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    line_lookup = load_line_lookup(args.line_embeddings)
    targets_df = pd.read_csv(args.targets_csv)

    train_ds = TextProbeDataset(args.h5_dir, args.field, line_lookup, targets_df,
                                args.train_manifest, args.line_filters)
    cont_mean, cont_std = train_ds.compute_norm()
    train_ds.cont_mean, train_ds.cont_std = cont_mean, cont_std

    val_ds = TextProbeDataset(args.h5_dir, args.field, line_lookup, targets_df,
                              args.val_manifest, args.line_filters,
                              cont_mean=cont_mean, cont_std=cont_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = TextProbe(num_targets=len(ALL_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}", flush=True)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    # tensors for normalization during training
    cm = torch.from_numpy(cont_mean).to(device)
    cs = torch.from_numpy(cont_std).to(device)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for embs, pad_mask, targets, target_masks in train_loader:
            embs = embs.to(device)
            pad_mask = pad_mask.to(device)
            targets = targets.to(device)
            target_masks = target_masks.to(device)

            pred = model(embs, pad_mask)
            loss_cont = masked_l1(pred[:, :N_CONT], targets[:, :N_CONT],
                                  target_masks[:, :N_CONT])
            loss_bin = masked_bce(pred[:, N_CONT:], targets[:, N_CONT:],
                                  target_masks[:, N_CONT:])
            loss = loss_cont + loss_bin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        results = evaluate(model, val_loader, device, cont_mean, cont_std)

        ef_mae = results.get("EF05", {}).get("mae", float("inf"))
        tag = ""
        if ef_mae < best_mae:
            best_mae = ef_mae
            torch.save(model.state_dict(), out / "best.pt")
            with open(out / "best_metrics.json", "w") as f:
                json.dump(results, f, indent=2)
            tag = f"  -> new best"

        print(f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"EF05_MAE={ef_mae:.4f}  lr={scheduler.get_last_lr()[0]:.1e}{tag}",
              flush=True)

    print(f"\nBest EF05 MAE: {best_mae:.4f}")
    with open(out / "best_metrics.json") as f:
        final = json.load(f)
    for col, m in sorted(final.items()):
        metric = "mae" if "mae" in m else "acc"
        print(f"  {col:>20s}: {m[metric]:.4f}  (n={m['n']:,})")


if __name__ == "__main__":
    main()
