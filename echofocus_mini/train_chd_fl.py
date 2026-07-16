"""Train EchoFocus CHD probe — single-site, merged, or federated.

Usage:
    # single-site internal (floor)
    python train_chd_fl.py \
        --train_manifests ../manifests/platon_train.txt \
        --local_steps 100 --batch_sizes 128

    # single-site external (floor)
    python train_chd_fl.py \
        --train_manifests ../manifests/study_outside_us_train.txt \
        --local_steps 10 --batch_sizes 6

    # merged (ceiling)
    python train_chd_fl.py \
        --train_manifests ../manifests/platon_train.txt ../manifests/study_outside_us_train.txt \
        --local_steps 100 --batch_sizes 128

    # federated
    python train_chd_fl.py \
        --train_manifests ../manifests/platon_train.txt ../manifests/study_outside_us_train.txt \
        --local_steps 100 10 --batch_sizes 128 6 --federated
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model_chd import EchoFocus


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CHDDataset(Dataset):
    def __init__(self, sids, emb_by_study, labels, clip_drop=0.5, train=True):
        self.sids = sids
        self.emb = emb_by_study
        self.labels = labels
        self.clip_drop = clip_drop
        self.train = train

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        sid = self.sids[idx]
        emb = self.emb[sid]

        if self.train and self.clip_drop > 0:
            keep = np.random.random(emb.shape[0]) >= self.clip_drop
            if not keep.any():
                keep[np.random.randint(emb.shape[0])] = True
            emb = emb[keep]

        return torch.from_numpy(emb), torch.tensor(self.labels[sid], dtype=torch.float32)


def collate_fn(batch):
    embs, labels = zip(*batch)
    lengths = [e.shape[0] for e in embs]
    max_len = max(lengths)
    B, D = len(embs), embs[0].shape[1]

    x = torch.zeros(B, max_len, D)
    mask = torch.ones(B, max_len, dtype=torch.bool)  # True = pad
    for i, e in enumerate(embs):
        x[i, :lengths[i]] = e
        mask[i, :lengths[i]] = False

    return x, mask, torch.stack(labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(paths):
    pool = {}
    for p in paths:
        data = np.load(p, allow_pickle=True)
        for emb, sid in zip(data["embeddings"], data["study_ids"].astype(str)):
            pool.setdefault(sid, []).append(emb)
    return {k: np.stack(v).astype(np.float32) for k, v in pool.items()}


def load_labels(path):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    meta = {"0", "1", "eid", "pid", "Gender", "Age"}
    diag_cols = [c for c in df.columns if c not in meta]
    df["eid"] = df["eid"].astype(str)
    labels = {row["eid"]: row[diag_cols].values.astype(np.float32) for _, row in df.iterrows()}
    return labels, diag_cols


def load_manifest(path):
    return set(Path(path).read_text().strip().splitlines())


def make_loader(manifest, emb_pool, labels, batch_size, clip_drop, train=True):
    sids = sorted(load_manifest(manifest) & set(emb_pool) & set(labels))
    ds = CHDDataset(sids, emb_pool, labels, clip_drop=clip_drop, train=train)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        drop_last=train, collate_fn=collate_fn, num_workers=0,
    )


def cycle(loader):
    while True:
        yield from loader


def avg_state_dicts(sds):
    return {k: torch.stack([sd[k].float() for sd in sds]).mean(0) for k in sds[0]}


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------

def train_steps(model, opt, it, loss_fn, device, n_steps):
    model.train()
    total = 0.0
    for _ in range(n_steps):
        x, mask, y = next(it)
        logits = model(x.to(device), mask.to(device))
        loss = loss_fn(logits, y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / n_steps


@torch.no_grad()
def evaluate(model, loader, device, diag_cols):
    model.eval()
    preds, targets = [], []
    for x, mask, y in loader:
        logits = model(x.to(device), mask.to(device))
        preds.append(logits.sigmoid().cpu())
        targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    aucs = {}
    for i, col in enumerate(diag_cols):
        y = targets[:, i]
        if 0 < y.sum() < len(y):
            aucs[col] = float(roc_auc_score(y, preds[:, i]))

    macro = float(np.mean(list(aucs.values()))) if aucs else 0.0
    return macro, aucs


def run_eval(model, val_loaders, val_names, device, diag_cols):
    results = {}
    for name, loader in zip(val_names, val_loaders):
        macro, aucs = evaluate(model, loader, device, diag_cols)
        results[name] = {"macro_auroc": macro, "per_code": aucs}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", nargs="+", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--train_manifests", nargs="+", required=True)
    p.add_argument("--val_manifests", nargs="+", required=True)
    p.add_argument("--local_steps", nargs="+", type=int, required=True)
    p.add_argument("--batch_sizes", nargs="+", type=int, required=True)
    p.add_argument("--federated", action="store_true")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip_drop", type=float, default=0.5)
    p.add_argument("--input_dim", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- data ---
    print("Loading embeddings...", flush=True)
    emb_pool = load_embeddings(args.embeddings)
    print(f"  {len(emb_pool):,} studies", flush=True)

    print("Loading labels...", flush=True)
    labels, diag_cols = load_labels(args.labels)
    print(f"  {len(labels):,} labeled, {len(diag_cols)} codes", flush=True)

    # --- val ---
    val_names = [Path(m).stem for m in args.val_manifests]
    val_loaders = [
        make_loader(m, emb_pool, labels, batch_size=64, clip_drop=0, train=False)
        for m in args.val_manifests
    ]
    for name, loader in zip(val_names, val_loaders):
        print(f"  val/{name}: {len(loader.dataset):,}", flush=True)

    # --- model ---
    n_targets = len(diag_cols)
    model = EchoFocus(input_dim=args.input_dim, n_targets=n_targets).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params, {n_targets} targets", flush=True)

    best_macro = 0.0
    history = []

    if args.federated:
        # ---------------------------------------------------------------
        # Federated
        # ---------------------------------------------------------------
        assert len(args.train_manifests) > 1
        assert len(args.local_steps) == len(args.train_manifests)
        assert len(args.batch_sizes) == len(args.train_manifests)

        site_names = [Path(m).stem for m in args.train_manifests]
        site_loaders = [
            make_loader(m, emb_pool, labels, bs, args.clip_drop, train=True)
            for m, bs in zip(args.train_manifests, args.batch_sizes)
        ]
        site_iters = [cycle(l) for l in site_loaders]

        for name, loader in zip(site_names, site_loaders):
            print(f"  train/{name}: {len(loader.dataset):,}", flush=True)
        print(f"\nFederated: {len(site_names)} sites, {args.rounds} rounds\n", flush=True)

        for rnd in range(1, args.rounds + 1):
            global_sd = copy.deepcopy(model.state_dict())
            site_sds, losses = [], []

            for i, (name, it, steps) in enumerate(
                zip(site_names, site_iters, args.local_steps)
            ):
                model.load_state_dict(copy.deepcopy(global_sd))
                opt = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                loss = train_steps(model, opt, it, loss_fn, device, steps)
                site_sds.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
                losses.append(loss)

            model.load_state_dict(avg_state_dicts(site_sds))

            loss_str = " | ".join(f"{n}: {l:.4f}" for n, l in zip(site_names, losses))
            print(f"Round {rnd}/{args.rounds}  {loss_str}", flush=True)

            if rnd % args.eval_every == 0 or rnd == args.rounds:
                results = run_eval(model, val_loaders, val_names, device, diag_cols)
                mean_macro = float(np.mean([r["macro_auroc"] for r in results.values()]))

                for vn, r in results.items():
                    print(f"  {vn}: macro={r['macro_auroc']:.4f}", flush=True)

                entry = {"round": rnd, "losses": dict(zip(site_names, losses)), **results}
                history.append(entry)

                if mean_macro > best_macro:
                    best_macro = mean_macro
                    torch.save(model.state_dict(), out / "best.pt")
                    print(f"  ** best: {mean_macro:.4f}", flush=True)

    else:
        # ---------------------------------------------------------------
        # Single-site or Merged
        # ---------------------------------------------------------------
        all_sids = set()
        for m in args.train_manifests:
            all_sids |= load_manifest(m) & set(emb_pool) & set(labels)
        all_sids = sorted(all_sids)

        bs = args.batch_sizes[0]
        steps_per_round = args.local_steps[0]
        ds = CHDDataset(all_sids, emb_pool, labels, clip_drop=args.clip_drop, train=True)
        loader = DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=True,
            collate_fn=collate_fn, num_workers=0,
        )
        it = cycle(loader)
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        print(f"  train: {len(ds):,} studies, batch={bs}", flush=True)
        print(f"\nBaseline: {args.rounds} rounds x {steps_per_round} steps\n", flush=True)

        for rnd in range(1, args.rounds + 1):
            loss = train_steps(model, opt, it, loss_fn, device, steps_per_round)
            print(f"Round {rnd}/{args.rounds}  loss: {loss:.4f}", flush=True)

            if rnd % args.eval_every == 0 or rnd == args.rounds:
                results = run_eval(model, val_loaders, val_names, device, diag_cols)
                mean_macro = float(np.mean([r["macro_auroc"] for r in results.values()]))

                for vn, r in results.items():
                    print(f"  {vn}: macro={r['macro_auroc']:.4f}", flush=True)

                entry = {"round": rnd, "loss": loss, **results}
                history.append(entry)

                if mean_macro > best_macro:
                    best_macro = mean_macro
                    torch.save(model.state_dict(), out / "best.pt")
                    print(f"  ** best: {mean_macro:.4f}", flush=True)

    with open(out / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best macro: {best_macro:.4f}", flush=True)


if __name__ == "__main__":
    main()
