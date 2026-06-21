"""Probe old JEPA embeddings with EchoFocus (mean-pool) on Fyler codes.

Same architecture and training as JEPA_probes/probe.py but reads from
pre-extracted embeddings (video_pretraining_v2 format).

Usage:
    python -u train_echofocusmini_chd.py \
        --embeddings ../video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
        --train_manifest ../manifests/train_50.txt \
        --val_manifest ../manifests/val_50.txt \
        --output_dir results/old_jepa_fyler
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from model_chd import EchoFocus



class StudyDataset(Dataset):
    def __init__(self, study_ids, emb_by_study, labels, n_videos=48):
        self.ids = study_ids
        self.emb = emb_by_study
        self.labels = labels
        self.n = n_videos

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        emb = self.emb[sid]
        if self.n is not None:
            n = emb.shape[0]
            sel = np.random.choice(n, self.n, replace=(n < self.n))
            emb = emb[sel]
        return torch.from_numpy(emb), torch.tensor(self.labels[sid], dtype=torch.float32)


def train_probe(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_sd = float('inf'), None

    for ep in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += loss_fn(model(x), y).item()
        v_loss /= len(val_loader)

        if v_loss < best_loss:
            best_loss = v_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: train={t_loss/len(train_loader):.4f} val={v_loss:.4f}", flush=True)

    model.load_state_dict(best_sd)
    return model


@torch.no_grad()
def eval_classification(model, loader, names, device):
    preds, labels = [], []
    for x, y in loader:
        preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    rows = []
    for i, name in enumerate(names):
        y = labels[:, i]
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        rows.append({'code': name, 'auroc': roc_auc_score(y, preds[:, i]),
             'auprc': average_precision_score(y, preds[:, i]),
             'n_pos': int(y.sum())})
    
    return pd.DataFrame(rows).sort_values('auroc', ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', required=True)
    p.add_argument('--train_manifest', required=True)
    p.add_argument('--val_manifest', required=True)
    p.add_argument('--fyler_labels', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_labels_v2.csv')
    p.add_argument('--fyler_lines', default='/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--min_pos', type=int, default=5)
    p.add_argument('--n_videos', type=int, default=48)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # load embeddings, group by study
    print(f"Loading {args.embeddings}...", flush=True)
    data = np.load(args.embeddings, allow_pickle=True)
    all_embs = data['embeddings']
    all_sids = data['study_ids'].astype(str)
    print(f"  {all_embs.shape[0]:,} clips, {all_embs.shape[1]}d", flush=True)

    emb_by_study = {}
    for emb, sid in zip(all_embs, all_sids):
        emb_by_study.setdefault(sid, []).append(emb)
    emb_by_study = {k: np.stack(v).astype(np.float32) for k, v in emb_by_study.items()}
    print(f"  {len(emb_by_study):,} studies", flush=True)

    # train/val from manifests
    train_sids = set(l.strip() for l in open(args.train_manifest))
    val_sids = set(l.strip() for l in open(args.val_manifest))

    # Fyler labels
    fyler_df = pd.read_csv(args.fyler_labels)
    fyler_df['sid'] = fyler_df['sid'].astype(str)
    fcols = [c for c in fyler_df.columns if c.startswith('fyler_')]

    lines_df = pd.read_csv(args.fyler_lines)
    code_map = dict(zip(lines_df['fyler_code'].astype(str).str.zfill(4), lines_df['line']))

    # filter codes with enough positives in train
    ft = fyler_df[fyler_df['sid'].isin(train_sids & set(emb_by_study))]
    valid_codes = [c for c in fcols if ft[c].sum() >= args.min_pos]
    print(f"  {len(valid_codes)} codes with >={args.min_pos} pos in train", flush=True)

    # vectorized label loading
    fyler_idx = fyler_df.set_index('sid')
    available = sorted(set(emb_by_study) & set(fyler_idx.index))
    label_matrix = fyler_idx.loc[available, valid_codes].values.astype(np.float32)
    all_labels = dict(zip(available, label_matrix))

    tr_ids = sorted(s for s in available if s in train_sids)
    va_ids = sorted(s for s in available if s in val_sids)
    print(f"  Fyler: {len(tr_ids)} train, {len(va_ids)} val", flush=True)

    # train
    print("Training...", flush=True)
    tr_dl = DataLoader(StudyDataset(tr_ids, emb_by_study, all_labels, args.n_videos),
                       batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    va_dl = DataLoader(StudyDataset(va_ids, emb_by_study, all_labels, n_videos=None),
                       batch_size=1, num_workers=0)

    model = EchoFocus(input_dim=all_embs.shape[1], n_targets=len(valid_codes)).to(device)
    model = train_probe(model, tr_dl, va_dl, args.epochs, args.lr, device)
    torch.save(model.state_dict(), Path(args.output_dir) / "best.pt")

    # eval
    names = [code_map.get(c.replace('fyler_', ''), c) for c in valid_codes]
    res = eval_classification(model, va_dl, names, device)
    res.to_csv(out / 'fyler_aurocs.csv', index=False)
    print(f"\nmean={res['auroc'].mean():.4f}  median={res['auroc'].median():.4f}  "
          f"({len(res)} codes)", flush=True)
    print(res.head(20).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
