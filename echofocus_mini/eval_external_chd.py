"""Evaluate trained EchoFocus CHD model on external data.

Usage:
    python -u eval_external_chd.py \
        --checkpoint results/chd_jepa_old_split/best.pt \
        --embeddings ../video_pretraining_v2/embeddings/jepa_outside.npz \
        --train_manifest ../manifests/train_50_nofetal.txt \
        --output_dir results/chd_jepa_old_split/external
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from model_chd import EchoFocus


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--train_manifest", required=True, help="Same train manifest used during training (to derive valid_codes)")
    p.add_argument("--fyler_labels", default="/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_labels_v2.csv")
    p.add_argument("--fyler_lines", default="/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv")
    p.add_argument("--min_pos", type=int, default=5)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    all_embs = data["embeddings"]
    all_sids = data["study_ids"].astype(str)
    input_dim = all_embs.shape[1]

    emb_by_study = {}
    for emb, sid in zip(all_embs, all_sids):
        emb_by_study.setdefault(sid, []).append(emb)
    emb_by_study = {k: np.stack(v, dtype=np.float32) for k, v in emb_by_study.items()}
    print(f"Loaded {len(emb_by_study)} external studies, {input_dim}d", flush=True)

    # Derive valid_codes (same logic as training)
    fyler_df = pd.read_csv(args.fyler_labels)
    fyler_df["sid"] = fyler_df["sid"].astype(str)
    fcols = [c for c in fyler_df.columns if c.startswith("fyler_")]

    train_sids = set(l.strip() for l in open(args.train_manifest))
    ft = fyler_df[fyler_df["sid"].isin(train_sids)]
    valid_codes = [c for c in fcols if ft[c].sum() >= args.min_pos]
    print(f"{len(valid_codes)} codes (min_pos={args.min_pos})", flush=True)

    lines_df = pd.read_csv(args.fyler_lines)
    code_map = dict(zip(lines_df["fyler_code"].astype(str).str.zfill(4), lines_df["line"]))
    code_names = [code_map.get(c.replace("fyler_", ""), c) for c in valid_codes]

    # External labels
    fyler_idx = fyler_df.set_index("sid")
    available = sorted(set(emb_by_study) & set(fyler_idx.index))
    if not available:
        print("No overlap between external embeddings and Fyler labels!")
        return
    label_matrix = fyler_idx.loc[available, valid_codes].values.astype(np.float32)
    print(f"{len(available)} external studies with Fyler labels", flush=True)

    # Load model
    model = EchoFocus(input_dim=input_dim, n_targets=len(valid_codes)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    # Evaluate (all clips per study, no subsampling)
    preds = []
    for sid in available:
        emb = torch.from_numpy(emb_by_study[sid]).unsqueeze(0).to(device)
        out = torch.sigmoid(model(emb)).squeeze(0).cpu().numpy()
        preds.append(out)
    preds = np.stack(preds)

    # Per-code AUROC
    rows = []
    for i, name in enumerate(code_names):
        y = label_matrix[:, i]
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        rows.append({"code": name, "auroc": roc_auc_score(y, preds[:, i]),
                      "n_pos": int(y.sum())})
    res = pd.DataFrame(rows).sort_values("auroc", ascending=False)

    print(f"\nmean={res['auroc'].mean():.4f}  median={res['auroc'].median():.4f}  "
          f"({len(res)} codes)", flush=True)
    print(res.head(20).to_string(index=False), flush=True)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "external_fyler_aurocs.csv", index=False)
        print(f"Saved {out / 'external_fyler_aurocs.csv'}", flush=True)


if __name__ == "__main__":
    main()
