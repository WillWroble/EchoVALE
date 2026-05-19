"""Ridge probe AUROC for every Fyler code on frozen embeddings.

Uses train/val splits from the embedding NPZ directly.
No line encoder — measures how much diagnostic signal lives
in the raw contrastive embeddings.

Usage:
    python -u eval_fyler.py \
        --embeddings results/full_contrast_v6/eval/embeddings_video.npz \
        --fyler_labels /lab-share/.../fyler_labels.csv \
        --output_dir results/full_contrast_v6/eval_fyler
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


def load_embeddings(path):
    npz = np.load(path)
    if "train" in npz:
        return (npz["train"].astype(np.float32), npz["train_ids"].astype(str).tolist(),
                npz["val"].astype(np.float32), npz["val_ids"].astype(str).tolist())
    raise ValueError("NPZ must contain train/val splits")


def load_fyler_labels(path):
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        code_cols = [c for c in reader.fieldnames if c.startswith('fyler_')]
        for row in reader:
            sid = str(int(float(row['sid'])))
            labels[sid] = {c: int(row[c]) for c in code_cols}
    all_codes = sorted([c.replace('fyler_', '') for c in code_cols], key=int)
    return labels, all_codes, code_cols


def build_labels(ids, labels_by_sid, code_cols):
    y = np.zeros((len(ids), len(code_cols)), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=bool)
    for i, sid in enumerate(ids):
        if sid in labels_by_sid:
            mask[i] = True
            for j, c in enumerate(code_cols):
                y[i, j] = labels_by_sid[sid][c]
    return y, mask


def load_fyler_descriptions(lines_csv):
    code_to_desc = {}
    with open(lines_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['fyler_code']
            if code not in code_to_desc:
                code_to_desc[code] = row['line']
    return code_to_desc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--fyler_labels", required=True)
    p.add_argument("--fyler_lines", default=None, help="fyler_lines.csv for descriptions")
    p.add_argument("--min_pos_train", type=int, default=5)
    p.add_argument("--min_pos_val", type=int, default=5)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest", default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    Z_train, train_ids, Z_val, val_ids = load_embeddings(args.embeddings)
    print(f"Train: {Z_train.shape}, Val: {Z_val.shape}", flush=True)


    if args.train_manifest:
        keep = set(str(int(float(x))) for x in Path(args.train_manifest).read_text().strip().splitlines())
        mask = [i for i, sid in enumerate(train_ids) if sid in keep]
        Z_train = Z_train[mask]
        train_ids = [train_ids[i] for i in mask]

    if args.val_manifest:
        keep = set(str(int(float(x))) for x in Path(args.val_manifest).read_text().strip().splitlines())
        mask = [i for i, sid in enumerate(val_ids) if sid in keep]
        Z_val = Z_val[mask]
        val_ids = [val_ids[i] for i in mask]


    labels_by_sid, all_codes, code_cols = load_fyler_labels(args.fyler_labels)
    print(f"Labels: {len(labels_by_sid)} studies, {len(all_codes)} codes", flush=True)

    code_to_desc = {}
    if args.fyler_lines:
        code_to_desc = load_fyler_descriptions(args.fyler_lines)

    y_train, tr_mask = build_labels(train_ids, labels_by_sid, code_cols)
    y_val, va_mask = build_labels(val_ids, labels_by_sid, code_cols)
    print(f"Matched: {tr_mask.sum()}/{len(train_ids)} train, {va_mask.sum()}/{len(val_ids)} val", flush=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(Z_train[tr_mask])
    X_va = scaler.transform(Z_val[va_mask])
    y_tr = y_train[tr_mask]
    y_va = y_val[va_mask]

    results = {}
    for j, code in enumerate(all_codes):
        n_pos_tr = int(y_tr[:, j].sum())
        n_pos_va = int(y_va[:, j].sum())
        if n_pos_tr < args.min_pos_train or n_pos_va < args.min_pos_val:
            continue
        if n_pos_tr == len(y_tr) or n_pos_va == len(y_va):
            continue

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr, y_tr[:, j])
        scores = clf.decision_function(X_va)
        auroc = roc_auc_score(y_va[:, j], scores)
        auprc = average_precision_score(y_va[:, j], scores)
        results[code] = {"auroc": round(auroc, 4), "auprc": round(auprc, 4),
                         "n_pos_train": n_pos_tr, "n_pos_val": n_pos_va}
        
    ranked = sorted(results.items(), key=lambda x: -x[1]["auroc"])

    print(f"\n{'Code':>6}  {'AUROC':>7}  {'N+tr':>6}  {'N+va':>6}  Description")
    print("-" * 80)
    for code, r in ranked[:50]:
        desc = code_to_desc.get(code, "")[:45]
        print(f"{code:>6}  {r['auroc']:>7.4f}  {r['n_pos_train']:>6}  {r['n_pos_val']:>6}  {desc}")

    aurocs = [r["auroc"] for r in results.values()]
    print(f"\n{len(results)} codes evaluated")
    print(f"Mean AUROC: {np.mean(aurocs):.4f}  Median: {np.median(aurocs):.4f}")

    for min_n in [5, 10, 25, 50, 100]:
        vals = [r["auroc"] for r in results.values() if r["n_pos_val"] >= min_n]
        if vals:
            print(f"  N+val>={min_n:>4}: {len(vals):>5} codes  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")

    with open(out / "fyler_aurocs.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out / "fyler_aurocs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "description", "auroc", "auprc", "n_pos_train", "n_pos_val"])
        for code, r in ranked:
            desc = code_to_desc.get(code, "")
            w.writerow([code, desc, r["auroc"], r["auprc"], r["n_pos_train"], r["n_pos_val"]])

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
