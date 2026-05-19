"""Evaluate per-Fyler-code AUROC using line tokenizer scores.

For each study, scores all Fyler anchor lines via cross-attention,
groups by code (max across severity variants), computes AUROC per code.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

from model import LineEncoder, CrossAttentionPool

CODE_RE = re.compile(r'\s*\[\d+\]\s*$')


def load_fyler_lines(path):
    """Returns (texts, codes) stripped of [code] suffix."""
    texts, codes = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            #texts.append(CODE_RE.sub('', row['line']).strip())
            #texts.append("• " + CODE_RE.sub('', row['line']).strip())
            texts.append("• " + row['line'].strip())
            codes.append(row['fyler_code'])
    return texts, codes


def load_fyler_labels(path, keep_sids=None):
    """Returns {sid_str: {code: 0/1}} and sorted list of all codes."""
    with open(path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        code_cols = [c for c in header if c.startswith('fyler_')]
        labels = {}
        for row in reader:
            sid = str(int(float(row['sid'])))
            if keep_sids and sid not in keep_sids:
                continue
            labels[sid] = {c.replace('fyler_', ''): int(row[c]) for c in code_cols}
    all_codes = sorted([c.replace('fyler_', '') for c in code_cols], key=int)
    return labels, all_codes


def encode_lines(texts, tokenizer, encoder, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            embs = encoder(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_embs.append(embs.cpu())
    return torch.cat(all_embs)

"""
def score_study(pool, line_embs, videos, device):
    line_embs = line_embs.to(device)
    videos_t = torch.from_numpy(videos).to(device)
    Q = pool.W_Q(line_embs)
    K = pool.W_K(videos_t)
    attn = (Q @ K.T) * pool.scale
    attended = attn.softmax(dim=-1) @ videos_t
    logits = (line_embs * attended).sum(dim=-1)
    return torch.sigmoid(logits).detach().cpu().numpy()
    #return torch.sigmoid(logits).cpu().numpy()
"""
def score_study(encoder, pool, line_embs, videos, device):
    line_embs = line_embs.to(device)
    videos_t = torch.from_numpy(videos).unsqueeze(0).to(device)
    video_mask = torch.ones(1, videos_t.shape[1], device=device)
    with torch.no_grad():
        attended = pool(line_embs.unsqueeze(0), videos_t, video_mask)
        logits = (line_embs.unsqueeze(0) * attended).sum(dim=-1)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fyler_lines", required=True, help="fyler_lines.csv")
    p.add_argument("--fyler_labels", required=True, help="fyler_labels.csv")
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max_videos", type=int, default=512)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = set(str(int(float(x))) for x in
                   Path(args.manifest).read_text().strip().splitlines())
    # Load Fyler data
    line_texts, line_codes = load_fyler_lines(args.fyler_lines)
    print(f"Loaded {len(line_texts)} Fyler lines", flush=True)

    labels_by_sid, all_codes = load_fyler_labels(args.fyler_labels, keep_sids=manifest)
    print(f"Loaded labels for {len(labels_by_sid)} studies, {len(all_codes)} codes", flush=True)

    # Build line index -> code mapping, and code -> line indices
    code_to_line_idx = defaultdict(list)
    for i, code in enumerate(line_codes):
        code_to_line_idx[code].append(i)


    
    cache_path = Path(args.output_dir) / "fyler_scores.npz"
    if cache_path.exists():
        print(f"Loading cached scores from {cache_path}", flush=True)
        cached = np.load(cache_path)
        line_scores = cached["scores"]
        #line_labels = cached["labels"]
        study_ids = cached["study_ids"].tolist()
        
        keep = [i for i, sid in enumerate(study_ids) if sid in manifest and sid in labels_by_sid]
        study_ids = [study_ids[i] for i in keep]
        line_scores = line_scores[keep]
        
        line_labels = np.zeros((len(study_ids), len(line_codes)), dtype=np.float32)
        for i, sid in enumerate(study_ids):
            for j in range(len(line_codes)):
                line_labels[i, j] = labels_by_sid[sid].get(line_codes[j], 0)
        print(f"Loaded {line_scores.shape[0]} studies x {line_scores.shape[1]} lines", flush=True)
    else:
        # Load video embeddings
        data = np.load(args.video_embeddings)
        embs, sids = data["embeddings"], data["study_ids"].astype(str)
        videos_by_study = {}
        for emb, sid in zip(embs, sids):
            sid = str(int(float(sid)))
            videos_by_study.setdefault(sid, []).append(emb)
        videos_by_study = {k: np.stack(v).astype(np.float32)
                           for k, v in videos_by_study.items()}

        # Intersect: manifest ∩ videos ∩ labels
        study_ids = sorted(sid for sid in manifest
                           if sid in videos_by_study and sid in labels_by_sid)
        print(f"{len(study_ids)} studies with videos + labels in manifest", flush=True)

        # Load model
        encoder = LineEncoder().to(device)
        pool = CrossAttentionPool().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        pool.load_state_dict(ckpt["attn_pool"])
        encoder.eval()
        pool.eval()

        # Encode all Fyler lines
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        line_embs = encode_lines(line_texts, tokenizer, encoder, device)
        print(f"Encoded {line_embs.shape[0]} lines -> {line_embs.shape[1]}d", flush=True)

        # Score each study
        line_scores = np.zeros((len(study_ids), len(line_codes)), dtype=np.float32)
        line_labels = np.zeros((len(study_ids), len(line_codes)), dtype=np.float32)

        for i, sid in enumerate(study_ids):
            if (i + 1) % 500 == 0:
                print(f"  Scoring {i+1}/{len(study_ids)}...", flush=True)

            vids = videos_by_study[sid]
            if vids.shape[0] > args.max_videos:
                idx = np.random.choice(vids.shape[0], args.max_videos, replace=False)
                vids = vids[idx]

            raw_scores = score_study(encoder, pool, line_embs, vids, device)

            for j in range(len(line_codes)):
                line_scores[i, j] = raw_scores[j]
                line_labels[i, j] = labels_by_sid[sid].get(line_codes[j], 0)


    # Compute AUROC per line
    results = {}
    for j in range(len(line_codes)):
        y = line_labels[:, j]
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            continue
        auc = roc_auc_score(y, line_scores[:, j])
        ap = average_precision_score(y, line_scores[:, j])
        results[j] = {"auroc": round(auc, 4), "auprc": round(ap, 4), "n_pos": n_pos,
                       "code": line_codes[j], "line": line_texts[j]}

    ranked = sorted(results.items(), key=lambda x: -x[1]["auroc"])

    print(f"\n{'Code':>6}  {'AUROC':>7}  {'N+':>6}  Line")
    print("-" * 80)
    for idx, r in ranked[:50]:
        print(f"{r['code']:>6}  {r['auroc']:>7.4f}  {r['n_pos']:>6}  {r['line'][:50]}")

    print(f"\n{len(results)} lines evaluated")
    aurocs = [r["auroc"] for r in results.values()]
    print(f"Mean AUROC: {np.mean(aurocs):.4f}  Median: {np.median(aurocs):.4f}")

    # Save
    out = Path(args.output_dir)
    with open(out / "fyler_aurocs.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(out / "fyler_scores.npz",
             study_ids=np.array(study_ids),
             codes=np.array(all_codes),
             scores=line_scores,
             labels=line_labels)
    print(f"\nSaved to {out}")

    with open(out / "fyler_aurocs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "line", "auroc", "auprc", "n_pos"])
        for idx, r in ranked:
            w.writerow([r["code"], r["line"].lstrip("• "), r["auroc"], r["auprc"], r["n_pos"]])



if __name__ == "__main__":
    main()
