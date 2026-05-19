"""Score all pool lines against each eval study via cross-attention.

Same output format as generate.py: hotspots + reference, now per-field.

Usage:
    python -u generate_lines.py \
        --checkpoint results/v2/best.pt \
        --video_embeddings /path/to/infonce_768_all.npz \
        --h5_dir /path/to/line_tokenizer/data \
        --pool_manifest /path/to/train_modern.txt \
        --eval_manifest /path/to/eval_manifest.txt \
        --output results/v2/line_scores.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from model import LineEncoder, CrossAttentionPool
from dataset import merge_soft_wraps


def load_video_embeddings_by_study(npz_path):
    data = np.load(npz_path)
    embs, sids = data["embeddings"], data["study_ids"].astype(str)
    by_study = {}
    for emb, sid in zip(embs, sids):
        by_study.setdefault(sid, []).append(emb)
    return {k: np.stack(v).astype(np.float32) for k, v in by_study.items()}


def collect_pool_lines(h5_path, manifest_path, line_filters=None):
    patterns = []
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(line_filters) if l.strip() and not l.startswith("#")]

    manifest = set(str(int(float(x))) for x in Path(manifest_path).read_text().strip().splitlines())

    unique = set()
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in manifest:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            lines = merge_soft_wraps(lines)
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            unique.update(lines)

    return sorted(unique)


def load_study_lines(h5_path, study_ids, line_filters=None):
    patterns = []
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(line_filters) if l.strip() and not l.startswith("#")]
    study_lines = {}

    ids_set = set(study_ids)
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in ids_set:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            lines = merge_soft_wraps(lines)
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            study_lines[sid] = lines
    return study_lines


@torch.no_grad()
def encode_lines(lines, encoder, tokenizer, device, batch_size=512):
    encoder.eval()
    all_embs = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        embs = encoder(tokens.input_ids.to(device), tokens.attention_mask.to(device))
        all_embs.append(embs.cpu())
        if (i // batch_size + 1) % 100 == 0:
            print(f"    Encoded {i + len(batch):,}/{len(lines):,}", flush=True)
    return torch.cat(all_embs, dim=0)


def get_or_encode_lines(field, h5_path, pool_manifest, line_filters, encoder, tokenizer, device, cache_dir):
    cache_path = f"{cache_dir}/{field}_pool_embs.npz"

    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        return data["lines"].tolist(), torch.from_numpy(data["embeddings"].astype(np.float32))

    lines = collect_pool_lines(h5_path, pool_manifest, line_filters)
    print(f"  Pool: {len(lines):,} unique lines", flush=True)
    embs = encode_lines(lines, encoder, tokenizer, device)
    print(f"  Encoded {len(lines):,} lines", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, lines=np.array(lines), embeddings=embs.numpy())
    print(f"  Cached to {cache_path}", flush=True)

    return lines, embs


@torch.no_grad()
def score_study(line_embs, videos, pool, device):
    line_embs = line_embs.to(device)
    videos_t = torch.from_numpy(videos).unsqueeze(0).to(device)
    video_mask = torch.ones(1, videos_t.shape[1], device=device)
    attended = pool(line_embs.unsqueeze(0), videos_t, video_mask)
    logits = (line_embs.unsqueeze(0) * attended).sum(dim=-1)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()

def find_hotspots(scores, line_embs, threshold=0.3, knn=10):
    active = np.where(scores > threshold)[0]
    if len(active) == 0:
        return []
    if len(active) == 1:
        return [[(int(active[0]), float(scores[active[0]]))]]

    # Cosine similarity on active lines
    active_embs = line_embs[active].numpy()
    norms = np.linalg.norm(active_embs, axis=1, keepdims=True) + 1e-8
    active_norm = active_embs / norms
    sim = active_norm @ active_norm.T

    # Mutual KNN graph
    n = len(active)
    k = min(knn, n - 1)
    top_k = np.argsort(-sim, axis=1)[:, 1:k+1]  # exclude self

    # Build adjacency: mutual = both must have each other in top-k
    adjacency = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in top_k[i]:
            if i in top_k[j]:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    hotspots = []
    for c in range(n_components):
        members = active[labels == c]
        ranked = sorted([(int(m), float(scores[m])) for m in members], key=lambda x: -x[1])
        hotspots.append(ranked)

    hotspots.sort(key=lambda h: -h[0][1])
    return hotspots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--pool_manifest", required=True)
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fields", nargs="+", default=["study_findings", "summary", "history"])
    p.add_argument("--line_filters", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--knn_findings", type=int, default=10)
    p.add_argument("--knn_summary", type=int, default=20)
    p.add_argument("--knn_history", type=int, default=10)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    encoder = LineEncoder().to(device)
    pool = CrossAttentionPool(dim=768).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    pool.eval()
    print("Loaded checkpoint", flush=True)

    # Load videos
    print("Loading video embeddings...", flush=True)
    video_embs = load_video_embeddings_by_study(args.video_embeddings)
    print(f"Loaded {len(video_embs):,} studies with video embeddings", flush=True)

    # Eval studies
    eval_ids = set(str(int(float(x))) for x in Path(args.eval_manifest).read_text().strip().splitlines())
    ids = [s for s in video_embs.keys() if s in eval_ids]
    print(f"Eval manifest: {len(ids):,} studies", flush=True)

    # Per-field: collect pool lines, encode (or load cache), load reference lines
    pool_lines = {}
    line_embs = {}
    line_to_idx = {}
    study_lines = {}
    novel_embs = {}
    novel_to_idx = {}
    cache_dir = Path(args.output).parent / "pool_cache"

    for field in args.fields:
        h5_path = f"{args.h5_dir}/{field}.h5"
        print(f"\n[{field}]", flush=True)

        pool_lines[field], line_embs[field] = get_or_encode_lines(
            field, h5_path, args.pool_manifest, args.line_filters,
            encoder, tokenizer, device, cache_dir
        )

        study_lines[field] = load_study_lines(h5_path, ids, args.line_filters)
        print(f"  Reference: {len(study_lines[field]):,} studies", flush=True)

        line_to_idx[field] = {l: i for i, l in enumerate(pool_lines[field])}
        novel = sorted(set(
            l for lines in study_lines[field].values() for l in lines
            if l not in line_to_idx[field]
        ))
        if novel:
            novel_embs[field] = encode_lines(novel, encoder, tokenizer, device)
            novel_to_idx[field] = {l: i for i, l in enumerate(novel)}
            print(f"  Novel: {len(novel):,} lines encoded", flush=True)
        else:
            novel_embs[field] = torch.zeros(0, 768)
            novel_to_idx[field] = {}

    # Generate per-study, per-field
    print(f"\nGenerating heatmaps...", flush=True)
    results = []
    knns = {
        "study_findings": args.knn_findings,
        "summary": args.knn_summary,
        "history": args.knn_history,
    }
    for i, sid in enumerate(ids):
        videos = video_embs[sid]
        result = {"study_id": sid}

        for field in args.fields:
            scores = score_study(line_embs[field], videos, pool, device)
            hotspots = find_hotspots(scores, line_embs[field], args.threshold, knns[field])

            hotspot_data = []
            for h_members in hotspots:
                top = h_members[:args.top_k]
                hotspot_data.append([
                    {"score": round(score, 4), "text": pool_lines[field][idx]}
                    for idx, score in top
                ])


            ref_lines = study_lines[field].get(sid, [])
            ref_data = []
            n_unmapped = 0

            # Batch-score novel lines for this study
            novel_ref = [line for line in ref_lines
                         if line not in line_to_idx[field] and line in novel_to_idx[field]]
            novel_scores = {}
            if novel_ref:
                idxs = [novel_to_idx[field][l] for l in novel_ref]
                ns = score_study(novel_embs[field][idxs], videos, pool, device)
                novel_scores = {l: float(ns[k]) for k, l in enumerate(novel_ref)}

            for line in ref_lines:
                idx = line_to_idx[field].get(line)
                if idx is not None:
                    ref_data.append({"text": line, "score": round(float(scores[idx]), 4)})
                elif line in novel_scores:
                    ref_data.append({"text": line, "score": round(novel_scores[line], 4), "ood": True})
                else:
                    ref_data.append({"text": line, "score": None})
                    n_unmapped += 1
            
            top_scores = np.sort(scores)[::-1]

            result[field] = {
                "n_hotspots": len(hotspot_data),
                "n_active": int((scores > args.threshold).sum()),
                "n_unmapped": n_unmapped,
                "top_5_scores": [round(float(s), 4) for s in top_scores[:5]],
                "hotspots": hotspot_data,
                "reference": ref_data,
            }

        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary stats
    print(f"\nSaved {len(results)} studies to {args.output}", flush=True)
    for field in args.fields:
        total_unmapped = sum(r[field]["n_unmapped"] for r in results)
        total_ref = sum(len(r[field]["reference"]) for r in results)
        print(f"  {field}: unmapped {total_unmapped}/{total_ref} ({100*total_unmapped/max(total_ref,1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
