"""Per-head cross-attention view analysis.

Shows which heads specialize in which views, and how specific lines
route differently across heads.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from model import LineEncoder, CrossAttentionPool

VIEW_COLS = ["parasternal_long", "parasternal_short", "subxiphoid_long",
             "apical_4_chamber", "other"]
VIEW_SHORT = ["PLAX", "PSAX", "A4C", "SubX", "Other"]

LINES = [
    "Bicuspid (bicommissural) aortic valve",
    "Aortic stenosis, subvalvar, discrete, membranous",
    "Dilated neo-aortic root",
    "Anomalous origin of a coronary artery from the pulmonary artery",
    "Anomalous origin of the left coronary artery from the right sinus",
    "Mitral regurgitation, moderate",
    "Tricuspid regurgitation, moderate",
    "Atrial septal defect, secundum",
    "Dilated right ventricle",
    "Pericardial effusion",
    "Abdominal situs inversus",
    "Tetralogy of Fallot",
]


@torch.no_grad()
def get_attention_per_head(lines_encoded, video_embs, pool, device):
    video_t = torch.from_numpy(video_embs).unsqueeze(0).float().to(device)
    lines_t = lines_encoded.unsqueeze(0).to(device)

    B, L, _ = lines_t.shape
    V = video_t.shape[1]
    h = pool.num_heads
    d = pool.head_dim

    Q = pool.W_Q(lines_t).view(B, L, h, d).transpose(1, 2)
    K = pool.W_K(video_t).view(B, V, h, d).transpose(1, 2)
    scores = torch.einsum("bhld,bhvd->bhlv", Q, K) * pool.scale
    weights = scores.softmax(dim=-1)

    # return (h, L, V) — per head, per line, per clip
    return weights.squeeze(0).cpu().numpy()


def build_view_lookup(view_labels_path):
    df = pd.read_csv(view_labels_path)
    lookup = {}
    for _, row in df.iterrows():
        eid = str(int(float(row["echo_id"])))
        vfname = str(row["video_filename"])
        for i, col in enumerate(VIEW_COLS):
            if col in df.columns and str(row[col]).strip().lower() == "true":
                lookup[(eid, vfname)] = VIEW_SHORT[i]
                break
        else:
            lookup[(eid, vfname)] = "Unknown"
    return lookup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--view_labels", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    encoder = LineEncoder().to(device)
    pool = CrossAttentionPool().to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    pool.eval()
    n_heads = pool.num_heads
    print(f"Loaded checkpoint ({n_heads} heads)", flush=True)

    #tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    tokens = tokenizer(LINES, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    lines_encoded = encoder(tokens.input_ids.to(device),
                            tokens.attention_mask.to(device)).detach()
    print(f"Encoded {len(LINES)} lines", flush=True)

    view_lookup = build_view_lookup(args.view_labels)
    print(f"View labels: {len(view_lookup)} videos", flush=True)

    labeled_sids = set(eid for eid, _ in view_lookup.keys())

    data = np.load(args.video_embeddings, allow_pickle=True)
    all_sids = data["study_ids"].astype(str)
    all_vids = data["video_ids"].astype(str)
    all_embs = data["embeddings"]

    print("Indexing view-labeled clips...", flush=True)
    study_data = defaultdict(lambda: {"embs": [], "views": []})
    for i in range(len(all_sids)):
        sid = str(int(float(all_sids[i])))
        if sid not in labeled_sids:
            continue
        vfname = all_vids[i].split("/")[-1]
        view = view_lookup.get((sid, vfname))
        if view is not None:
            study_data[sid]["embs"].append(i)
            study_data[sid]["views"].append(view)

    val_ids = set(str(int(float(x)))
                  for x in Path(args.val_manifest).read_text().strip().splitlines())

    # Accumulate: (head, line, view) -> sum of attention, count
    attn_sum = np.zeros((n_heads, len(LINES), len(VIEW_SHORT)))
    clip_count = np.zeros((n_heads, len(LINES), len(VIEW_SHORT)))
    view_to_idx = {v: i for i, v in enumerate(VIEW_SHORT)}
    n_studies = 0

    for sid, sd in study_data.items():
        #if sid not in val_ids:
        #    continue
        if len(sd["embs"]) == 0:
            continue

        clip_embs = all_embs[sd["embs"]].astype(np.float32)
        clip_views = sd["views"]
        attn = get_attention_per_head(lines_encoded, clip_embs, pool, device)
        # attn: (n_heads, n_lines, n_clips)

        n_studies += 1
        for ci, view in enumerate(clip_views):
            vi = view_to_idx.get(view, -1)
            if vi < 0:
                continue
            for hi in range(n_heads):
                for li in range(len(LINES)):
                    attn_sum[hi, li, vi] += attn[hi, li, ci]
                    clip_count[hi, li, vi] += 1

    print(f"\nStudies analyzed (val): {n_studies}")

    # Compute avg attention per (head, line, view)
    avg_attn = np.where(clip_count > 0, attn_sum / clip_count, 0)
    views_used = ["PLAX", "PSAX", "A4C", "SubX"]
    vi_used = [view_to_idx[v] for v in views_used]

    # 1. Head baseline profiles (averaged across all lines)
    print(f"\n{'='*80}")
    print("HEAD BASELINE PROFILES (avg across all lines)")
    print(f"{'='*80}")
    print(f"{'Head':>6s} | {'PLAX':>8s} | {'PSAX':>8s} | {'A4C':>8s} | {'SubX':>8s} | Peak")
    print("-" * 65)
    for hi in range(n_heads):
        vals = [avg_attn[hi, :, vi].mean() for vi in vi_used]
        peak = views_used[np.argmax(vals)]
        print(f"  {hi:>4d} | " + " | ".join(f"{v:>8.5f}" for v in vals) + f" | {peak}")

    # 2. Per-line per-head view preference
    print(f"\n{'='*80}")
    print("PER-LINE PER-HEAD VIEW PREFERENCE")
    print(f"{'='*80}")

    short_names = {
        "Bicuspid (bicommissural) aortic valve": "Bicuspid AV",
        "Aortic stenosis, subvalvar, discrete, membranous": "Subvalvar AS",
        "Dilated neo-aortic root": "Neo-aortic root",
        "Anomalous origin of a coronary artery from the pulmonary artery": "ALCAPA",
        "Anomalous origin of the left coronary artery from the right sinus": "Anomalous LCA",
        "Mitral regurgitation, moderate": "Mitral regurg.",
        "Tricuspid regurgitation, moderate": "Tricuspid regurg.",
        "Atrial septal defect, secundum": "ASD secundum",
        "Dilated right ventricle": "Dilated RV",
        "Pericardial effusion": "Pericardial eff.",
        "Abdominal situs inversus": "Situs inversus",
        "Tetralogy of Fallot": "TOF",
    }

    for li, line in enumerate(LINES):
        name = short_names.get(line, line[:20])
        print(f"\n  {name}")
        print(f"  {'Head':>6s} | {'PLAX':>8s} | {'PSAX':>8s} | {'A4C':>8s} | {'SubX':>8s} | Peak")
        print(f"  " + "-" * 63)
        for hi in range(n_heads):
            vals = [avg_attn[hi, li, vi] for vi in vi_used]
            peak = views_used[np.argmax(vals)]
            print(f"  {hi:>6d} | " + " | ".join(f"{v:>8.5f}" for v in vals) + f" | {peak}")


if __name__ == "__main__":
    main()
