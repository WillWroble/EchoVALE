"""Analyze cross-attention view preferences for specific lines.

Usage:
    python -u view_attention.py \
        --checkpoint results/v24/best.pt \
        --video_embeddings .../jepa_clips_4x768_fixed.npz \
        --view_labels .../view_labels.csv \
        --train_manifest .../train_50_nofetal.txt \
        --val_manifest .../val_50_nofetal.txt
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
VIEW_SHORT = ["PLAX", "PSAX", "SubX", "A4C", "Other"]

LINES = [
    # HLHS
    "Hypoplastic left heart syndrome",
    "Hypoplastic left ventricle",
    "Hypoplastic aortic arch",
    "Single right ventricle",
    # TOF
    "Tetralogy of Fallot",
    "Right ventricular outflow tract obstruction",
    "Ventricular septal defect, membranous",
    "Overriding aorta",
    "Right ventricular hypertrophy",
    # Parasternal
    "Bicuspid (bicommissural) aortic valve",
    "Aortic stenosis, subvalvar, discrete, membranous",
    "Dilated neo-aortic root",
    "Aneurysm, ascending aorta",
    # Coronary (PSAX)
    "Anomalous origin of a coronary artery from the pulmonary artery",
    "Anomalous origin of the left coronary artery from the right sinus",
    "Atypical coronary arteries in d-loop TGA",
    # Apical / A4C
    "Hypertrophic cardiomyopathy, apical",
    "Mitral regurgitation, moderate",
    "Atrial septal defect, secundum",
    "Dilated right ventricle",
    "Tricuspid regurgitation, moderate",
    "Pericardial effusion",
    # Subxiphoid
    "Abdominal situs inversus",
]
@torch.no_grad()
def get_attention_weights(lines_encoded, video_embs, pool, device):
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

    return weights.squeeze(0).mean(dim=0).cpu().numpy()


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

    # Load model
    encoder = LineEncoder().to(device)
    pool = CrossAttentionPool().to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    pool.eval()
    print("Loaded checkpoint", flush=True)

    # Encode lines
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    tokens = tokenizer(LINES, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    lines_encoded = encoder(tokens.input_ids.to(device),
                            tokens.attention_mask.to(device)).detach()
    print(f"Encoded {len(LINES)} lines", flush=True)

    # Load view labels → {(echo_id, video_filename): view}
    vdf = pd.read_csv(args.view_labels)
    view_lookup = {}
    for _, row in vdf.iterrows():
        eid = str(int(float(row["echo_id"])))
        vfname = str(row["video_filename"])
        for i, col in enumerate(VIEW_COLS):
            if col in vdf.columns and str(row[col]).strip().lower() == "true":
                view_lookup[(eid, vfname)] = VIEW_SHORT[i]
                break
        else:
            view_lookup[(eid, vfname)] = "Unknown"
    print(f"View labels: {len(view_lookup)} videos", flush=True)

    # Get set of study IDs that have view labels
    labeled_sids = set(eid for eid, _ in view_lookup.keys())

    # Load NPZ — index only clips belonging to view-labeled studies
    data = np.load(args.video_embeddings, allow_pickle=True)
    all_sids = data["study_ids"].astype(str)
    all_vids = data["video_ids"].astype(str)

    # Find clips that match view-labeled videos
    print("Indexing view-labeled clips...", flush=True)
    study_data = defaultdict(lambda: {"embs": [], "views": []})
    n_matched = 0
    for i in range(len(all_sids)):
        sid = str(int(float(all_sids[i])))
        if sid not in labeled_sids:
            continue
        vfname = all_vids[i].split("/")[-1]
        view = view_lookup.get((sid, vfname))
        if view is None:
            continue
        study_data[sid]["embs"].append(i)
        study_data[sid]["views"].append(view)
        n_matched += 1
    print(f"Matched {n_matched:,} clips across {len(study_data):,} studies", flush=True)

    # Load only matched embeddings
    all_embs = data["embeddings"]

    # Manifests
    train_ids = set(str(int(float(x)))
                    for x in Path(args.train_manifest).read_text().strip().splitlines())
    val_ids = set(str(int(float(x)))
                  for x in Path(args.val_manifest).read_text().strip().splitlines())

    # Accumulate per-view attention
    attn_sum = {s: {line: defaultdict(float) for line in LINES}
                for s in ["train", "val", "combined"]}
    clip_count = {s: {line: defaultdict(int) for line in LINES}
                  for s in ["train", "val", "combined"]}
    n_studies = {"train": 0, "val": 0, "combined": 0}

    for sid, sd in study_data.items():
        if sid in train_ids:
            split = "train"
        elif sid in val_ids:
            split = "val"
        else:
            continue

        clip_embs = all_embs[sd["embs"]].astype(np.float32)
        clip_views = sd["views"]
        attn = get_attention_weights(lines_encoded, clip_embs, pool, device)

        n_studies[split] += 1
        n_studies["combined"] += 1

        for li, line in enumerate(LINES):
            for ci, view in enumerate(clip_views):
                for s in [split, "combined"]:
                    attn_sum[s][line][view] += attn[li, ci]
                    clip_count[s][line][view] += 1

    # Print results
    print(f"\n{'='*80}")
    print(f"Studies analyzed: train={n_studies['train']}, val={n_studies['val']}, "
          f"combined={n_studies['combined']}")
    print(f"{'='*80}")

    all_views = VIEW_SHORT + ["Unknown"]
    for split in ["train", "val", "combined"]:
        print(f"\n--- {split.upper()} ---")
        for line in LINES:
            print(f"\n  Line: \"{line}\"")
            views_with_data = [(v, attn_sum[split][line][v], clip_count[split][line][v])
                               for v in all_views if clip_count[split][line][v] > 0]
            if not views_with_data:
                print("    No data")
                continue

            total_attn = sum(a for _, a, _ in views_with_data)
            for view, total, n in sorted(views_with_data, key=lambda x: -x[1]/x[2]):
                avg = total / n
                pct = 100 * total / total_attn if total_attn > 0 else 0
                print(f"    {view:>8s}: avg_attn={avg:.4f}  share={pct:5.1f}%  (n_clips={n:,})")


if __name__ == "__main__":
    main()
