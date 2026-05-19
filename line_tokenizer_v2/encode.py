"""Encode all unique lines through trained LineEncoder."""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer
from dataset import merge_soft_wraps

from model import LineEncoder

MAX_LENGTH = 128


def collect_unique_lines(h5_path, line_filters=None):
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE) for l in open(line_filters)
                    if l.strip() and not l.startswith("#")]
    else:
        patterns = []

    unique = set()
    with h5py.File(h5_path, "r") as f:
        for sid in f.keys():
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                     for x in f[sid][()]]
            lines = merge_soft_wraps(lines)  # add this
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            unique.update(lines)
    print(f"Collected {len(unique):,} unique lines", flush=True)
    return sorted(unique)


@torch.no_grad()
def encode_all(model, tokenizer, lines, batch_size, device):
    model.eval()
    all_embs = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt")
        embs = model(tokens.input_ids.to(device), tokens.attention_mask.to(device))
        all_embs.append(embs.cpu().numpy())
        if (i // batch_size + 1) % 100 == 0:
            print(f"  {i + len(batch):,}/{len(lines):,}", flush=True)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    model = LineEncoder().to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt)
        print("Loaded checkpoint", flush=True)
    else:
        print("Using pretrained BERT (no checkpoint)", flush=True)

    lines = collect_unique_lines(args.h5_path, args.line_filters)
    embeddings = encode_all(model, tokenizer, lines, args.batch_size, device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embeddings=embeddings, lines=np.array(lines))
    print(f"Saved {embeddings.shape} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
