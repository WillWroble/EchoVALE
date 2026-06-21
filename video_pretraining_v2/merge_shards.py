"""Merge npz shards from array job into single file."""

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob("shard_*.npz"))
    print(f"Found {len(shards)} shards", flush=True)

    all_embs, all_sids, all_vids = [], [], []
    for shard_path in shards:
        d = np.load(shard_path, allow_pickle=True)
        all_embs.append(d["embeddings"])
        all_sids.append(d["study_ids"])
        all_vids.append(d["video_ids"])
        print(f"  {shard_path.name}: {len(d['embeddings'])} clips", flush=True)

    embeddings = np.concatenate(all_embs)
    #study_ids = np.concatenate(all_sids)
    study_ids = np.concatenate(all_sids).astype(np.int64)
    video_ids = np.concatenate(all_vids)

    print(f"Total: {len(embeddings)} clips, {len(np.unique(study_ids))} studies", flush=True)
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}", flush=True)

    np.savez(args.output, embeddings=embeddings, study_ids=study_ids, video_ids=video_ids)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
