"""Merge shards into per-study HDF5 files.

Usage:
    python -u merge_shards_hdf5.py \
        --shard_dir embeddings/clip_encoder_v3_shards \
        --hdf5_output_dir Echo_JEPA_Embeddings/clip_encoder_1024d
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--hdf5_output_dir", required=True)
    p.add_argument("--avi_base", default="lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k")
    args = p.parse_args()

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob("shard_*.npz"))
    print(f"Found {len(shards)} shards", flush=True)

    all_embs, all_sids, all_vids = [], [], []
    for sp in shards:
        d = np.load(sp, allow_pickle=True)
        all_embs.append(d["embeddings"])
        all_sids.append(d["study_ids"])
        all_vids.append(d["video_ids"])
        print(f"  {sp.name}: {len(d['embeddings'])} clips", flush=True)

    embeddings = np.concatenate(all_embs)
    study_ids = np.concatenate(all_sids)
    video_ids = np.concatenate(all_vids)
    print(f"Total: {len(embeddings)} clips, {len(np.unique(study_ids))} studies", flush=True)

    # group: study -> video -> list of clip embeddings
    studies = defaultdict(lambda: defaultdict(list))
    for emb, sid, vid in zip(embeddings, study_ids, video_ids):
        fname = Path(str(vid)).name
        studies[str(sid)][fname].append(emb)

    # write per-study HDF5
    hdf5_dir = Path(args.hdf5_output_dir)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for sid, videos in studies.items():
        h5path = hdf5_dir / f"{sid}_trim_embed.hdf5"
        with h5py.File(h5path, "w") as f:
            f.create_group(f"{sid}_trim")
            for vname, clips in videos.items():
                grp = f.create_group(f"{args.avi_base}/{sid}_trim/{vname}")
                grp.create_dataset("emb", data=np.stack(clips).astype(np.float32))
        n_written += 1
        if n_written % 5000 == 0:
            print(f"  {n_written} studies written", flush=True)

    print(f"Done: {n_written} HDF5 files → {hdf5_dir}", flush=True)


if __name__ == "__main__":
    main()
