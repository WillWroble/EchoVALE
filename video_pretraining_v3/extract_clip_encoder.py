"""Single-GPU clip encoder extraction for SLURM array job.

Each task processes a chunk of studies, outputs one npz shard.
Merge shards with merge_shards.py after all tasks complete.

Adapted from video_pretraining_v2/extract_jepa_clips.py:
replaces patches.mean(dim=1) with AttentivePooler CLS.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA")

from src.models.vision_transformer import vit_large
from src.models.attentive_pooler import AttentivePooler
from src.hub.backbones import _clean_backbone_key
from src.datasets.video_dataset import VideoDataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


class ClipTransform:
    def __init__(self, resolution=224):
        self.resolution = resolution

    def __call__(self, buffer):
        x = torch.from_numpy(buffer).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        x = x.permute(1, 0, 2, 3)
        return (x - MEAN) / STD


def load_encoder(checkpoint_path, device):
    model = vit_large(
        img_size=(224, 224), patch_size=16, num_frames=16,
        tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("target_encoder") or ckpt.get("encoder")
    state = _clean_backbone_key(state)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_pooler(v3_checkpoint, device, embed_dim=1024, num_heads=16, depth=4):
    pooler = AttentivePooler(
        num_queries=1, embed_dim=embed_dim,
        num_heads=num_heads, depth=depth,
    )
    ckpt = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["pooler"].items()}
    pooler.load_state_dict(state)
    pooler.to(device).eval()
    for p in pooler.parameters():
        p.requires_grad_(False)
    return pooler


def extract_study_id(path):
    sid = path.split("/")[-2]
    if sid.endswith("_trim"):
        sid = sid[:-5]
    return sid


def build_task_csv(avi_manifest, study_manifest, task_id, n_tasks, output_csv):
    keep = set(Path(study_manifest).read_text().split())
    all_videos = []
    with open(avi_manifest) as f:
        for line in f:
            path = line.split(maxsplit=1)[0]
            sid = extract_study_id(path)
            if sid in keep:
                all_videos.append((path, sid))

    all_studies = sorted(set(sid for _, sid in all_videos))
    my_studies = set(all_studies[task_id::n_tasks])
    my_videos = [(p, s) for p, s in all_videos if s in my_studies]

    sid_to_int = {s: i for i, s in enumerate(sorted(my_studies))}
    lines = [f"{path} {sid_to_int[sid]}" for path, sid in my_videos]

    Path(output_csv).write_text("\n".join(lines) + "\n")
    with open(str(output_csv).replace(".csv", ".sid_map.txt"), "w") as f:
        for sid, i in sorted(sid_to_int.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{sid}\n")

    return len(my_studies), len(my_videos), sid_to_int


def collate_fn(batch):
    all_clips, all_labels = [], []
    for buffer_list, label, _ in batch:
        for clip in buffer_list:
            all_clips.append(clip)
            all_labels.append(label)
    return torch.stack(all_clips), all_labels


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--avi_manifest", required=True)
    p.add_argument("--study_manifest", required=True)
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--v3_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_clips", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_tasks", type=int, default=128)
    args = p.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = args.n_tasks

    device = torch.device("cuda")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build task-specific CSV
    tmp_csv = Path(args.output_dir) / f"tmp_task_{task_id}.csv"
    n_studies, n_videos, sid_to_int = build_task_csv(
        args.avi_manifest, args.study_manifest, task_id, n_tasks, tmp_csv
    )
    int_to_sid = {v: k for k, v in sid_to_int.items()}

    print(f"Task {task_id}/{n_tasks}: {n_studies} studies, {n_videos} videos", flush=True)

    if n_videos == 0:
        print("  No videos, saving empty shard", flush=True)
        np.savez(
            Path(args.output_dir) / f"shard_{task_id:03d}.npz",
            embeddings=np.zeros((0, 1024), dtype=np.float16),
            study_ids=np.array([], dtype=object),
            video_ids=np.array([], dtype=object),
        )
        tmp_csv.unlink()
        return

    # Dataset
    transform = ClipTransform(resolution=224)
    dataset = VideoDataset(
        data_paths=[str(tmp_csv)],
        frames_per_clip=16,
        frame_step=None,
        fps=8,
        num_clips=args.num_clips,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        filter_short_videos=False,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Models
    encoder = load_encoder(args.encoder_checkpoint, device)
    pooler = load_pooler(args.v3_checkpoint, device)
    print(f"  Encoder + pooler loaded", flush=True)

    # Extract
    all_embs, all_sids, all_vids = [], [], []
    for step, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patches = encoder(clips)
            embs = pooler(patches).squeeze(1)
        embs = embs.float().cpu().numpy()

        for i, label in enumerate(labels):
            all_embs.append(embs[i])
            all_sids.append(int_to_sid[label])
            all_vids.append(dataset.samples[step * args.batch_size + i // args.num_clips])

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}, {len(all_embs)} clips", flush=True)

    # Save shard
    embeddings = np.stack(all_embs).astype(np.float16)
    study_ids = np.array(all_sids, dtype=object)
    video_ids = np.array(all_vids, dtype=object)

    out_path = Path(args.output_dir) / f"shard_{task_id:03d}.npz"
    np.savez(out_path, embeddings=embeddings, study_ids=study_ids, video_ids=video_ids)
    print(f"Task {task_id}: saved {len(embeddings)} clips to {out_path}", flush=True)

    tmp_csv.unlink()
    Path(str(tmp_csv).replace(".csv", ".sid_map.txt")).unlink()


if __name__ == "__main__":
    main()
