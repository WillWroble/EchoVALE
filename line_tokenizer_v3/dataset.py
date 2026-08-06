"""Study-level dataset with live AVI loading for end-to-end training.

Loads clips directly from AVIs, merges lines across fields into one pool,
supports Bernoulli video dropout and shared batch negatives.
"""

import re
import numpy as np
import h5py
import torch
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset

import sys
#sys.path.insert(0, '/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes')
from encoder import load_and_sample_clips


def merge_soft_wraps(lines):
    if not lines:
        return []
    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if line[0].islower() or prev.endswith('-'):
            sep = '' if prev.endswith('-') else ' '
            merged[-1] = prev.rstrip('-') + sep + line
        else:
            merged.append(line)
    return merged


class StudyAVIDataset(Dataset):

    def __init__(self, h5_dir, study_ids, data_dir, fields, K, M,
                 subsample_t=1e-3, clip_dropout=0.8, max_clips=32,
                 num_clips_per_video=2, line_filters=None):
        self.K = K
        self.M = M
        self.data_dir = Path(data_dir)
        self.clip_dropout = clip_dropout
        self.max_clips = max_clips
        self.num_clips_per_video = num_clips_per_video

        if line_filters:
            patterns = [re.compile(l.strip(), re.IGNORECASE) for l in open(line_filters)
                        if l.strip() and not l.startswith("#")]
        else:
            patterns = []

        def keep(line):
            return not any(p.search(line) for p in patterns)

        # Load and merge lines across fields
        study_set = set(study_ids)
        self.study_lines = {}

        for field in fields:
            h5_path = f"{h5_dir}/{field}.h5"
            with h5py.File(h5_path, "r") as f:
                for sid_raw in f.keys():
                    sid = str(int(float(sid_raw)))
                    if sid not in study_set:
                        continue
                    lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                             for x in f[sid_raw][()]]
                    lines = merge_soft_wraps(lines)
                    lines = [l.lstrip("\u2022 ").strip() for l in lines]
                    lines = [l for l in lines if keep(l)]
                    if sid not in self.study_lines:
                        self.study_lines[sid] = set()
                    self.study_lines[sid].update(lines)

        self.study_lines = {sid: list(lines) for sid, lines in self.study_lines.items()
                            if len(lines) >= K}
        self.study_ids = [s for s in study_ids if s in self.study_lines]
        print(f"StudyAVIDataset: {len(self.study_ids):,} studies, "
              f"{len(fields)} fields merged", flush=True)

        # Frequency statistics for merged pool
        counter = Counter()
        for lines in self.study_lines.values():
            counter.update(lines)
        total = sum(counter.values())
        print(f"  {len(counter):,} unique lines, {total:,} total occurrences", flush=True)

        self.line_keep_prob = {
            line: min(1.0, np.sqrt(subsample_t / (count / total)))
            for line, count in counter.items()
        }

        self.all_lines = list(counter.keys())
        freqs = np.array([counter[l] for l in self.all_lines], dtype=np.float64)
        freqs = freqs ** 0.75
        self.neg_probs = freqs / freqs.sum()

        # Pre-tokenize
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
        all_unique = list(set(l for lines in self.study_lines.values() for l in lines))
        print(f"  Pre-tokenizing {len(all_unique):,} unique lines ...", flush=True)
        enc = tokenizer(all_unique, padding="max_length", truncation=True,
                        max_length=128, return_tensors="np")
        self.token_ids = {}
        self.token_masks = {}
        for i, line in enumerate(all_unique):
            self.token_ids[line] = enc["input_ids"][i]
            self.token_masks[line] = enc["attention_mask"][i]

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines = self.study_lines[sid]

        # Load clips from AVIs
        study_dir = self.data_dir / f"{sid}_trim"
        avis = sorted(study_dir.glob("*.avi")) if study_dir.exists() else []

        # Bernoulli video dropout
        if self.clip_dropout > 0 and len(avis) > 2:
            keep_mask = np.random.rand(len(avis)) > self.clip_dropout
            if keep_mask.sum() < 2:
                keep_mask[:2] = True
            avis = [a for a, k in zip(avis, keep_mask) if k]

        all_clips = []
        for avi in avis:
            clips = load_and_sample_clips(avi, num_clips=self.num_clips_per_video)
            if clips is not None:
                all_clips.append(clips)
            if len(all_clips) * self.num_clips_per_video >= self.max_clips:
                break

        if all_clips:
            clips = torch.cat(all_clips, dim=0)[:self.max_clips]
        else:
            clips = torch.zeros(1, 3, 16, 224, 224)

        # Sample positives with freq downsampling
        kept = [l for l in lines if np.random.rand() < self.line_keep_prob.get(l, 1.0)]
        if len(kept) < self.K:
            kept = lines
        sel = np.random.choice(len(kept), size=self.K, replace=False)
        positives = [kept[i] for i in sel]

        ids = np.stack([self.token_ids[l] for l in positives])
        masks = np.stack([self.token_masks[l] for l in positives])

        return clips, ids, masks, clips.shape[0]

    def sample_negatives(self, M):
        idx = np.random.choice(len(self.all_lines), M, replace=False, p=self.neg_probs)
        lines = [self.all_lines[i] for i in idx]
        ids = torch.stack([torch.from_numpy(self.token_ids[l]) for l in lines])
        masks = torch.stack([torch.from_numpy(self.token_masks[l]) for l in lines])
        return ids, masks


def collate_fn(batch):
    clips_list, ids_list, masks_list, n_clips_list = zip(*batch)

    all_clips = torch.cat(clips_list, dim=0)
    all_ids = torch.from_numpy(np.concatenate(ids_list))
    all_masks = torch.from_numpy(np.concatenate(masks_list))
    n_clips = list(n_clips_list)

    return all_clips, all_ids, all_masks, n_clips
