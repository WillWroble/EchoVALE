"""Skip-gram dataset with per-study video embeddings for cross-attention pooling."""

import re
import numpy as np
import h5py
import torch
from collections import Counter
from torch.utils.data import Dataset


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


def load_videos_by_study(npz_path):
    """Group flat (N_videos, 768) array by study_id."""
    data = np.load(npz_path)
    embs = data["embeddings"]
    sids = data["study_ids"].astype(str)
    by_study = {}
    for emb, sid in zip(embs, sids):
        sid_clean = str(int(float(sid)))
        by_study.setdefault(sid_clean, []).append(emb)
    by_study = {k: np.stack(v).astype(np.float32) for k, v in by_study.items()}
    print(f"Loaded {len(by_study):,} studies from {npz_path}", flush=True)
    return by_study


FIELD_CONFIG = {
    'study_findings': {'K': 2, 'M': 10},
    'summary': {'K': 2, 'M': 10},
    'history': {'K': 1, 'M': 5},
}


class SkipGramDataset(Dataset):

    def __init__(self, h5_dir, study_ids, videos_by_study, field='study_findings',
                 subsample_t=1e-3, max_videos=128, line_filters=None):
        self.field = field
        self.K = FIELD_CONFIG[field]['K']
        self.M = FIELD_CONFIG[field]['M']
        self.max_videos = max_videos

        if line_filters:
            patterns = [re.compile(l.strip(), re.IGNORECASE) for l in open(line_filters)
                        if l.strip() and not l.startswith("#")]
        else:
            patterns = []

        def keep(line):
            return not any(p.search(line) for p in patterns)

        # Load lines per study
        embs_set = set(study_ids) & set(videos_by_study.keys())
        self.study_lines = {}

        h5_path = f"{h5_dir}/{field}.h5"
        with h5py.File(h5_path, "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid in embs_set:
                    lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                             for x in f[sid_raw][()]]
                    lines = merge_soft_wraps(lines)
                    lines = [l for l in lines if keep(l)]
                    if len(lines) >= self.K:
                        self.study_lines[sid] = lines

        self.study_ids = list(self.study_lines.keys())
        self.videos_by_study = videos_by_study
        print(f"SkipGramDataset[{field}]: {len(self.study_ids):,} studies", flush=True)

        # Line frequency counts
        counter = Counter()
        for lines in self.study_lines.values():
            counter.update(lines)
        total = sum(counter.values())
        print(f"  {len(counter):,} unique lines, {total:,} total occurrences", flush=True)

        # Subsample keep probability (word2vec formula)
        self.line_keep_prob = {
            line: min(1.0, np.sqrt(subsample_t / (count / total)))
            for line, count in counter.items()
        }

        # Negative sampling pool (freq^0.75)
        self.all_lines = list(counter.keys())
        freqs = np.array([counter[l] for l in self.all_lines], dtype=np.float64)
        freqs = freqs ** 0.75
        self.neg_probs = freqs / freqs.sum()

        # Pre-tokenize all unique lines
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
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
        line_set = set(lines)
        videos = self.videos_by_study[sid]

        # Cap videos
        if videos.shape[0] > self.max_videos:
            sub = np.random.choice(videos.shape[0], self.max_videos, replace=False)
            videos = videos[sub]

        
        # Positive selection with frequency downsampling
        """
        kept = [l for l in lines if np.random.rand() < self.line_keep_prob.get(l, 1.0)]
        if len(kept) < self.K:
            kept = lines
        """
        kept = lines
        sel = np.random.choice(len(kept), size=self.K, replace=False)
        positives = [kept[i] for i in sel]

        # Negative selection, reject if in anchor study
        negatives = []
        while len(negatives) < self.M:
            #i = np.random.choice(len(self.all_lines), p=self.neg_probs)
            i = np.random.choice(len(self.all_lines))
            candidate = self.all_lines[i]
            if candidate not in line_set:
                negatives.append(candidate)

        all_lines = positives + negatives
        ids = np.stack([self.token_ids[l] for l in all_lines])
        masks = np.stack([self.token_masks[l] for l in all_lines])
        labels = [1.0] * self.K + [0.0] * self.M
        return ids, masks, videos, labels


def collate_fn(batch):
    """Pad videos to batch max. Returns flat lines + per-study video tensors."""
    max_vids = max(v.shape[0] for _, _, v, _ in batch)
    dim = batch[0][2].shape[1]
    B = len(batch)

    all_ids, all_masks, all_labels = [], [], []
    videos = np.zeros((B, max_vids, dim), dtype=np.float32)
    video_mask = np.zeros((B, max_vids), dtype=np.float32)

    for i, (ids, masks, vids, labels) in enumerate(batch):
        all_ids.append(ids)
        all_masks.append(masks)
        all_labels.extend(labels)
        n = vids.shape[0]
        videos[i, :n] = vids
        video_mask[i, :n] = 1.0

    return (
        torch.from_numpy(np.concatenate(all_ids)),
        torch.from_numpy(np.concatenate(all_masks)),
        torch.from_numpy(videos),
        torch.from_numpy(video_mask),
        torch.tensor(all_labels),
    )
