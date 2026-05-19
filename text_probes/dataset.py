"""Dataset for text-only probes: line embeddings → measurement prediction."""

import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CONTINUOUS_COLS = [
    "AA01", "AR01", "EF05", "LD05", "LE05", "LE07", "LM12", "LS04",
    "MA02", "MP01", "PA02", "TA01", "LA34", "RA06", "RV19", "ST39", "ST49", "RV32",
]
BINARY_COLS = [
    "1470s", "1450", "1411", "1530", "3520", "1730", "1813",
    "1650", "1610", "1639", "3436", "1401", "3608", "1610_1639_3436", "is_EF",
]
ALL_COLS = CONTINUOUS_COLS + BINARY_COLS
N_CONT = len(CONTINUOUS_COLS)
N_BIN = len(BINARY_COLS)


def merge_soft_wraps(lines):
    merged = []
    for line in lines:
        if merged and (line and line[0].islower()
                       or (merged[-1] and merged[-1].endswith("-"))):
            merged[-1] = merged[-1].rstrip("-") + line
        else:
            merged.append(line)
    return merged


def load_line_lookup(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    lines, embs = data["lines"].astype(str), data["embeddings"]
    return dict(zip(lines, embs))


class TextProbeDataset(Dataset):
    def __init__(self, h5_dir, field, line_lookup, targets_df,
                 manifest_path, line_filters=None, max_lines=128,
                 cont_mean=None, cont_std=None):
        self.line_lookup = line_lookup
        self.max_lines = max_lines
        self.cont_mean = cont_mean
        self.cont_std = cont_std

        patterns = []
        if line_filters and Path(line_filters).exists():
            patterns = [re.compile(l.strip(), re.IGNORECASE)
                        for l in open(line_filters)
                        if l.strip() and not l.startswith("#")]

        manifest = set(str(int(float(x)))
                       for x in Path(manifest_path).read_text().strip().splitlines())

        # target lookup: eid → (values, mask)
        target_lookup = {}
        for _, row in targets_df.iterrows():
            eid = str(int(float(row["eid"])))
            vals = np.array([row.get(c, np.nan) for c in ALL_COLS], dtype=np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, 0.0)
            target_lookup[eid] = (vals, mask)

        h5_path = Path(h5_dir) / f"{field}.h5"
        self.studies, self.study_lines, self.targets, self.target_masks = [], {}, {}, {}

        with h5py.File(h5_path, "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid not in manifest or sid not in target_lookup:
                    continue
                lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                         for x in f[sid_raw][()]]
                lines = merge_soft_wraps(lines)
                if patterns:
                    lines = [l for l in lines
                             if not any(p.search(l) for p in patterns)]
                lines = [l for l in lines if l in self.line_lookup]
                if not lines:
                    continue
                self.studies.append(sid)
                self.study_lines[sid] = lines
                self.targets[sid], self.target_masks[sid] = target_lookup[sid]

        print(f"TextProbeDataset[{field}]: {len(self.studies):,} studies", flush=True)

    def compute_norm(self):
        vals = np.stack([self.targets[s] for s in self.studies])
        masks = np.stack([self.target_masks[s] for s in self.studies])
        mean = np.zeros(N_CONT, dtype=np.float32)
        std = np.ones(N_CONT, dtype=np.float32)
        for i in range(N_CONT):
            m = masks[:, i].astype(bool)
            if m.sum() > 1:
                mean[i] = vals[m, i].mean()
                std[i] = vals[m, i].std().clip(min=1e-6)
        return mean, std

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        sid = self.studies[idx]
        lines = self.study_lines[sid]
        if len(lines) > self.max_lines:
            sel = np.random.choice(len(lines), self.max_lines, replace=False)
            lines = [lines[i] for i in sorted(sel)]
        embs = np.stack([self.line_lookup[l] for l in lines])
        targets = self.targets[sid].copy()
        mask = self.target_masks[sid].astype(np.float32)
        if self.cont_mean is not None:
            targets[:N_CONT] = (targets[:N_CONT] - self.cont_mean) / self.cont_std
        return (torch.from_numpy(embs),
                torch.from_numpy(targets),
                torch.from_numpy(mask))


def collate_fn(batch):
    embs_list, targets_list, masks_list = zip(*batch)
    B = len(batch)
    max_len = max(e.shape[0] for e in embs_list)
    dim = embs_list[0].shape[1]
    padded = torch.zeros(B, max_len, dim)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, e in enumerate(embs_list):
        n = e.shape[0]
        padded[i, :n] = e
        pad_mask[i, :n] = True
    return padded, pad_mask, torch.stack(targets_list), torch.stack(masks_list)
