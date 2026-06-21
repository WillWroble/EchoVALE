"""Extract per-line binary labels from report H5 files.

Produces two CSVs compatible with eval_line_aurocs.py:
  line_vocab.csv   — line, fyler_code
  line_labels.csv  — sid, fyler_code (sparse positive pairs)

Usage:
    python -u extract_lines.py \
        --manifest manifests/platon_val.txt \
        --h5_dir line_tokenizer/data \
        --line_filters line_tokenizer/ignore_patterns.txt \
        --output_dir line_labels
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import h5py


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


def load_filters(path):
    if not path:
        return []
    return [re.compile(l.strip(), re.IGNORECASE)
            for l in open(path) if l.strip() and not l.startswith("#")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--fields", nargs="+",
                   default=["study_findings", "summary", "history"])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = set(str(int(float(x))) for x in
                   Path(args.manifest).read_text().strip().splitlines())
    print(f"Manifest: {len(manifest):,} studies", flush=True)

    filters = load_filters(args.line_filters)

    def keep(line):
        return not any(p.search(line) for p in filters)

    # Collect lines per study
    study_lines = defaultdict(set)
    for field in args.fields:
        h5_path = f"{args.h5_dir}/{field}.h5"
        print(f"Scanning {h5_path}...", flush=True)
        with h5py.File(h5_path, "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid not in manifest:
                    continue
                raw = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                       for x in f[sid_raw][()]]
                lines = merge_soft_wraps(raw)
                lines = [l.lstrip("\u2022 ").strip() for l in lines]
                lines = [l for l in lines if l and keep(l)]
                study_lines[sid].update(lines)

    print(f"{len(study_lines):,} studies with lines", flush=True)

    # Build sorted vocabulary
    vocab = sorted(set(l for lines in study_lines.values() for l in lines))
    print(f"{len(vocab):,} unique lines", flush=True)

    line_to_code = {line: f"{i+1:06d}" for i, line in enumerate(vocab)}

    # Write line vocab
    vocab_path = out / "line_vocab.csv"
    with open(vocab_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "fyler_code"])
        for line in vocab:
            w.writerow([line, line_to_code[line]])
    print(f"Wrote {vocab_path}", flush=True)

    # Write sparse labels
    labels_path = out / "line_labels.csv"
    n_pairs = 0
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "fyler_code"])
        for sid in sorted(study_lines.keys(), key=int):
            for line in sorted(study_lines[sid]):
                w.writerow([sid, line_to_code[line]])
                n_pairs += 1

    print(f"Wrote {labels_path} ({n_pairs:,} positive pairs)", flush=True)


if __name__ == "__main__":
    main()
