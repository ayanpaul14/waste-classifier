"""
prepare_dataset.py
Splits a flat dataset (one folder per class) into train/val/test splits.

Usage:
    python prepare_dataset.py --src raw_dataset --dst dataset --split 0.7 0.15 0.15
"""

import os
import shutil
import random
import argparse
from pathlib import Path

def prepare(src_dir, dst_dir, splits=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1"
    splits_names = ['train', 'val', 'test']

    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * splits[0])
        n_val   = int(n * splits[1])
        partitions = [
            images[:n_train],
            images[n_train:n_train + n_val],
            images[n_train + n_val:]
        ]

        for split_name, partition in zip(splits_names, partitions):
            out_dir = dst / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in partition:
                shutil.copy2(img, out_dir / img.name)

        print(f"{class_dir.name}: {len(partitions[0])} train | "
              f"{len(partitions[1])} val | {len(partitions[2])} test")

    print("\nDataset split complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='raw_dataset', help='Source folder with class subfolders')
    parser.add_argument('--dst', default='dataset', help='Output folder')
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    args = parser.parse_args()
    prepare(args.src, args.dst, tuple(args.split))