import os
import shutil
import random

INPUT_DIR = r"input here"  # change to your input directory
OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.8

random.seed(42)  # reproducible split

tracks = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
random.shuffle(tracks)

split_idx = int(len(tracks) * TRAIN_RATIO)
train_tracks = tracks[:split_idx]
valid_tracks = tracks[split_idx:]

for folder, names in [(train_tracks, "train"), (valid_tracks, "valid")]:
    out_path = os.path.join(OUTPUT_DIR, names)
    os.makedirs(out_path, exist_ok=True)
    for t in folder:
        src = os.path.join(INPUT_DIR, t)
        dst = os.path.join(out_path, t)
        shutil.copytree(src, dst)

print(f"Train: {len(train_tracks)} tracks")
print(f"Validation: {len(valid_tracks)} tracks")

