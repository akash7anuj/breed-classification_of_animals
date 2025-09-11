import os
import shutil
import random

# Paths
DATASET_DIR = "dataset_cleaned"   # cleaned dataset ka folder
OUTPUT_DIR = "dataset_split"      # new folder jisme train/val/test hoga

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output dirs
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Process each class
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_files = images[:n_train]
    val_files = images[n_train:n_train+n_val]
    test_files = images[n_train+n_val:]

    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        split_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(split_dir, f)
            shutil.copy(src, dst)

print("✅ Dataset successfully split into train/val/test folders!")
