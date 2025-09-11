import os
import hashlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
from collections import Counter
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = "dataset/train"   # apne dataset ka path yaha do
MIN_SIZE = 128                  # minimum resolution
BLUR_THRESHOLD = 100            # blur detection threshold
RESIZE_SHAPE = (224, 224)       # consistent size for all images
CLEANED_DIR = "dataset_cleaned" # cleaned dataset save hoga

# ---------------------------
# 1. Remove Duplicate Images
# ---------------------------
def remove_duplicates(folder):
    hashes = {}
    duplicates = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                if filehash in hashes:
                    duplicates.append(filepath)
                    os.remove(filepath)  # delete duplicate
                else:
                    hashes[filehash] = filepath
    print(f"✅ Removed {len(duplicates)} duplicate images.")

# ---------------------------
# 2. Remove Blurry Images
# ---------------------------
def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def remove_blurry(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if is_blurry(path):
                os.remove(path)
                count += 1
    print(f"✅ Removed {count} blurry images.")

# ---------------------------
# 3. Remove Very Small / Corrupted Images
# ---------------------------
def clean_small_and_corrupt(folder, min_size=MIN_SIZE):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                if img.size[0] < min_size or img.size[1] < min_size:
                    os.remove(path)
                    count += 1
            except:
                os.remove(path)
                count += 1
    print(f"✅ Removed {count} small/corrupted images.")

# ---------------------------
# 4. Resize Images & Convert to RGB
# ---------------------------
def resize_and_standardize(input_folder, output_folder, size=RESIZE_SHAPE):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = os.path.basename(root)
                save_dir = os.path.join(output_folder, class_name)
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize(size)
                    img.save(os.path.join(save_dir, file))
                except:
                    continue
    print(f"✅ All images resized to {size} and converted to RGB.")

# ---------------------------
# 5. Visualize Random Samples
# ---------------------------
def visualize_samples(folder, num_samples=5):
    classes = os.listdir(folder)
    for cls in classes:
        folder_path = os.path.join(folder, cls)
        if not os.path.isdir(folder_path):
            continue
        samples = random.sample(os.listdir(folder_path), min(num_samples, len(os.listdir(folder_path))))
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
        fig.suptitle(f"Class: {cls}")
        for ax, img_file in zip(axes, samples):
            img = Image.open(os.path.join(folder_path, img_file))
            ax.imshow(img)
            ax.axis("off")
        plt.show()

# ---------------------------
# 6. Handle Class Imbalance (Simple Oversampling Augmentation)
# ---------------------------
def balance_dataset(folder):
    dataset = ImageFolder(folder)
    counts = Counter([label for _, label in dataset.samples])
    max_count = max(counts.values())

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(RESIZE_SHAPE)
    ])

    for cls_idx, cls_name in enumerate(dataset.classes):
        class_folder = os.path.join(folder, cls_name)
        current_count = len(os.listdir(class_folder))
        if current_count < max_count:
            extra_needed = max_count - current_count
            print(f"⚡ Augmenting {cls_name}: {extra_needed} images")
            loader = DataLoader([s for s in dataset.samples if s[1] == cls_idx], batch_size=1, shuffle=True)
            for i, (img_path, _) in enumerate(loader):
                if i >= extra_needed:
                    break
                img = Image.open(img_path[0]).convert("RGB")
                img = transform(img)
                save_path = os.path.join(class_folder, f"aug_{i}.jpg")
                img.save(save_path)

    print("✅ Dataset balanced with augmentation.")

# ---------------------------
# MAIN PIPELINE
# ---------------------------
if __name__ == "__main__":
    print("🔹 Starting Data Cleaning Pipeline...")
    remove_duplicates(DATASET_DIR)
    remove_blurry(DATASET_DIR)
    clean_small_and_corrupt(DATASET_DIR)
    resize_and_standardize(DATASET_DIR, CLEANED_DIR)
    visualize_samples(CLEANED_DIR)
    balance_dataset(CLEANED_DIR)
    print("🎉 Data cleaning completed successfully!")
