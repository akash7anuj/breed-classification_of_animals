from PIL import Image
import os

min_size = 128  # minimum resolution (width & height)

bad_images = []
for root, _, files in os.walk("dataset/train"):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            if img.size[0] < min_size or img.size[1] < min_size:
                bad_images.append(path)
        except:
            bad_images.append(path)  # corrupted image
print("Small or corrupted images:", bad_images)
print(f"Total small or corrupted images found: {len(bad_images)}")