import cv2
import os

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

blurry_images = []
for root, _, files in os.walk("dataset/train"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            if is_blurry(path):
                blurry_images.append(path)

print("Blurry images:", blurry_images)
print(f"Total blurry images found: {len(blurry_images)}")