import matplotlib.pyplot as plt
import os
import random
from PIL import Image

base_folder = "dataset/train"
classes = os.listdir(base_folder)

for cls in classes:
    folder = os.path.join(base_folder, cls)
    samples = random.sample(os.listdir(folder), 5)  # 5 random images
    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    fig.suptitle(f"Class: {cls}")
    for ax, img_file in zip(axes, samples):
        img = Image.open(os.path.join(folder, img_file))
        ax.imshow(img)
        ax.axis("off")
    plt.show()
