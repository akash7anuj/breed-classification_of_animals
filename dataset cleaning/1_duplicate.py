import os
import hashlib
from PIL import Image

def find_duplicates(folder):
    hashes = {}
    duplicates = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                if filehash in hashes:
                    duplicates.append(filepath)   # duplicate mila
                else:
                    hashes[filehash] = filepath
    return duplicates

# Example use:
duplicates = find_duplicates("dataset/train")
print("Duplicate images:", duplicates)
