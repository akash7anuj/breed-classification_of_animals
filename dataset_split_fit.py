# split dataset into train, val, test sets using PyTorch during training without creating separate folders

from torchvision import datasets
from torch.utils.data import random_split

dataset = datasets.ImageFolder("dataset_cleaned", transform=None)
n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)
n_test = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")