"""Dataset loading utilities for training CNN classifiers."""

import itertools

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

from config.settings import Settings


SPECTRAL_LABELS = [
    "R", "G", "B", "Edge Red", "NIR", "EXG", "EXGR", "NGRDI", "NGBDI",
    "RGRI", "GBRI", "CIVE", "VEG", "RGBVI", "MGRVI", "NDVI", "RVI",
    "DVI", "EVI", "REVI", "NDRE", "RERVI", "REDVI", "SCCI",
]


def cargar_database(combination=("R", "G", "B")):
    """Load image dataset for a given 3-band combination.

    Returns (device, train_dataset, val_dataset, num_classes).
    """
    combinations = list(itertools.combinations(SPECTRAL_LABELS, 3))
    pos = combinations.index(combination)
    selected = combinations[pos]
    folder_name = "-".join(selected)

    folder_path = Settings.DATASTORE_DIR / folder_name

    transform = transforms.Compose([transforms.ToTensor()])
    image_dataset = datasets.ImageFolder(root=str(folder_path), transform=transform)

    train_size = int(0.7 * len(image_dataset))
    val_size = len(image_dataset) - train_size
    train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

    num_classes = len(image_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, train_dataset, val_dataset, num_classes
