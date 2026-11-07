import os
import cv2
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UnderwaterDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        degraded = create_nextgen_degradation(image)
        if self.transform is not None:
            aug = self.transform(image=image, degraded=degraded)
            image = aug["image"]
            degraded = aug["degraded"]
        return degraded, image


def create_nextgen_degradation(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    depth = np.random.uniform(0.4, 0.9)
    # wavelength-dependent attenuation (R,G,B)
    attn = np.array([
        np.exp(-depth * 2.4),  # R
        np.exp(-depth * 1.6),  # G
        np.exp(-depth * 0.9),  # B
    ], dtype=np.float32)
    img = img * attn
    # backscatter haze
    haze = np.random.uniform(0.02, 0.12)
    img = img * (1 - haze) + haze * np.array([0.1, 0.25, 0.35], dtype=np.float32)
    # spatially varying blur
    k = np.random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (k, k), 0)
    # signal-dependent noise
    sigma = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img + noise * img, 0, 1)
    return (img * 255).astype(np.uint8)


def build_transforms(size: int = 256) -> Tuple[A.Compose, A.Compose]:
    train = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={"degraded": "image"})
    val = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={"degraded": "image"})
    return train, val


def create_loaders(train_dir: str, val_dir: str, batch_size: int = 6, num_workers: int = 2):
    train_t, val_t = build_transforms()
    train_ds = UnderwaterDataset(train_dir, transform=train_t)
    val_ds = UnderwaterDataset(val_dir, transform=val_t)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )



