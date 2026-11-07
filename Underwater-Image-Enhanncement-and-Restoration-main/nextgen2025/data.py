from pathlib import Path
from typing import Callable, Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random


class UnderwaterPairs(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        input_dir: str = "input",
        target_dir: str = "target",
        suffix: Tuple[str, ...] = (".jpg", ".png", ".jpeg"),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.input_dir = self.root / input_dir
        self.target_dir = self.root / target_dir
        self.transform = transform
        self.target_transform = target_transform
        input_candidates = sorted([p for p in self.input_dir.rglob("*") if p.suffix.lower() in suffix])
        self.samples: List[Tuple[Path, Path]] = []
        for inp in input_candidates:
            rel = inp.relative_to(self.input_dir)
            # Try to match any common target extension
            matched = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = (self.target_dir / rel).with_suffix(ext)
                if cand.exists():
                    matched = cand
                    break
            if matched is not None:
                self.samples.append((inp, matched))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp_path, tgt_path = self.samples[idx]
        image = Image.open(inp_path).convert("RGB")
        target = Image.open(tgt_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


def build_transforms(image_size: int = 256, augment_blur_dark: bool = False):
    ops = [T.Resize((image_size, image_size), antialias=True)]
    if augment_blur_dark:
        ops.extend([
            T.ColorJitter(brightness=(0.3, 1.0), contrast=(0.5, 1.2), saturation=(0.6, 1.2), hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5),
        ])
    ops.append(T.ToTensor())
    return T.Compose(ops)


def build_dataloader(root: str, image_size: int = 256, batch_size: int = 8, num_workers: int = 4, shuffle: bool = True, augment_blur_dark: bool = False) -> DataLoader:
    transform = build_transforms(image_size, augment_blur_dark=augment_blur_dark)
    ds = UnderwaterPairs(root=root, transform=transform, target_transform=transform)
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No paired samples found under '{root}'. Expected 'input/' and 'target/' with matching filenames (.png/.jpg/.jpeg)."
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


