import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from .models import build_model
from .metrics import uiqm, uciqe


class SingleFolderImages(Dataset):
    def __init__(self, folder: str, image_size: int = 256):
        self.root = Path(folder)
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        self.transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), str(p)


def evaluate_noref(data_root: str, checkpoint: str, model_variant: str = "dual_stage_base", device: str = "cuda", image_size: int = 256, batch_size: int = 4, num_workers: int = 2):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = SingleFolderImages(Path(data_root) / "input", image_size=image_size)
    if len(ds) == 0:
        raise FileNotFoundError(f"No images found under {Path(data_root) / 'input'}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(model_variant).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    total_uiqm = 0.0
    total_uciqe = 0.0
    count = 0
    with torch.no_grad():
        for images, _ in dl:
            images = images.to(device)
            pred, _, _, _ = model(images)
            total_uiqm += float(uiqm(pred).detach().cpu()) * images.size(0)
            total_uciqe += float(uciqe(pred).detach().cpu()) * images.size(0)
            count += images.size(0)
    uiqm_mean = total_uiqm / max(count, 1)
    uciqe_mean = total_uciqe / max(count, 1)
    print(f"UIQM: {uiqm_mean:.3f} | UCIQE: {uciqe_mean:.3f}")
    return {"uiqm": uiqm_mean, "uciqe": uciqe_mean}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root containing input/ only")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="dual_stage_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    evaluate_noref(**vars(args))


if __name__ == "__main__":
    main()


