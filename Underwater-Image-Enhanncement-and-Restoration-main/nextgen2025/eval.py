import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import UnderwaterPairs, build_transforms
from .metrics import summarize_metrics
from .models import build_model


def evaluate(data_root: str, checkpoint: str, model_variant: str = "dual_stage_base", device: str = "cuda", image_size: int = 256, batch_size: int = 4, num_workers: int = 2):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = build_transforms(image_size)
    ds = UnderwaterPairs(root=data_root, transform=transform, target_transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(model_variant).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    agg = {"psnr": 0.0, "ssim": 0.0, "uiqm": 0.0, "uciqe": 0.0}
    count = 0
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)
            pred, _, _, _ = model(images)
            m = summarize_metrics(pred, targets)
            n = images.size(0)
            agg["psnr"] += m["psnr"] * n
            agg["ssim"] += m["ssim"] * n
            agg["uiqm"] += m["uiqm"] * n
            agg["uciqe"] += m["uciqe"] * n
            count += n
    agg = {k: v / max(count, 1) for k, v in agg.items()}
    print(f"PSNR: {agg['psnr']:.3f} | SSIM: {agg['ssim']:.3f} | UIQM: {agg['uiqm']:.3f} | UCIQE: {agg['uciqe']:.3f}")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="dual_stage_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()


