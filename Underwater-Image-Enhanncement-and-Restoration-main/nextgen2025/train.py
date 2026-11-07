import argparse
from pathlib import Path

import torch
from torch import optim
from torch.cuda import amp

from .data import build_dataloader
from .losses import RestorationLoss
from .models import build_model


def train(
    data_root: str,
    out_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    lr: float = 2e-4,
    epochs: int = 50,
    model_variant: str = "dual_stage_base",
    device: str = "cuda",
    augment_blur_dark: bool = False,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dl = build_dataloader(
        data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        augment_blur_dark=augment_blur_dark,
    )
    model = build_model(model_variant).to(device)
    criterion = RestorationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                pred, coarse, _, _ = model(images)
                loss, parts = criterion(pred, targets, coarse)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} l1 {parts['l1']:.4f} ssim {parts['ssim']:.4f}")
        ckpt_path = out_path / f"model_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        print(f"saved {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing input/ and target/")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_variant", type=str, default="dual_stage_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augment_blur_dark", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()


