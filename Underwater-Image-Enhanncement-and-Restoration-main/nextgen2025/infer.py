import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from .models import build_model


def load_image(path: str, size: int = 0):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size))
    tensor = T.ToTensor()(img).unsqueeze(0)
    return img, tensor


def save_image(tensor: torch.Tensor, path: str):
    img = T.ToPILImage()(tensor.clamp(0, 1).squeeze(0).cpu())
    img.save(path)


def infer(input_path: str, output_path: str, checkpoint: str = "", model_variant: str = "dual_stage_base", device: str = "cuda", size: int = 0):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model(model_variant).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    _, x = load_image(input_path, size)
    x = x.to(device)
    with torch.no_grad():
        pred, _, _, _ = model(x)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(pred, output_path)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_variant", type=str, default="dual_stage_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=0)
    args = parser.parse_args()
    infer(args.input, args.output, args.checkpoint, args.model_variant, args.device, args.size)


if __name__ == "__main__":
    main()


