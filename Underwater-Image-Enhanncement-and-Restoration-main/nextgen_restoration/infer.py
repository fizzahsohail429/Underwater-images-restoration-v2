import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .models import NextGenRestorer


def load_image(path: str, size: int = 256):
    img = Image.open(path).convert('RGB')
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(img).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    t = (t.squeeze(0).detach().cpu().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(1, 2, 0)
    t = (t * 255).astype(np.uint8)
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)


def restore_folder(weights: str, in_dir: str, out_dir: str = "nextgen_restored", size: int = 256, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    model = NextGenRestorer().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    with torch.no_grad():
        for name in os.listdir(in_dir):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(in_dir, name)
            x = load_image(path, size=size).to(device)
            y = model(x)
            out = tensor_to_image(y)
            cv2.imwrite(os.path.join(out_dir, name), out)


if __name__ == "__main__":
    restore_folder("nextgen_outputs/nextgen_best.pth", "aquarium_pretrain/test/images")




