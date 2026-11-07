import math
from typing import Dict

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse + 1e-8)


def ssim_simple(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size // 2)
    sigma1 = F.avg_pool2d(pred * pred, window_size, 1, window_size // 2) - mu1 * mu1
    sigma2 = F.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu2 * mu2
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu1 * mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return (ssim_n / (ssim_d + 1e-6)).mean()


def summarize_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "psnr": float(psnr(pred, target).detach().cpu()),
            "ssim": float(ssim_simple(pred, target).detach().cpu()),
            # UIQM/UCIQE are typically no-reference; compute on prediction only
            "uiqm": float(uiqm(pred).detach().cpu()),
            "uciqe": float(uciqe(pred).detach().cpu()),
        }


# Below are lightweight approximations of UIQM and UCIQE to enable quick experiments.
# They capture their main components (colorfulness/contrast/saturation) without external deps.

def _rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    # expects [B,3,H,W] in [0,1]
    # convert to XYZ
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    def inv_gamma(u):
        return torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
    r = inv_gamma(r)
    g = inv_gamma(g)
    b = inv_gamma(b)
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    # reference white D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn
    def f(t):
        eps = 0.008856
        k = 903.3
        return torch.where(t > eps, t ** (1/3), (k * t + 16) / 116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def uiqm(img: torch.Tensor) -> torch.Tensor:
    # UIQM components (UCIQE variant is separate). Here we approximate:
    # UIQM ≈ c1*UICONM + c2*UISM + c3*UIConM
    # We compute: colorfulness via ab-chroma std, sharpness via gradient magnitude, contrast via L-channel std
    lab = _rgb_to_lab(img)
    L = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]
    chroma = torch.sqrt(a * a + b * b)
    uiconm = chroma.std(dim=[1, 2, 3])
    # sharpness via Sobel gradients on L
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=L.dtype, device=L.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=L.dtype, device=L.device).view(1, 1, 3, 3)
    gx = F.conv2d(L, sobel_x, padding=1)
    gy = F.conv2d(L, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    uism = grad_mag.mean(dim=[1, 2, 3])
    uicontrast = L.std(dim=[1, 2, 3])
    # weights from literature (approx)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    val = c1 * uiconm + c2 * uism + c3 * uicontrast
    return val.mean()


def uciqe(img: torch.Tensor) -> torch.Tensor:
    # UCIQE ≈ c1*σ_c + c2*con_l + c3*μ_s (Lab-based)
    lab = _rgb_to_lab(img)
    L = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]
    chroma = torch.sqrt(a * a + b * b)
    sigma_c = chroma.std(dim=[1, 2, 3])
    con_l = L.std(dim=[1, 2, 3])
    # saturation proxy: chroma normalized by L
    mu_s = (chroma / (L + 1e-6)).mean(dim=[1, 2, 3])
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    val = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return val.mean()


