from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mu1 = F.avg_pool2d(img1, self.window_size, 1, self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, self.window_size // 2)
        sigma1 = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size // 2) - mu1 * mu1
        sigma2 = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size // 2) - mu2 * mu2
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size // 2) - mu1 * mu2
        ssim_n = (2 * mu1 * mu2 + self.C1) * (2 * sigma12 + self.C2)
        ssim_d = (mu1 ** 2 + mu2 ** 2 + self.C1) * (sigma1 + sigma2 + self.C2)
        ssim_map = ssim_n / (ssim_d + 1e-6)
        return ssim_map.mean()


class RestorationLoss(nn.Module):
    """Composite loss supporting L1, SSIM, and auxiliary supervision of stage-1 outputs."""

    def __init__(self, w_l1: float = 1.0, w_ssim: float = 0.5, w_coarse: float = 0.2):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_coarse = w_coarse
        self.l1 = nn.L1Loss()
        self.ssim = SSIM()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, coarse: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l1 = self.l1(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        coarse_l1 = self.l1(coarse, target)
        total = self.w_l1 * l1 + self.w_ssim * ssim_loss + self.w_coarse * coarse_l1
        return total, {"l1": l1.detach(), "ssim": ssim_loss.detach(), "coarse_l1": coarse_l1.detach()}


