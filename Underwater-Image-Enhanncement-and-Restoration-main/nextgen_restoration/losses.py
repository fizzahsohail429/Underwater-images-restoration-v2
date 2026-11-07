import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

try:
    import piq  # SSIM
except Exception:  # pragma: no cover
    piq = None


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def dark_channel(x: torch.Tensor, window: int = 15) -> torch.Tensor:
    pad = window // 2
    x_min, _ = torch.min(x, dim=1, keepdim=True)
    x_pad = F.pad(x_min, (pad, pad, pad, pad), mode="replicate")
    x_unf = x_pad.unfold(2, window, 1).unfold(3, window, 1)
    dc, _ = torch.min(x_unf, dim=-1)
    dc, _ = torch.min(dc, dim=-1)
    return dc


class CompositeRestorationLoss(nn.Module):
    def __init__(self, w_l1: float = 0.4, w_ssim: float = 0.3, w_color: float = 0.2, w_dcp: float = 0.1):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_color = w_color
        self.w_dcp = w_dcp
        self.l1 = nn.L1Loss()
        self._ssim = piq.SSIMLoss(data_range=2.0) if piq is not None else None

    def color_constancy(self, x: torch.Tensor) -> torch.Tensor:
        means = [torch.mean(x[:, i:i+1]) for i in range(3)]
        return (means[0] - means[1]).abs() + (means[1] - means[2]).abs() + (means[0] - means[2]).abs()

    def dcp_reg(self, x: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        dc_out = dark_channel(torch.clamp((x + 1) / 2, 0, 1))
        dc_in = dark_channel(torch.clamp((degraded + 1) / 2, 0, 1))
        return F.l1_loss(dc_out, dc_in)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, degraded: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = self.l1(pred, target)
        if self._ssim is not None:
            ssim = self._ssim((pred + 1), (target + 1))  # data_range=2
        else:
            # fallback: edge-aware l1
            gx = rgb_to_gray(pred)
            gy = rgb_to_gray(target)
            ssim = F.l1_loss(gx, gy)
        color = self.color_constancy(pred)
        dcp = self.dcp_reg(pred, degraded)
        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_color * color + self.w_dcp * dcp
        return total, {
            "l1": float(l1.detach().cpu()),
            "ssim": float(ssim.detach().cpu()),
            "color": float(color.detach().cpu()),
            "dcp": float(dcp.detach().cpu()),
            "total": float(total.detach().cpu()),
        }



