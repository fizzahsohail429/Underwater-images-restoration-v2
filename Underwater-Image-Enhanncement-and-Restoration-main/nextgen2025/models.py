from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 3):
        super().__init__()
        c = base_channels
        self.inc = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 8)
        self.up1 = Up(c * 8, c * 4)
        self.up2 = Up(c * 4, c * 2)
        self.up3 = Up(c * 2, c)
        self.up4 = Up(c, c)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class PhysicsGuidedDehaze(nn.Module):
    """
    Physics-inspired first stage estimating transmission and ambient light.
    Produces a coarse restored image that guides the refinement stage.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.encoder = UNet(in_channels=in_channels, base_channels=base_channels, out_channels=6)

    @staticmethod
    def _recover_image(x: torch.Tensor, t: torch.Tensor, a: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        t = t.clamp(0.05, 1.0)
        a = a.clamp(0.0, 1.0)
        restored = (x - a) / (t + eps) + a
        return restored.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.encoder(x)
        t = pred[:, 0:3, :, :].sigmoid()
        a = pred[:, 3:6, :, :].sigmoid()
        coarse = self._recover_image(x, t, a)
        return coarse, t, a


class ColorRefiner(nn.Module):
    """Refinement network operating on concatenated input and coarse output."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32, out_channels: int = 3):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, base_channels=base_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


class DualStageRestorer(nn.Module):
    """
    Dual-stage underwater image restorer:
    - Stage 1: PhysicsGuidedDehaze estimates transmission and ambient light and recovers a coarse image.
    - Stage 2: ColorRefiner enhances colors and removes residual artifacts.
    Returns final restored image and intermediate outputs for auxiliary losses.
    """

    def __init__(self, base_channels_stage1: int = 32, base_channels_stage2: int = 32):
        super().__init__()
        self.stage1 = PhysicsGuidedDehaze(in_channels=3, base_channels=base_channels_stage1)
        self.stage2 = ColorRefiner(in_channels=6, base_channels=base_channels_stage2, out_channels=3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse, t, a = self.stage1(x)
        refined = self.stage2(torch.cat([x, coarse], dim=1)).sigmoid()
        return refined, coarse, t, a


def build_model(variant: str = "dual_stage_base") -> nn.Module:
    if variant == "dual_stage_base":
        return DualStageRestorer(base_channels_stage1=32, base_channels_stage2=32)
    if variant == "dual_stage_large":
        return DualStageRestorer(base_channels_stage1=48, base_channels_stage2=48)
    raise ValueError(f"Unknown model variant: {variant}")


