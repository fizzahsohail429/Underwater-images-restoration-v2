import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        x_attn = torch.cat([max_out, mean_out], dim=1)
        x_attn = self.conv(x_attn)
        x_attn = self.sigmoid(x_attn)
        return x * x_attn


class MHSA(nn.Module):
    """Lightweight multi-head self-attention over downsampled feature map."""

    def __init__(self, channels: int, num_heads: int = 4, downsample: int = 2):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.downsample = downsample
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.downsample > 1:
            x_pool = F.avg_pool2d(x, self.downsample)
        else:
            x_pool = x
        qkv = self.qkv(x_pool)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, -1)
        k = k.view(b, self.num_heads, head_dim, -1)
        v = v.view(b, self.num_heads, head_dim, -1)
        attn = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', q, k) / (head_dim ** 0.5), dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).contiguous().view(b, c, *x_pool.shape[-2:])
        out = self.proj(out)
        if self.downsample > 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out + x


class DualBranchBlock(nn.Module):
    """Novel: Dual-branch local-global block (CNN + MHSA) with channel/spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            DepthwiseSeparableConv(channels, channels),
        )
        self.global_attn = MHSA(channels, num_heads=4, downsample=2)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = self.local(x)
        xg = self.global_attn(x)
        x_cat = torch.cat([xl, xg], dim=1)
        x_fused = self.fuse(x_cat)
        x_fused = self.ca(x_fused)
        x_fused = self.sa(x_fused)
        return x + x_fused


class ColorCorrectionHead(nn.Module):
    """Light color correction via 3x3 matrix and per-channel scales."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 + channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        stats = self.pool(x)
        params = self.mlp(stats).view(b, -1)
        mat = params[:, :9].view(b, 3, 3)
        gains = params[:, 9:].view(b, c, 1, 1).tanh() + 1.0
        x_flat = x.view(b, c, -1).transpose(1, 2)
        x_corr = torch.bmm(x_flat, mat).transpose(1, 2).view(b, c, h, w)
        return x_corr * gains


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.db = DualBranchBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.db(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.db = DualBranchBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.db(x)


class NextGenRestorer(nn.Module):
    """Next-gen underwater restorer: Dual-branch UNet with attention and color head."""

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock(base_ch, base_ch * 2)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = DualBranchBlock(base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.color = ColorCorrectionHead(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b) + e2
        d2 = self.dec2(d3) + e1
        d1 = self.dec1(d2) + s
        out = self.head(d1)
        out = self.color(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



