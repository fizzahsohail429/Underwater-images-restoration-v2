"""
Improved Underwater Image Restoration Model with Novel Contributions
==================================================================

An advanced deep learning model for underwater image restoration with novel contributions
that should outperform U-Net and ResNet architectures.
"""

import hashlib
import json
import os
import platform
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Single source for the settings in Table 2 and Equations (3)–(4).
EVALUATED_SETTINGS = {
    "framework": "PyTorch",
    "input_resolution": [256, 256],
    "batch_size": 6,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
    },
    "epochs": 30,
    "normalization": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "max_pixel_value": 255.0,
        "formula": "(u/255 - 0.5) / 0.5",
        "tensor_range": [-1.0, 1.0],
    },
    "loss_weights": {
        "l1": 0.4,
        "mse": 0.3,
        "sobel": 0.2,
        "degradation_consistency": 0.1,
    },
}

DEGRADATION_SPEC = {
    "order": "Split the original images first. Draw a synthetic input only from an image already assigned to one split.",
    "depth_factor": {"distribution": "uniform", "low": 0.3, "high": 0.8},
    "attenuation": {
        "form": "exp(-depth_factor * rate)",
        "rates": {"R": 2.0, "G": 1.5, "B": 0.8},
    },
    "gaussian_scatter": {
        "distribution": "normal",
        "mean": 0.0,
        "std": {"distribution": "uniform", "low": 0.1, "high": 0.3},
    },
    "blur": {
        "type": "gaussian",
        "kernel": "odd integer in {3, 5, 7, 9}",
        "sigma": "OpenCV default for the selected kernel",
    },
    "intensity_noise": {
        "distribution": "normal",
        "mean": 0.0,
        "sigma": {"distribution": "uniform", "low": 0.02, "high": 0.08},
        "form": "sample * image",
    },
}

# These albumentations are composed only into the training transform.
TRAIN_ONLY_TRANSFORMS = [
    {"name": "HorizontalFlip", "p": 0.5},
    {"name": "VerticalFlip", "p": 0.3},
    {"name": "RandomRotate90", "p": 0.5},
    {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.7},
    {"name": "RandomBrightnessContrast", "p": 0.5},
    {"name": "GaussNoise", "var_limit": [10.0, 50.0], "p": 0.3},
    {"name": "GaussianBlur", "blur_limit": [3, 7], "p": 0.2},
]

HISTORY_KEYS = (
    "train_loss", "val_loss",
    "train_l1", "train_mse", "train_edge", "train_physics",
    "val_l1", "val_mse", "val_edge", "val_physics",
)
HISTORY_NAMES = {
    "l1_loss": "l1",
    "mse_loss": "mse",
    "edge_loss": "edge",
    "physics_loss": "physics",
}


def package_versions():
    import importlib.metadata as metadata
    versions = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "albumentations", "pillow", "matplotlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_configuration(run_device, executed=None):
    """Configuration of the code path that is actually called."""
    config = {
        "table2": {
            "framework": EVALUATED_SETTINGS["framework"],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            "device": "CUDA if available, otherwise CPU",
            "device_used": str(run_device),
            "cuda_available": torch.cuda.is_available(),
            "input_resolution": EVALUATED_SETTINGS["input_resolution"],
            "batch_size": EVALUATED_SETTINGS["batch_size"],
            "optimizer": EVALUATED_SETTINGS["optimizer"],
            "learning_rate": EVALUATED_SETTINGS["learning_rate"],
            "weight_decay": EVALUATED_SETTINGS["weight_decay"],
            "scheduler": EVALUATED_SETTINGS["scheduler"],
            "epochs_configured": EVALUATED_SETTINGS["epochs"],
        },
        "loss_weights": EVALUATED_SETTINGS["loss_weights"],
        "normalization": EVALUATED_SETTINGS["normalization"],
        "history_keys_initialized": list(HISTORY_KEYS),
        "history_keys_appended": list(HISTORY_KEYS),
        "synthetic_degradation": DEGRADATION_SPEC,
        "transforms_train_only": TRAIN_ONLY_TRANSFORMS,
        "transforms_all_splits": [
            "Resize to the configured input resolution",
            "Normalize with mean 0.5 and std 0.5",
        ],
        "synthetic_degradation_splits": "train, validation, and test, each drawn after that image's split assignment",
        "validation_and_test_degradation": "fixed by SHA-256 of the filename so the saved input does not change between epochs",
    }
    if executed:
        config["executed"] = executed
    return config

class MultiScaleAttentionModule(nn.Module):
    """
    Novel Contribution 1: Multi-Scale Attention Module
    Captures both local and global features with adaptive attention weights
    """
    
    def __init__(self, channels, reduction=16):
        super(MultiScaleAttentionModule, self).__init__()
        self.channels = channels
        
        # Multi-scale feature extraction
        self.scale1 = nn.Conv2d(channels, channels//4, kernel_size=1)
        self.scale2 = nn.Conv2d(channels, channels//4, kernel_size=3, padding=1, dilation=1)
        self.scale3 = nn.Conv2d(channels, channels//4, kernel_size=3, padding=2, dilation=2)
        self.scale4 = nn.Conv2d(channels, channels//4, kernel_size=3, padding=4, dilation=4)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        # Multi-scale feature extraction
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)
        scale4 = self.scale4(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([scale1, scale2, scale3, scale4], dim=1)
        
        # Channel attention
        channel_weights = self.channel_attention(multi_scale)
        channel_refined = multi_scale * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_attention(channel_refined)
        spatial_refined = channel_refined * spatial_weights
        
        # Feature fusion
        output = self.fusion(spatial_refined)
        
        return output + x  # Residual connection

class AdaptiveColorCorrectionModule(nn.Module):
    """
    Novel Contribution 2: Adaptive Color Correction Module
    Learns underwater-specific color transformations dynamically
    """
    
    def __init__(self, channels):
        super(AdaptiveColorCorrectionModule, self).__init__()
        
        # Color transformation network
        self.color_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 9, kernel_size=1),  # 3x3 color transformation matrix
            nn.Tanh()
        )
        
        # Underwater-specific color enhancement
        self.underwater_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate adaptive color transformation
        color_params = self.color_net(x)  # [B, 9, 1, 1]
        color_matrix = color_params.view(batch_size, 3, 3)  # [B, 3, 3]
        
        # Apply color transformation
        x_reshaped = x.view(batch_size, channels, -1)  # [B, C, H*W]
        x_reshaped = x_reshaped.permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply color matrix transformation
        color_corrected = torch.bmm(x_reshaped, color_matrix)  # [B, H*W, 3]
        color_corrected = color_corrected.permute(0, 2, 1)  # [B, 3, H*W]
        color_corrected = color_corrected.view(batch_size, channels, height, width)
        
        # Apply underwater-specific enhancement
        enhancement_weights = self.underwater_enhancement(color_corrected)
        enhanced = color_corrected * enhancement_weights
        
        return enhanced

class PhysicsInformedLoss(nn.Module):
    """
    Novel Contribution 3: Physics-Informed Loss Function
    Incorporates underwater imaging physics into the loss function
    """
    
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # MSE loss weight
        self.gamma = gamma  # Physics-based loss weight
        self.delta = delta  # Perceptual loss weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Sobel filters for edge detection. Registered buffers follow the loss device.
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        
    def edge_loss(self, pred, target):
        """Edge preservation loss using one-pixel zero padding."""
        # Convert to grayscale for edge detection
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Apply Sobel filters
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return self.l1_loss(pred_edges, target_edges)
    
    def underwater_physics_loss(self, pred, degraded):
        """Underwater physics-based loss"""
        # Simulate underwater degradation on prediction
        # Blue-green tint simulation
        blue_green_tint = torch.tensor([0.7, 0.9, 0.6], device=pred.device).view(1, 3, 1, 1)
        degraded_pred = pred * blue_green_tint
        
        # Blur simulation
        degraded_pred = F.avg_pool2d(degraded_pred, kernel_size=3, stride=1, padding=1)
        
        # Compare with original degraded image
        return self.mse_loss(degraded_pred, degraded)
    
    def forward(self, pred, target, degraded):
        # Standard losses
        l1_loss = self.l1_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)
        
        # Physics-informed losses
        edge_loss = self.edge_loss(pred, target)
        physics_loss = self.underwater_physics_loss(pred, degraded)
        
        # Combined loss
        total_loss = (self.alpha * l1_loss + 
                     self.beta * mse_loss + 
                     self.gamma * edge_loss + 
                     self.delta * physics_loss)
        
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'mse_loss': mse_loss.item(),
            'edge_loss': edge_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item()
        }

class HybridEncoderDecoder(nn.Module):
    """
    Novel Contribution 4: Hybrid Encoder-Decoder Architecture
    Combines the best of U-Net, ResNet, and attention mechanisms
    """
    
    def __init__(self, in_channels=3, out_channels=3, attention_mode="multiscale", use_color=True):
        super(HybridEncoderDecoder, self).__init__()
        if attention_mode not in ("multiscale", "none", "conv3x3"):
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
        self.attention_mode = attention_mode
        self.use_color = use_color
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with residual connections and attention
        self.encoder1 = self._make_encoder_block(64, 128)
        self.encoder2 = self._make_encoder_block(128, 256)
        self.encoder3 = self._make_encoder_block(256, 512)
        self.encoder4 = self._make_encoder_block(512, 1024)
        
        # Bottleneck with multi-scale attention, or the control replacement.
        self.bottleneck = nn.Sequential(
            self._refine(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            self._refine(1024)
        )
        
        # Decoder blocks accept the concatenated skip width.
        # Decoder 4 upsamples the 1024-channel bottleneck to 512.
        # Each later decoder consumes that map concatenated with the matching encoder map.
        self.decoder4 = self._make_decoder_block(1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 128, 64)
        
        # RGB head consumes Decoder 1 concatenated with the 64-channel stem.
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Adaptive color correction. The no-color ablation omits this module.
        if use_color:
            self.color_correction = AdaptiveColorCorrectionModule(out_channels)
        else:
            self.color_correction = nn.Identity()

    def _refine(self, channels):
        """Feature refinement used by every encoder, bottleneck, and decoder stage."""
        if self.attention_mode == "multiscale":
            return MultiScaleAttentionModule(channels)
        if self.attention_mode == "conv3x3":
            # Capacity-matched stand-in: one 3x3 convolution in place of multiscale attention.
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        return nn.Identity()
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create encoder block with residual connections and attention"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self._refine(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with skip connections"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self._refine(out_channels)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.initial_conv(x)
        enc2 = self.encoder1(enc1)
        enc3 = self.encoder2(enc2)
        enc4 = self.encoder3(enc3)
        enc5 = self.encoder4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder path. Each concatenation is consumed by the next layer,
        # whose input width equals the upsampled channels plus the skip channels.
        dec4 = self.decoder4(bottleneck)          # 512 x 32 x 32
        dec4 = torch.cat([dec4, enc4], dim=1)     # 1024 x 32 x 32
        
        dec3 = self.decoder3(dec4)                # 256 x 64 x 64
        dec3 = torch.cat([dec3, enc3], dim=1)     # 512 x 64 x 64
        
        dec2 = self.decoder2(dec3)                # 128 x 128 x 128
        dec2 = torch.cat([dec2, enc2], dim=1)     # 256 x 128 x 128
        
        dec1 = self.decoder1(dec2)                # 64 x 256 x 256
        dec1 = torch.cat([dec1, enc1], dim=1)     # 128 x 256 x 256
        
        # Final output
        output = self.final_conv(dec1)
        
        # Apply adaptive color correction
        output = self.color_correction(output)
        
        return output


def build_paper_variant(name):
    """Architectural variants used by the ablation and cost tables."""
    specs = {
        "e-urm": {"attention_mode": "multiscale", "use_color": True},
        "no-attention": {"attention_mode": "none", "use_color": True},
        "no-color": {"attention_mode": "multiscale", "use_color": False},
        "capacity-matched": {"attention_mode": "conv3x3", "use_color": False},
    }
    if name not in specs:
        raise ValueError(f"Unknown variant: {name}")
    return HybridEncoderDecoder(**specs[name])


def list_images(image_dir):
    names = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return [os.path.join(image_dir, name) for name in sorted(names)]


def split_originals(image_paths, seed=0, ratios=(0.8, 0.1, 0.1)):
    """Assign every original to one split before any synthetic degradation is drawn."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")
    order = np.random.default_rng(seed).permutation(len(image_paths))
    ordered = [image_paths[i] for i in order]
    n = len(ordered)
    n_train = int(round(n * ratios[0]))
    n_valid = int(round(n * ratios[1]))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    return {
        "train": ordered[:n_train],
        "valid": ordered[n_train:n_train + n_valid],
        "test": ordered[n_train + n_valid:],
    }


def _degradation_rng(filename, is_training):
    if is_training:
        return np.random.default_rng()
    digest = hashlib.sha256(os.path.basename(filename).encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def create_advanced_degradation(image, rng):
    """Attenuation, Gaussian scatter, blur, then intensity-dependent noise.

    Ranges are the values in DEGRADATION_SPEC. Call this only after the source
    image has been assigned to a split.
    """
    img_float = image.astype(np.float32) / 255.0
    depth = DEGRADATION_SPEC["depth_factor"]
    rates = DEGRADATION_SPEC["attenuation"]["rates"]
    scatter = DEGRADATION_SPEC["gaussian_scatter"]
    noise = DEGRADATION_SPEC["intensity_noise"]

    depth_factor = float(rng.uniform(depth["low"], depth["high"]))
    absorption = np.array([
        np.exp(-depth_factor * rates["R"]),
        np.exp(-depth_factor * rates["G"]),
        np.exp(-depth_factor * rates["B"]),
    ], dtype=np.float32)
    img_float = img_float * absorption

    scattering_strength = float(rng.uniform(scatter["std"]["low"], scatter["std"]["high"]))
    img_float = img_float + rng.normal(0.0, scattering_strength, img_float.shape).astype(np.float32)

    blur_kernel_size = int(rng.integers(3, 9))
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    img_float = cv2.GaussianBlur(img_float, (blur_kernel_size, blur_kernel_size), 0)

    noise_level = float(rng.uniform(noise["sigma"]["low"], noise["sigma"]["high"]))
    img_float = img_float + rng.normal(0.0, noise_level, img_float.shape).astype(np.float32) * img_float

    img_float = np.clip(img_float, 0, 1)
    degraded = (img_float * 255).astype(np.uint8)
    params = {
        "depth_factor": depth_factor,
        "absorption_rgb": [float(v) for v in absorption],
        "scatter_std": scattering_strength,
        "blur_kernel": blur_kernel_size,
        "noise_sigma": noise_level,
    }
    return degraded, params


class ImprovedUnderwaterDataset(Dataset):
    """Paired examples whose degraded input is created after the split."""

    def __init__(self, image_paths, transform=None, is_training=True, apply_synthetic_degradation=True):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.is_training = is_training
        self.apply_synthetic_degradation = apply_synthetic_degradation
        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def load_pair(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not self.apply_synthetic_degradation:
            return img_path, image, image.copy(), {}
        rng = _degradation_rng(img_path, self.is_training)
        degraded, params = create_advanced_degradation(image, rng)
        return img_path, image, degraded, params

    def __getitem__(self, idx):
        _path, image, degraded, _params = self.load_pair(idx)
        if self.transform:
            augmented = self.transform(image=image, degraded=degraded)
            image = augmented["image"]
            degraded = augmented["degraded"]
        return degraded, image

def create_improved_transforms(image_size=256):
    """Create enhanced data augmentation transforms"""
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'degraded': 'image'})
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'degraded': 'image'})
    
    return train_transform, val_transform

def save_checkpoint(path, model, optimizer, scheduler, epoch, history, config, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "configuration": config,
    }, path)


def train_improved_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler=None,
    num_epochs=30, device=device, output_dir="runs/evaluated", config=None,
):
    """Train the improved underwater image restoration model"""
    model.to(device)
    history = {key: [] for key in HISTORY_KEYS}
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics = {'l1_loss': [], 'mse_loss': [], 'edge_loss': [], 'physics_loss': []}
        
        for batch_idx, (degraded, target) in enumerate(train_loader):
            degraded, target = degraded.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(degraded)
            loss, metrics = criterion(output, target, degraded)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            for key in train_metrics:
                train_metrics[key].append(metrics[key])
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}', flush=True)
            if batch_idx > 0 and batch_idx % 40 == 0:
                save_checkpoint(
                    os.path.join(output_dir, "last.pt"),
                    model, optimizer, scheduler, epoch + 1, history, config, best_val_loss,
                )
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = {'l1_loss': [], 'mse_loss': [], 'edge_loss': [], 'physics_loss': []}
        
        with torch.no_grad():
            for degraded, target in val_loader:
                degraded, target = degraded.to(device), target.to(device)
                output = model(degraded)
                loss, metrics = criterion(output, target, degraded)
                
                val_losses.append(loss.item())
                for key in val_metrics:
                    val_metrics[key].append(metrics[key])
        
        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Append under the same names that were initialized in HISTORY_KEYS.
        for key, short_name in HISTORY_NAMES.items():
            history[f'train_{short_name}'].append(float(np.mean(train_metrics[key])))
            history[f'val_{short_name}'].append(float(np.mean(val_metrics[key])))
        missing = [key for key in HISTORY_KEYS if key not in history or len(history[key]) != epoch + 1]
        if missing:
            raise KeyError(f"Training-history keys do not match the initialized names: {missing}")
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Train L1: {history["train_l1"][-1]:.4f}, Val L1: {history["val_l1"][-1]:.4f}')
        print(f'  Train Edge: {history["train_edge"][-1]:.4f}, Val Edge: {history["val_edge"][-1]:.4f}')
        print('-' * 50)
        
        # Save best model together with the configuration used for this run.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                os.path.join(output_dir, "checkpoint.pt"),
                model, optimizer, scheduler, epoch + 1, history, config, best_val_loss,
            )
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
    
    return history

def plot_improved_training_history(history):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # L1 loss
    axes[0, 1].plot(history['train_l1'], label='Train L1', color='blue')
    axes[0, 1].plot(history['val_l1'], label='Validation L1', color='red')
    axes[0, 1].set_title('L1 Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L1 Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MSE loss
    axes[0, 2].plot(history['train_mse'], label='Train MSE', color='blue')
    axes[0, 2].plot(history['val_mse'], label='Validation MSE', color='red')
    axes[0, 2].set_title('MSE Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MSE Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Edge loss
    axes[1, 0].plot(history['train_edge'], label='Train Edge', color='blue')
    axes[1, 0].plot(history['val_edge'], label='Validation Edge', color='red')
    axes[1, 0].set_title('Edge Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Edge Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Physics loss
    axes[1, 1].plot(history['train_physics'], label='Train Physics', color='blue')
    axes[1, 1].plot(history['val_physics'], label='Validation Physics', color='red')
    axes[1, 1].set_title('Physics Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Physics Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning curves comparison
    axes[1, 2].plot(history['train_loss'], label='Train Total', color='blue', alpha=0.7)
    axes[1, 2].plot(history['val_loss'], label='Val Total', color='red', alpha=0.7)
    axes[1, 2].plot(history['train_l1'], label='Train L1', color='green', alpha=0.7)
    axes[1, 2].plot(history['val_l1'], label='Val L1', color='orange', alpha=0.7)
    axes[1, 2].set_title('All Losses Comparison')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_improved_results(model, test_loader, device, num_samples=5):
    """Visualize improved restoration results"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (degraded, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            degraded, target = degraded.to(device), target.to(device)
            restored = model(degraded)
            
            # Convert tensors to numpy arrays
            degraded_np = degraded[0].cpu().numpy().transpose(1, 2, 0)
            target_np = target[0].cpu().numpy().transpose(1, 2, 0)
            restored_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize images
            degraded_np = (degraded_np * 0.5 + 0.5).clip(0, 1)
            target_np = (target_np * 0.5 + 0.5).clip(0, 1)
            restored_np = (restored_np * 0.5 + 0.5).clip(0, 1)
            
            # Plot images
            axes[i, 0].imshow(degraded_np)
            axes[i, 0].set_title('Degraded Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(restored_np)
            axes[i, 1].set_title('Improved Restoration')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_np)
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('improved_restoration_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def set_reproducibility_seed(seed=2026):
    """Seed Python, NumPy, and PyTorch before dataset iteration and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_sources():
    """Print every dataset the project can read, and which one is compared now."""
    from dataset_catalog import source_rows

    print("Dataset catalog")
    print("Current visual comparison: UIEB archive/raw-890")
    for row in source_rows():
        split = row.get("split", "all")
        print(
            f"  {row['source']} [{split}]: {row['image_count']} images, "
            f"paired {row['paired_count']}, comparison {row['in_current_comparison']}"
        )
    print("Training the current UIEB run: python improved_underwater_restoration.py --train")


def load_uieb_split_paths():
    """Use the locked UIEB lists when they exist, so training matches the records."""
    from dataset_catalog import UIEB_RAW, uieb_raw_paths

    record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_records")
    lists = {
        "train": os.path.join(record_dir, "uieb_train.txt"),
        "valid": os.path.join(record_dir, "uieb_validation.txt"),
        "test": os.path.join(record_dir, "uieb_test.txt"),
    }
    if all(os.path.isfile(path) for path in lists.values()):
        loaded = {}
        for split, list_path in lists.items():
            with open(list_path, encoding="utf-8") as handle:
                loaded[split] = [
                    os.path.join(UIEB_RAW, line.strip())
                    for line in handle
                    if line.strip()
                ]
        return loaded
    paths = uieb_raw_paths()
    order = np.random.default_rng(2026).permutation(len(paths))
    ordered = [paths[i] for i in order]
    n_train = int(round(len(ordered) * 0.8))
    n_valid = int(round(len(ordered) * 0.1))
    return {
        "train": ordered[:n_train],
        "valid": ordered[n_train:n_train + n_valid],
        "test": ordered[n_train + n_valid:],
    }


def main():
    """Train E-URM on the locked UIEB raw-890 split."""
    set_reproducibility_seed(2026)
    print("Starting Improved Underwater Image Restoration Training")
    print("Random seed: 2026")
    print("Current images: UIEB archive/raw-890")
    print("Other named datasets are read by dataset_catalog.py and are not in this comparison.")
    report_sources()

    split_paths = load_uieb_split_paths()
    
    # Create transforms
    train_transform, val_transform = create_improved_transforms(image_size=256)
    
    # Create datasets
    train_dataset = ImprovedUnderwaterDataset(split_paths["train"], transform=train_transform, is_training=True)
    val_dataset = ImprovedUnderwaterDataset(split_paths["valid"], transform=val_transform, is_training=False)
    test_dataset = ImprovedUnderwaterDataset(split_paths["test"], transform=val_transform, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create improved model
    model = HybridEncoderDecoder()
    print(f"Improved model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function and optimizer
    criterion = PhysicsInformedLoss(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    print("\nStarting improved training...")
    run_config = build_configuration(device)
    run_config["seed"] = 2026
    run_config["checkpoint_rule"] = "Keep the epoch with the lowest mean validation loss."
    run_config["checkpoint_path"] = "runs/evaluated/checkpoint.pt"
    history = train_improved_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, device=device, output_dir="runs/evaluated", config=run_config,
    )
    
    # Plot training history
    print("\nPlotting improved training history...")
    plot_improved_training_history(history)
    
    # Load best model and visualize results
    print("\nLoading best model and visualizing results...")
    model.load_state_dict(torch.load("runs/evaluated/checkpoint.pt", map_location=device)["model"])
    visualize_improved_results(model, test_loader, device, num_samples=5)
    
    print("\nImproved training completed!")
    print("Checkpoint saved as: runs/evaluated/checkpoint.pt")
    print("Training plots saved as: improved_training_history.png")
    print("Results visualization saved as: improved_restoration_results.png")

def verify_implementation(image_size=256):
    """Run the architecture, loss, and degradation checks named in the manuscript."""
    model = HybridEncoderDecoder()
    model.eval()
    x = torch.randn(1, 3, image_size, image_size)
    captured = {}

    def save(name):
        def _hook(_module, _inputs, output):
            captured[name] = tuple(output.shape)
        return _hook

    hooks = [
        model.initial_conv.register_forward_hook(save("stem")),
        model.encoder1.register_forward_hook(save("encoder1")),
        model.encoder2.register_forward_hook(save("encoder2")),
        model.encoder3.register_forward_hook(save("encoder3")),
        model.encoder4.register_forward_hook(save("encoder4")),
        model.bottleneck.register_forward_hook(save("bottleneck")),
        model.decoder4.register_forward_hook(save("decoder4_before_skip")),
        model.decoder3.register_forward_hook(save("decoder3_before_skip")),
        model.decoder2.register_forward_hook(save("decoder2_before_skip")),
        model.decoder1.register_forward_hook(save("decoder1_before_skip")),
        model.final_conv.register_forward_hook(save("rgb_head")),
        model.color_correction.register_forward_hook(save("color_correction")),
    ]
    with torch.no_grad():
        y = model(x)
    for hook in hooks:
        hook.remove()

    print("Residual encoder-decoder")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    for name in (
        "stem", "encoder1", "encoder2", "encoder3", "encoder4", "bottleneck",
        "decoder4_before_skip", "decoder3_before_skip", "decoder2_before_skip",
        "decoder1_before_skip", "rgb_head", "color_correction",
    ):
        print(f"  {name}: {captured[name]}")
    print(f"  output: {tuple(y.shape)}")
    print("  encoder stages: 4 stride-2 blocks, widths 64 -> 128 -> 256 -> 512 -> 1024")
    print("  residual additions: inside each multiscale attention block (output + input)")
    print("  skip concatenations consumed by the next layer:")
    print("    decoder4 512 + encoder3 512 = 1024 -> decoder3")
    print("    decoder3 256 + encoder2 256 = 512 -> decoder2")
    print("    decoder2 128 + encoder1 128 = 256 -> decoder1")
    print("    decoder1 64 + stem 64 = 128 -> RGB head")

    attention = model.encoder1[3]
    print("Multiscale attention")
    print(f"  branches: 1x1, dilation 1, dilation 2, dilation 4")
    print(f"  branch out channels: {attention.scale1.out_channels} each, input {attention.channels}")
    print(f"  channel attention reduction: {attention.channels} -> {attention.channels // 16} -> {attention.channels}")
    print(f"  spatial attention: {attention.spatial_attention[0].kernel_size} conv to 1 channel, then sigmoid")

    color = model.color_correction
    print("Adaptive color correction")
    print(f"  global pool: {color.color_net[0]}")
    print("  matrix network: 3 -> 64 -> 32 -> 9, final tanh, applied as a 3x3 RGB matrix")
    print("  spatial gate: 3x3 conv, batch norm, ReLU, 3x3 conv, sigmoid")

    criterion = PhysicsInformedLoss(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    target = torch.randn_like(y)
    total, parts = criterion(y, target, x)
    print("Training objective")
    print(f"  weights L1={criterion.alpha}, MSE={criterion.beta}, Sobel={criterion.gamma}, degradation={criterion.delta}")
    print(f"  normalization: albumentations mean=0.5, std=0.5, max_pixel_value=255 -> (u/255 - 0.5) / 0.5")
    print(f"  dummy loss keys: {sorted(parts)}")
    print(f"  history keys written each epoch: train/val loss, l1, mse, edge, physics")
    print(f"  dummy total: {float(total):.4f}")

    print("Synthetic degradation, drawn independently on each image after the directory split")
    print("  channel attenuation: depth ~ Uniform(0.3, 0.8); scale = exp(-depth * rate)")
    print("  attenuation rates: R=2.0, G=1.5, B=0.8")
    print("  Gaussian scatter: std ~ Uniform(0.1, 0.3), then added to the image")
    print("  blur: Gaussian, odd kernel in {3, 5, 7, 9}")
    print("  intensity-dependent noise: sigma ~ Uniform(0.02, 0.08), noise = N(0, sigma) * image")
    print("  train split only: horizontal flip, vertical flip, random 90-degree rotation,")
    print("    color jitter, brightness/contrast, extra Gaussian noise, extra Gaussian blur")
    print("  validation and test: resize and the same normalization, plus the synthetic degradation above")

    print("Evaluation")
    print("  current comparison and training source: archive/raw-890")
    print("  other sources are listed by dataset_catalog.source_rows()")
    print("  batch size 6, epochs 30, AdamW lr 0.001, weight decay 1e-4")
    print("  ReduceLROnPlateau factor 0.5, patience 5, checkpoint = lowest validation loss")
    print("  PSNR, SSIM, UIQM, and UCIQE are not computed by this script")
    return captured


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_implementation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--train":
        main()
    else:
        report_sources() 