"""PSNR, SSIM, UIQM, and UCIQE used for the saved records.

PSNR and SSIM compare a prediction with the original image after both have been
mapped to [0, 1]. They are not scores against a separate reference-890 file.

UIQM follows Panetta, Gao, and Agaian (2016): UICM with 10% asymmetric
alpha-trimming, UISM as the EME of Sobel magnitude, and UIConM as PLIP log-AMEE.
The combination weights are 0.0282, 0.2953, and 0.6765.

UCIQE follows Yang and Sowmya (2015): 0.4680 * chroma std + 0.2745 * luminance
contrast + 0.2576 * mean saturation, in CIE Lab.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def denormalize(tensor):
    return torch.clamp(tensor.float() * 0.5 + 0.5, 0.0, 1.0)


def psnr_per_image(prediction, target):
    scores = []
    for pred, truth in zip(prediction, target):
        mse = torch.mean((pred - truth) ** 2).item()
        scores.append(99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse))
    return scores


def _gaussian_window():
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).view(1, 1, 11, 11)
    return window


_WINDOW = _gaussian_window()


def ssim_per_image(prediction, target):
    window = _WINDOW.to(prediction.device)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for pred, truth in zip(prediction, target):
        channel_scores = []
        for channel in range(pred.shape[0]):
            x = pred[channel:channel + 1].unsqueeze(0)
            y = truth[channel:channel + 1].unsqueeze(0)
            mu_x = F.conv2d(x, window)
            mu_y = F.conv2d(y, window)
            sigma_x = F.conv2d(x * x, window) - mu_x ** 2
            sigma_y = F.conv2d(y * y, window) - mu_y ** 2
            sigma_xy = F.conv2d(x * y, window) - mu_x * mu_y
            numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
            channel_scores.append(torch.mean(numerator / denominator).item())
        scores.append(float(np.mean(channel_scores)))
    return scores


def _trimmed_stats(values, alpha=0.1):
    ordered = np.sort(np.asarray(values, dtype=np.float64).ravel())
    count = ordered.size
    left = int(math.ceil(alpha * count))
    right = int(math.floor(alpha * count))
    kept = ordered[left:count - right] if count - right > left else ordered
    mean = float(np.mean(kept))
    std = float(np.sqrt(np.mean((kept - mean) ** 2)))
    return mean, std


def _eme(channel, block=8):
    height, width = channel.shape
    height -= height % block
    width -= width % block
    if height < block or width < block:
        return 0.0
    view = channel[:height, :width]
    total = 0.0
    count = 0
    for y in range(0, height, block):
        for x in range(0, width, block):
            patch = view[y:y + block, x:x + block]
            high = float(np.max(patch))
            low = float(np.min(patch))
            if low <= 1e-8 or high <= 0:
                continue
            total += math.log(high / low)
            count += 1
    if count == 0:
        return 0.0
    return 2.0 * total / count


def _sobel_magnitude(channel):
    grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def _plip_log_amee(gray, block=8):
    gray = np.clip(gray, 0.0, 1.0)
    height, width = gray.shape
    height -= height % block
    width -= width % block
    if height < block or width < block:
        return 0.0
    view = gray[:height, :width]
    total = 0.0
    blocks = 0
    for y in range(0, height, block):
        for x in range(0, width, block):
            patch = view[y:y + block, x:x + block]
            high = float(np.max(patch))
            low = float(np.min(patch))
            if low >= 1.0:
                continue
            subtracted = (high - low) / max(1.0 - low, 1e-8)
            added = high + low - high * low
            if added <= 1e-8 or subtracted <= 1e-8:
                continue
            ratio = np.clip(subtracted / added, 1e-8, 1.0)
            total += ratio * math.log(ratio)
            blocks += 1
    if blocks == 0:
        return 0.0
    return -total / blocks


def uiqm_image(rgb_unit):
    """rgb_unit is H x W x 3 in [0, 1]."""
    image = np.clip(np.asarray(rgb_unit, dtype=np.float64), 0.0, 1.0)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = (red - green).ravel()
    yb = ((red + green) / 2.0 - blue).ravel()
    mu_rg, std_rg = _trimmed_stats(rg)
    mu_yb, std_yb = _trimmed_stats(yb)
    uicm = -0.0268 * math.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * math.sqrt(std_rg ** 2 + std_yb ** 2)
    scaled = image * 255.0
    uism = (
        0.299 * _eme(_sobel_magnitude(scaled[:, :, 0].astype(np.float32)))
        + 0.587 * _eme(_sobel_magnitude(scaled[:, :, 1].astype(np.float32)))
        + 0.114 * _eme(_sobel_magnitude(scaled[:, :, 2].astype(np.float32)))
    )
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    uiconm = _plip_log_amee(gray)
    return 0.0282 * uicm + 0.2953 * uism + 0.6765 * uiconm


def _rgb_to_lab(rgb_unit):
    try:
        from skimage.color import rgb2lab
        return rgb2lab(np.clip(rgb_unit, 0.0, 1.0))
    except Exception:
        uint8 = np.clip(rgb_unit * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
        lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
        lab[:, :, 1] -= 128.0
        lab[:, :, 2] -= 128.0
        return lab


def uciqe_image(rgb_unit):
    lab = _rgb_to_lab(np.asarray(rgb_unit, dtype=np.float64))
    lightness = lab[:, :, 0]
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    sigma_c = float(np.std(chroma))
    saturation = chroma / np.sqrt(chroma ** 2 + lightness ** 2 + 1e-8)
    mu_s = float(np.mean(saturation))
    ordered = np.sort(lightness.ravel())
    tail = max(1, int(round(0.01 * ordered.size)))
    contrast = float(np.mean(ordered[-tail:]) - np.mean(ordered[:tail]))
    return 0.4680 * sigma_c + 0.2745 * contrast + 0.2576 * mu_s


def tensor_to_rgb_list(tensor):
    array = denormalize(tensor).detach().cpu().numpy()
    return [np.transpose(image, (1, 2, 0)) for image in array]


def interval(values):
    count = len(values)
    mean = float(np.mean(values))
    if count < 2:
        return mean, None, None
    # df = 88 critical value is not reused; use a normal 1.96 bound only as fallback
    # when scipy is absent. Prefer the Student-t critical value.
    variance = float(np.sum((np.asarray(values) - mean) ** 2) / (count - 1))
    half_factor = _t_critical(count - 1)
    half = half_factor * math.sqrt(variance / count)
    return mean, mean - half, mean + half


def _t_critical(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        if df >= 120:
            return 1.96
        if df >= 60:
            return 2.00
        return 2.04
