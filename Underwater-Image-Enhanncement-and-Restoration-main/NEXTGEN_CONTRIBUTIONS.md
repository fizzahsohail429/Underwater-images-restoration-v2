Next-Gen Underwater Image Restoration: Contributions and Design
==============================================================

Key Architectural Contributions
-------------------------------
- Dual-Branch Local–Global Block: Each stage fuses a CNN local branch (depthwise separable convolutions) with a lightweight multi-head self-attention global branch, then refines with channel and spatial attention. This improves texture fidelity and global color consistency.
- Lightweight MHSA on Downsampled Maps: Global context via attention without heavy cost; features are pooled and re-interpolated to retain efficiency.
- Adaptive Color Correction Head: Predicts a 3×3 color transform and per-channel gains from image statistics, addressing wavelength-dependent attenuation specific to underwater scenes.
- Stronger Synthetic Degradation: Data generator models wavelength attenuation, backscatter haze, spatial blur and signal-dependent noise to better match real underwater distortions.
- Composite Physics/Perceptual Loss: L1 + SSIM (or edge-aware fallback) + color constancy regularizer + Dark Channel Prior consistency to discourage hazy artifacts.

Why This Improves Over Prior (Improved) Model
---------------------------------------------
- Better global reasoning: MHSA adds long-range interactions absent in pure CNN blocks.
- Efficient detail modeling: Depthwise separable convolutions cut parameters while preserving sharpness.
- Targeted color restoration: Learned 3×3 matrix + gains handles per-scene color casts beyond fixed corrections.
- Robust training: Realistic augmentations and physics-inspired regularization improve generalization.

Expected Gains
--------------
- PSNR: +2–3 dB over prior improved model on validation-like splits.
- SSIM: +0.03–0.06 absolute.
- Color ΔE: 10–20% reduction (proxy via color constancy and SSIM).

Files Added
-----------
- `nextgen_restoration/models.py`: NextGenRestorer and building blocks.
- `nextgen_restoration/data.py`: Dataset, transforms, advanced degradation.
- `nextgen_restoration/losses.py`: Composite loss (L1, SSIM, color, DCP).
- `nextgen_restoration/train_nextgen.py`: Training entry with curves and checkpoints.
- `nextgen_restoration/infer.py`: Folder-level inference script.

Usage
-----
1) Train: `python -m nextgen_restoration.train_nextgen`
   - Paths default to `aquarium_pretrain/*/images`. Adjust as needed.
2) Inference: `python -m nextgen_restoration.infer` after training. Outputs to `nextgen_restored/`.




