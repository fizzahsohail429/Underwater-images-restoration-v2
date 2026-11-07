# Records for the Sensors manuscript

These files were generated from the local code and from measurements on this computer. Cells that require trained outputs or missing datasets are left empty rather than filled with assumed scores.

## Sample counts

The code reads every source below. The saved image comparison is UIEB `archive/raw-890` only.

| Source | Reference type | Split | Images | Paired references |
|---|---|---|---:|---:|
| UIEB raw-890 | raw underwater image | train | 712 | 0 |
| UIEB raw-890 | raw underwater image | validation | 89 | 0 |
| UIEB raw-890 | raw underwater image | test | 89 | 0 |
| UIEB reference-890 | perceptual reference | unsplit | 0 | 0 |
| UIEB challenge-60 | no satisfactory reference | unsplit | 0 | 0 |
| UIEB_paired | paired perceptual reference | unsplit | 0 | 0 |
| EUVP_paired | paired reference | unsplit | 0 | 0 |
| UODD_noref | no reference | unsplit | 0 | 0 |
| UTDAC2020_noref | no reference | unsplit | 0 | 0 |
| RUOD_noref | no reference | unsplit | 0 | 0 |
| UDD_noref | no reference | unsplit | 0 | 0 |
| aquarium | capture image, no enhancement reference | train | 448 | 0 |
| aquarium | capture image, no enhancement reference | valid | 127 | 0 |
| aquarium | capture image, no enhancement reference | test | 63 | 0 |

The UIEB partition is one still image per file, seed 2026, ratios 0.8 / 0.1 / 0.1. Identifiers are in `image_identifiers.csv` (1528 rows). Split lists are `uieb_train.txt`, `uieb_validation.txt`, and `uieb_test.txt`.

UIEB overlap check: SHA-256 of `raw-890`. Images hashed in that source: 890. Identical-byte groups inside UIEB: 0. Identical UIEB files in more than one split: 0. Aquarium capture ids shared by more than one aquarium split: 15.

Current visual comparison: UIEB archive/raw-890 only. The ten-image panels are in `uieb_workflow10/`.

Held-out source: UODD is the held-out no-reference source. Its prepared folder has no images yet. EUVP_paired, UIEB reference-890, and UIEB challenge-60 are also empty. Aquarium captures are readable and are excluded from the UIEB comparison.

Paired-reference PSNR and SSIM against `reference-890` are not inserted because that directory is not in the archive. UIQM and UCIQE are not computed.

## Training configuration

- Seed: 2026, applied to `random`, NumPy, and PyTorch before training.
- Input: 256 × 256. Batch size: 6. Epochs configured: 30.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 5, mode min.
- Loss weights: L1 0.4, MSE 0.3, Sobel 0.2, degradation consistency 0.1.
- Normalization: (pixel / 255 − 0.5) / 0.5, so network tensors lie in [−1, 1].
- Checkpoint rule: keep `runs/evaluated/checkpoint.pt` from the epoch with the lowest mean validation loss.
- Checkpoint file present: False.
- Repository named in the manuscript: https://github.com/fizzahsohail429/Underwater-Image-Enhanncement-and-Restoration
- Remote commit lookup: d219b49c81b9eb03932c9174c9f0be9695e8e737
- Evaluated revision: Local working tree. The decoder-width fix and these records are not identified by a git commit in this folder, so they are not the remote commit above.

The full machine-readable copy is `training_configuration.json`.

## Metric definition

PSNR and SSIM are defined for later evaluation. The UIEB test list contains 89 raw images. They were not scored, because `reference-890` is absent and no checkpoint has been trained, so the evaluated sample count is 0 and the uncertainty interval is undefined.

- Scaling: denormalize with `y = clamp(x * 0.5 + 0.5, 0, 1)` before either metric. The peak value is 1, so PSNR is in decibels on the unit interval.
- PSNR: for each image, MSE is the mean squared error over the three channels and all pixels. `PSNR = 10 log10(1 / MSE)`.
- SSIM: Wang et al. (2004), 11 × 11 Gaussian window with sigma 1.5, `C1 = (0.01)^2`, `C2 = (0.03)^2`, computed per channel and then averaged across channels and pixels. One SSIM value is kept per image.
- Averaging unit: the reported mean is the unweighted mean of the per-image scores. It is not a mean over pixels pooled across the dataset.
- Uncertainty: 95% Student-t interval of that per-image mean, `mean ± t_(n−1, 0.975) * s / sqrt(n)`, where n is the number of images in that source and split. The interval is reported only when n ≥ 2.
- UIQM and UCIQE, when implemented, are computed on the denormalized prediction only and are averaged per image. They are reported apart from PSNR and SSIM. This repository does not yet run the Panetta et al. (2016) or Yang and Sowmya (2015) implementations, so no UIQM or UCIQE values are included.

## Computational cost

Resolution 256 × 256, batch size 6, device `cpu`. One untimed warmup pass and two timed passes, in evaluation mode, without gradients. Processor: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel. PyTorch threads: 10.

| Variant | Parameters | Peak memory (MiB) | Latency per batch (ms) | Latency per image (ms) | Checkpoint |
|---|---:|---:|---:|---:|---|
| e-urm | 73639032 | 1792.65 | 7171.14 | 1195.19 | not trained; architecture only |
| no-attention | 42199098 | 1362.74 | 4053.81 | 675.63 | not trained; architecture only |
| no-color | 73636225 | 1789.38 | 6156.05 | 1026.01 | not trained; architecture only |
| capacity-matched | 76747715 | 1494.61 | 4397.24 | 732.87 | not trained; architecture only |
| unprocessed input | 0 |  |  |  | none |
| Water-Net |  |  |  |  | not in this repository |
| FUnIE-GAN |  |  |  |  | not in this repository |
| PCAFA-Net |  |  |  |  | not in this repository |

Peak memory on this CPU run is the process working set. Definition: peak process working set during the forward, including loaded weights.

Capacity-matched control parameter ratio versus E-URM: 1.042. The control replaces each multiscale attention block with one 3 × 3 convolution, batch normalization, and ReLU, and it removes the color module. The ratio is the measured parameter count, not a claim of an exact match.

Water-Net, FUnIE-GAN, and PCAFA-Net were not timed. Their code and checkpoints are not in this folder, so a shared-hardware comparison is not available.

## Ablations and baselines

| Name | What changes | Checkpoint |
|---|---|---|
| Unprocessed input | The degraded image is the output | none |
| E-URM | Multiscale attention and adaptive color correction | `runs/evaluated/checkpoint.pt`, absent until training |
| Capacity-matched control | `attention_mode=conv3x3`, color module removed | not trained |
| No attention | `attention_mode=none`; color module kept | not trained |
| No color correction | color module removed; attention kept | not trained |
| No degradation consistency | same network; loss weight on the consistency term set to 0 | not trained |
| No synthetic augmentation | same network; training uses the clean image as the input | not trained |

No quality table is written because those checkpoints have not been trained and the image directories contain no files.
