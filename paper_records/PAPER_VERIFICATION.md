# What the current code verifies, what was extracted, and what is missing

This file was written while UIEB training was still on epoch 1. It records only checks and numbers that the code on disk can support. Empty cells are empty because the required file or checkpoint is not available.

## Verified against the manuscript

These components exist in `improved_underwater_restoration.py` and were checked with a forward and backward pass. The record is `architecture_check.json`.

| Paper component | Result of the check |
|---|---|
| Residual encoder–decoder | Stem 7×7 maps 3 channels to 64. Four stride-2 encoders go 64 → 128 → 256 → 512 → 1024. The bottleneck is attention, 3×3, attention. Four stride-2 4×4 decoders follow. The residual add is inside each multiscale attention block (`output + input`), not around the stride-2 stages. Parameter count: 73,639,032. Input and output of the check: 2 × 3 × 256 × 256. |
| Decoder skip widths | Decoder 4 reads 1024 and writes 512. After concatenation with the 512-channel encoder skip, decoder 3 reads 1024. Decoder 2 reads 512. Decoder 1 reads 256. The RGB head reads 128. Forward and backward both passed. |
| Multiscale attention | Four branches: 1×1, and 3×3 with dilation 1, 2, and 4. Each branch outputs one quarter of the channels. Channel attention reduces by 16. Spatial attention is a 7×7 convolution to one channel and a sigmoid. |
| Adaptive color correction | Global average pooling, then 3 → 64 → 32 → 9 with a final tanh, reshaped to a 3×3 RGB matrix. A spatial gate follows: 3×3, batch norm, ReLU, 3×3, sigmoid. |
| Training objective | L1 0.4, MSE 0.3, Sobel 0.2, degradation consistency 0.1. History keys are `train_l1`, `train_mse`, `train_edge`, `train_physics` and the matching `val_` names. |
| Normalization | `(pixel / 255 − 0.5) / 0.5`, so tensors lie in [−1, 1]. The network ends in tanh. |
| Synthetic degradation | Depth Uniform(0.3, 0.8). Channel scale `exp(−depth × rate)` with rates R = 2.0, G = 1.5, B = 0.8. Gaussian scatter std Uniform(0.1, 0.3). Blur kernel is an odd size in {3, 5, 7, 9}. Intensity-noise sigma Uniform(0.02, 0.08), and the noise is multiplied by the image. Validation and test draws are seeded by the SHA-256 of the filename. Flips, rotation, color jitter, extra noise, and extra blur are training-only. |
| Splits | UIEB test list is locked at seed 2026, ratios 0.8 / 0.1 / 0.1: 712 / 89 / 89. The 15 images below are the first 15 names in `uieb_test.txt`. |
| PSNR and SSIM definitions | Peak value 1 after denormalizing to [0, 1]. SSIM uses an 11×11 Gaussian window, sigma 1.5, C1 = (0.01)², C2 = (0.03)², averaged over RGB channels and then kept as one value per image. |
| Computational cost already measured | On CPU, 256×256, batch 6: E-URM 73,639,032 parameters, 1792.65 MiB working set, 7171.14 ms per batch. The other architecture variants are in `computational_cost.csv`. Those timings are architecture-only. They are not trained-model scores. |

## Outputs extracted now

Fifteen UIEB test images were selected: `716_img_.png`, `341_img_.png`, `476_img_.png`, `867_img_.png`, `680_img_.png`, `798_img_.png`, `893_img_.png`, `528_img_.png`, `689_img_.png`, `299_img_.png`, `789_img_.png`, `67_img_.png`, `730_img_.png`, `613_img_.png`, `674_img_.png`.

### Reference comparison

`available_now/uieb_reference_metrics.csv` has one row per image with columns `image_id`, `dataset`, `method`, `PSNR`, `SSIM`. The method is E-URM. PSNR and SSIM are blank.

Panels are in `available_now/`. Each panel is original input, reference, and E-URM output. The reference panel says the reference is not on disk. The E-URM panel says no checkpoint has been saved. `contact_sheet.png` stacks the fifteen panels.

No paired reference file was found for any of the fifteen images. `archive/UIEB` contains 890 raw images and no `reference-890` folder. EUVP `trainA` has 1,478 images and no reference folder. Aquarium has capture images and no enhancement reference.

### Synthetic degradation, kept separate

`available_now/uieb_synthetic_metrics.csv` is a different file. Its `evaluation` column is `synthetic degradation; target is the original image`. The method is `unprocessed synthetic input`, which is the degraded image before any network. These numbers are not reference-paired scores and they are not E-URM scores.

On these 15 images the unprocessed synthetic input has mean PSNR 13.117 dB and mean SSIM 0.404. Per-image values:

| Image | PSNR | SSIM |
|---|---:|---:|
| 716_img_.png | 16.511 | 0.541 |
| 341_img_.png | 13.465 | 0.338 |
| 476_img_.png | 13.009 | 0.493 |
| 867_img_.png | 15.536 | 0.637 |
| 680_img_.png | 9.496 | 0.293 |
| 798_img_.png | 15.051 | 0.396 |
| 893_img_.png | 10.365 | 0.304 |
| 528_img_.png | 13.730 | 0.515 |
| 689_img_.png | 14.579 | 0.486 |
| 299_img_.png | 10.815 | 0.196 |
| 789_img_.png | 14.426 | 0.342 |
| 67_img_.png | 11.596 | 0.369 |
| 730_img_.png | 13.496 | 0.466 |
| 613_img_.png | 9.780 | 0.103 |
| 674_img_.png | 14.905 | 0.586 |

## Missing

Training of UIEB is still inside epoch 1 of 30. `runs/uieb/checkpoint.pt` and `runs/evaluated/checkpoint.pt` do not exist yet, so an E-URM image cannot be saved and E-URM PSNR/SSIM cannot be calculated. EUVP and aquarium training have not started. Transfer scores are not available.

The following manuscript items cannot be filled from the files on disk:

- Paired PSNR and SSIM of E-URM against a UIEB reference image.
- A visual panel that contains the real reference and a trained E-URM output.
- UIQM and UCIQE on a trained output. The functions exist in `quality_metrics.py`, but they have not been run on a saved model.
- Trained ablation checkpoints: capacity-matched control, no attention, no color, no degradation consistency, and no synthetic augmentation.
- External baselines Water-Net, FUnIE-GAN, and PCAFA-Net. Their code and checkpoints are not in this folder.
- Held-out transfer. The UODD folder is empty, and EUVP and aquarium have not been trained or scored with a finished checkpoint.
- Numerical claims for the abstract, discussion, and conclusions. Those sentences need the trained test table first.

When epoch 1 finishes, the same 15 images can be filled by loading `runs/uieb/checkpoint.pt`, writing the E-URM image into the third panel, and putting reference PSNR/SSIM into `uieb_reference_metrics.csv` only for images that have a reference file.
