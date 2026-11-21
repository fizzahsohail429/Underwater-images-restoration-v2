# Underwater Image Enhancement and Restoration

This repository contains E-URM, the encoder-decoder used for the UIEB reference experiment. The reported run learns a mapping from a raw UIEB image to the matching image in `reference-890`.

The model class is `HybridEncoderDecoder` in `improved_underwater_restoration.py`. `build_paper_variant("e-urm")` builds the reported model: multiscale attention and adaptive color correction. It has 73,639,032 parameters, counted with `sum(p.numel() for p in model.parameters())`. That count includes BatchNorm scale and shift. It excludes buffers such as BatchNorm running statistics. The constructor does not freeze any parameter.

## Repository

```bash
git clone https://github.com/fizzahsohail429/Underwater-images-restoration-v2.git
cd Underwater-images-restoration-v2
git lfs pull
pip install -r requirements.txt
```

Git LFS is required for the selected checkpoint. The decoder-width correction is in `80d7096`. The reference-pair trainer is in `4a9d333`. The selected checkpoint and its configuration were published in `514e874`. The ablation trainer and pair audit were added in `fda03a3`.

The completed run used Python 3.11.9, PyTorch 2.7.1+cpu, torchvision 0.22.1, and Windows on CPU. There was no CUDA and no gradient clipping.

## Files that produce the reported results

| Purpose | File |
|---|---|
| Model | `improved_underwater_restoration.py` |
| E-URM training and test | `train_uieb_reference.py` |
| Attention, color, and capacity-matched runs | `train_uieb_variants.py` |
| PSNR and SSIM | `paper_records/quality_metrics.py` |
| Pair and duplicate audit | `paper_records/audit_uieb_pairs.py` |
| Shape check | `shape_check()` in `paper_records/train_three_sources.py` |
| Locked split | `paper_records/uieb_train.txt`, `uieb_validation.txt`, `uieb_test.txt` |

`train_uieb_reference.py` is the script for the E-URM numbers. `train_uieb_variants.py` repeats that protocol for one named variant and writes to `runs/uieb_variants/`. It does not modify `runs/uieb_reference/`.

## Dataset

Pairing is by filename stem. The image folders stay on the local machine and are not in this Git repository.

```
archive/
├── UIEB/              # 890 raw underwater images (model input)
└── reference-890/     # 890 reference images (training target)

paper_records/
├── uieb_train.txt         # 712 image ids
├── uieb_validation.txt    # 89 image ids
└── uieb_test.txt          # 89 image ids
```

The split uses seed 2026 at ratios 0.8 / 0.1 / 0.1. The same 89-image test list is used for every reported comparison. It is not used for training or for choosing a checkpoint.

Check the split and the pairs:

```bash
python train_uieb_reference.py --phase verify
python paper_records/audit_uieb_pairs.py
```

The audit checked 890 pairs: none missing, no filename in more than one split, and no exact raw or reference duplicate. The filenames do not contain a scene or sequence id, so images were not grouped by scene.

`archive/challenge-60` has 60 images and no reference. The local EUVP folder has raw images only, so it is not a paired external test.

## Training and test

Place the raw images in `archive/UIEB` and the matching reference images in `archive/reference-890`, then run:

```bash
python train_uieb_reference.py --phase train --learning-rate 0.0001
python train_uieb_reference.py --phase test --learning-rate 0.0001
```

The test command scores `runs/uieb_reference/checkpoints/best_val_psnr.pt` once.

| Setting | Value used for the reported run |
|---|---|
| Image size | 256×256 |
| Batch size | 6 |
| Epochs | 30 |
| Optimizer | AdamW, learning rate 0.0001, weight decay 1e-4 |
| Scheduler | ReduceLROnPlateau, mode max, factor 0.5, patience 5, stepped on validation PSNR |
| Loss | L1 0.4, MSE 0.3, Sobel 0.2, degradation consistency 0.0 |
| Checkpoint | Highest mean validation PSNR |

The manuscript learning rate of 0.001 produced non-finite gradients on this task, so the completed run used 0.0001. The loss weights are applied as written and are not rescaled to sum to 1. The degradation-consistency term compares a tinted blur of the prediction with the raw input, so its weight is 0 when the target is `reference-890`.

Seed 2026 is set by `set_reproducibility_seed` before the model is built. The training loader uses `torch.Generator().manual_seed(2026)`, `shuffle=True`, and `num_workers=0`. Validation and test loaders are not shuffled. `torch.use_deterministic_algorithms` and `cudnn.deterministic` are not set. The reported training was resumed at epoch 5, so epochs 5–30 did not continue one uninterrupted random-number stream.

The configuration saved with that run is `runs/uieb_reference/uieb_experiment_config.json`. Its `git_revision` field remains `2bc0bf573c080f45557996c22fcee6898f11aafb`, which was the revision when training started. The publication commit is `514e874`.

## Preprocessing and PSNR/SSIM

These settings are the ones used by `train_uieb_reference.py` and `train_uieb_variants.py` for training, validation, and the held-out test. Validation PSNR in the training log and PSNR/SSIM in the test table use `psnr_per_image` and `ssim_per_image` in `paper_records/quality_metrics.py`.

### Preprocessing

- Read with OpenCV, then convert BGR to RGB.
- Resize the raw input and the reference together to 256×256. Both images go through one Albumentations `Resize`, so they receive the same interpolation. The call does not override the default, which is OpenCV `INTER_LINEAR`.
- No crop, flip, rotation, color jitter, noise, blur, or synthetic degradation on this UIEB run. The training, validation, and test loaders all use the validation transform.
- Normalize each channel with mean 0.5 and standard deviation 0.5. With Albumentations `max_pixel_value=255`, this is `(pixel / 255 - 0.5) / 0.5`. Network inputs are in `[-1, 1]`.
- Before a metric, map a tensor back with `clamp(x * 0.5 + 0.5, 0, 1)`. The unprocessed input, the model output, and the reference are all mapped this way.

### PSNR

- Compare one image with the resized reference. The data range is `[0, 1]` and the peak is 1.
- Mean squared error is over every pixel and all three channels.
- `PSNR = 10 * log10(1 / MSE)`. If MSE is at most `1e-12`, the score is recorded as 99.

### SSIM

- Wang et al. (2004), with `K1 = 0.01` and `K2 = 0.03`, so `C1 = 0.0001` and `C2 = 0.0009` at peak 1.
- 11×11 Gaussian window, sigma 1.5, normalized so the window sums to 1.
- Valid convolution: no padding, so the map is 246×246 for a 256×256 image.
- SSIM is computed on each RGB channel, then the three channel scores are averaged.

### Sobel term

The Sobel term converts RGB to grayscale with weights 0.299, 0.587, and 0.114, applies the standard 3×3 Sobel kernels with one-pixel zero padding, takes the gradient magnitude, and uses L1 against the reference.

## Model

`HybridEncoderDecoder` uses a 7×7 stem from 3 to 64 channels, then encoders at 128, 256, 512, and 1024 channels. The bottleneck applies attention, a 3×3 convolution, and attention again. Four stride-2 4×4 transposed convolutions decode the image. Each skip connection is concatenated before the next decoder. After the decoder-width correction the decoder inputs are 1024, 1024, 512, and 256, and the RGB head receives 128 channels. The head uses `Tanh`, and the color module is applied after that.

The color module uses global average pooling, a 3→64→32→9 map, a `tanh` 3×3 color matrix, and a spatial sigmoid gate. E-URM contains ten multiscale attention modules. The attention residual adds the module output to its input.

Run only the shape check. Executing `paper_records/train_three_sources.py` as a program also starts other training.

```bash
python -c "import torch; from paper_records.train_three_sources import shape_check; shape_check(torch.device('cpu'))"
```

The saved result is `paper_records/architecture_check.json`: forward and backward passed, input and output `[2, 3, 256, 256]`.

## Checkpoint and outputs

The selected E-URM checkpoint is:

`runs/uieb_reference/checkpoints/best_val_psnr.pt`

It is epoch 29. Its SHA-256 is `176c0673f6fa3d4d0ae03ff7ce7537109185cfba418b79fbc9f63675719220c9`. This is the file stored with Git LFS. The other files under `runs/` are local.

A completed E-URM run also writes:

- `runs/uieb_reference/uieb_experiment_config.json`
- `runs/uieb_reference/uieb_split.json`
- `runs/uieb_reference/training_log.csv`
- `runs/uieb_reference/uieb_test_results.csv`
- `runs/uieb_reference/uieb_test_summary.json`

`best_val_ssim.pt` is saved during training and is not the checkpoint used for the test table.

## Controlled comparisons

Each finished variant uses the same split, preprocessing, 30 epochs, learning rate 0.0001, and highest-validation-PSNR rule. The test scores below are means over the same 89 images.

```bash
python train_uieb_variants.py --variant no-color --phase train --learning-rate 0.0001
python train_uieb_variants.py --variant no-color --phase test --learning-rate 0.0001
```

Replace `no-color` with `no-attention` or `capacity-matched`. Each selected checkpoint is `runs/uieb_variants/<name>/checkpoints/best_val_psnr.pt`, with `test_summary.json` and `training_log.csv` in that variant folder. Those variant checkpoints are local. They are not in the GitHub history.

| Method | Selected epoch | Validation PSNR | Test PSNR | Test SSIM | PSNR change vs input |
|---|---:|---:|---:|---:|---:|
| Unprocessed input | — | — | 17.874 | 0.7719 | 0 |
| E-URM | 29 | 22.912 | 22.217 | 0.8825 | +4.342 |
| No color correction | 30 | 22.943 | 22.212 | 0.8820 | +4.338 |
| Capacity-matched | 29 | 21.087 | 20.915 | 0.8672 | +3.041 |
| No attention | 30 | 21.045 | 20.748 | 0.8662 | +2.873 |

On this test list, E-URM improves mean PSNR and mean SSIM relative to the unprocessed input. Removing attention, and replacing attention with the capacity-matched convolution, lowers test PSNR. Removing color correction leaves test PSNR within 0.005 dB of E-URM.

`consistency` and `synthetic` are defined in `train_uieb_variants.py` and have not been trained. Water-Net, FUnIE-GAN, and PCAFA-Net are not implemented or scored in this repository.

`paper_records/computational_cost.csv` records a separate forward-only timing of the untrained architectures at 256×256, batch 6, on CPU. The E-URM row is marked architecture only: 7,171.14 ms per batch and a peak process working set of 1,792.65 MiB. It is not a timing of the trained checkpoint.

## License

This project is licensed under the MIT License.
