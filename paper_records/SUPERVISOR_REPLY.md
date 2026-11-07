# Reply to the review requests

This note answers the requested checks from the code and files that are on disk now. Items that are not finished are stated as not finished. No score was filled in without a reference image or a saved checkpoint.

## Code version

This folder is a local git repository. The root commit on `master` is `80d709618e740217db4b9a093c72e57b7003ed73`, message "Record the evaluated E-URM revision, decoder fix, and experiment files." It has not been pushed.

The manuscript names this public repository:

https://github.com/fizzahsohail429/Underwater-Image-Enhanncement-and-Restoration

The `main` branch on that remote was last looked up as commit `d219b49c81b9eb03932c9174c9f0be9695e8e737`. Local commit `80d7096` is a new history. It is not `d219b49`. Updating GitHub `main` to this history would require a force push, which has not been done.

The runnable architecture check for this tree is `paper_records/architecture_check.json`. Reference scores written after that commit are in the working tree and are not inside `80d7096`.

## Training configuration

Saved in `paper_records/training_configuration.json`.

| Setting | Value |
|---|---|
| Framework | PyTorch 2.7.1, torchvision 0.22.1 |
| Other packages | NumPy 2.2.6, OpenCV 4.12.0.88, albumentations 2.0.8, Pillow 11.3.0, matplotlib 3.10.3 |
| Python | 3.11.9 on Windows 10 |
| Device | CPU. CUDA is not available. |
| Input | 256 × 256, normalized with (pixel / 255 − 0.5) / 0.5 |
| Batch size | 6 |
| Epochs configured | 30 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 5, mode min |
| Loss | L1 0.4, MSE 0.3, Sobel 0.2, degradation consistency 0.1 |
| Seed | 2026 for Python, NumPy, and PyTorch |
| Checkpoint rule | Keep the epoch with the lowest mean validation loss |

## Checkpoints

No selected checkpoint is on disk.

`runs/uieb/checkpoint.pt` and `runs/evaluated/checkpoint.pt` are absent. UIEB training is running again from the start of epoch 1 of 30 (`paper_records/train_three_sources.py`). The script writes `runs/uieb/last.pt` every 40 batches and writes `checkpoint.pt` only after a finished epoch whose mean validation loss is the best so far. EUVP and aquarium have not started. The ablation checkpoints have not been trained. Pictures from an untrained initialization are not used as results.

## Dataset splits

| Source | Train | Validation | Test | List files | Notes |
|---|---:|---:|---:|---|---|
| UIEB `archive/UIEB` | 712 | 89 | 89 | `uieb_train.txt`, `uieb_validation.txt`, `uieb_test.txt` | Seed 2026, ratios 0.8 / 0.1 / 0.1. SHA-256 found no duplicate across splits. |
| EUVP `archive/EUVP/Paired/underwater_dark/trainA` | 1182 | 148 | 148 | `euvp_train.txt`, `euvp_validation.txt`, `euvp_test.txt` | Same seed and ratios. This folder has inputs only. |
| Aquarium pretrain | 448 | 127 | 63 | `aquarium_train.txt`, `aquarium_validation.txt`, `aquarium_test.txt` | The folders already supplied these splits. Fifteen capture ids appear in more than one split, so this set is not used as a clean held-out test. |

`archive/UIEB` has the 890 raw images. The official reference archive is extracted to `archive/reference-890` (890 files, filenames match the raw stems). It is the UIEB reference-890 release, password from the authors' page. Those reference images are not committed, because the UIEB authors forbid redistribution.

## Decoder channel mismatch

This is fixed in `improved_underwater_restoration.py`, and a forward plus backward pass succeeded.

| Stage | Channels going into the layer |
|---|---:|
| Decoder 4 | 1024 |
| Decoder 3, after the skip concatenation | 1024 |
| Decoder 2, after the skip concatenation | 512 |
| Decoder 1, after the skip concatenation | 256 |
| RGB head, after the stem concatenation | 128 |

Decoder 4 still writes 512 channels. The next layer now expects the concatenated width, 1024, instead of 512. The checked model has 73,639,032 parameters. Input and output were both 2 × 3 × 256 × 256.

## First comparison

Ten images were taken from the locked UIEB test list. The difficult scene is `716_img_.png`: a strong blue cast, distant water with almost no visible structure, and a very dark foreground.

Each panel is the original input, the matching reference, and a blank E-URM slot that says no checkpoint has been saved. The untrained network is not shown.

Files:

- Panels and `contact_sheet.png` in `paper_records/uieb_workflow10/`
- `paper_records/uieb_workflow10/difficult_case.txt`
- Table: `paper_records/uieb_workflow10/uieb_workflow10.csv` and `uieb_workflow10.xlsx`
- If that CSV is open and locked, the same rows are in `uieb_workflow10_unlocked.csv`
- All 89 test images: `paper_records/uieb_test_unprocessed_vs_reference.csv` and `uieb_test_unprocessed_vs_reference.txt`

Columns are image ID, dataset, method, PSNR, and SSIM. There is one row per image and method. The unprocessed-input row is the raw image against the same-size reference, range [0, 1], peak 1. On all 89 test images that mean is 17.814 dB PSNR (95% interval 16.787 to 18.840) and 0.781 SSIM (95% interval 0.754 to 0.807). Every test name had a reference, and none needed resizing. The difficult case `716_img_.png` is 9.515 dB and 0.625 SSIM. E-URM rows stay blank.

Synthetic-degradation scores are kept out of that table. For the first 15 UIEB test images, the unprocessed degraded input compared with the original image is in `paper_records/available_now/uieb_synthetic_metrics.csv`. That file has its own evaluation column. On those 15 images the mean is 13.117 dB PSNR and 0.404 SSIM. That is the degradation itself, not E-URM.

## Not ready for the next step

These are not done. Their cells stay empty:

- Trained E-URM output, and its PSNR, SSIM, UIQM, and UCIQE
- Ablations: no attention, no color correction, no degradation consistency, no synthetic augmentation, and the capacity-matched control
- Water-Net, FUnIE-GAN, and PCAFA-Net. Their code and weights are not in this folder, so they were not timed and were not scored
- Transfer after EUVP and aquarium training. Those runs start only after the 30 UIEB epochs finish

The unprocessed-input scores above are a baseline against the reference. They are not a trained enhancement result.
