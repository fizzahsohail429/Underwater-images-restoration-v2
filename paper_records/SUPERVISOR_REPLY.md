# Reply to the review requests

This note answers the requested checks from the code and files that are on disk now. Items that are not finished are stated as not finished. No score was filled in without a reference image or a saved checkpoint.

## Code version

The project folder is not a git repository, so this working tree does not have a local commit hash.

The manuscript names this public repository:

https://github.com/fizzahsohail429/Underwater-Image-Enhanncement-and-Restoration

The `main` branch on that remote was last looked up as commit `d219b49c81b9eb03932c9174c9f0be9695e8e737`. The local files are not that commit. The decoder-width correction and the records in `paper_records/` exist only in this working tree.

The runnable architecture check for this tree is `paper_records/architecture_check.json`.

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

No checkpoint file is on disk.

`runs/uieb/checkpoint.pt` and `runs/evaluated/checkpoint.pt` are absent. UIEB training is still inside epoch 1 of 30, so the script has not yet reached the validation pass that writes the checkpoint. EUVP and aquarium have not started. The ablation checkpoints have not been trained.

## Dataset splits

| Source | Train | Validation | Test | List files | Notes |
|---|---:|---:|---:|---|---|
| UIEB `archive/UIEB` | 712 | 89 | 89 | `uieb_train.txt`, `uieb_validation.txt`, `uieb_test.txt` | Seed 2026, ratios 0.8 / 0.1 / 0.1. SHA-256 found no duplicate across splits. |
| EUVP `archive/EUVP/Paired/underwater_dark/trainA` | 1182 | 148 | 148 | `euvp_train.txt`, `euvp_validation.txt`, `euvp_test.txt` | Same seed and ratios. This folder has inputs only. |
| Aquarium pretrain | 448 | 127 | 63 | `aquarium_train.txt`, `aquarium_validation.txt`, `aquarium_test.txt` | The folders already supplied these splits. Fifteen capture ids appear in more than one split, so this set is not used as a clean held-out test. |

`archive/UIEB` has 890 raw images and no reference folder. A paired reference file was not found for the comparison images.

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

For each scene the panel is original input, capacity-matched baseline, E-URM, and reference. The reference slot says the reference is unavailable. The baseline and E-URM images in this set are an untrained initialization, because no checkpoint has been saved. They show that the figure path runs. They are not trained enhancement results.

Files:

- Panels and `contact_sheet.png` in `paper_records/uieb_workflow10/`
- `paper_records/uieb_workflow10/difficult_case.txt`
- Table: `paper_records/uieb_workflow10/uieb_workflow10.csv` and `uieb_workflow10.xlsx`
- If that CSV is open and locked, the same rows are in `uieb_workflow10_unlocked.csv`

Columns are image ID, dataset, method, PSNR, and SSIM. There is one row per image and method. PSNR and SSIM are blank because no paired reference file exists, so those scores were not invented.

Synthetic-degradation scores are kept out of that table. For the first 15 UIEB test images, the unprocessed degraded input compared with the original image is in `paper_records/available_now/uieb_synthetic_metrics.csv`. That file has its own evaluation column. On those 15 images the mean is 13.117 dB PSNR and 0.404 SSIM. That is the degradation itself, not E-URM.

## Not ready for the next step

These were left for after the first comparison, and they are not done:

- Trained E-URM, baseline, and ablation outputs
- PSNR and SSIM against UIEB reference images
- UIQM and UCIQE on a trained output
- Water-Net, FUnIE-GAN, and PCAFA-Net
- Transfer onto an external test set

The files above are what can be shared now. The remaining runs need the UIEB reference images and a finished checkpoint.
