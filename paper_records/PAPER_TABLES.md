# Records for the Sensors manuscript

The measured UIEB experiment is the one in `README.md`. Input is `archive/UIEB`. Target is `archive/reference-890`. The script is `train_uieb_reference.py`. The selected checkpoint is `runs/uieb_reference/checkpoints/best_val_psnr.pt`, epoch 29.

`paper_records/training_configuration.json` keeps the earlier intended settings, including learning rate 0.001 and a consistency weight of 0.1. The completed run used learning rate 0.0001, consistency weight 0.0, and the checkpoint with the highest validation PSNR. That run is recorded in `runs/uieb_reference/uieb_experiment_config.json`.

## Sample counts

| Source | Split | Images | Paired references |
|---|---|---:|---:|
| archive/UIEB | train | 712 | 712 |
| archive/UIEB | validation | 89 | 89 |
| archive/UIEB | test | 89 | 89 |
| archive/reference-890 | unsplit | 890 | 890 |
| archive/challenge-60 | unsplit | 60 | 0 |
| EUVP underwater_dark/trainA | unsplit | 1478 | 0 |

The split uses seed 2026 and ratios 0.8 / 0.1 / 0.1. Pairing is by filename stem. `paper_records/uieb_pair_audit.json` records 890 checked pairs, no missing pair, no filename overlap, and no exact duplicate. Filenames do not contain a scene id. Challenge-60 has no reference images. The EUVP folder has no `trainB` reference folder. Aquarium images are not part of this comparison.

## Completed test

Means over the locked 89 test images. Each model row is the checkpoint with the highest validation PSNR.

| Method | Epoch | Validation PSNR | Test PSNR | Test SSIM | PSNR change vs input |
|---|---:|---:|---:|---:|---:|
| Unprocessed input | — | — | 17.874 | 0.7719 | 0 |
| E-URM | 29 | 22.912 | 22.217 | 0.8825 | +4.342 |
| No color correction | 30 | 22.943 | 22.212 | 0.8820 | +4.338 |
| Capacity-matched | 29 | 21.087 | 20.915 | 0.8672 | +3.041 |
| No attention | 30 | 21.045 | 20.748 | 0.8662 | +2.873 |

The per-image E-URM file is `runs/uieb_reference/uieb_test_results.csv`. Variant summaries are under `runs/uieb_variants/`. Those variant checkpoints are local.

Water-Net, FUnIE-GAN, and PCAFA-Net have no scores. The consistency and synthetic variants have not been trained.

## Computational cost

`paper_records/computational_cost.csv` is a forward-only timing of the untrained architectures at 256×256, batch 6, on CPU. The E-URM row is marked architecture only: 73,639,032 parameters, 7,171.14 ms per batch, peak process working set 1,792.65 MiB. It is not a timing of the trained checkpoint.
