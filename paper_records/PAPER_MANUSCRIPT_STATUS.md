# Manuscript status

Updated after the UIEB reference run and the three controlled variants.

| Item | Status |
|---|---|
| Corrected model | `improved_underwater_restoration.py`, decoder correction in commit `80d7096` |
| Training and test script | `train_uieb_reference.py`, commit `4a9d333` |
| Selected checkpoint | `runs/uieb_reference/checkpoints/best_val_psnr.pt`, epoch 29, published with Git LFS in commit `514e874` |
| Configuration | `runs/uieb_reference/uieb_experiment_config.json`. The saved `git_revision` field is `2bc0bf5` |
| README | Matches this run, including preprocessing, PSNR/SSIM, and the comparison table |
| Pair audit | `paper_records/uieb_pair_audit.json`, 890 pairs |
| Ablations | No color, no attention, and capacity-matched are trained and scored locally |
| External methods | Water-Net, FUnIE-GAN, and PCAFA-Net are not scored |
| External paired test | Not available. Challenge-60 has no reference. EUVP `trainB` is absent |
| Repeated run | Not done |

The conclusion that is supported is the comparison with the unprocessed input and the three finished variants. A comparison with Water-Net, FUnIE-GAN, or PCAFA-Net is not supported.
