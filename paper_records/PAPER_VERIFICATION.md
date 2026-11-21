# Verification of the completed UIEB run

This file replaces the earlier note that said training was still in epoch 1. That note described a different task. The completed reference-890 run is finished.

- Model: `HybridEncoderDecoder` through `build_paper_variant("e-urm")` in `improved_underwater_restoration.py`.
- Trainer and test: `train_uieb_reference.py`.
- Input: `archive/UIEB`. Target: `archive/reference-890`, same filename stem.
- Split: `paper_records/uieb_train.txt` (712), `uieb_validation.txt` (89), `uieb_test.txt` (89). Seed 2026. The test list was not used to choose the checkpoint.
- Selected checkpoint: `runs/uieb_reference/checkpoints/best_val_psnr.pt`, epoch 29, validation PSNR 22.911585. SHA-256 `176c0673f6fa3d4d0ae03ff7ce7537109185cfba418b79fbc9f63675719220c9`.
- Test means: unprocessed PSNR 17.874277 and SSIM 0.771901; E-URM PSNR 22.216550 and SSIM 0.882475; PSNR change +4.342273.
- Public commits: decoder correction `80d7096`, trainer `4a9d333`, checkpoint and configuration `514e874`, ablation trainer and pair audit `fda03a3`. The configuration field `git_revision` remains `2bc0bf5` because that was the revision when training started.
- Pair audit: `paper_records/audit_uieb_pairs.py` writes `uieb_pair_audit.json`.
- Shape check: `shape_check()` in `paper_records/train_three_sources.py`. Running that file as a program starts a different training task.

The attention, color, and capacity-matched variants finished under `train_uieb_variants.py`. Their test means are in `README.md`. A second inference pass, a second full training run, Water-Net, FUnIE-GAN, PCAFA-Net, and a paired external test have not been run.
