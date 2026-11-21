# Current status for the supervisor

The public repository is https://github.com/fizzahsohail429/Underwater-images-restoration-v2. `README.md` is the file to follow. The older repository at commit `d219b49` still has the decoder-width mismatch and is not this experiment.

What is finished:

- E-URM was trained for 30 epochs on raw UIEB images with `reference-890` as the target. The selected checkpoint is epoch 29, `runs/uieb_reference/checkpoints/best_val_psnr.pt`. Its SHA-256 is `176c0673f6fa3d4d0ae03ff7ce7537109185cfba418b79fbc9f63675719220c9`. The checkpoint and `runs/uieb_reference/uieb_experiment_config.json` are in commit `514e874`.
- On the locked 89-image test list, the unprocessed input has mean PSNR 17.874 and SSIM 0.7719. E-URM has mean PSNR 22.217 and SSIM 0.8825. The mean PSNR change is +4.342.
- The no-color, no-attention, and capacity-matched variants were trained and scored with the same split, preprocessing, and checkpoint rule. The table is in `README.md` and `paper_records/PAPER_TABLES.md`.
- All 890 filename pairs were checked. None are missing, none overlap across splits, and none are exact file duplicates.

What is not finished:

- Water-Net, FUnIE-GAN, and PCAFA-Net have not been scored. Their official code and a matching split are not in this repository.
- Challenge-60 has 60 images and no reference. EUVP is present as 1478 raw images and has no reference folder here.
- The consistency and synthetic extensions have not been trained.
- One E-URM training run was completed, with a resume at epoch 5. A second inference pass and a second full training run have not been done.
- A Zenodo deposit has not been made.

The learning rate in the manuscript, 0.001, produced non-finite gradients. The completed run used 0.0001. The degradation-consistency weight was 0.
