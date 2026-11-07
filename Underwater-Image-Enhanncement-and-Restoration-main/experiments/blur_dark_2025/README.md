Blur-and-Dark Underwater Restoration (2025 Experiment)

Dataset layout:

input/  # blurred and/or dark underwater images
target/ # corresponding reference or enhanced targets (if available)

Run:

python -m Underwater-Image-Enhanncement-and-Restoration-main.experiments.blur_dark_2025.run_experiment --data_root PATH/TO/DATA --workdir runs/blur_dark --image_size 256 --batch_size 8 --epochs 20

Outputs:
- checkpoints/ model_epoch_*.pt
- metrics.json (PSNR/SSIM)


