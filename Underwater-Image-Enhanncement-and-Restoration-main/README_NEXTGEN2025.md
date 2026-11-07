# NextGen2025 Underwater Restoration

## What this adds
- Dual-stage model: physics-guided dehaze + UNet color refiner
- Training, inference, and evaluation (PSNR, SSIM, UIQM, UCIQE)
- Experiment: `experiments/blur_dark_2025` for blurred/dark underwater images

## Why it’s valuable (2025)
- Handles darkness and blur via coarse-to-fine design
- Physically meaningful intermediates (transmission, ambient light)
- Simple, fast, and easy to reproduce

## Run the experiment
Dataset layout:
- `input/` blurred/dark underwater images
- `target/` corresponding references (if available)

Train + eval:
```bash
python -m Underwater-Image-Enhanncement-and-Restoration-main.experiments.blur_dark_2025.run_experiment \
  --data_root /path/to/dataset \
  --workdir runs/blur_dark \
  --image_size 256 --batch_size 8 --epochs 20
```

Single-image inference:
```bash
python -m Underwater-Image-Enhanncement-and-Restoration-main.nextgen2025.infer \
  --input sample.jpg --output restored.png \
  --checkpoint runs/blur_dark/checkpoints/model_epoch_20.pt
```

## Research contributions to highlight
- Physics-guided first stage for transmission/ambient estimation and coarse recovery
- Data-driven second stage refining color/contrast and artifacts
- Auxiliary supervision improves stability and plausibility
- UIQM/UCIQE-aware evaluation beyond PSNR/SSIM
