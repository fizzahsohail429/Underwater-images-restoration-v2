# Underwater Image Restoration Deep Learning Model

This project implements a comprehensive deep learning model for underwater image restoration that addresses three main challenges:

1. **Color Correction and Enhancement** - Restores natural colors that are lost due to water absorption
2. **Deblurring and Sharpening** - Removes blur caused by water scattering and turbulence
3. **Visibility Improvement** - Enhances overall visibility and contrast

## Features

### Model Architecture
- **Multi-scale Processing**: Handles different levels of detail simultaneously
- **Attention Mechanisms**: Channel and spatial attention for focusing on important features
- **Residual Connections**: Deep residual blocks for better gradient flow
- **Skip Connections**: Preserves fine details during upsampling

### Key Components

1. **Channel Attention Module**: Focuses on important color channels
2. **Spatial Attention Module**: Identifies important spatial locations
3. **Multi-scale Block**: Processes images at different scales (1x1, 3x3, 5x5, 7x7)
4. **Residual Blocks**: Deep feature processing with attention
5. **Upsampling Blocks**: Progressive restoration with skip connections

### Loss Function
The model uses a combined loss function:
- **L1 Loss**: Pixel-wise accuracy
- **MSE Loss**: Overall structure preservation
- **Perceptual Loss**: High-level feature matching using VGG features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fizzahsohail429/Underwater-images-restoration-v2.git
cd Underwater-images-restoration-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The UIEB experiment uses paired raw images and reference images. Pairing is by filename stem. The image folders are kept locally and are not part of this Git repository.

```
archive/
├── UIEB/              # 890 raw underwater images (model input)
└── reference-890/     # 890 reference images (training target)

paper_records/
├── uieb_train.txt         # 712 image ids
├── uieb_validation.txt    # 89 image ids
└── uieb_test.txt          # 89 image ids
```

The split is locked with seed 2026 at ratios 0.8 / 0.1 / 0.1. The test list is not used for training or checkpoint selection.

## Usage

### Training on UIEB

Place the raw images in `archive/UIEB` and the matching reference images in `archive/reference-890`, then run:

```bash
python train_uieb_reference.py --phase train --learning-rate 0.0001
```

Score the held-out test list once, using the checkpoint with the highest validation PSNR:

```bash
python train_uieb_reference.py --phase test --learning-rate 0.0001
```

### Training Parameters

- **Image Size**: 256x256 pixels
- **Batch Size**: 6
- **Learning Rate**: 0.0001 (0.001 produced non-finite gradients on this task)
- **Epochs**: 30
- **Loss Weights**: L1 (0.4), MSE (0.3), Sobel (0.2), degradation consistency (0.0 on this UIEB reference task)
- **Checkpoint**: highest mean validation PSNR

## Preprocessing and PSNR/SSIM

These settings are the ones used by `train_uieb_reference.py` for training, validation, and the held-out test. The architecture check uses the same 256×256 RGB input size. Validation PSNR in the training log and PSNR/SSIM in the test table are computed by the same functions in `paper_records/quality_metrics.py`.

### Preprocessing

- Read with OpenCV, then convert BGR to RGB.
- Resize the raw input and the reference together to 256×256. Both images go through one Albumentations `Resize`, so they receive the same interpolation. The call does not override the default, which is OpenCV `INTER_LINEAR`.
- No crop, flip, rotation, color jitter, noise, blur, or synthetic degradation on this UIEB run. The training, validation, and test loaders all use the validation transform.
- Normalize each channel with mean 0.5 and standard deviation 0.5. With Albumentations `max_pixel_value=255`, this is `(pixel / 255 - 0.5) / 0.5`. Network inputs are in `[-1, 1]`.
- Before a metric, map a tensor back with `clamp(x * 0.5 + 0.5, 0, 1)`. The unprocessed input, the E-URM output, and the reference are all mapped this way.

### PSNR

- Compare one image with the resized reference. The data range is `[0, 1]` and the peak is 1.
- Mean squared error is over every pixel and all three channels.
- `PSNR = 10 * log10(1 / MSE)`. If MSE is at most `1e-12`, the score is recorded as 99.
- The unprocessed input and E-URM are each scored against that same reference.

### SSIM

- Wang et al. (2004), with `K1 = 0.01` and `K2 = 0.03`, so `C1 = 0.0001` and `C2 = 0.0009` at peak 1.
- 11×11 Gaussian window, sigma 1.5, normalized so the window sums to 1.
- Valid convolution: no padding, so the map is 246×246 for a 256×256 image.
- SSIM is computed on each RGB channel, then the three channel scores are averaged.
- The unprocessed input and E-URM use this same window and the same resized reference.

### Model Outputs

The training process generates:
- `best_underwater_restoration_model.pth`: Best trained model
- `training_history.png`: Training metrics visualization
- `restoration_results.png`: Sample restoration results

## Model Architecture Details

### Input Processing
- Converts degraded underwater images to feature maps
- Applies multi-scale feature extraction
- Uses attention mechanisms for feature selection

### Feature Processing
- Deep residual blocks with attention
- Multi-scale convolution operations
- Progressive feature refinement

### Output Generation
- Upsampling with skip connections
- Final convolution with Tanh activation
- Output in range [-1, 1] for restoration

## Performance Metrics

The model tracks multiple metrics during training:
- **Total Loss**: Combined loss value
- **L1 Loss**: Pixel-wise reconstruction accuracy
- **MSE Loss**: Mean squared error
- **Perceptual Loss**: High-level feature similarity

## Visualization

The model provides comprehensive visualization:
1. **Training History**: Loss curves and metrics over epochs
2. **Restoration Results**: Side-by-side comparison of degraded, restored, and ground truth images
3. **Attention Maps**: Visualization of attention mechanisms (if implemented)

## Customization

### Modifying Model Architecture
- Adjust the number of residual blocks in `ResidualBlock`
- Change the number of channels in each layer
- Modify attention mechanisms

### Adjusting Loss Weights
```python
criterion = CombinedLoss(alpha=0.7, beta=0.2, gamma=0.1)
```

### Data Augmentation
Modify the transforms in `create_transforms()` function:
```python
train_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # Add more augmentations as needed
])
```

## Technical Details

### Degradation Simulation
The model creates synthetic degraded images by:
- Adding blue-green tint (common in underwater images)
- Applying Gaussian blur (water scattering)
- Reducing contrast and brightness
- Adding noise

### Attention Mechanisms
- **Channel Attention**: Uses global average and max pooling
- **Spatial Attention**: Uses channel-wise average and max operations
- Both use shared MLPs and sigmoid activation

### Multi-scale Processing
- 1x1 convolution: Point-wise feature extraction
- 3x3 convolution: Local feature extraction
- 5x5 convolution: Medium-range feature extraction
- 7x7 convolution: Long-range feature extraction

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM recommended

## License

This project is licensed under the MIT License.

## Citation

If you use this model in your research, please cite:
```bibtex
@article{underwater_restoration_2024,
  title={Deep Learning for Underwater Image Restoration},
  author={Your Name},
  journal={Journal of Computer Vision},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on the GitHub repository. 