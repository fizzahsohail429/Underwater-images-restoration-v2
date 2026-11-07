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
git clone <repository-url>
cd underwater-image-restoration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The model expects the following dataset structure:
```
aquarium_pretrain/
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
```

## Usage

### Training the Model

1. Place your underwater images in the appropriate directories
2. Run the training script:
```bash
python underwater_image_restoration.py
```

### Training Parameters

- **Image Size**: 256x256 pixels
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Epochs**: 30 (configurable)
- **Loss Weights**: L1 (0.7), MSE (0.3), Perceptual (0.2)

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