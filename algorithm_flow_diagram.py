"""
Algorithm Flow Diagram
=====================

Creates a focused flow diagram showing the step-by-step process
of the improved underwater image restoration algorithm.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_algorithm_flow():
    """Create the algorithm flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Improved Underwater Image Restoration Algorithm Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input
    input_box = FancyBboxPatch((5, 8.5), 2, 0.6, boxstyle="round,pad=0.1",
                              facecolor='#E6F3FF', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 8.8, 'Degraded Underwater Image', fontsize=12, ha='center', fontweight='bold')
    
    # Step 1: Multi-Scale Attention Processing
    step1_box = FancyBboxPatch((0.5, 7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor='#FFE6E6', edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2, 7.8, 'Step 1: Multi-Scale Attention Module', fontsize=11, ha='center', fontweight='bold')
    ax.text(2, 7.4, '• 4 parallel convolution paths', fontsize=9, ha='center')
    ax.text(2, 7.1, '• Channel & spatial attention', fontsize=9, ha='center')
    ax.text(2, 6.8, '• Adaptive feature fusion', fontsize=9, ha='center')
    
    # Step 2: Hybrid Encoder-Decoder
    step2_box = FancyBboxPatch((4.5, 7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor='#E6FFE6', edgecolor='black', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(6, 7.8, 'Step 2: Hybrid Architecture', fontsize=11, ha='center', fontweight='bold')
    ax.text(6, 7.4, '• U-Net skip connections', fontsize=9, ha='center')
    ax.text(6, 7.1, '• ResNet residual learning', fontsize=9, ha='center')
    ax.text(6, 6.8, '• Attention mechanisms', fontsize=9, ha='center')
    
    # Step 3: Adaptive Color Correction
    step3_box = FancyBboxPatch((8.5, 7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor='#FFF2E6', edgecolor='black', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(10, 7.8, 'Step 3: Adaptive Color Correction', fontsize=11, ha='center', fontweight='bold')
    ax.text(10, 7.4, '• Dynamic color matrices', fontsize=9, ha='center')
    ax.text(10, 7.1, '• Underwater enhancement', fontsize=9, ha='center')
    ax.text(10, 6.8, '• Depth-aware processing', fontsize=9, ha='center')
    
    # Intermediate output
    inter_box = FancyBboxPatch((5, 5.5), 2, 0.6, boxstyle="round,pad=0.1",
                              facecolor='#F0E6FF', edgecolor='black', linewidth=2)
    ax.add_patch(inter_box)
    ax.text(6, 5.8, 'Restored Image', fontsize=12, ha='center', fontweight='bold')
    
    # Physics-Informed Loss
    loss_box = FancyBboxPatch((0.5, 4), 4, 1, boxstyle="round,pad=0.1",
                             facecolor='#FFD700', edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(2.5, 4.8, 'Physics-Informed Loss Function', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, 4.4, '• L1 Loss (0.4) + MSE Loss (0.3)', fontsize=9, ha='center')
    ax.text(2.5, 4.1, '• Edge Loss (0.2) + Physics Loss (0.1)', fontsize=9, ha='center')
    ax.text(2.5, 3.8, '• Sobel edge detection + Beer-Lambert simulation', fontsize=9, ha='center')
    
    # Advanced Degradation
    deg_box = FancyBboxPatch((5.5, 4), 4, 1, boxstyle="round,pad=0.1",
                            facecolor='#E5CCFF', edgecolor='black', linewidth=2)
    ax.add_patch(deg_box)
    ax.text(7.5, 4.8, 'Advanced Degradation Simulation', fontsize=11, ha='center', fontweight='bold')
    ax.text(7.5, 4.4, '• Beer-Lambert law (wavelength absorption)', fontsize=9, ha='center')
    ax.text(7.5, 4.1, '• Rayleigh scattering simulation', fontsize=9, ha='center')
    ax.text(7.5, 3.8, '• Signal-dependent noise modeling', fontsize=9, ha='center')
    
    # Training Process
    train_box = FancyBboxPatch((1, 2.5), 8, 1, boxstyle="round,pad=0.1",
                              facecolor='#CCFFCC', edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(5, 3.3, 'Training Process', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 2.9, '• AdamW Optimizer + Learning Rate Scheduler', fontsize=10, ha='center')
    ax.text(5, 2.6, '• 30 Epochs with early stopping + Best model saving', fontsize=10, ha='center')
    
    # Performance Metrics
    metrics_box = FancyBboxPatch((0.5, 1), 11, 1, boxstyle="round,pad=0.1",
                                facecolor='#FFFFCC', edgecolor='black', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(6, 1.8, 'Performance Evaluation Metrics', fontsize=12, ha='center', fontweight='bold')
    ax.text(6, 1.4, 'PSNR (dB) | SSIM | Color Accuracy (%) | Edge Preservation (%) | Real-world Generalization (%)', fontsize=10, ha='center')
    ax.text(6, 1.1, 'Expected Improvements: +3dB | +0.075 | +20% | +25% | +35%', fontsize=10, ha='center', fontweight='bold')
    
    # Draw arrows
    # Input to steps
    ax.arrow(6, 8.5, -2.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 8.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 8.5, 2.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Steps to intermediate output
    ax.arrow(2, 7, 1.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 7, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(10, 7, -1.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Intermediate to loss and degradation
    ax.arrow(5, 5.5, -2, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, 5.5, 2, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Loss and degradation to training
    ax.arrow(2.5, 4, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 4, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Training to metrics
    ax.arrow(5, 2.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Novel contributions highlight
    ax.text(6, 0.5, '★ 5 Novel Contributions: Multi-Scale Attention | Adaptive Color Correction | Physics-Informed Loss | Hybrid Architecture | Advanced Degradation ★', 
            fontsize=11, ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('algorithm_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_flow():
    """Create a comparison flow diagram"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Simple Model Flow
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.text(4, 7.5, 'Simple Model Flow', fontsize=14, fontweight='bold', ha='center')
    
    # Simple model components
    simple_boxes = [
        {'name': 'Input\nImage', 'pos': (4, 6.5), 'color': '#E6F3FF'},
        {'name': 'Basic\nEncoder', 'pos': (2, 5.5), 'color': '#FFE6E6'},
        {'name': 'Simple\nDecoder', 'pos': (6, 5.5), 'color': '#E6FFE6'},
        {'name': 'Basic Color\nCorrection', 'pos': (4, 4.5), 'color': '#FFF2E6'},
        {'name': 'L1 + MSE\nLoss', 'pos': (4, 3.5), 'color': '#F0E6FF'},
        {'name': 'Output', 'pos': (4, 2.5), 'color': '#CCFFCC'}
    ]
    
    for box in simple_boxes:
        rect = FancyBboxPatch((box['pos'][0]-0.8, box['pos'][1]-0.4), 1.6, 0.8,
                             boxstyle="round,pad=0.05", facecolor=box['color'],
                             edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(box['pos'][0], box['pos'][1], box['name'], fontsize=9, ha='center', va='center')
    
    # Draw simple model arrows
    for i in range(len(simple_boxes)-1):
        ax1.arrow(simple_boxes[i]['pos'][0], simple_boxes[i]['pos'][1]-0.4, 
                 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Improved Model Flow
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.text(4, 7.5, 'Improved Model Flow', fontsize=14, fontweight='bold', ha='center')
    
    # Improved model components
    improved_boxes = [
        {'name': 'Input\nImage', 'pos': (4, 6.5), 'color': '#E6F3FF'},
        {'name': 'Multi-Scale\nAttention', 'pos': (1, 5.5), 'color': '#FFE6E6'},
        {'name': 'Hybrid\nEncoder', 'pos': (3, 5.5), 'color': '#FFE6E6'},
        {'name': 'Hybrid\nDecoder', 'pos': (5, 5.5), 'color': '#E6FFE6'},
        {'name': 'Adaptive Color\nCorrection', 'pos': (7, 5.5), 'color': '#E6FFE6'},
        {'name': 'Physics-Informed\nLoss', 'pos': (2, 4.5), 'color': '#FFF2E6'},
        {'name': 'Advanced\nDegradation', 'pos': (6, 4.5), 'color': '#FFF2E6'},
        {'name': 'Multi-Component\nTraining', 'pos': (4, 3.5), 'color': '#F0E6FF'},
        {'name': 'Enhanced\nOutput', 'pos': (4, 2.5), 'color': '#CCFFCC'}
    ]
    
    for box in improved_boxes:
        rect = FancyBboxPatch((box['pos'][0]-0.8, box['pos'][1]-0.4), 1.6, 0.8,
                             boxstyle="round,pad=0.05", facecolor=box['color'],
                             edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(box['pos'][0], box['pos'][1], box['name'], fontsize=9, ha='center', va='center')
    
    # Draw improved model arrows
    # Input to processing
    ax2.arrow(4, 6.5, -2, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(4, 6.5, -0.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(4, 6.5, 0.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(4, 6.5, 2, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Processing to loss/degradation
    ax2.arrow(2, 5.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(6, 5.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Loss/degradation to training
    ax2.arrow(2, 4.5, 1.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(6, 4.5, -1.5, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Training to output
    ax2.arrow(4, 3.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Performance comparison
    ax1.text(4, 1.5, 'Simple Model Performance:', fontsize=10, ha='center', fontweight='bold')
    ax1.text(4, 1.2, '• Basic restoration', fontsize=9, ha='center')
    ax1.text(4, 0.9, '• Limited color accuracy', fontsize=9, ha='center')
    ax1.text(4, 0.6, '• Standard loss function', fontsize=9, ha='center')
    
    ax2.text(4, 1.5, 'Improved Model Performance:', fontsize=10, ha='center', fontweight='bold')
    ax2.text(4, 1.2, '• Enhanced restoration', fontsize=9, ha='center')
    ax2.text(4, 0.9, '• Superior color accuracy', fontsize=9, ha='center')
    ax2.text(4, 0.6, '• Physics-informed loss', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create flow diagrams"""
    print("Creating algorithm flow diagrams...")
    
    # Create main algorithm flow
    create_algorithm_flow()
    print("Algorithm flow diagram saved as: algorithm_flow_diagram.png")
    
    # Create comparison flow
    create_comparison_flow()
    print("Model comparison flow saved as: model_comparison_flow.png")

if __name__ == "__main__":
    main() 