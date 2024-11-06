# CTA-Net for DR Grading

This repository provides a PyTorch implementation of a multi-branch neural network for image classification. The model combines local and global branches, leveraging both convolutional and transformer-based layers for comprehensive feature extraction.

## Overview
()

The model consists of:
- **Local Branch**: Convolutional layers for capturing fine-grained, localized features.
- **Global Branch**: Transformer layers for capturing broader, contextual information.
- **Feature Fusion**: Combines local and global features to enhance performance on classification tasks.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Torchvision

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Renjacklove/CTA-Net.git
   cd CTA-Net
