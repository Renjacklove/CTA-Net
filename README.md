# CTA-Net for DR Grading

This repository provides a PyTorch implementation of a multi-branch neural network for image classification. The model combines local and global branches, leveraging both convolutional and transformer-based layers for comprehensive feature extraction.

## Overview
![Model Architecture](PNG/model.png)

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

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14046093.svg)](https://doi.org/10.5281/zenodo.14046093)

|--DMDC
    |--ARAD_1K 
    |--dataset 
        |--Train_Spec
            |--ARAD_1K_0001.mat
            |--ARAD_1K_0002.mat
            ： 
            |--ARAD_1K_0950.mat
  	  |--Train_RGB
            |--ARAD_1K_0001.jpg
            |--ARAD_1K_0002.jpg
            ： 
            |--ARAD_1K_0950.jpg
        |--Valid_Spec
            |--ARAD_1K_0901.mat
            |--ARAD_1K_0902.mat
            ： 
            |--ARAD_1K_0950.mat
  	  |--Valid_RGB
            |--ARAD_1K_0901.jpg
            |--ARAD_1K_0902.jpg
            ： 
            |--ARAD_1K_0950.jpg
        |--Test_RGB
            |--ARAD_1K_0951.jpg
            |--ARAD_1K_0952.jpg
            ： 
            |--ARAD_1K_1000.jpg
        |--split_txt
            |--train_list.txt
            |--valid_list.txt
        |--mask.mat
