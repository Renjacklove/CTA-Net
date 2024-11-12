## Overview
![Model Architecture](PNG/model.png)

# CTANet: Enhanced Diabetic Retinopathy Grading via Dual-Branch Network with Adaptive Focus Fusion

This repository contains the code and data for our paper, *"CTANet: Enhanced Diabetic Retinopathy Grading via Dual-Branch Network with Adaptive Focus Fusion"*.

## Overview

Diabetic Retinopathy (DR) is a leading cause of vision impairment, requiring accurate grading for timely intervention. CTANet is a dual-branch network designed to enhance DR grading accuracy. It includes:
- **Local Branch (ODStarNet)**: Focuses on extracting fine lesion details.
- **Global Branch (Swin Transformer)**: Captures comprehensive structural and spatial context.
- **Adaptive Focus Fusion (AFF) Module**: Dynamically balances local and global feature integration for improved accuracy.

Our experiments on the APTOS 2019 and DDR datasets demonstrate that CTANet achieves state-of-the-art performance, surpassing existing methods.

## Requirements and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Renjacklove/CTA-Net.git
    cd CTA-Net
    ```

2. **Install dependencies**:
    We recommend using a virtual environment. The required dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3. **Setup environment**:
    Ensure you are using Python 3.8 or later and have an NVIDIA GPU with CUDA installed for optimal performance.

## Dataset Preparation

CTANet relies on two datasets for diabetic retinopathy grading:
- **APTOS 2019 Dataset**: Available from [Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)
- **DDR Dataset**: Available from [Kaggle](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset)

### Steps to Download and Organize Data
1. Download the datasets from Kaggle (requires a Kaggle account).
2. Place the datasets in the following directory structure:
    ```
    CTA-Net/
    ├── data/
    │   ├── aptos2019/
    │   │   ├── train_images/
    │   │   └── test_images/
    │   └── ddr/
    │       ├── train_images/
    │       └── test_images/
    ```

3. **Data Preprocessing**:
   Run the following command to preprocess the data if necessary:
   ```bash
   python preprocess_data.py --input data/aptos2019 --output data/aptos2019/processed

## Training and Testing

### Quick Start
To train and test the CTANet model on the APTOS 2019 dataset, follow these steps:

1. **Training the model**:
    ```bash
    python train.py --config configs/aptos2019.yaml
    ```

2. **Testing the model**:
    ```bash
    python test.py --config configs/aptos2019.yaml
    ```

3. **Evaluation on DDR Dataset**:
    Modify the configuration file or specify the dataset path for the DDR dataset.
    ```bash
    python train.py --config configs/ddr.yaml
    python test.py --config configs/ddr.yaml
    ```

### One-click Run
You can also use the `run_experiment.sh` script to perform training and testing in one step:
```bash
bash run_experiment.sh


## FAQ

### 1. How can I resolve dependency issues?
   - Ensure you have installed all dependencies listed in `requirements.txt`. If issues persist, consider using `pip install --upgrade <package_name>` for problematic packages.

### 2. Can I run this on CPU?
   - CTANet is optimized for GPU. While it can run on a CPU, the training and testing time will be significantly longer.

### 3. How do I set up a specific random seed for reproducibility?
   - Random seeds can be set in `train.py` to ensure reproducible results:
     ```python
     import torch
     import numpy as np
     import random
     
     seed = 42
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     ```

## License and Citation

This code is made available under the MIT License. If you find our work helpful, please cite our paper:



## Contact

If you have any questions or run into issues, please reach out:
- **Email**: 152514845@qq.com




[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14046093.svg)](https://doi.org/10.5281/zenodo.14046093)
