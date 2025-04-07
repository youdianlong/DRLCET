```markdown
# DRL$_{\rm{CET}}$: Disentangled Representation Learning with Causal Effect Transmission

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of the paper *"Disentangled Representation Learning with Causal Effect Transmission in Variational Autoencoder"*.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [License](#license)

---

## Environment Setup

1. **Create Conda Environment**:
   ```bash
   conda create -n drlcet python=3.8
   conda activate drlcet
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Training the Model
Run the following command to train the DRL$_{\rm{CET}}$ model:
```bash
python train.py \
    --dataset <dataset_name> \          # e.g., "celeba", "c3dtree"
    --data_dir <path_to_dataset> \      # path to dataset directory
    --seed <random_seed> \              # random seed (default: 42)
    --latent_dim <dimension> \          # latent space dimension (default: 50)
    --labels <label_indices>            # comma-separated label indices (e.g., "smile")
```

**Example**:
```bash
python train.py --dataset celeba --data_dir ./data/celeba --seed 42 --latent_dim 50 --labels "smile"
```

---

## Configuration
All hyperparameters are defined in `config.py`. Key arguments include:

| Argument       | Description                                                                  |
|----------------|------------------------------------------------------------------------------|
| `--dataset`    | Dataset name (e.g., `celeba`, `c3dtree`)                                     |
| `--data_dir`   | Directory path for dataset                                                   |
| `--seed`       | Random seed for reproducibility                                              |
| `--latent_dim` | Dimension of the latent space                                                |
| `--labels`     | Indices of causal attributes to disentangle (comma-separated, e.g., "smile") |
| `--batch_size` | Training batch size (default: `128`)                                         |
| `--lr`         | Learning rate (default: `1e-4`)                                              |

---

## Datasets
The following datasets are used in this project:

1. **CelebA Dataset**  
   - **Description**: CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with over 200K celebrity images, each annotated with 40 attributes.  
   - **Download Link**: [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
   - **Usage**: Place the dataset in `./data/celeba`.

2. **C3dtree Dataset**  
   - **Description**: A custom dataset for causal disentanglement tasks.  
   - **Download Link**: [C3dtree Dataset](https://drive.google.com/file/d/1Q1ysD2ParC3nZmLAczwFqYIZa-PWP5Gb/view?usp=sharing)  
   - **Usage**: Place the dataset in `./data/c3dtree`.

---

## Reproducing Results
To reproduce experiments from the paper:
1. **Dataset Preparation**:  
   Ensure datasets are placed in `./data/<dataset_name>` (e.g., `./data/celeba`).

2. **Run Training Script**:
   ```bash
   # Example for CelebA with latent_dim=64 and labels 2,5,7
   python train.py --dataset celeba --data_dir ./data/celeba --latent_dim 50 --labels "smile"
   ```

3. **Monitor Training**:  
   Logs and model checkpoints will be saved to `./logs/`.

---

## Project Structure
```
DRLCET/
├── config.py             # Configuration parameters
├── train.py              # Main training script
├── requirements.txt      # Dependency list
├── models/
│   ├── encoder.py        # Encoder architecture
│   ├── decoder.py        # Decoder architecture
│   └── discriminator.py  # Discriminator module
├── utils/
│   ├── data_loader.py    # Dataset loading utilities
│   └── losses.py         # Custom loss functions
└── logs/                 # Training logs and checkpoints
```

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
```

### Key Features:
1. **Dataset Links**: Added direct download links for CelebA and C3dtree datasets.
2. **Clear Installation Guide**: Step-by-step environment setup with Conda.
3. **Parameter Documentation**: Detailed explanations of critical arguments.
4. **Reproducibility Focus**: Explicit commands for replicating experiments.
5. **Modular Structure**: Organized codebase for easy customization.

Let me know if you need further adjustments! 🚀