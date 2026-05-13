<p align="center">
  <h1 align="center">SA-P2SB: Structure-Aware Schrödinger Bridge for Scene-Level Point Cloud Denoising</h1>
  <h3 align="center"><a href="">📚Paper</a> | <a href="">💾Code</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/overview.png " width="100%">
  </a>
</p>

## Requirements

We recommend creating a clean conda environment before installation.

```bash
conda create -n P2SB python=3.10
conda activate P2SB
```

### Environment

The code has been tested under the following environment:

- Ubuntu 22.04
- Python 3.10
- CUDA 12.8
- PyTorch 2.11
- NVIDIA RTX 5060Ti GPU

### Main Dependencies

- torch
- torchvision
- torchaudio
- open3d
- pytorch3d
- numpy
- scipy
- scikit-learn
- matplotlib
- opencv-python
- trimesh
- point-cloud-utils

Install the required packages using:

```bash
pip install -r requirements.txt
```

### PyTorch Installation

Please install PyTorch according to your CUDA version following the official instructions:

[PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/?utm_source=chatgpt.com)

### PyTorch3D Installation

Please install PyTorch3D manually following the official instructions:

[PyTorch3D Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md?utm_source=chatgpt.com)

### CUDA Extensions

Some CUDA-based operators used in this project (e.g., Chamfer Distance, EMD, PointNet2, and PointOps) may require manual compilation depending on your environment.

## Data Preparation

### PUNet Dataset

The PUNet dataset is used for synthetic point cloud denoising experiments.

Please download the dataset from the official repository:

[PUNet Dataset](https://github.com/yulequan/PU-Net?utm_source=chatgpt.com)

After downloading, organize the files as follows:

```text
data/
├── PUNet/
│   ├── pointclouds/
│   ├── train/
│   └── test/
```

### PCNet Dataset

The PCNet dataset is used for real-scene point cloud denoising evaluation.

Please download the dataset from the official repository or project page:

[PCNet Project Page](https://github.com/minghanz/PCNet?utm_source=chatgpt.com)

Organize the dataset as follows:

```text
data/
├── PCNet/
│   ├── train/
│   ├── val/
│   └── test/
```

### ScanNet++ Dataset

The ScanNet++ dataset is used for large-scale real-world scene evaluation.

Please download the dataset from the official website:

[ScanNet++ Official Website](https://kaldir.vc.in.tum.de/scannetpp/?utm_source=chatgpt.com)

After downloading, place the processed scenes under:

```text
data/
├── ScanNetpp/
│   ├── scene0000/
│   ├── scene0001/
│   └── ...
```

### Dataset Structure

The final dataset directory should be organized as:

```text
data/
├── PUNet/
├── PCNet/
└── ScanNetpp/
```

Please modify the dataset paths in the configuration files before training or evaluation.

## Training

Before training, please modify the corresponding configuration file in the `configs/` directory according to your dataset path and experimental settings.

### Training Command

Run the following command to train the model:

```bash
python train.py \
    --config <CONFIG FILE> \
    --save_dir <SAVE DIRECTORY> \
    --wandb_project <WANDB PROJECT NAME> \
    --wandb_entity <WANDB ENTITY NAME>
```

### Example

```bash
python train.py \
    --config configs/punet.yaml \
    --save_dir experiments/punet \
    --wandb_project P2SB \
    --wandb_entity your_wandb_name
```

### Available Arguments

For all available arguments, run:

```bash
python train.py --help
```

This will also provide instructions for multi-GPU training.

### Weights & Biases (WandB)

This project uses Weights & Biases for experiment tracking and visualization.

Before training, please login to WandB:

```bash
wandb login
```

For more details, visit:

[Weights & Biases Official Website](https://wandb.ai/site?utm_source=chatgpt.com)

## Evaluation

After inference, you can evaluate the denoised point clouds using the provided evaluation script.

The evaluation script supports:

- Chamfer Distance (CD)
- Point-to-Face Distance (P2F)
- Face-to-Point Distance (F2P)
- Point-to-Mesh Distance (P2M)

### Evaluation Command

```bash
python evaluate.py \
    --pred_dir <DENOISED_POINT_CLOUD_DIR> \
    --gt_dir <GROUND_TRUTH_DIR>
```

### Unit Sphere Normalization

To evaluate under unit sphere normalization, add the `--normalize` flag:

```bash
python evaluate.py \
    --pred_dir results/punet \
    --gt_dir data/PUNet/test \
    --normalize
```

### Output

The evaluation script will:

- Compute CD, P2F, F2P, and P2M metrics
- Print the averaged results in a formatted table
- Save all evaluation results to:

```text
evaluation_results.csv
```

### Supported File Formats

The evaluation script supports:

- `.ply`
- `.xyz`

Ground-truth meshes should contain triangle faces for computing P2F/F2P/P2M metrics.
Otherwise, these metrics will be recorded as `NaN`.

## Process Your Own Point Clouds

### Indoor Scenes

```bash
python denoise_room.py \
    --room_path <ROOM PATH> \
    --model_path <MODEL PATH> \
    --out_path <OUTPUT PATH>
```

For all available arguments:

```bash
python denoise_room.py --help
```

### Object-Level Point Clouds

```bash
python denoise_object.py \
    --data_path <PATH TO XYZ FILE> \
    --save_path <OUTPUT FILE> \
    --model_path <MODEL PATH>
```

For all available arguments:

```bash
python denoise_object.py --help
```

# SA-P2SB

This repository provides the official implementation of our manuscript submitted to *The Visual Computer*.

If you find this code useful for your research, please consider citing our related manuscript.
