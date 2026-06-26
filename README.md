
# Learning Decentralized PointGoal Navigation

## Overview

This repository contains the code of the paper **"Learning Decentralized PointGoal Navigation"**. The paper explores applying MARL to address the problem of PointGoal navigation in multi-robot settings, enabling agents to navigate efficiently in unknown environments while accounting for the presence of other agents.

![](https://github.com/TakieddineSOUALHI/MARLPointNAV/blob/main/video.gif)

<div align="center">
  <img src="https://github.com/TakieddineSOUALHI/MARLPointNAV/blob/main/0.gif" alt="Subfigure 1" width="20%" style="border: 2px solid black; margin: 5px;">
  <img src="https://github.com/TakieddineSOUALHI/MARLPointNAV/blob/main/1.gif" alt="Subfigure 2" width="20%" style="border: 2px solid black; margin: 5px;">
  <img src="https://github.com/TakieddineSOUALHI/MARLPointNAV/blob/main/2.gif" alt="Subfigure 3" width="20%" style="border: 2px solid black; margin: 5px;">
  <img src="https://github.com/TakieddineSOUALHI/MARLPointNAV/blob/main/3.gif" alt="Subfigure 4" width="20%" style="border: 2px solid black; margin: 5px;">
</div>

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)


## Installation

### Prerequisites

- Python 3.8
- CUDA-capable GPU (required for `headless_tensor` rendering mode used during training)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/TakieddineSOUALHI/MARLPointNAV.git
   cd MARLPointNAV
   ```

2. **Create the conda environment from the provided file**

   ```bash
   conda env create -f environment.yml
   conda activate igibson
   ```

   This installs all dependencies including PyTorch 2.1.1 (CUDA), pyequilib, PyBullet, OpenCV, gym, and everything else required.

3. **Install the package in editable mode**

   ```bash
   pip install -e .
   ```


## Datasets

Two datasets are required:

### Gibson Dataset

Place the Gibson scene meshes under `igibson/data/g_dataset/`. Each scene folder (e.g. `Allensville/`) must contain at minimum the file `<SceneName>_mesh_texture.obj`.

```
igibson/data/g_dataset/
├── Allensville/
│   ├── Allensville_mesh_texture.obj
│   └── ...
├── Beechwood/
│   └── ...
└── ...
```

> Download link: https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md

### CollaVN Reset Dataset

This dataset contains per-scene JSON files that define agent spawn positions and goals for each episode. Place it under `igibson/data/dataset/`:

```
igibson/data/dataset/
└── commongoal/
    ├── train/
    │   └── 2/
    │       ├── Allensville.json
    │       └── ...
    └── test/
        └── 2/
            └── hard/
                └── ...
```

> Download link: (https://github.com/Haiyang-W/MAVN)

No configuration changes are needed — the code resolves the dataset path from `igibson/data/dataset/` automatically.


## Usage

### Training

```bash
bash train.sh
```

Or launch directly:

```bash
python ./igibson/onpolicy/scripts/train/train_bot.py \
    --experiment_name training \
    --env_name bot \
    --algorithm_name rmappo \
    --scenario_name common_goal_two_bots \
    --num_agents 2 \
    --episode_length 80
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--num_agents` | `2` | Number of robots |
| `--episode_length` | `80` | Steps per episode |
| `--n_rollout_threads` | — | Number of parallel environments |
| `--experiment_name` | — | Name used for logging and checkpoints |

### Evaluation

```bash
python ./igibson/onpolicy/scripts/train/train_bot.py \
    --experiment_name eval \
    --env_name bot \
    --algorithm_name rmappo \
    --scenario_name common_goal_two_bots \
    --num_agents 2 \
    --episode_length 160 \
    --use_eval True \
    --model_dir <path_to_checkpoint>
```


## Citation

If you use this code, please cite our work:

```bibtex
@ARTICLE{soualhi2025marl,
  author={Soualhi, Takieddine and Crombez, Nathan and Ruichek, Yassine and Lombard, Alexandre and Galland, Stéphane},
  journal={IEEE Robotics and Automation Letters}, 
  title={Learning Decentralized Multi-Robot PointGoal Navigation}, 
  year={2025},
  volume={10},
  number={4},
  pages={4117-4124},
  doi={10.1109/LRA.2025.3550798}}

```
