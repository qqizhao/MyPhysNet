## Introduction

This is a partially unofficial reimplementation of [PhysNet](https://arxiv.org/abs/1905.02419) (BMCV 2019) training on UBFC-rPPG dataset.

## Quick Start

1. Install dependencies
```bash
bash setup.sh
pip install -r requirements.txt
```

2. Set your config in `configs/config.py`

3. Train the model
```bash
python main.py
```

## Acknowledgement

- [PhysNet](https://github.com/ZitongYu/PhysNet)
- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
- [PhysNet](https://github.com/MayYoY/PhysNet)