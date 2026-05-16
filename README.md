<p align="center">
  <img src="docs/assets/pixelvote6d.png" alt="PixelVote6D logo" width="420" />
</p>

<h1 align="center">PixelVote6D</h1>

<p align="center">
  From-scratch PVNet (Pixel-wise Voting Network) implementation for 6D object pose estimation.
</p>

<p align="center">
  <img src="docs/assets/demo.gif" alt="PixelVote6D inference demo" width="720" />
</p>

<p align="center">
  Offline inference demo showing pose axes, predicted mask, keypoints, and overlay.
</p>

This project was developed based on the research paper by Peng et al. (see [Citation](#citation)).

The project was built mainly as a research and learning exercise to explore the following topics:

- Implementing a computer vision research paper from scratch in PyTorch
- Deep learning model training and fine-tuning with transfer learning
- Image segmentation and dense output prediction (mask + vector field regression)
- Dataset creation and synthetic data generation with Sim-to-Real transfer
- End-to-end 6D pose estimation pipeline: from raw image to 3D object pose

## Pipeline

```text
image -> PVNet -> mask + vector field -> RANSAC keypoints -> SolvePnP -> 6D pose
```

## Repository Focus

- Train a PVNet-style model for object keypoint voting
- Run offline inference on image folders
- Run realtime inference with a webcam
- Generate and use synthetic and self-labeled data

The repository tries to stay as simple and readable as possible, minimal abstractions, no unnecessary complexity. It is meant to be a self-contained research project that can be read quickly and used as a reference to understand the topic 

## Quick Start

Install dependencies:

```bash
pip install -e .
```

**Get a dataset.** All official BOP datasets are hosted on the [BOP HuggingFace Hub](https://huggingface.co/bop-benchmark). Download any of them with:

```bash
pip install huggingface-hub
huggingface-cli download bop-benchmark/ycbv --repo-type dataset --local-dir dataset/ycbv
```

Replace `ycbv` with any dataset name (`lm`, `lmo`, `tless`, `hope`, `hb`, ...).

Train:

```bash
python scripts/train.py --config configs/train.yaml --dataset-root dataset/
```

## Project Layout

- runnable entry points live under `scripts/`
- reusable geometry, model, and dataset code lives under `src/pixelvote6d/`

## BOP Format

The dataset loader follows the [BOP benchmark](https://bop.felk.cvut.cz) format, the standard reference for 6DoF object pose estimation. This means any BOP-compatible dataset can be used directly with this training code by pointing the config to the right directory.

## Further Reading

These are personal notes accumulated along the way written as questions came up or things were learned and discovered. They serve as a way to explain and solidify understanding, and as a reference to come back to later. Anyone curious about the approach is welcome to read them.

- [Architecture notes](docs/architecture.md)
- [Training notes](docs/training.md)
- [RANSAC notes](docs/ransac.md)

## Status

This is a personal research and showcase project rather than a general-purpose framework.

The main value of the repository is the implementation, experiments, and documentation of the approach.

## Citation

This project was developed based on:

```bibtex
@inproceedings{peng2019pvnet,
  title={PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation},
  author={Peng, Sida and Liu, Yuan and Huang, Qixing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2019}
}
```