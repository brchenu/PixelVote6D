# PixelVote6D

PixelVote6D is a from-scratch reimplementation of PVNet for 6D object pose estimation.

The project was built as a research and learning exercise around PyTorch training, synthetic data generation, sim-to-real transfer, vector-field supervision, RANSAC-based keypoint voting, and pose recovery with SolvePnP.

## Pipeline

```text
image -> PVNet -> mask + vector field -> RANSAC keypoints -> SolvePnP -> 6D pose
```

## Repository Focus

- train a PVNet-style model for object keypoint voting
- run offline inference on image folders
- run realtime inference with a webcam
- generate and use synthetic and self-labeled data

## Quick Start

Install dependencies in your environment:

```bash
pip install -e .
```

Train:

```bash
python train.py --obj-id 1 --dataset drill drill_hd --epochs 20
```

Offline inference:

```bash
python inference.py \
  --images dataset/realfootage/drill2/frames/ \
  --calib dataset/realfootage/drill2/calibration/ \
  --checkpoint checkpoints/2026-04-02_14-56-01_obj1_drill_hd+drill_cut+sl_drill2+sl_real/checkpoint.pth \
  --keypoints dataset/drill/models/obj_000001_keypoints.txt \
  --output output/inference.mp4
```

Realtime demo:

```bash
python realtime.py
```

## Project Layout

- training and inference entry points currently live at the repository root
- reusable geometry, model, and dataset code is being migrated into a `src/` package
- datasets, checkpoints, and generated outputs stay local and are ignored by Git

## Further Reading

- [Architecture notes](docs/architecture.md)
- [Training notes](docs/training.md)
- [RANSAC notes](docs/ransac.md)
- [Data notes](docs/data.md)
- [Migration plan](docs/migration-plan.md)

## Status

This is a personal research and showcase project rather than a general-purpose framework.

The main value of the repository is the implementation, experiments, and documentation of the approach.