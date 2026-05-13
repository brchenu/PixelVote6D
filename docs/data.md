# Data Notes

## Synthetic And Real Data

This project mixes synthetic data, real captures, and self-labeled data.

The goal is to reduce the sim-to-real gap while still benefiting from the scale and controllability of rendered data.

## Vector Field Label Generation

To build supervision targets for one image:

1. project the 3D keypoints into image space
2. use the object mask to determine foreground pixels
3. for each foreground pixel, compute the normalized direction toward each projected keypoint
4. store all keypoint directions as a stacked vector field

For `K` keypoints, this produces `2K` channels.

## PBR And Domain Gap

Physically based rendering helps produce synthetic images that are closer to real images, but there is still a domain gap.

One practical issue in this project is that Blender renders tend to keep the object centered, while real footage does not. Spatial augmentation helps bridge that mismatch.

## Self-Labeling

The repository also includes support for self-labeled data generated from inference outputs.

That data can be useful as a lightweight way to adapt the model toward real captures once the base model is already good enough to produce reasonable pseudo-labels.

## Repository Advice

For GitHub, keep large datasets outside the repository itself.

Recommended split:

- GitHub: code, docs, figures, a tiny sample, and usage examples
- external hosting: full generated datasets and selected checkpoints

If you publish the data, prefer one curated dataset release instead of many intermediate generations.