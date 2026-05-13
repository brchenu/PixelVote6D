# Architecture Notes

## Overview

The model is a PVNet-style fully convolutional network that predicts:

- a foreground mask
- a 2D vector field for each keypoint

The full pipeline is:

```text
image -> PVNet model -> RANSAC -> SolvePnP -> 6D pose
```

## Model Structure

The network follows a U-Net-like encoder-decoder design.

- encoder backbone: ResNet18
- decoder: repeated skip connection, convolution, and upsampling blocks
- output head: foreground mask and keypoint vector field

Input shape:

- `C, H, W`

Output shape:

- vector field: `2K` channels for `K` keypoints
- mask: `1` channel

That gives a total output channel count of `2K + 1`.

## Why A Vector Field

For a keypoint $k$ with image coordinate $(x_k, y_k)$ and a foreground pixel $p=(x_p, y_p)$, the supervision target is the unit vector pointing from the pixel toward that keypoint:

$$
V_k(p) = \frac{(x_k - x_p, y_k - y_p)}{\sqrt{(x_k - x_p)^2 + (y_k - y_p)^2}}
$$

Predicting these local directions is more robust than regressing absolute keypoint coordinates directly from the image.

## Skip Connections

The current implementation keeps skip features from the ResNet backbone at several resolutions:

- `skip_1`: after `resnet.relu`
- `skip_2`: after `resnet.layer1`
- `skip_3`: after `resnet.layer2`
- deeper features: `layer3`, `layer4`, and bottleneck output

These skip connections preserve spatial detail that is useful for dense mask and vector-field prediction.

## Decoder Summary

The decoder progressively upsamples and fuses encoder features:

1. Fuse bottleneck features with mid-level features.
2. Upsample and fuse with earlier backbone features.
3. Upsample to full resolution.
4. Concatenate with the input image before the final prediction head.

## Backbone Notes

The model uses ResNet18 as a starting point, but the later backbone stages are modified with dilation and stride changes to preserve a denser output resolution than the default classifier version.

This is important because pose estimation relies on dense per-pixel predictions rather than a single global feature vector.