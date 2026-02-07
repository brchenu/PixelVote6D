# Notes on PVNet archi

### Skip connections

- skip_1 (1/2 res) after: resnet.relu (64 channels)
- skip_2 (1/4 res) after: resnet.layer1 (64 channels)
- skip_3 (1/8 res) after: resnet.layer2 (128 channels)
- skip_4 (1/16 res) after: resnet.layer3 (256 channels)
- skip_5 Bottleneck (1/32 res) after: resnet.layer4 (512 channels)


## Decoder Architecture (U-Net Style Upsampling)

The paper describes: **"repeatedly perform skip connection, convolution and upsampling on the feature map, until its size reaches H × W"**

Based on paper Figure 2(b), the decoder has 4 upsampling stages:

| Stage | Operations | Resolution | Channels In | Channels Out | Skip From | Notes |
|-------|------------|------------|-------------|--------------|-----------|-------|
| **Decoder Stage 1** | | | | | | |
| ├─ Skip Connection | Concatenate | 1/8× | 512 + 128 = 640 | - | layer2 (x8s) | Fuse deep + mid features |
| ├─ Conv-BN-ReLU | 3×3 convolution | 1/8× | 640 | 256 | - | Feature refinement |
| ├─ Conv-BN-ReLU | 3×3 convolution | 1/8× | 256 | 256 | - | Additional processing |
| └─ Bilinear Upsample | 2× upsampling | 1/4× | 256 | 256 | - | Spatial upsampling |
| **Decoder Stage 2** | | | | | | |
| ├─ Skip Connection | Concatenate | 1/4× | 256 + 64 = 320 | - | layer1 (x4s) | Add early features |
| ├─ Conv-BN-ReLU | 3×3 convolution | 1/4× | 320 | 128 | - | Feature refinement |
| └─ Bilinear Upsample | 2× upsampling | 1/2× | 128 | 128 | - | Spatial upsampling |
| **Decoder Stage 3** | | | | | | |
| ├─ Skip Connection | Concatenate | 1/2× | 128 + 64 = 192 | - | conv1 (x2s) | Add finest features |
| ├─ Conv-BN-ReLU | 3×3 convolution | 1/2× | 192 | 64 | - | Feature refinement |
| └─ Bilinear Upsample | 2× upsampling | 1× | 64 | 64 | - | Spatial upsampling |
| **Output Head** | | | | | | |
| ├─ Skip Connection | Concatenate | 1× | 64 + 3 = 67 | - | Input image | Add raw RGB |
| ├─ Conv-BN-ReLU | 3×3 convolution | 1× | 67 | 64 | - | Final processing |
| └─ Conv 1×1 | Prediction head | 1× | 64 | K×2 + (C+1) | - | Output vectors + labels |

**Output Channels**:
- **K × 2**: Unit vectors for K keypoints (2D directions: x, y)
- **C + 1**: Semantic labels (C object classes + 1 background)

---

### ResNet forward pass

Orignial ResNet Architecture:

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 1000]                 --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 64, 56, 56]           --
│    └─BasicBlock: 2-1                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
│    └─BasicBlock: 2-2                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-9                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-10                 [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 56, 56]           128
│    │    └─ReLU: 3-12                   [1, 64, 56, 56]           --
├─Sequential: 1-6                        [1, 128, 28, 28]          --
│    └─BasicBlock: 2-3                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-13                 [1, 128, 28, 28]          73,728
│    │    └─BatchNorm2d: 3-14            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-15                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-16                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-17            [1, 128, 28, 28]          256
│    │    └─Sequential: 3-18             [1, 128, 28, 28]          8,448
│    │    └─ReLU: 3-19                   [1, 128, 28, 28]          --
│    └─BasicBlock: 2-4                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-20                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-21            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-22                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-23                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-24            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-25                   [1, 128, 28, 28]          --
├─Sequential: 1-7                        [1, 256, 14, 14]          --
│    └─BasicBlock: 2-5                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-26                 [1, 256, 14, 14]          294,912
│    │    └─BatchNorm2d: 3-27            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-28                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-29                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-30            [1, 256, 14, 14]          512
│    │    └─Sequential: 3-31             [1, 256, 14, 14]          33,280
│    │    └─ReLU: 3-32                   [1, 256, 14, 14]          --
│    └─BasicBlock: 2-6                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-33                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-34            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-35                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-36                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-37            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-38                   [1, 256, 14, 14]          --
├─Sequential: 1-8                        [1, 512, 7, 7]            --
│    └─BasicBlock: 2-7                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-39                 [1, 512, 7, 7]            1,179,648
│    │    └─BatchNorm2d: 3-40            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-42                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-43            [1, 512, 7, 7]            1,024
│    │    └─Sequential: 3-44             [1, 512, 7, 7]            132,096
│    │    └─ReLU: 3-45                   [1, 512, 7, 7]            --
│    └─BasicBlock: 2-8                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-46                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-47            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-49                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-50            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 1000]                 513,000
==========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.81
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 39.75
Params size (MB): 46.76
Estimated Total Size (MB): 87.11
==========================================================================================