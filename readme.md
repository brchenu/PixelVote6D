# PVNet re-implementation from scratch

This project contains an implementation of the PVNet (Pixel Voting Network) model, which goal is to find keypoints of an object in order to then solve its 6DOF position.


This project was mainly done in a project based learning approach, in order to understand better model creation with Pytorch, pre-training, transfer learning, data manipulation and generation, syntehtic data generation and sim-to-real concept and so on.


Model architecure: 

The PVNet model use a FCN (Fully Convolutional Network)
 
It is based on a "U-Net" like architecture, with en Encoder and a Decoder.
The Encoder is based on the ResNet18 architecture, its role is to extract features, it compress the input into a meaningful feature map.

The Decoder, is a serie of upsampling layers that bring image back to it's original resolution.

Input: C, H, W
Backbone: ResNet18
Output: 
- Vector field (C, H, W) where C = 2K + 1, where K is number of keypoints
- Mask (mask is a probabilty map of pixels, where its the proba that each pixels are part of the detected object)

ResNet18 is used as a backbone but its modified.

### Output details

Vector field 

For a single keypoint K, every pixels p inside the object, mask is assigned a 2D unit vector $V_k(p)$

- Keypoint $(x_k, y_k)$
- Current pixel $(x_p, y_p)$

$$ 
 V_k(p) =  \frac{(x_k - x_p, y_k - y_p)}{\sqrt{(x_k - x_p)² + (y_k - y_p)²}}
$$

The term "field", if done for every pixels you end up with a map directions, looking like a magnetic field.

#### Vfield training data

To generate the vector field label data from a list of 3D keypoints with the transformation matrix of the object relative to the camera here are the steps:

1. Project keypoints to get 2D info (x_k, y_k)

2. For every pixels (u, v) in yout object mask
        - calculate the direction towards the keypoints.
        - normalize it 

3. Save this as a 2 channel image, one channel for X and one channel for Y, (if you have 8 keypoints it will gives you 16 channel 8x2 components)


## Training 

### Binary Cross-Entropy (BCE) for Mask Prediction

For per-pixel object masks, each pixel represents a binary classification:

0 : background  
1 : object

The network predics raw logits for each pixel, which are real numbers, sigmoid activation converts these logits into probabilites [0, 1] (BCE require probabilites as input).

BCE measures the divergence between the predicted probability and the ground truth:

- Encourages high probability for object pixels
- Encourages low probability for background pixels
- Penalizes confident but wrong predictions more than linear losses like MSE

> In our case we use BCEWithLogitsLoss (PyTorch) which integrates sigmoid + BCE in a numerically stable way, so raw logits can be used directly.

#### When to choose BCE

- When the target is binary (0/1) per element (pixel, keypoint presence, etc.)
- When predictions should be interpreted as probabilities
- When confident mistakes should be penalized heavily

### SmoothL1Loss

Vector loss: Loss only calculated for pixels inside the ground truth mask, the formula looks difference betweeen $V_{pred}$ and $V_{truth}$ 

(?) what is $V_{truth}$ since mask can be different
(?) why use and mask and a vfield and not directly uniquely create vector for pixels in the mask ? why do we need both ? 

### PBR (Physically Based Rendering)


## Architecture

The model backbone is based on ResNet18. 
ResNet make use of skip connections, 


Image -> PVNet Model -> RANSAC -> SolvePnP -> 6DOF Pose


### RANSAC

The RANSAC part of the pipeline is here to find the best keypoints possible from the predicted vector field.
To do that the RANSAC pipeline performs different stages:

#### 1. Hypothesis compute 

1. Pick 2 random pixels inside the object mask
2. Use the predicted vector from those pixels to compute the ray intersection
3. The computed intersection point is our Hypothesis H

#### 2. The scoring (consensus part)

Now with our hypothesis (H) we want to ask for every other vector in our mask vfield, if they agree to this keypoint Hypothesis H.

For every pixel in the mask $p_i$

1. Compute vector from pixel to H

$u_i = normalize(H - p_i)$

2. Dot product between the comupted vector hypo $u_i$ and the actual prediction $v_i$

$score_i = v_i . u_i$

3. Check if the $score_i$ is high enough usually > 0.99  

#### 3. Determine final keypoints

Once we have our hypothesis and their associated scores, we can determine the final keypoints.

Multiples way to get the final keypoints:

- Select the keypoints with the highest scores (simplest method)
- Perform a Weighted Mean, where you weight each hypothesis by it's score

$$\mu = \frac{\sum (score_i \cdot H_i)}{\sum score_i}$$

- Or like in PVNet paper treat our hypothesis and scores as a **Spatial Probability Distribution**

##### Case when No valid hypothesis

No valid hypothesis can happen when, for one keypints every sampled hypo was either: 

1. singluar, form nearly parallel vectors
2. out of bounds

That can happen if: 

- The vector field for that keypoint is noisy or inconsistent
- Object heavily occluded or absent
- Mask roughly right but direction prediction are bad

##### Ray Intersection via Carmer's Rule

So in the hypothesis step we need to find the intersection point H of two point pixels $p_1, p_2$ with predicted direction vectors $v_1, v_2$. This requires solving the linear system:

$p_1 + t1 * v_1 = p_2 + t_2 * v_2$

Which can be written as the matrix equation $Ax = b$:$$\begin{bmatrix} v_{1x} & -v_{2x} \\ v_{1y} & -v_{2y} \end{bmatrix} \begin{bmatrix} t_1 \\ t_2 \end{bmatrix} = \begin{bmatrix} p_{2x} - p_{1x} \\ p_{2y} - p_{1y} \end{bmatrix}$$


Instead of using an iterative solver (like `torch.linalg.solve`), we use **Cramer’s Rule** to solve for the distance $t_1$ directly using determinants:$$t_1 = \frac{\det(A_{replace\_col1})}{\det(A)}$$The final intersection point is then: $H = p_1 + t_1 v_1$.

Tow main advantages:

- Compute efficiency
- Numerical Safety By calculating the determinant ($\det A $), we can immediately identify parallel rays (where $\det \approx 0$) and mask them as invalid (NaN) before they cause a crash or numerical instability.

# Notes on PVNet arch

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


#### File structure note

my-repo/
│
├── pyproject.toml
├── README.md
├── requirements.txt   (or only pyproject.toml)
├── .gitignore
│
├── src/
│   └── myproject/
│       ├── __init__.py
│       │
│       ├── training/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── losses.py
│       │
│       ├── inference/
│       │   ├── inference.py
│       │   └── batch_infer.py
│       │
│       ├── models/
│       │   ├── model.py
│       │   └── layers.py
│       │
│       ├── data/
│       │   ├── dataset.py
│       │   ├── loaders.py
│       │   └── preprocessing.py
│       │
│       ├── utils/
│       │   ├── logging.py
│       │   ├── io.py
│       │   └── math.py
│       │
│       ├── geometry/
│       │   ├── ransac.py
│       │   ├── bop_toolkit/
│       │   └── view_kp.py
│       │
│       └── pipelines/
│           └── pipeline.py
│
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── self_label.py
│   └── export_model.py
│
├── configs/
│   ├── train.yaml
│   ├── model.yaml
│   └── dataset.yaml
│
├── tests/
│   ├── test_model.py
│   ├── test_pipeline.py
│   └── test_utils.py
│
├── assets/
├── checkpoints/
├── outputs/
└── datasets/   (or external path only)