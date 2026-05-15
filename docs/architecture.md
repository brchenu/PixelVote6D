# Model architecure 

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

3. Save this as a 2 channel image, one channel for X and one channel for Y, (if you have 8 keypoints it will gives you 16 channel 8x2 components