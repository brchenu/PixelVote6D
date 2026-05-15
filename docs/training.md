# Training 

## Binary Cross-Entropy (BCE) for Mask Prediction

For per-pixel object masks, each pixel represents a binary classification:

0 : background  
1 : object

The network predics raw logits for each pixel, which are real numbers, sigmoid activation converts these logits into probabilites [0, 1] (BCE require probabilites as input).

BCE measures the divergence between the predicted probability and the ground truth:

- Encourages high probability for object pixels
- Encourages low probability for background pixels
- Penalizes confident but wrong predictions more than linear losses like MSE

> In our case we use BCEWithLogitsLoss (PyTorch) which integrates sigmoid + BCE in a numerically stable way, so raw logits can be used directly.

### When to choose BCE

- When the target is binary (0/1) per element (pixel, keypoint presence, etc.)
- When predictions should be interpreted as probabilities
- When confident mistakes should be penalized heavily

## SmoothL1Loss

Vector loss: Loss only calculated for pixels inside the ground truth mask, the formula looks difference betweeen $V_{pred}$ and $V_{truth}$ 

(?) what is $V_{truth}$ since mask can be different
(?) why use and mask and a vfield and not directly uniquely create vector for pixels in the mask ? why do we need both ? 