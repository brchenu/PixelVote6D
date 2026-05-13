# RANSAC Notes

## Purpose

The network predicts dense direction fields, not the final keypoint coordinates directly.

RANSAC is used to turn those noisy pixel-wise votes into a stable 2D keypoint estimate for each keypoint.

## Step 1: Hypothesis Generation

For one keypoint:

1. sample two pixels inside the predicted object mask
2. read the predicted direction vector at each sampled pixel
3. intersect the two rays to produce a keypoint hypothesis

If the rays are close to parallel, the hypothesis is treated as invalid.

## Step 2: Consensus Scoring

For every hypothesis, compute agreement with other foreground pixels.

For a foreground pixel $p_i$, define the direction from the pixel to the hypothesis $H$:

$$
u_i = \text{normalize}(H - p_i)
$$

Compare that with the predicted vector $v_i$ using a dot product:

$$
\text{score}_i = v_i \cdot u_i
$$

If the score is high enough, the pixel counts as an inlier.

## Step 3: Final Keypoint Estimate

After generating many hypotheses, combine them into a final keypoint estimate.

The implementation currently uses a softmax-weighted average of hypotheses, which keeps the estimate stable while still favoring better hypotheses.

## Ray Intersection With Cramer's Rule

The two rays are written as:

$$
p_1 + t_1 v_1 = p_2 + t_2 v_2
$$

This gives the linear system:

$$
\begin{bmatrix}
v_{1x} & -v_{2x} \\
v_{1y} & -v_{2y}
\end{bmatrix}
\begin{bmatrix}
t_1 \\
t_2
\end{bmatrix}
=
\begin{bmatrix}
p_{2x} - p_{1x} \\
p_{2y} - p_{1y}
\end{bmatrix}
$$

Using Cramer's rule makes the implementation efficient and gives a simple determinant test for singular or near-singular cases.

If the determinant is near zero, the sampled rays are effectively parallel and the hypothesis is marked invalid.

## Failure Modes

No valid hypothesis for a keypoint can happen when:

- the predicted vector field is too noisy
- the object is severely occluded
- the mask is weak and does not isolate the object properly
- too many sampled ray pairs are nearly parallel