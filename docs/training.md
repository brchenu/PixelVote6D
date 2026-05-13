# Training Notes

## Training Targets

The network predicts two things:

- a binary foreground mask
- a vector field toward each object keypoint

The vector loss is only computed on foreground pixels.

## Mask Loss

Foreground segmentation is trained with binary cross-entropy on logits using `BCEWithLogitsLoss`.

This is appropriate because each pixel is a binary prediction:

- `0`: background
- `1`: object

Using the logits version keeps the optimization numerically stable while still representing a probability after sigmoid.

## Vector Field Loss

The vector field is trained with `SmoothL1Loss`.

In practice, the loss is computed channel-wise and then masked so that only foreground pixels contribute. Background pixels are ignored because the target direction is only defined for pixels belonging to the object.

## Why Predict Both Mask And Vector Field

The mask and vector field solve different problems:

- the mask decides which pixels belong to the object
- the vector field tells each object pixel how to vote toward each keypoint

Without the mask, noisy background vectors would corrupt the RANSAC voting stage.

## Data Sources

The training pipeline supports:

- direct BOP-style rendered or labeled data
- mixed datasets through concatenation
- self-labeled datasets generated from inference outputs
- optional spatial augmentation to reduce the gap between centered synthetic renders and real footage

## Practical Training Notes

- checkpoints store both model state and optimizer state
- runs are saved into timestamped output folders
- dataset mixing can be controlled with sampling weights
- cosine annealing is used to decay the learning rate over a run