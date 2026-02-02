import numpy as np


def generate_vector_field(
    height: int, width: int, mask: np.ndarray, keypoints: np.ndarray
) -> np.ndarray:
    # Create coordinate grid: (height, width, 2) with last dimension as [x, y]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack((x, y), axis=-1)  # (H, W, 2)

    O = coords[None, :, :, :]  # (1, H, W, 2)
    P = keypoints[:, None, None, :]  # (K, 1, 1, 2)

    vector_field = P - O  # (K, H, W, 2)
    vector_field = vector_field.transpose(1, 2, 0, 3)

    # Normalize vectors
    vector_field_norm = np.linalg.norm(vector_field, axis=-1, keepdims=True) + 1e-8
    vector_field = vector_field / vector_field_norm

    vector_field = vector_field.reshape(height, width, -1)  # (H, W, K*2)

    bin_mask = mask > 0
    vector_field = vector_field * bin_mask[:, :, None]

    return vector_field
