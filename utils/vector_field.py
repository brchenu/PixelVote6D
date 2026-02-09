import numpy as np


def generate_vector_field(
    height: int, width: int, mask: np.ndarray, keypoints: np.ndarray
) -> np.ndarray:
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    coords = np.stack((x, y), axis=-1)  # (H, W, 2)

    O = coords[None, :, :, :]  # (1, H, W, 2)
    P = keypoints[:, None, None, :]  # (K, 1, 1, 2)
    vector_field = P - O  # (K, H, W, 2)

    norm = np.sqrt(np.sum(vector_field**2, axis=-1, keepdims=True))
    vector_field = vector_field / (norm + 1e-8)

    # (K, H, W, 2) -transpose-> (K, 2, H, W) -reshape-> (K*2, H, W)
    vector_field = vector_field.transpose(3, 0, 1, 2).reshape(-1, height, width)

    # Apply mask (convert to boolean once)
    bin_mask = mask.astype(bool)
    vector_field = vector_field * bin_mask[None, :, :]

    return vector_field
