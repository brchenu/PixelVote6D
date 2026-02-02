import numpy as np


def generate_vector_field(
    height: int, width: int, mask: np.ndarray, keypoints: np.ndarray
) -> np.ndarray:
    # Create coordinate grid: (height, width, 2) with last dimension as [x, y]
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    coords = np.stack((x, y), axis=-1)  # (H, W, 2)

    O = coords[None, :, :, :]  # (1, H, W, 2)
    P = keypoints[:, None, None, :]  # (K, 1, 1, 2)
    vector_field = P - O  # (K, H, W, 2)

    # 1e-8 to avoid division by zero
    norm = np.sqrt(np.sum(vector_field ** 2, axis=-1, keepdims=True)) + 1e-8
    vector_field = vector_field / norm
    
    # Reshape: (H, W, K, 2) -> (H, W, K*2)
    vector_field = vector_field.transpose(1, 2, 0, 3).reshape(height, width, -1)

    # Apply mask (convert to boolean once)
    bin_mask = mask.astype(bool)
    vector_field = vector_field * bin_mask[:, :, None]

    return vector_field
