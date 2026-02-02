import numpy as np


def project_points(
    points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    assert (
        K.shape == (3, 3)
        and R.shape == (3, 3)
        and t.shape == (3, 1)
        and points_3d.shape[1] == 3
    )

    points_2d = K @ (R @ points_3d.T + t)
    points_2d = points_2d[:2, :] / points_2d[2, :]
    return points_2d.T
