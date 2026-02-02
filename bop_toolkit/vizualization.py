import cv2
import numpy as np


def show_vector_field(
    img: np.ndarray,
    mask: np.ndarray,
    vector_field: np.ndarray,
    keypoints: np.ndarray,
    step: int = 20,
):
    """
    Visualize the vector field on the image with keypoints.
    Args:
        img: (H, W, 3) input image
        mask: (H, W) binary mask
        vector_field: (H, W, K*2) vector field
        keypoints: (K, 2) 2D keypoints in the image coordinates
        step: int, step size for drawing arrows
    Returns:
        None
    """
    # Draw first keypoints and its vector field
    img_masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8) * 255)
    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_masked, (x, y), 4, (0, 255, 0), -1)

        # Draw vector field arrows
        for row in range(0, img.shape[0], step):
            for col in range(0, img.shape[1], step):
                if mask[row, col] > 0:
                    vx = vector_field[row, col, i * 2]
                    vy = vector_field[row, col, i * 2 + 1]
                    cv2.arrowedLine(
                        img_masked,
                        (col, row),
                        (int(col + vx), int(row + vy)),
                        (255, 0, 0),
                        1,
                        tipLength=0.03,
                    )

        cv2.imshow("Masked Image", img_masked)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        cv2.destroyAllWindows()


def draw_keypoints(image: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray:
    for point in keypoints_2d:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image


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
