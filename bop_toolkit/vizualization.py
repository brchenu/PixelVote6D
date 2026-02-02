import cv2
import numpy as np


def show_vector_field(
    img: np.ndarray,
    mask: np.ndarray,
    vector_field: np.ndarray,
    keypoints: np.ndarray,
    step: int = 20,
    scale_mode: str = "normalized"
):
    """
    Visualize the vector field on the image with keypoints.
    Args:
        img: (H, W, 3) input image
        mask: (H, W) binary mask
        vector_field: (H, W, K*2) normalized vector field
        keypoints: (K, 2) 2D keypoints in image coordinates
        step: int, step size for drawing arrows
        scale_mode: str, "normalized" shows unit vectors, "full" shows displacement vectors
    Returns:
        None
    """
    ARROW_TIP_LENGTH = 0.03
    img_masked_base = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8) * 255)
    
    for i, point in enumerate(keypoints):
        # Create a fresh copy for each keypoint to avoid accumulation
        img_masked = img_masked_base.copy()
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_masked, (x, y), 4, (0, 255, 0), -1)

        # Draw vector field arrows
        for row in range(0, img.shape[0], step):
            for col in range(0, img.shape[1], step):
                if mask[row, col] <= 0:
                    continue
                
                if scale_mode == "full":
                    # Show full displacement from pixel to keypoint
                    vx = x - col
                    vy = y - row
                else:
                    # Show normalized direction
                    vx = vector_field[row, col, i * 2]
                    vy = vector_field[row, col, i * 2 + 1]
                
                cv2.arrowedLine(
                    img_masked,
                    (col, row),
                    (int(col + vx), int(row + vy)),
                    (255, 0, 0),
                    1,
                    tipLength=ARROW_TIP_LENGTH,
                )

        cv2.imshow(f"Vector Field - Keypoint {i+1}/{len(keypoints)}", img_masked)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
    
    cv2.destroyAllWindows()


def draw_keypoints(image: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray:
    for point in keypoints_2d:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image
