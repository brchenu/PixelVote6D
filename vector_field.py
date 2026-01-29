import os
import cv2
import numpy as np
from dataloader import load_data, project_points

scene = "000010"
obj_id = 10


def show_vector_field(
    img: np.ndarray, mask: np.ndarray, vector_field: np.ndarray, step: int = 20
):
    # Draw first keypoints and its vector field
    img_masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8) * 255)
    for i, point in enumerate(project_keypoints):
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_masked, (x, y), 4, (0, 255, 0), -1)

        # Draw vector field arrows
        step = 20
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
    # vector_field_norm = np.linalg.norm(vector_field, axis=-1, keepdims=True) + 1e-8
    # vector_field = vector_field / vector_field_norm

    vector_field = vector_field.reshape(height, width, -1)  # (H, W, K*2)

    bin_mask = mask > 0
    vector_field = vector_field * bin_mask[:, :, None]

    return vector_field


for idx, (keypoints, img, mask, K, R, t) in enumerate(load_data(scene, obj_id, "lm")):

    project_keypoints = project_points(keypoints, K, R, t)
    vector_field = generate_vector_field(
        img.shape[0], img.shape[1], mask, project_keypoints
    )
    print(f"vector_field shape: {vector_field.shape}")

    show_vector_field(img, mask, vector_field, step=20)
