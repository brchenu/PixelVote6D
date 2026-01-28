import os
import re
import numpy as np
import cv2
import orjson


def draw_keypoints(image: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray:
    for point in keypoints_2d:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def porject_points(
    points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    assert (
        K.shape == (3, 3)
        and R.shape == (3, 3)
        and t.shape == (3, 1)
        and points_3d.shape[1] == 3
    )
    print(f"points_3d shape: {points_3d.shape}")

    points_2d = K @ (R @ points_3d.T + t)
    print(f"points_2d (homogeneous):\n{points_2d}")
    points_2d = points_2d[:2, :] / points_2d[2, :]
    return points_2d.T


def get_transfo(poses: list, id: int) -> np.ndarray:
    for pose in poses:
        if pose["obj_id"] == id:
            matrix = np.array(pose["cam_R_m2c"]).reshape(3, 3)
            translation = np.array(pose["cam_t_m2c"]).reshape(3, 1)
            transfo = np.eye(4)
            transfo[:3, :3] = matrix
            transfo[:3, 3:] = translation
            return transfo


def get_Rt(poses: list, id: int) -> np.ndarray:
    for pose in poses:
        if pose["obj_id"] == id:
            rotation = np.array(pose["cam_R_m2c"]).reshape(3, 3)
            translation = np.array(pose["cam_t_m2c"]).reshape(3, 1)
            return rotation, translation


def load_data(obj_id: int):
    BASE_DIR = os.path.join("dataset", "lm")

    padded_id = str(obj_id).zfill(6)

    keypoints = np.loadtxt(
        os.path.join(BASE_DIR, "models", f"obj_{padded_id}_keypoints.txt")
    )

    train_dir = os.path.join(BASE_DIR, "train_pbr", padded_id)

    pose_file = os.path.join(train_dir, "scene_gt.json")
    camera_file = os.path.join(train_dir, "scene_camera.json")

    with open(camera_file, "rb") as f:
        cameras = orjson.loads(f.read())

    K = np.array(cameras["0"]["cam_K"]).reshape(3, 3)

    with open(pose_file, "rb") as f:
        poses = orjson.loads(f.read())

    rgb_dir = os.path.join(train_dir, "rgb")
    img_files = sorted(os.listdir(rgb_dir))

    for idx, img_filename in enumerate(img_files):
        img_path = os.path.join(rgb_dir, img_filename)
        img = cv2.imread(img_path)

        curr_poses = poses[str(idx)]
        R, t = get_Rt(curr_poses, int(padded_id))

        projected_keypoints = porject_points(keypoints, K, R, t)

        print(f"Projected keypoints (2D):\n{projected_keypoints}")

        img = draw_keypoints(img, projected_keypoints)

        yield keypoints, img

egg_box = 10

for idx, (keypoints, img) in enumerate(load_data(egg_box)):
    print(f"Sample {idx}\n")

    cv2.imshow("Color image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
