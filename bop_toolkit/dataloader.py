import os
import re
import numpy as np
import cv2
import orjson
from mask_finder import find_masks_for_object


def get_Rt(poses: list, obj_id: int) -> tuple:
    """Extract rotation and translation for specific object from scene poses.

    Args:
        poses: List of pose dictionaries for current scene
        obj_id: Object ID to find

    Returns:
        Tuple of (rotation, translation) or (None, None) if not found
    """
    for pose in poses:
        if pose["obj_id"] == obj_id:
            rotation = np.array(pose["cam_R_m2c"]).reshape(3, 3)
            translation = np.array(pose["cam_t_m2c"]).reshape(3, 1)
            return rotation, translation
    return None, None


def load_data(dataset_dir: str, scene: str, obj_id: int):
    """Generator that yields training samples for a specific object.

    Args:
        obj_id: BOP object ID (e.g., 10 for egg_box)
        dataset: Dataset name ("lm" or "lmo")

    Yields:
        Tuple of (keypoints_3d, annotated_image, K, R, t)
    """

    # Load 3D keypoints
    keypoints_path = os.path.join(
        dataset_dir, "models", f"obj_{str(obj_id).zfill(6)}_keypoints.txt"
    )
    keypoints = np.loadtxt(keypoints_path)

    train_dir = os.path.join(dataset_dir, "train_pbr", scene)
    pose_file = os.path.join(train_dir, "scene_gt.json")
    camera_file = os.path.join(train_dir, "scene_camera.json")

    # Load all poses and cameras once
    with open(camera_file, "rb") as f:
        cameras = orjson.loads(f.read())

    with open(pose_file, "rb") as f:
        poses = orjson.loads(f.read())

    masks = find_masks_for_object(dataset_dir, scene, obj_id)

    rgb_dir = os.path.join(train_dir, "rgb")
    img_files = sorted(os.listdir(rgb_dir))

    for idx, img_filename in enumerate(img_files):
        img_path = os.path.join(rgb_dir, img_filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        # Get camera intrinsics for current frame
        K = np.array(cameras[str(idx)]["cam_K"]).reshape(3, 3)

        # Get pose for current frame
        curr_poses = poses[str(idx)]
        R, t = get_Rt(curr_poses, obj_id)

        if R is None or t is None:
            print(f"Warning: Object {obj_id} not found in frame {idx}")
            continue

        mask_filename = masks.get(str(idx))
        mask_path = os.path.join(train_dir, "mask", mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        yield keypoints, img, mask, K, R, t
