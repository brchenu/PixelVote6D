import os
import re
import numpy as np
import cv2
import orjson
from mask_finder import find_masks_for_object

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
    keypoints_path = os.path.join(dataset_dir, "models", f"obj_{str(obj_id).zfill(6)}_keypoints.txt")
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