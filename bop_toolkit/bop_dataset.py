import os
from typing import Optional

import cv2
import numpy as np
import orjson
import torch
from torch.utils.data import Dataset

from .data_transfrom import PVNetTransform
from utils.algebra import project_points
from utils.vector_field import generate_vector_field


class BOPSceneIndex:
    """
    Helper class to build an index of dataset samples by aggregating frame-level metadata
    (scene, frame id, mask filename) and camera parameters (R, t, K)
    into a structured collection for efficient retrieval and processing.

    sample is composed of: scene name, frame id, mask filename, R, t, K
    """

    def __init__(self, data_dir: str, obj_id: int):
        self.data_dir = data_dir
        self.obj_id = obj_id

        # Each entry: (scene_name, frame_idx_str, mask_filename, R, t, K)
        self.samples: list[dict] = []
        self._build_index()

    def _build_index(self) -> None:
        scenes = sorted(os.listdir(self.data_dir))

        for scene in scenes:
            scene_path = os.path.join(self.data_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            # scene_gt contains the id and poses of each object in each frame.
            gt_path = os.path.join(scene_path, "scene_gt.json")

            # scene_camera contains the camera intrinsics for each frame.
            cam_path = os.path.join(scene_path, "scene_camera.json")

            with open(gt_path, "rb") as f:
                poses = orjson.loads(f.read())

            with open(cam_path, "rb") as f:
                cameras = orjson.loads(f.read())

            for frame_id_str, frame_poses in poses.items():
                # Check if our object is in this frame
                R, t = self._get_Rt(frame_poses, self.obj_id)

                if R is None:
                    continue

                # Derive mask filename from object index in frame
                mask_filename = self._find_mask(self.obj_id, frame_poses, frame_id_str)

                if mask_filename is None:
                    continue

                K = np.array(cameras[frame_id_str]["cam_K"]).reshape(3, 3)

                self.samples.append(
                    {
                        "scene": scene,
                        "frame": frame_id_str,
                        "mask_filename": mask_filename,
                        "R": R,
                        "t": t,
                        "K": K,
                    }
                )

    @staticmethod
    def _find_mask(obj_id: int, frame_poses: list, frame_id_str: str) -> Optional[str]:
        """Find the mask filename
        The mask filename is formatted as follow: IMID_GTID.png
        where:
           - IMID is an image ID corresponding to the frame_id_str
           - GTID is the index of the object in the list of poses for that
             specific frame (IMID) given in the scene_gt.json file.
        """

        for idx, obj in enumerate(frame_poses):
            if obj["obj_id"] == obj_id:
                return f"{int(frame_id_str):06d}_{idx:06d}.png"
        return None

    @staticmethod
    def _get_Rt(
        poses: list, obj_id: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract rotation and translation for a specific object.

        Args:
            poses: list of pose dictionaries for the current frame.
            obj_id: object ID to find.

        Returns:
            Tuple of (R, t) or (None, None) if object not found.
        """
        for pose in poses:
            if pose["obj_id"] == obj_id:
                R = np.array(pose["cam_R_m2c"]).reshape(3, 3)
                t = np.array(pose["cam_t_m2c"]).reshape(3, 1)
                return R, t
        return None, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class BOPDirectDataset(Dataset):
    """PyTorch Dataset that reads directly from the BOP file structure.

    Loads images, masks, and computes vector fields on-the-fly for a
    single target object across training or test scenes.

    Args:
        dataset_dir: root directory of the BOP dataset (e.g., "dataset/ycbv").
        obj_id: BOP object ID to load (e.g., 10 for egg_box).
        testing: bool = False, if True loads data from test folder
        transform: Optional[PVNetTransform] = None,
    """

    def __init__(
        self,
        dataset_dir: str,
        obj_id: int,
        transform: Optional[PVNetTransform] = None,
        testing_mode: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.obj_id = obj_id
        self.transform = transform or PVNetTransform()
        self.testing_mode = testing_mode

        # Resolve the data directory for this split
        dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        sub = f"{dataset_name}_test_all/test" if testing_mode else "train_pbr"
        self.data_dir = os.path.join(dataset_dir, sub)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load 3D keypoints for the object
        keypoints_path = os.path.join(
            dataset_dir, "models", f"obj_{str(obj_id).zfill(6)}_keypoints.txt"
        )

        if not os.path.exists(keypoints_path):
            raise FileNotFoundError(
                f"Keypoints file not found: {keypoints_path}. "
                f"Generate keypoints first using utils/keypoints_picker.py."
            )

        self.keypoints_3d = np.loadtxt(keypoints_path)

        self.index = BOPSceneIndex(self.data_dir, obj_id)

        if len(self.index) == 0:
            raise ValueError(
                f"No samples found for object {obj_id} in {self.data_dir}. "
                f"Check that the object exists in the scene_gt.json files."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, mask, vector_field):
                - image: (3, H, W) float32 tensor, normalized.
                - mask: (1, H, W) float32 tensor.
                - vector_field: (K*2, H, W) float32 tensor.
        """
        sample = self.index[idx]
        scene = sample["scene"]
        frame = sample["frame"]
        mask_filename = sample["mask_filename"]
        R = sample["R"]
        t = sample["t"]
        K = sample["K"]

        scene_dir = os.path.join(self.data_dir, scene)

        # Load image
        extension = "png" if self.testing_mode else "jpg"
        img_path = os.path.join(scene_dir, "rgb", f"{int(frame):06d}.{extension}")
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = os.path.join(scene_dir, "mask", mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not load mask: {mask_path}")

        keypoints_2d = project_points(self.keypoints_3d, K, R, t)

        # Apply transforms
        image, mask, keypoints_2d = self.transform(image, mask, keypoints_2d)

        h, w = image.shape[1], image.shape[2]

        # Generate vector field on-the-fly after transformations
        vector_field = generate_vector_field(
            height=h,
            width=w,
            mask=mask.squeeze(0).numpy(),
            keypoints=keypoints_2d,
        )

        tensor_vfield = torch.from_numpy(vector_field).float()

        return image, mask, tensor_vfield, keypoints_2d
