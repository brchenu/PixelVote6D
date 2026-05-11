import os
import cv2
import numpy as np
import orjson
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset

from .data_transfrom import PVNetTransformV2
from utils.vector_field import generate_vector_field


class SelfLabelDataset(Dataset):
    """Dataset for self-labeled real footage produced by batch_infer.py.

    Reads original frames from the source directory recorded in dataset.json,
    loads predicted masks and 2D keypoints, and generates vector fields on-the-fly.

    Returns the same (image, mask, vfield, keypoints_2d) tuple as BOPDirectDataset.

    Args:
        infer_dir: batch_infer.py output folder (must contain dataset.json,
                   mask_visib/, 2d_keypoints/).
        augment: if True, apply random augmentations (blur, color jitter)
                 matching PVNetRandomTranform. Default False uses deterministic
                 transform only.
    """

    def __init__(self, infer_dir: str, augment: bool = False):
        self.infer_dir = infer_dir

        meta_path = os.path.join(infer_dir, "dataset.json")
        with open(meta_path, "rb") as f:
            meta = orjson.loads(f.read())

        self.frames_dir = meta["frames_dir"]
        self.frame_ids = meta["frame_ids"]

        if len(self.frame_ids) == 0:
            raise ValueError(f"No frames listed in {meta_path}")

        # Image-only transform: resize + crop + normalize (same pipeline as inference).
        # Mask and keypoints are already in 224x224 crop space from batch_infer.py,
        # so they must NOT go through spatial transforms again.
        self._img_transform = PVNetTransformV2().transform

        # Optional image-only augmentations (applied after the base transform)
        self._augment = augment
        if augment:
            self._aug = v2.Compose([
                v2.RandomApply([v2.GaussianBlur(5)], p=0.2),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]

        # --- Load original frame and apply image transform ---
        img_path = os.path.join(self.frames_dir, f"{frame_id}.png")
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image only: resize + crop + normalize → (3, 224, 224)
        image = self._img_transform(image)
        if self._augment:
            image = self._aug(image)

        # --- Load predicted mask (already 224x224, 0/255) ---
        mask_path = os.path.join(self.infer_dir, "mask_visib", f"{frame_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not load mask: {mask_path}")
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # (1, 224, 224)

        # --- Load predicted 2D keypoints (already in 224x224 crop space) ---
        kp_path = os.path.join(self.infer_dir, "2d_keypoints", f"{frame_id}.txt")
        keypoints_2d = np.loadtxt(kp_path)  # (K, 2)

        h, w = image.shape[1], image.shape[2]

        # Generate vector field on-the-fly from mask + keypoints
        vector_field = generate_vector_field(
            height=h,
            width=w,
            mask=mask.squeeze(0).numpy(),
            keypoints=keypoints_2d,
        )
        tensor_vfield = torch.from_numpy(vector_field).float()

        return image, mask, tensor_vfield, keypoints_2d
