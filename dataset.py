import torch
import os
import numpy as np


class BOPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str):
        self.dataset = dataset_path
        self.idx = sorted(os.listdir(dataset_path))

        self.cam_file = "camera_params.npz"
        self.img_file = "image.npy"
        self.mask_file = "mask.npy"
        self.kp_file = "keypoints.npy"
        self.vfield_file = "vector_field.npy"

    def __getitem__(self, idx):
        sample_path = os.path.join(self.dataset, self.idx[idx])

        cam_params = np.load(os.path.join(sample_path, self.cam_file))
        image = np.load(os.path.join(sample_path, self.img_file))
        mask = np.load(os.path.join(sample_path, self.mask_file))
        keypoints = np.load(os.path.join(sample_path, self.kp_file))
        vector_field = np.load(os.path.join(sample_path, self.vfield_file))

        return cam_params, image, mask, keypoints, vector_field

    def __len__(self):
        return len(self.idx)
