import torch
import os
import numpy as np
from utils.vector_field import generate_vector_field
from data_transfrom import PVNetTransform

class BOPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str):
        self.dataset = dataset_path
        self.idx = sorted(os.listdir(dataset_path))

        self.img_file = "image.npy"
        self.mask_file = "mask.npy"
        self.kp_file = "2d_keypoints.npy"

        self.transform = PVNetTransform()

    def __getitem__(self, idx):
        sample_path = os.path.join(self.dataset, self.idx[idx])

        image = np.load(os.path.join(sample_path, self.img_file))
        mask = np.load(os.path.join(sample_path, self.mask_file))
        keypoints = np.load(os.path.join(sample_path, self.kp_file))

        image, mask, keypoints = self.transform(image, mask, keypoints)

        h, w = image.shape[1], image.shape[2]

        # Becareful to generate the vector field
        # after applying the different transformations to the image
        vector_field = generate_vector_field(
            height=h,
            width=w,
            mask=mask.squeeze(0).numpy(),
            keypoints=keypoints,
        )

        tensor_vfield = torch.from_numpy(vector_field).float()

        # return image, mask, vector_field
        return image, mask, tensor_vfield

    def __len__(self):
        return len(self.idx)
