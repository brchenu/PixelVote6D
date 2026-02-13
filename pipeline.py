import cv2
import os
import torch
import numpy as np
from model import PVNet
from pathlib import Path
from bop_toolkit.bop_dataset import BOPDirectDataset
from bop_toolkit.data_transfrom import PVNetTransform


def intersect_line(p1, v1, p2, v2):
    A = np.array([v1, -v2]).T
    b = p2 - p1
    t = np.linalg.solve(A, b)
    return p1 + t[0] * v1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

pvnet = PVNet()
pvnet.load_state_dict(
    torch.load("checkpoints/pvnet_checkpoint.pth", weights_only=True)[
        "model_state_dict"
    ]
)
pvnet.to(device)

BASE_DIR = Path(__file__).resolve().parent
dataset_path = os.path.join(BASE_DIR, "dataset", "ycbv")

dataset = BOPDirectDataset(
    dataset_dir=dataset_path, obj_id=1, transform=PVNetTransform()
)

image, mask, vfield, keypoints_2d = dataset[0]

# img = PVNetTransform.unnormalize_image(image)
# img = img.permute(1, 2, 0).cpu().numpy()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


class PVNetRansac:
    def __init__(self, data: torch.Tensor, iter: int):
        self.reshaped_data = data.reshape(
            data.size(0) // 2, 2, data.size(1), data.size(2)
        )
        self.iter = iter

    @staticmethod
    def single_hypothesis(data: torch.Tensor):
        assert data.dim() == 3 and data.size(0) == 2  # 2, H, W

        h, w = data.size()[1:]
        idx = torch.randperm(h * w)[:2]

        def coords(idx: int) -> tuple[int, int]:
            row = idx // w
            col = idx % w
            return (col, row)

        p1 = coords(idx[0])
        p2 = coords(idx[1])

        vec1_vals = data[:, p1[1], p1[0]]
        vec2_vals = data[:, p2[1], p2[0]]

        assert vec1_vals.dim() == 1 and vec1_vals.size(0) == 2

        return intersect_line(p1, vec1_vals, p2, vec2_vals)

    @staticmethod
    def batched_hypothesis(data: torch.Tensor):
        """Batch hypothesis over B keypoits"""

        assert data.dim() == 4 and data.size(1) == 2  # B, 2, H, W

        h, w = data.size()[2:]
        idx = torch.randperm(h * w)[:2]

        assert idx.dim() == 1 and idx.size(0) == 2

        def coords(idx: int) -> tuple[int, int]:
            row = idx // w
            col = idx % w
            return (col, row)

        p1 = torch.Tensor(coords(idx[0]))
        p2 = torch.Tensor(coords(idx[1]))

        assert p1.dim() == 1 and p1.size(0) == 2

        vec1_vals = data[:, :, p1[1], p1[0]]
        vec2_vals = data[:, :, p2[1], p2[0]]

        assert (
            vec1_vals.dim() == 2
            and vec1_vals.size(0) == data.size(0)
            and vec1_vals.size(1) == 2
        )

        # Vectorized intersect line

        A = torch.stack([vec1_vals, -vec2_vals], dim=0).T

        p1 = p1[None, :].repeat(data.size(0), 0)
        p2 = p2[None, :].repeat(data.size(0), 0)
        b = p2 - p1

        assert b.dim == 2 and b.size(0) == data.size(0) and b.size(1) == 2

        t = torch.linalg.solve(A, b)

        assert t.size(0) == data.size(0)

        return p1 + t[:, 0] * vec1_vals

    def score(self, hypothesis: torch.Tensor):
        assert hypothesis.dim() == 2
        assert hypothesis.size(1) == 2  # B, 2

        h, w = self.data.size()[2:]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        coords = torch.stack([x, y], dim=0).float()  # 2, H, W

        hypothesis = hypothesis[:, :, None, None]  # B, 2, 1, 1
        coords = coords[None, :, :, :].repeat(hypothesis.size(0), 1, 1, 1)  # B, 2, H, W

        hypo_vec = hypothesis - coords  # B, 2, H, W
        hypo_vec = hypo_vec / torch.norm(hypo_vec, dim=1, keepdim=True)  # B, 2, H, W

        hypo_vec = hypo_vec.reshape(hypo_vec.size(0), 2, -1)
        data_vec = self.reshaped_data.reshape(self.reshaped_data.size(0), 2, -1)

        assert hypo_vec.dim() == 3 and hypo_vec.size(1) == 2
        assert data_vec.dim() == 3 and data_vec.size(1) == 2

        # dot_product = torch.sum(hypo_vec * data_vec, dim=1)  # B, H*W
        scores = hypo_vec @ data_vec  # B, H*W

        THRESHOLD = 0.9
        inliers = scores > THRESHOLD

        consensus = inliers.sum(dim=1)  # B

        return consensus, hypothesis.squeeze()  # B, 2

    def ransac(self):

        hypo_points = self.batched_hypothesis(self.reshaped_data)

        iter_hypo_points = []
        iter_consensus = []
        for _ in range(self.iter):
            consensus, hypo_points = self.score(hypo_points)
            iter_hypo_points.append(hypo_points)
            iter_consensus.append(consensus)

        iter_hypo_points = torch.stack(iter_hypo_points, dim=0)  # I, B, 2
        iter_consensus = torch.stack(iter_consensus, dim=0)  # I, B

        # get max of each batch for iter
        best_consensus, best_idx = iter_consensus.max(dim=0)  # B
        best_hypo_points = iter_hypo_points[best_idx, :, :]  # B, 2
