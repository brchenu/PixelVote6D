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
pvnet.load_state_dict(torch.load("checkpoints/pvnet_checkpoint.pth", weights_only=True)["model_state_dict"])
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

class PVNetRansac():
    def __init__(self, data, iter:int):
        self.data = data
        self.iter = iter
    
    @staticmethod
    def vectorized_hypothesis(data: torch.Tensor):
        assert data.dim == 3

        h, w = data.size[1:]
        idx = torch.randperm(h*w)[:2]

        assert idx.dim == 1 and idx.size(0) == 2

        def coords(idx: int) -> tuple[int, int]:
            row = idx // w 
            col = idx % w
            return (col, row)

        p1 = torch.Tensor(coords(idx[0]))
        p2 = torch.Tensor(coords(idx[1]))

        assert p1.dim == 1 and p1.size(0) == 2
        assert p2.dim == 1 and p2.size(0) == 2

        vec1_vals = data[:, p1[1], p1[0]]
        vec2_vals = data[:, p2[1], p2[0]]

        # Vectorized intersect line
        
        A = torch.stack([vec1_vals, -vec2_vals], dim=1).T

        b = p2 - p1
        b = b.unsqueeze(0)

        assert b.dim == 2 and b.size(1) == 2

        b = b.repeat(8, 1) 

        t = torch.linalg.solve(A, b)

        return p1 + t[:, 0] * vec1_vals 
    
    def score(self, hypothesis: tuple[int, int]):
        # TODO
        pass 

    def ransac(self):
        for c in range(self.data.size(0)):
            point_h = self.vectorized_hypothesis(self.data)
            # score()