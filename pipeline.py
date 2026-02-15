import cv2
import os
import torch
import numpy as np
from model import PVNet
from pathlib import Path
from bop_toolkit.bop_dataset import BOPDirectDataset, BOPSubSet
from bop_toolkit.data_transfrom import PVNetTransform
from ransac import PVNetRansac

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
    dataset_dir=dataset_path,
    obj_id=1,
    transform=PVNetTransform(),
    subset=BOPSubSet.REAL,
)

image, mask, vfield, keypoints_2d = dataset[0]

# img = PVNetTransform.unnormalize_image(image)
# img = img.permute(1, 2, 0).cpu().numpy()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()