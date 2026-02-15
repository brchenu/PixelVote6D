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

OBJ_ID = 15
BASE_DIR = Path(__file__).resolve().parent
dataset_path = os.path.join(BASE_DIR, "dataset", "ycbv")

# load 3d keypoints
keypoints_3d = np.loadtxt(os.path.join(dataset_path, "models", f"obj_{str(OBJ_ID).zfill(6)}_keypoints.txt"))
print(f"3D keypoints shape: {keypoints_3d.shape}")

pvnet = PVNet()
pvnet.load_state_dict(
    torch.load(
        "checkpoints/pvnet_2026-02-15_10-08-26_epoch2_obj15_ycbv.pth", weights_only=True
    )["model_state_dict"]
)
pvnet.to(device)
pvnet.eval()


dataset = BOPDirectDataset(
    dataset_dir=dataset_path,
    obj_id=OBJ_ID,
    transform=PVNetTransform(),
    subset=BOPSubSet.REAL,
)

datasetloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False
)

for image, mask, vfield, keypoints_2d in datasetloader:

    image = image.to(device)

    pred_mask, pred_vfield = pvnet(image)

    print(f"pred_mask shape: {pred_mask.shape}")
    print(f"pred_vfield shape: {pred_vfield.shape}")
    
    pred_mask = pred_mask.squeeze()
    pred_vfield = pred_vfield.squeeze(0)

    print(f"device pred_mask: {pred_mask.device}")
    print(f"device pred_vfield: {pred_vfield.device}") 

    # pred_mask = pred_mask.cpu()
    # pred_vfield = pred_vfield.cpu() 

    pvnet_ransac = PVNetRansac(mask=pred_mask, vfield=pred_vfield, num_iter=500)
    final_keypoints = pvnet_ransac.ransac()

    img = PVNetTransform.unnormalize_image(image.squeeze())
    img = img.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # draw keypoints on image
    for kp in final_keypoints:
        x, y = int(kp[0].item()), int(kp[1].item())
        cv2.circle(img_bgr, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("image", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()