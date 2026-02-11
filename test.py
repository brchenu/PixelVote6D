import os
import torch
import cv2
import numpy as np
from model import PVNet
from pathlib import Path
from bop_toolkit.data_transfrom import PVNetTransform
from bop_toolkit.bop_dataset import BOPDirectDataset
from bop_toolkit.vizualization import show_vector_field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

pvnet = PVNet()
pvnet.load_state_dict(
    torch.load("pvnet_checkpoint.pth", weights_only=True)["model_state_dict"]
)

dataset_path = os.path.join(BASE_DIR, "dataset", "ycbv")

dataset = BOPDirectDataset(
    dataset_dir=dataset_path, obj_id=1, testing_mode=True, transform=PVNetTransform()
)

datasetloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)
pvnet.to(device)
pvnet.eval()

for image, mask, vfield, keypoints_2d in datasetloader:
    image = image.to(device)
    pred_mask, pred_vfield = pvnet(image)

    show_vector_field(
        img=image[0].cpu().numpy().transpose(1, 2, 0),
        mask=pred_mask[0].squeeze(0).detach().cpu().numpy(),
        vector_field=pred_vfield[0].detach().cpu().numpy(),
        keypoints=keypoints_2d[0].numpy(),
        step=20,
        scale_mode="full",
        mask_img=True,
        resize_factor=2,
    )

    # img = image[0].cpu().numpy().transpose(1, 2, 0)
    # img = (img * 255).clip(0, 255).astype(np.uint8)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # prob_map = torch.sigmoid(pred_mask[0])
    # binary_mask = (prob_map > 0.5).float()

    # mask_np = binary_mask.squeeze().cpu().numpy().astype(np.uint8)  # 0/1
    # mask_3c = np.repeat(mask_np[:, :, None], 3, axis=2)

    # masked_img = img_bgr * mask_3c  # zero out background

    # # after masked_img is created
    # orig_w, orig_h = 640, 480
    # resized = cv2.resize(masked_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Masked Image", resized)
    # cv2.waitKey(0)

cv2.destroyAllWindows()