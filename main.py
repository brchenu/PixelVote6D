import os
import cv2
import torch
import torchvision
import numpy as np
from model import PVNet
from dataset import BOPDataset
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

BASE_DIR = Path(__file__).resolve().parent

transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

dataset_path = os.path.join(BASE_DIR, "dataset", "clean", "ycbv", "scene_000010")
dataset = BOPDataset(dataset_path=dataset_path)

datasetloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=8, shuffle=True
)

bce_loss = torch.nn.BCEWithLogitsLoss()
smooth_l1_loss = torch.nn.SmoothL1Loss()

pvnet = PVNet()

# Move model to GPU
pvnet = pvnet.to(device)

optimizer = torch.optim.Adam(pvnet.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    mask_total_loss = 0.0
    vfield_total_loss = 0.0
    for image, mask, vfield in datasetloader:

        image = image.to(device)
        mask = mask.to(device)
        vfield = vfield.to(device)

        pred_mask, pred_vfield = pvnet(image)

        mask_loss = bce_loss(pred_mask, mask)
        vfield_loss = (
            smooth_l1_loss(pred_vfield, vfield) * 10
        )  # Scale vfield loss to balance with mask loss

        mask_total_loss += mask_loss.item()
        vfield_total_loss += vfield_loss.item()

        optimizer.zero_grad()
        total_loss = mask_loss + vfield_loss
        total_loss.backward()
        optimizer.step()

    print(
        f"Average Mask Loss: {mask_total_loss / len(datasetloader):.4f} | Average VField Loss: {vfield_total_loss / len(datasetloader):.4f}"
    )

test_dataset_path = os.path.join(
    BASE_DIR, "dataset", "clean", "ycbv", "test", "scene_000000"
)
test_datasetloader = torch.utils.data.DataLoader(
    BOPDataset(dataset_path=test_dataset_path), batch_size=1, shuffle=False
)

with torch.no_grad():
    for image, mask, vfield in test_datasetloader:
        image = image.to(device)
        pred_mask, pred_vfield = pvnet(image)

        # image: (1,3,H,W) in range [0,1] from torchvision transforms
        img = image[0].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        prob_map = torch.sigmoid(pred_mask[0])
        binary_mask = (prob_map > 0.5).float()

        mask_np = binary_mask.squeeze().cpu().numpy().astype(np.uint8)  # 0/1
        mask_3c = np.repeat(mask_np[:, :, None], 3, axis=2)

        masked_img = img_bgr * mask_3c  # zero out background

        # after masked_img is created
        orig_w, orig_h = 640, 480
        resized = cv2.resize(
            masked_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Masked Image", resized)
        cv2.waitKey(0)

cv2.destroyAllWindows()
