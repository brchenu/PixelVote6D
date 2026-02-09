import os
import torch
import torchvision
from model import PVNet
from dataset import BOPDataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

dataset_path = os.path.join(BASE_DIR, "dataset", "clean", "scene_000010")
dataset = BOPDataset(dataset_path=dataset_path)

datasetloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

bce_loss = torch.nn.BCEWithLogitsLoss()
smooth_l1_loss = torch.nn.SmoothL1Loss()

pvnet = PVNet()

optimizer = torch.optim.Adam(pvnet.parameters(), lr=1e-3)

EPOCHS = 3

for epoch in range(EPOCHS):
    mask_total_loss = 0.0
    vfield_total_loss = 0.0
    for image, mask, vfield in datasetloader:
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

    print(f"Average Mask Loss: {mask_total_loss / len(datasetloader):.4f} | Average VField Loss: {vfield_total_loss / len(datasetloader):.4f}")

# test_dataset_path = os.path.join(BASE_DIR, "dataset", "clean", "test", "scene_000011")
# test_datasetloader = torch.utils.data.DataLoader(BOPDataset(dataset_path=test_dataset_path), batch_size=1, shuffle=False)

# with torch.no_grad():
#     for image, mask, vfield in test_datasetloader:
#         pred_mask, pred_vfield = pvnet(image)