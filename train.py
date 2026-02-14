import os
import time
import torch
import torchvision
from bop_toolkit.bop_dataset import BOPDirectDataset
from model import PVNet
from pathlib import Path
from bop_toolkit.data_transfrom import PVNetTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

BASE_DIR = Path(__file__).resolve().parent

transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

OBJ_ID = 1
DATASET_NAME = "ycbv"

dataset_path = os.path.join(BASE_DIR, "dataset", DATASET_NAME)

# dataset = BOPDataset(dataset_path=dataset_path)
dataset = BOPDirectDataset(
    dataset_dir=dataset_path, obj_id=OBJ_ID, transform=PVNetTransform()
)

print(f"Dataset size: {len(dataset)} samples")

datasetloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=8, shuffle=True
)

bce_loss = torch.nn.BCEWithLogitsLoss()
smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="none")  # We'll apply mask manually, so no reduction here

pvnet = PVNet()
pvnet = pvnet.to(device)  # Move model to GPU if available

optimizer = torch.optim.Adam(pvnet.parameters(), lr=1e-3)

EPOCHS = 3

for epoch in range(EPOCHS):
    mask_total_loss = 0.0
    vfield_total_loss = 0.0
    for image, mask, vfield, keypoints_2d in datasetloader:

        # Move data to GPU if available
        image = image.to(device)
        mask = mask.to(device)
        vfield = vfield.to(device)

        pred_mask, pred_vfield = pvnet(image)

        mask_loss = bce_loss(pred_mask, mask)

        loss_map = smooth_l1_loss(pred_vfield, vfield)   # [B, 2K, H, W]

        mask_binary = (mask > 0.5).float()               # [B, 1, H, W]
        valid_pixels = mask_binary.sum()

        num_channels = loss_map.shape[1]
        mask_binary = mask_binary.repeat(1, num_channels, 1, 1)

        vfield_loss = (loss_map * mask_binary).sum()
        vfield_loss = vfield_loss / (valid_pixels + 1e-6)
        vfield_loss = vfield_loss / num_channels

        mask_total_loss += mask_loss.item()
        vfield_total_loss += vfield_loss.item()

        optimizer.zero_grad()
        total_loss = mask_loss + vfield_loss
        total_loss.backward()
        optimizer.step()

    print(
        f"Average Mask Loss: {mask_total_loss / len(datasetloader):.4f} | Average VField Loss: {vfield_total_loss / len(datasetloader):.4f}"
    )

current_mask_loss = mask_total_loss / len(datasetloader)
current_vfield_loss = vfield_total_loss / len(datasetloader)

# ==== Save Checkpoint ====

current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_path = os.path.join(
    BASE_DIR, "checkpoints", f"pvnet_checkpoint_obj{OBJ_ID}_{DATASET_NAME}_{current_date}.pth"
)
checkpoints = {
    "epoch": epoch,
    "model_state_dict": pvnet.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "mask_loss": current_mask_loss,
    "vfield_loss": current_vfield_loss,
}
torch.save(checkpoints, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")
