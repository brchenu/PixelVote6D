import os
import time
import logging
import torch
from bop_toolkit.bop_dataset import BOPDirectDataset, BOPSubSet
from model import PVNet
from pathlib import Path
from bop_toolkit.data_transfrom import PVNetTransform

BASE_DIR = Path(__file__).resolve().parent

OBJ_ID = 15
DATASET_NAME = "ycbv"

EPOCHS = 20

current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
filename_suffix = f"{current_date}_epoch{EPOCHS - 1}_obj{OBJ_ID}_{DATASET_NAME}"
report_path = os.path.join(
    BASE_DIR, "checkpoints", f"train_report_{filename_suffix}.txt"
)

# Initalize logging
logger = logging.getLogger("pvnet_train")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = logging.FileHandler(report_path, mode="w", encoding="utf-8")
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Initialize network, optimizer, and loss functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = BOPDirectDataset(
    dataset_dir=os.path.join(BASE_DIR, "dataset", DATASET_NAME),
    obj_id=OBJ_ID,
    transform=PVNetTransform(),
    subset=BOPSubSet.TRAIN,
)

datasetloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=10, shuffle=True
)

# Loss functions
bce_loss = torch.nn.BCEWithLogitsLoss()
smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="none")

checkpoint = "checkpoints/pvnet_2026-02-14_20-19-55_epoch19_obj15_ycbv.pth" 

pvnet = PVNet()
# pvnet.load_state_dict(
#     torch.load(checkpoint, weights_only=True)[
#         "model_state_dict"
#     ]
# )
pvnet = pvnet.to(device)  # Try to move model to GPU

optimizer = torch.optim.Adam(pvnet.parameters(), lr=1e-4)

logger.info("Using device: %s", device)
logger.info("Dataset size: %d samples", len(dataset))
logger.info("Object ID: %d | Dataset: %s | Epochs: %d", OBJ_ID, DATASET_NAME, EPOCHS)

start_time = time.time()

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

        loss_map = smooth_l1_loss(pred_vfield, vfield)  # [B, 2K, H, W]

        mask_binary = (mask > 0.5).float()  # [B, 1, H, W]
        valid_pixels = mask_binary.sum()

        num_channels = loss_map.shape[1]
        mask_binary = mask_binary.repeat(1, num_channels, 1, 1)

        vfield_loss = (loss_map * mask_binary).sum()
        vfield_loss = vfield_loss / (valid_pixels + 1e-6)
        vfield_loss = vfield_loss / num_channels

        mask_total_loss += mask_loss.item()
        vfield_total_loss += vfield_loss.item()

        optimizer.zero_grad()
        total_loss = mask_loss + vfield_loss * 2.0
        total_loss.backward()
        optimizer.step()

    avg_mask_loss = mask_total_loss / len(datasetloader)
    avg_vfield_loss = vfield_total_loss / len(datasetloader)

    logger.info(
        "Epoch %d/%d | mask_loss=%.6f | vfield_loss=%.6f",
        epoch + 1,
        EPOCHS,
        avg_mask_loss,
        avg_vfield_loss,
    )

end_time = time.time()
elapsed_time = end_time - start_time
logger.info("Training completed in %.2f seconds", elapsed_time)

current_mask_loss = mask_total_loss / len(datasetloader)
current_vfield_loss = vfield_total_loss / len(datasetloader)

# ==== Save Checkpoint ====

checkpoint_path = os.path.join(BASE_DIR, "checkpoints", f"pvnet_{filename_suffix}.pth")
checkpoints = {
    "epoch": epoch,
    "model_state_dict": pvnet.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "mask_loss": current_mask_loss,
    "vfield_loss": current_vfield_loss,
}
torch.save(checkpoints, checkpoint_path)

logger.info("Checkpoint saved to %s", checkpoint_path)