import time
import torch
import yaml
import argparse
from pathlib import Path

from pixelvote6d.models import PVNet
from pixelvote6d.training import build_dataset, build_sampler
from pixelvote6d.training import init_logger, log_starting_info, create_output_dir


def create_output_dir(output_dir: str, run_name: str) -> Path:
    """Create a unique directory for this training run and return its path."""
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PVNet model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file with training parameters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/dataset",
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )

    args = parser.parse_args()

    # --- Load and parse config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = create_output_dir(Path(args.output), Path(args.config))
    report_path = output_dir / "report.txt"

    logger = init_logger(report_path)

    # --- Dataset

    concat_dataset, datasets, weights = build_dataset(
        config["training"], Path(args.dataset_root)
    )

    # --- Weighted sampling

    # Weights Sample only if weights are not all default (1.0)
    if any(w != 1.0 for w in weights):
        sampler = build_sampler(weights, datasets)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=shuffle,
        sampler=sampler,
    )

    # --- Model ---
    pvnet = PVNet()

    total_epochs = config["epochs"]

    if args.load:
        checkpoint = torch.load(args.load, weights_only=True)
        pvnet.load_state_dict(checkpoint["model_state_dict"])
        resumed_epochs = checkpoint.get("epoch", 0)
        total_epochs = resumed_epochs + args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pvnet = pvnet.to(device)

    # --- Loss & optimizer ---
    bce_loss = torch.nn.BCEWithLogitsLoss()
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="none")

    optimizer = torch.optim.Adam(pvnet.parameters(), lr=args.lr)

    if args.load and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Cosine annealing: smoothly decays lr from args.lr down to eta_min over the run.
    # eta_min=1e-6 avoids fully freezing the model in the final epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    if args.load and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    log_starting_info(logger, config, len(concat_dataset), resuming=(args.load is not None))

    # --- Training loop ---
    start_time = time.time()

    for epoch in range(args.epochs):
        mask_total_loss = 0.0
        vfield_total_loss = 0.0

        for image, mask, vfield, keypoints_2d in dataloader:
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
            vfield_loss = (loss_map * mask_binary).sum() / (
                valid_pixels * num_channels + 1e-6
            )

            total_loss = mask_loss + vfield_loss * 2.0
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            mask_total_loss += mask_loss.item()
            vfield_total_loss += vfield_loss.item()

        avg_mask_loss = mask_total_loss / len(dataloader)
        avg_vfield_loss = vfield_total_loss / len(dataloader)

        scheduler.step()

        logger.info(
            "Epoch %d/%d (total %d) | mask_loss=%.6f | vfield_loss=%.6f | lr=%.2e",
            epoch + 1,
            config["epochs"],
            total_epochs,
            avg_mask_loss,
            avg_vfield_loss,
            scheduler.get_last_lr()[0],
        )

    logger.info("Training completed in %.2f seconds", time.time() - start_time)

    # --- Save checkpoint ---
    checkpoint_path = output_dir / "checkpoint.pth"
    torch.save(
        {
            "epoch": total_epochs,
            "model_state_dict": pvnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "mask_loss": mask_total_loss / len(dataloader),
            "vfield_loss": vfield_total_loss / len(dataloader),
        },
        checkpoint_path,
    )

    logger.info("Checkpoint saved : %s", checkpoint_path)
    logger.info("Report saved     : %s", report_path)
