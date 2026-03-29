import os
import time
import logging
import torch
import argparse
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from bop_toolkit.bop_dataset import BOPDirectDataset, BOPSubSet
from model import PVNet
from pathlib import Path
from bop_toolkit.data_transfrom import PVNetRandomTranform

BASE_DIR = Path(__file__).resolve().parent


def init_logger(report_path: Path) -> logging.Logger:
    """Initialize a logger that writes to both console and a file."""
    logger = logging.getLogger("pvnet_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(report_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def create_run_dir(output_dir: str, run_name: str) -> Path:
    """Create a unique directory for this training run and return its path."""
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PVNet model")
    parser.add_argument(
        "--obj-id", type=int, required=True, help="Object ID to train on"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+", required=True,
        help="One or more dataset names (e.g., --dataset drill drill_hd)"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Sampling weight for each dataset (must match number of --dataset entries). "
        "E.g. --weights 0.45 0.45 0.10",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for outputs. A timestamped sub-folder is created for each run. "
        "Defaults to ./output if not specified.",
    )

    args = parser.parse_args()

    # --- Output directory setup ---
    output_base = Path(args.output_dir) if args.output_dir else Path.cwd() / "output"
    dataset_tag = "+".join(args.dataset)
    run_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_obj{args.obj_id}_{dataset_tag}"
    run_dir = create_run_dir(str(output_base), run_name)

    report_path = run_dir / "report.txt"
    checkpoint_path = run_dir / "checkpoint.pth"

    logger = init_logger(report_path)

    for name in args.dataset:
        print(f"name: {name}")

    # --- Dataset & dataloader ---
    transform = PVNetRandomTranform()
    datasets = [
        BOPDirectDataset(
            dataset_dir=os.path.join(BASE_DIR, "dataset", name),
            obj_id=args.obj_id,
            transform=transform,
            subset=BOPSubSet.TRAIN,
        )
        for name in args.dataset
    ]
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # --- Weighted sampling ---
    sampler = None
    shuffle = True
    if args.weights is not None:
        assert len(args.weights) == len(datasets), (
            f"Number of weights ({len(args.weights)}) must match "
            f"number of datasets ({len(datasets)})"
        )
        sample_weights = []
        for ds, w in zip(datasets, args.weights):
            sample_weights.extend([w / len(ds)] * len(ds))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))
        shuffle = False  # mutually exclusive with sampler

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=10,
        shuffle=shuffle, sampler=sampler,
    )

    # --- Model ---
    pvnet = PVNet()
    prev_epochs = 0

    if args.load:
        checkpoint = torch.load(args.load, weights_only=True)
        pvnet.load_state_dict(checkpoint["model_state_dict"])
        prev_epochs = checkpoint.get("epoch", 0) + 1

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

    # --- Log run configuration ---
    logger.info("=" * 60)
    logger.info("Run directory    : %s", run_dir)
    logger.info("Device           : %s", device)
    logger.info("Dataset(s)       : %s  (obj_id=%d)", dataset_tag, args.obj_id)
    if args.weights:
        for name, ds, w in zip(args.dataset, datasets, args.weights):
            logger.info("  %-16s: %d samples, weight=%.2f", name, len(ds), w)
    logger.info("Dataset size     : %d samples", len(dataset))
    logger.info("Batch size       : %d", args.batch_size)
    logger.info("Learning rate    : %g", args.lr)
    logger.info("Epochs this run  : %d", args.epochs)
    if args.load:
        logger.info("Resumed from     : %s", args.load)
        logger.info(
            "Epochs in checkpoint : %d  →  total after run: %d",
            prev_epochs,
            prev_epochs + args.epochs,
        )
    else:
        logger.info("Training from scratch  →  total after run: %d", args.epochs)
    logger.info("=" * 60)

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
            args.epochs,
            prev_epochs + epoch + 1,
            avg_mask_loss,
            avg_vfield_loss,
            scheduler.get_last_lr()[0],
        )

    logger.info("Training completed in %.2f seconds", time.time() - start_time)

    # --- Save checkpoint ---
    torch.save(
        {
            "epoch": prev_epochs + args.epochs - 1,
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
