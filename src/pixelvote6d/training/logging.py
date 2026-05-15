import logging
from pathlib import Path


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


def log_starting_info(
    logger: logging.Logger, config: dict, dataset_size: int, resuming: bool
):
    logger.info("=" * 50)
    logger.info("Device           : %s", config["device"])
    # logger.info("Dataset(s)       : %s  (obj_id=%d)", dataset_tag, args.obj_id)
    # all_names = list(args.dataset) + (
    #     [Path(d).name for d in args.self_label] if args.self_label else []
    # )
    # if args.weights:
    #     for name, ds, w in zip(all_names, datasets, args.weights):
    #         logger.info("  %-17s: %d samples, weight=%.2f", name, len(ds), w)
    logger.info("Dataset size     : %d samples", dataset_size)
    logger.info("Batch size       : %d", config["batch_size"])
    logger.info("Learning rate    : %g", config["lr"])
    logger.info("Epochs this run  : %d", config["epochs"])
    if resuming:
        logger.info("Resumed from     : %s", config["load"])
    else:
        logger.info(
            "Training from scratch  →  total after run: %d", config["num_epochs"]
        )
    logger.info("=" * 50)
