from pathlib import Path
from pixelvote6d.dataset import BOPDirectDataset, BOPSubSet
from pixelvote6d.dataset import PVNetRandomTransform
from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler


def build_dataset(
    config: dict, dataset_root: Path
) -> tuple[ConcatDataset, list[BOPDirectDataset], list[float]]:

    datasets = []
    weights = []

    for entry in config["datasets"]:

        path = dataset_root / entry["path"]

        ds = BOPDirectDataset(
            dataset_dir=str(path),
            obj_id=config["training"]["obj_id"],
            transform=PVNetRandomTransform(spatial=True),
            subset=BOPSubSet.TRAIN,
        )

        datasets.append(ds)
        weights.append(entry.get("weight", 1.0))

    if not datasets:
        raise ValueError("No datasets specified in config")

    return ConcatDataset(datasets), datasets, weights

def build_sampler(weights: list[float], datasets: list[BOPDirectDataset]) -> WeightedRandomSampler:
    weights = [w / len(weights) for w in weights]  # Normalize weights

    sample_weights = []
    for ds, w in zip(datasets, weights):
        sample_weights.extend([w] * len(ds))

    # weights are extended because WeightedRandomSampler expects a weight for each sample in the concatenated dataset
    return WeightedRandomSampler(sample_weights, num_samples=sum(len(ds) for ds in datasets))