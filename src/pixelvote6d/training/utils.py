from pathlib import Path
from datetime import datetime
import shutil


def create_output_dir(output_dir: Path, config_path: Path) -> Path:
    """
    Create a timestamped run directory and copy the config file into it.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config.yaml")

    return run_dir