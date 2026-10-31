import argparse
import importlib
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.config import load_config

CONFIG_DIR = Path("configs")


def available_tasks() -> list[str]:
    """Every configs/<task>.yaml is a task."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def run_task(name: str) -> None:
    cfg_path = CONFIG_DIR / f"{name}.yaml"
    cfg = load_config(cfg_path)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_dir / "config_used.yaml")

    module = importlib.import_module(f"src.tasks.{name}.run")
    module.run(cfg, out_dir)
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    tasks = available_tasks()
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=tasks + ["all"])
    args = p.parse_args()

    for t in (tasks if args.task == "all" else [args.task]):
        run_task(t)