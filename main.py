import argparse
import importlib
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.common.config import load_config

PACKAGES = ["tasks", "models"]  # where run code can live


def resolve_config(arg: str) -> Path:
    """Accept configs/chronos.yaml, or just 'chronos'."""
    p = Path(arg)
    if p.exists():
        return p
    shortcut = Path("configs") / f"{arg}.yaml"
    if shortcut.exists():
        return shortcut
    raise FileNotFoundError(f"Config not found: {arg}")


def find_module(name: str) -> str:
    """Find src/tasks/<name>/run.py or src/models/<name>/run.py."""
    for pkg in PACKAGES:
        if Path(f"src/{pkg}/{name}/run.py").exists():
            return f"src.{pkg}.{name}.run"
    raise FileNotFoundError(f"No run.py found for '{name}' in src/tasks/ or src/models/")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_config(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)

    # code to run: optional `module:` key in the yaml, otherwise the yaml file name
    name = cfg.get("module", cfg_path.stem)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / cfg_path.stem / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": str(cfg_path),
        "module": name,
        "timestamp": stamp,
        "description": cfg.get("experiment", {}).get("description"),
        "dataset_id": cfg.get("data", {}).get("id"),
        "dataset_version": cfg.get("data", {}).get("version"),
        "seed": cfg["seed"],
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"Running {cfg_path}")
    module = importlib.import_module(find_module(name))
    module.run(cfg, out_dir)
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one or more config files.")
    p.add_argument("configs", nargs="+", help="path(s) to yaml, e.g. configs/chronos.yaml")
    args = p.parse_args()

    for c in args.configs:
        run_config(resolve_config(c))