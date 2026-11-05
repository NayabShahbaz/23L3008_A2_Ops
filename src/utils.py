"""Shared helpers: config loading, seeding, MLflow/DagsHub setup, git/DVC tags."""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    """Task 2.4: fix every source of randomness we control."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)  # seeds python, numpy and TF together
        # Full op determinism (tf.config.experimental.enable_op_determinism) is
        # stricter but breaks some GPU augmentation ops; mention this in the report.
    except ImportError:
        pass


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def dvc_data_version() -> str:
    """md5 of the tracked dataset directory, read from data/raw.dvc."""
    dvc_file = ROOT / "data" / "raw.dvc"
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return meta["outs"][0].get("md5", "unknown")


def setup_mlflow(experiment: str | None = None):
    """Task 3.2: tracking URI and credentials come ONLY from env vars / .env."""
    import mlflow
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "23L3008_A2_Ops"))
    return mlflow
