"""Shared utilities: seeding, device selection, logging, config loading, run tracking."""
from __future__ import annotations
import csv, json, logging, os, random, subprocess, time
from pathlib import Path
from typing import Any
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]  # repository root


def get_logger(name: str) -> logging.Logger:
    """Console logger. Never log raw customer data, only aggregates and ids."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(os.environ.get("PRESAGE_LOG", "INFO"))
    return log


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda', 'mps' or 'cpu' (never assume an NVIDIA GPU)."""
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


REGISTRY = ROOT / "research" / "experiment_registry.csv"
_FIELDS = ["timestamp", "experiment", "model", "dataset", "seed", "n_features", "metrics_json", "params_json", "train_seconds", "infer_seconds", "commit"]


def log_experiment(experiment: str, model: str, dataset: str, seed: int, metrics: dict, params: dict | None = None,
                   n_features: int = 0, train_seconds: float = 0.0, infer_seconds: float = 0.0) -> None:
    """Append to experiment_registry.csv and (best effort) to a local MLflow file store."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    row = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), experiment=experiment, model=model, dataset=dataset, seed=seed,
               n_features=n_features, metrics_json=json.dumps(metrics, default=float), params_json=json.dumps(params or {}, default=str),
               train_seconds=round(train_seconds, 3), infer_seconds=round(infer_seconds, 4), commit=git_hash())
    with open(REGISTRY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new: w.writeheader()
        w.writerow(row)
    if os.environ.get("PRESAGE_MLFLOW", "1") == "1":
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=model):
                mlflow.log_params({"dataset": dataset, "seed": seed, "commit": row["commit"], **{k: str(v)[:200] for k, v in (params or {}).items()}})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
                mlflow.log_metrics({"train_seconds": train_seconds, "infer_seconds": infer_seconds})
        except Exception as e:  # tracking must never break an experiment
            get_logger("tracking").warning("mlflow logging skipped: %s", e)
