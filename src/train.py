"""Tasks 2.2-2.5, 3.3-3.5 - train a model and track everything in MLflow (DagsHub).

Examples
  python -m src.train --model cnn --tracking manual --run-name cnn_baseline
  python -m src.train --model transfer --tracking manual --run-name mnv2_ft
  python -m src.train --model transfer --tracking autolog --run-name mnv2_autolog
  # Task 3.5 one-change-at-a-time runs:
  python -m src.train --model transfer --lr 0.0005           --run-name mnv2_lr5e-4
  python -m src.train --model transfer --no-augment          --run-name mnv2_noaug
  python -m src.train --model transfer --unfreeze 40         --run-name mnv2_unfreeze40
  python -m src.train --model transfer --img-size 160        --run-name mnv2_160
  python -m src.train --model transfer --backbone EfficientNetB0 --run-name effb0

Outputs: models/<run>.keras, reports/figures/<run>_*.png, MLflow run with
params / per-epoch metrics / test metrics / tags / artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.data_pipeline import make_dataset  # noqa: E402
from src.evaluate import full_evaluation  # noqa: E402
from src.models import (  # noqa: E402
    build_baseline_cnn,
    build_transfer,
    compile_model,
    trainable_layers,
    unfreeze_last,
)
from src.utils import ROOT, dvc_data_version, git_commit, load_params, set_seeds, setup_mlflow  # noqa: E402

MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual tracking: log every metric Keras reports at the end of each epoch."""

    def __init__(self, mlflow, prefix: str = "", offset: int = 0):
        super().__init__()
        self.mlflow, self.prefix, self.offset = mlflow, prefix, offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.mlflow.log_metrics({f"{self.prefix}{k}": float(v) for k, v in logs.items()},
                                step=epoch + self.offset)


def callbacks(ckpt: Path, p: dict) -> list:
    t = p["train"]
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=t["reduce_lr_factor"],
                                          patience=t["reduce_lr_patience"], verbose=1),
    ]


def plot_curves(history: dict, out: Path, title: str, stage_split: int | None = None) -> tuple[Path, int]:
    best = int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history[key], label=f"train {key}")
        ax.plot(history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best, color="red", ls="--", label=f"best epoch = {best + 1}")
        if stage_split:
            ax.axvline(stage_split - 0.5, color="grey", ls=":", label="start fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title}: {key}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, best + 1


def merge(h1: dict, h2: dict) -> dict:
    return {k: h1.get(k, []) + h2.get(k, []) for k in set(h1) | set(h2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--unfreeze", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--strategy", default="max_fpr", choices=["max_fpr", "f1"])
    a = ap.parse_args()

    p = load_params()
    set_seeds(p["seed"])
    size = a.img_size or p["image"]["size"]
    augment = p["train"]["augment"] and not a.no_augment
    tr = p["transfer"]
    backbone = a.backbone or tr["backbone"]
    run_name = a.run_name or (f"cnn_{a.tracking}" if a.model == "cnn" else f"{backbone}_{a.tracking}")
    MODELS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS / f"{run_name}.keras"

    train_ds = make_dataset("train", p, training=True, augment=augment, img_size=size)
    val_ds = make_dataset("val", p, training=False, img_size=size)

    mlflow = setup_mlflow()
    if a.tracking == "autolog":
        mlflow.tensorflow.autolog(log_models=True, checkpoint=False)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
            "seed": p["seed"], "model_type": a.model, "tracking": a.tracking,
            "backbone": backbone if a.model == "transfer" else "none",
        })
        base_params = {"img_size": size, "batch_size": p["train"]["batch_size"], "augment": augment,
                       "seed": p["seed"], "threshold_strategy": a.strategy}
        if a.tracking == "manual":
            mlflow.log_params(base_params)

        if a.model == "cnn":
            lr = a.lr or p["train"]["learning_rate"]
            epochs = a.epochs or p["train"]["epochs"]
            model = compile_model(build_baseline_cnn(size, p["baseline_cnn"]["filters"],
                                                     p["baseline_cnn"]["dropout"]), lr)
            if a.tracking == "manual":
                mlflow.log_params({"lr": lr, "epochs": epochs, "filters": p["baseline_cnn"]["filters"],
                                   "dropout": p["baseline_cnn"]["dropout"]})
            cbs = callbacks(ckpt, p) + ([MLflowEpochLogger(mlflow)] if a.tracking == "manual" else [])
            hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs).history
            curves, best_epoch = plot_curves(hist, FIGS / f"{run_name}_curves.png", run_name)
            stage_info = {"trainable_layers": trainable_layers(model)}
        else:
            lr1 = a.lr or tr["stage1_lr"]
            lr2 = lr1 * (tr["stage2_lr"] / tr["stage1_lr"])
            n_unfreeze = a.unfreeze if a.unfreeze is not None else tr["unfreeze_last_n"]
            ep1 = a.epochs or tr["stage1_epochs"]          # --epochs N -> N epochs per stage
            ep2 = a.epochs or tr["stage2_epochs"]
            model, base = build_transfer(size, backbone, tr["head_dropout"])
            if a.tracking == "manual":
                mlflow.log_params({"backbone": backbone, "stage1_lr": lr1, "stage2_lr": lr2,
                                   "stage1_epochs": ep1, "stage2_epochs": ep2,
                                   "unfreeze_last_n": n_unfreeze, "head_dropout": tr["head_dropout"]})

            # Stage 1: frozen backbone, train the head only
            compile_model(model, lr1)
            stage1_trainable = trainable_layers(model)
            cbs = callbacks(ckpt, p) + ([MLflowEpochLogger(mlflow)] if a.tracking == "manual" else [])
            h1 = model.fit(train_ds, validation_data=val_ds, epochs=ep1, callbacks=cbs).history
            plot_curves(h1, FIGS / f"{run_name}_stage1_curves.png", f"{run_name} stage 1")

            # Stage 2: unfreeze last block, much smaller LR (recompile is required)
            unfreeze_last(base, n_unfreeze)
            compile_model(model, lr2)
            stage2_trainable = trainable_layers(model)
            off = len(h1["loss"])
            cbs = callbacks(ckpt, p) + ([MLflowEpochLogger(mlflow, offset=off)] if a.tracking == "manual" else [])
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=ep2, callbacks=cbs).history
            plot_curves(h2, FIGS / f"{run_name}_stage2_curves.png", f"{run_name} stage 2")
            hist = merge(h1, h2)
            curves, best_epoch = plot_curves(hist, FIGS / f"{run_name}_curves.png", run_name, stage_split=off)
            stage_info = {"stage1_trainable": stage1_trainable, "stage2_trainable": stage2_trainable}

        print(f"\nBest epoch (lowest val_loss): {best_epoch}")
        mlflow.log_metric("best_epoch", best_epoch)

        # Final test metrics (test set used once, threshold picked on val)
        test_m, figs = full_evaluation(model, p, run_name, a.strategy, img_size=size)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_m.items()})
        print("Test metrics:", json.dumps(test_m, indent=2))

        # Artifacts (Task 3.4)
        layers_file = ROOT / "reports" / "metrics" / f"{run_name}_trainable_layers.json"
        layers_file.write_text(json.dumps(stage_info, indent=2))
        freeze = ROOT / "reports" / "metrics" / "pip_freeze.txt"
        freeze.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                         capture_output=True, text=True).stdout)
        for f in [curves, layers_file, freeze, *figs]:
            mlflow.log_artifact(str(f), artifact_path="reports")
        mlflow.log_artifacts(str(ROOT / p["data"]["splits_dir"]), artifact_path="splits")
        model.save(ckpt)                                           # best weights (restore_best_weights)
        mlflow.log_artifact(str(ckpt), artifact_path="keras_file")
        if a.tracking == "manual":
            mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"\nRun id: {run.info.run_id}\nSaved model: {ckpt}")


if __name__ == "__main__":
    main()
