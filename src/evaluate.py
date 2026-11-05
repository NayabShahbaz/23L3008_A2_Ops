"""Task 2.5 (evaluation) and 2.6 (error analysis).

Threshold is chosen on VALIDATION data, then the test set is evaluated ONCE.
Positive class = AI-generated (label 1). A false positive = a real photo called
fake. Default strategy "max_fpr" keeps that error at or below 5% on validation;
"f1" maximises F1 instead.

Standalone:  python -m src.evaluate --model models/<run>.keras [--strategy f1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ROOT, load_params  # noqa: E402


def predict(model, ds) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def choose_threshold(y: np.ndarray, p: np.ndarray, strategy: str = "max_fpr", max_fpr: float = 0.05) -> float:
    if strategy == "f1":
        grid = np.linspace(0.05, 0.95, 91)
        return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])
    fpr, _, thr = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    return float(min(max(thr[ok[-1]], 0.0), 1.0)) if len(ok) else 0.5


def metrics_at(y, p, t) -> dict:
    yhat = (p >= t).astype(int)
    return {
        "threshold": round(t, 4),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }


def plot_confusion(y, p, t, out: Path, title: str) -> Path:
    cm = confusion_matrix(y, (p >= t).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "ai"])
    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.set_title(f"{title} (t={t:.2f})")
    disp.figure_.tight_layout()
    disp.figure_.savefig(out, dpi=150)
    plt.close(disp.figure_)
    return out


def plot_roc(y, p, out: Path, title: str) -> Path:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def misclassified_grid(y, p, t, test_csv: Path, out: Path, n: int = 10) -> Path:
    df = pd.read_csv(test_csv)
    df["prob_ai"], df["pred"] = p, (p >= t).astype(int)
    wrong = df[df.pred != df.label].copy()
    wrong["conf_err"] = (wrong.prob_ai - t).abs()
    wrong = wrong.sort_values("conf_err", ascending=False).head(n)   # most confident mistakes
    wrong.to_csv(out.with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, (_, r) in zip(axes.flat, wrong.iterrows()):
        with Image.open(ROOT / r.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"true={r['class']}  p(ai)={r.prob_ai:.2f}\n{Path(r.path).name[:28]}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    fig.suptitle("Misclassified test images (write a hypothesis for each in the report)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def full_evaluation(model, params: dict, tag: str, strategy: str = "max_fpr",
                    img_size: int | None = None) -> tuple[dict, list[Path]]:
    from src.data_pipeline import make_dataset

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    yv, pv = predict(model, make_dataset("val", params, training=False, img_size=img_size))
    t = choose_threshold(yv, pv, strategy)
    yt, pt = predict(model, make_dataset("test", params, training=False, img_size=img_size))  # test: once
    m = metrics_at(yt, pt, t)
    files = [
        plot_confusion(yt, pt, t, fig_dir / f"{tag}_confusion.png", f"{tag} test confusion"),
        plot_roc(yt, pt, fig_dir / f"{tag}_roc.png", f"{tag} test ROC"),
        misclassified_grid(yt, pt, t, ROOT / params["data"]["splits_dir"] / "test.csv",
                           fig_dir / f"{tag}_misclassified.png"),
    ]
    out = ROOT / "reports" / "metrics" / f"{tag}_test_metrics.json"
    out.write_text(json.dumps(m, indent=2))
    files.append(out)
    return m, files


def main() -> None:
    from tensorflow import keras

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategy", default="max_fpr", choices=["max_fpr", "f1"])
    ap.add_argument("--img-size", type=int, default=None)
    a = ap.parse_args()
    model = keras.models.load_model(a.model)
    m, files = full_evaluation(model, load_params(), Path(a.model).stem, a.strategy, a.img_size)
    print(json.dumps(m, indent=2))
    print("Saved:", *[str(f) for f in files], sep="\n  ")


if __name__ == "__main__":
    main()
