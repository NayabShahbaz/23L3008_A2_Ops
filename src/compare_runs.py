"""Task 3.5 / 3.6 - compare MLflow runs and list run ids.

  python -m src.compare_runs                  # runs whose name starts with "cmp_" -> table + chart
  python -m src.compare_runs --prefix mnv2    # any other name prefix
  python -m src.compare_runs --all            # every run in the experiment (find run ids for 3.6)

Outputs: reports/metrics/run_comparison.csv, reports/figures/run_comparison.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import ROOT, setup_mlflow  # noqa: E402

PARAMS = ["backbone", "stage1_lr", "lr", "augment", "unfreeze_last_n", "img_size", "stage1_epochs"]
METRICS = ["test_roc_auc", "test_f1", "test_accuracy", "test_precision", "test_recall", "test_threshold", "best_epoch"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cmp_")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    mlflow = setup_mlflow()  # sets the active experiment (MLFLOW_EXPERIMENT_NAME)
    runs = mlflow.search_runs(output_format="pandas")
    if runs.empty:
        raise SystemExit("No runs found in this experiment.")
    runs["name"] = runs.get("tags.mlflow.runName", pd.Series(dtype=str))

    if a.all:
        cols = ["name", "run_id", "status", "start_time", "metrics.test_roc_auc", "metrics.test_f1"]
        print(runs[[c for c in cols if c in runs]].to_string(index=False))
        return

    sel = runs[runs.name.fillna("").str.startswith(a.prefix) & (runs.status == "FINISHED")].copy()
    if sel.empty:
        raise SystemExit(f"No finished runs named '{a.prefix}*'. Use --all to see what exists.")
    table = pd.DataFrame({"run": sel.name, "run_id": sel.run_id.str[:8]})
    for p in PARAMS:
        if f"params.{p}" in sel:
            table[p] = sel[f"params.{p}"].values
    for m in METRICS:
        if f"metrics.{m}" in sel:
            table[m] = sel[f"metrics.{m}"].round(4).values
    table = table.dropna(axis=1, how="all").sort_values("test_roc_auc", ascending=False)

    out_csv = ROOT / "reports" / "metrics" / "run_comparison.csv"
    table.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(table)), 4.5))
    x = range(len(table))
    w = 0.38
    ax.bar([i - w / 2 for i in x], table.test_roc_auc, w, label="ROC-AUC")
    ax.bar([i + w / 2 for i in x], table.test_f1, w, label="F1")
    for i, (auc, f1) in enumerate(zip(table.test_roc_auc, table.test_f1)):
        ax.text(i - w / 2, auc + 0.01, f"{auc:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f1 + 0.01, f"{f1:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table.run, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test score")
    ax.set_title("Experiment comparison (test set)")
    ax.legend()
    fig.tight_layout()
    out_png = ROOT / "reports" / "figures" / "run_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nBest by ROC-AUC: {table.iloc[0].run} (run {table.iloc[0].run_id})")
    print(f"Saved {out_csv}\nSaved {out_png}")


if __name__ == "__main__":
    main()
