"""Task 2.7 (apply part) - Keras Tuner on VisionGuard.

Searches learning rate, dropout and number of unfrozen backbone layers with
RandomSearch (>= 8 trials). Each trial: one fine-tuning stage for a few epochs.

Run:  python -m src.tune [--trials 8 --epochs 5]
Output: reports/metrics/tuner_results.csv (+ printed best hyperparameters)
"""
import argparse

import keras_tuner as kt
import pandas as pd

from src.data_pipeline import make_dataset
from src.models import build_transfer, compile_model, unfreeze_last
from src.utils import ROOT, load_params, set_seeds


def main() -> None:
    p = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=p["tuner"]["max_trials"])
    ap.add_argument("--epochs", type=int, default=p["tuner"]["epochs"])
    a = ap.parse_args()
    set_seeds(p["seed"])
    size, backbone = p["image"]["size"], p["transfer"]["backbone"]

    def build(hp: kt.HyperParameters):
        model, base = build_transfer(size, backbone, hp.Float("dropout", 0.1, 0.6, step=0.1))
        unfreeze_last(base, hp.Choice("unfreeze_last_n", [0, 10, 20, 40, 60]))
        return compile_model(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))

    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=a.trials, seed=p["seed"],
                            directory=str(ROOT / "tuner_runs"), project_name="visionguard", overwrite=True)
    tuner.search(make_dataset("train", p, training=True), validation_data=make_dataset("val", p, training=False),
                 epochs=a.epochs, verbose=2)

    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=a.trials):
        rows.append({"trial": t.trial_id, **t.hyperparameters.values, "val_loss": t.score})
    df = pd.DataFrame(rows).sort_values("val_loss")
    out = ROOT / "reports" / "metrics" / "tuner_results.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
