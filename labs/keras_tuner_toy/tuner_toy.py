"""Task 2.7 (toy part) - tune learning rate and units on Fashion-MNIST.

Run:  python labs/keras_tuner_toy/tuner_toy.py [--algo hyperband]
"""
import argparse
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

HERE = Path(__file__).resolve().parent


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["random", "hyperband"], default="random")
    a = ap.parse_args()
    keras.utils.set_random_seed(42)
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    x, y = x[:15000] / 255.0, y[:15000]

    common = dict(objective="val_accuracy", directory=str(HERE / "tuner_runs"), overwrite=True, seed=42)
    if a.algo == "random":
        tuner = kt.RandomSearch(build, max_trials=10, project_name="toy_random", **common)
    else:
        tuner = kt.Hyperband(build, max_epochs=9, factor=3, project_name="toy_hyperband", **common)
    tuner.search(x, y, validation_split=0.2, epochs=5, verbose=2)

    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
            for t in tuner.oracle.get_best_trials(num_trials=50)]
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    df.to_csv(HERE / f"toy_results_{a.algo}.csv", index=False)
    print(df.to_string(index=False))
    tuner.results_summary(3)
    print("\nBEST:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
