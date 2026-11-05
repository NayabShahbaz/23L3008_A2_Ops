"""Task 3.1 / 3.2 - tiny Keras model on Fashion-MNIST tracked with MLflow.

Local (3.1):   python labs/mlflow_toy/mlflow_toy.py --lr 0.001
               python labs/mlflow_toy/mlflow_toy.py --lr 0.01
               mlflow ui        -> http://127.0.0.1:5000, select both runs -> Compare
DagsHub (3.2): python labs/mlflow_toy/mlflow_toy.py --lr 0.001 --dagshub
"""
import argparse
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.tensorflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tensorflow import keras  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dagshub", action="store_true", help="log to DagsHub instead of ./mlruns")
    a = ap.parse_args()

    if a.dagshub:
        load_dotenv(ROOT / ".env")                          # token comes from .env, never from code
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("toy_fashion_mnist")
    keras.utils.set_random_seed(42)

    (x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
    x, xt = x[:10000] / 255.0, xt / 255.0
    y = y[:10000]
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(a.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
        mlflow.set_tags({"purpose": "toy", "dataset": "fashion_mnist"})
        mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 64})

        class EpochLog(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

        h = model.fit(x, y, validation_split=0.2, epochs=a.epochs, callbacks=[EpochLog()], verbose=2)
        loss, acc = model.evaluate(xt, yt, verbose=0)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})

        fig, ax = plt.subplots()
        ax.plot(h.history["loss"], label="train")
        ax.plot(h.history["val_loss"], label="val")
        ax.set_title(f"loss (lr={a.lr})")
        ax.legend()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "loss_curve.png"
            fig.savefig(f)
            mlflow.log_artifact(str(f))
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
