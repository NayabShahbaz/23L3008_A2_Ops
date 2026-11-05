"""Task 3.6 - load the model by alias from the MLflow registry and classify one image.

Run:  python -m src.predict_from_registry path/to/image.jpg
"""
import argparse

import mlflow.tensorflow
import tensorflow as tf

from app.preprocess import decode_and_resize
from src.utils import load_params, setup_mlflow


def main() -> None:
    p = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    setup_mlflow()
    uri = f"models:/{p['registry']['model_name']}@{p['registry']['alias']}"
    print(f"Loading {uri} ...")
    model = mlflow.tensorflow.load_model(uri)
    x = decode_and_resize(tf.io.read_file(a.image), p["image"]["size"])[None]
    prob = float(model(x, training=False)[0, 0])
    label = "AI-generated" if prob >= a.threshold else "Real"
    print(f"{a.image}: {label}  (p_ai={prob:.4f}, threshold={a.threshold})")


if __name__ == "__main__":
    main()
