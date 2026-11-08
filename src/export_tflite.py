"""Task 7 - convert the champion Keras model to TFLite for a small (512 MB) host,
and PROVE it still predicts the same on the test split.

    python -m src.export_tflite                       # models/champion.keras -> models/champion.tflite
    python -m src.export_tflite --threshold 0.8432

Checks (on data/splits/test.csv):
  1. numpy preprocessing (app/preprocess_np.py) vs TF preprocessing -> max pixel difference
  2. TFLite P(AI) vs Keras P(AI) -> max difference and label agreement at the threshold
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from app import preprocess, preprocess_np  # noqa: E402


def convert(src: Path, dst: Path) -> None:
    model = keras.models.load_model(src)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, verbose=False)                       # SavedModel (Keras 3 way)
        tflite = tf.lite.TFLiteConverter.from_saved_model(tmp).convert()
    dst.write_bytes(tflite)
    print(f"{src} ({src.stat().st_size / 1e6:.1f} MB) -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/champion.keras")
    ap.add_argument("--dst", default="models/champion.tflite")
    ap.add_argument("--split", default="data/splits/test.csv")
    ap.add_argument("--threshold", type=float, default=float(os.getenv("DECISION_THRESHOLD", "0.8432")))
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)

    convert(src, dst)

    keras_model = keras.models.load_model(src)
    interp = tf.lite.Interpreter(model_path=str(dst))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

    rows = pd.read_csv(a.split)
    pix_diff, p_keras, p_lite, t_keras, t_lite = [], [], [], [], []
    for path in rows["path"]:
        data = Path(path).read_bytes()
        x_tf = preprocess.decode_and_resize(data, a.img_size).numpy()
        x_np = preprocess_np.decode_and_resize(data, a.img_size)
        pix_diff.append(float(np.abs(x_tf - x_np).max()))

        t0 = time.perf_counter()
        p_keras.append(float(keras_model(x_tf[None], training=False)[0, 0]))
        t_keras.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        interp.set_tensor(inp, x_np[None].astype(np.float32))   # the full serving path: numpy pixels
        interp.invoke()
        p_lite.append(float(interp.get_tensor(out)[0, 0]))
        t_lite.append(time.perf_counter() - t0)

    p_keras, p_lite = np.array(p_keras), np.array(p_lite)
    agree = float(np.mean((p_keras >= a.threshold) == (p_lite >= a.threshold)))
    report = {
        "images": len(rows),
        "keras_mb": round(src.stat().st_size / 1e6, 1),
        "tflite_mb": round(dst.stat().st_size / 1e6, 1),
        "max_pixel_diff_numpy_vs_tf": round(max(pix_diff), 4),
        "max_prob_diff": float(np.abs(p_keras - p_lite).max()),
        "mean_prob_diff": float(np.abs(p_keras - p_lite).mean()),
        "label_agreement_at_threshold": agree,
        "threshold": a.threshold,
        "keras_ms_per_image": round(1000 * float(np.median(t_keras)), 1),
        "tflite_ms_per_image": round(1000 * float(np.median(t_lite)), 1),
    }
    print(json.dumps(report, indent=2))
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics/tflite_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
