"""Bonus - Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

Run:  python -m src.gradcam --model models/<run>.keras [--threshold 0.5]
Output: reports/figures/gradcam.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from app.preprocess import decode_and_resize  # noqa: E402
from src.models import last_conv_layer  # noqa: E402
from src.utils import ROOT, load_params  # noqa: E402


def gradcam(model, img: tf.Tensor, layer_name: str) -> np.ndarray:
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        fmap, pred = grad_model(img[None])
        score = pred[:, 0]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    p = load_params()
    size = p["image"]["size"]
    model = keras.models.load_model(a.model)
    layer = last_conv_layer(model)
    df = pd.read_csv(ROOT / p["data"]["splits_dir"] / "test.csv").sample(frac=1, random_state=p["seed"])

    correct, wrong = [], []
    for _, r in df.iterrows():
        img = decode_and_resize(tf.io.read_file(str(ROOT / r.path)), size)
        prob = float(model(img[None], training=False)[0, 0])
        bucket = correct if int(prob >= a.threshold) == r.label else wrong
        if len(bucket) < 4:
            bucket.append((img, prob, r["class"]))
        if len(correct) == 4 and len(wrong) == 4:
            break

    fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
    for col, (img, prob, cls) in enumerate(correct + wrong):
        cam = gradcam(model, img, layer)
        tag = "correct" if col < len(correct) else "WRONG"
        axes[0, col].imshow(img.numpy().astype("uint8"))
        axes[0, col].set_title(f"{tag}: true={cls} p(ai)={prob:.2f}", fontsize=8)
        axes[1, col].imshow(img.numpy().astype("uint8"))
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM on layer '{layer}'")
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "gradcam.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
