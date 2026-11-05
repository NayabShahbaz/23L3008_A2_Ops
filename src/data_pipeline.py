"""Task 2.1 - tf.data input pipeline.

Stages: read split CSV -> (path, label) dataset -> [shuffle, train only] ->
read file bytes -> decode + resize (shared with serving) -> batch ->
[augment, train only] -> prefetch.

Run to save one augmented batch figure:
    python -m src.data_pipeline
"""
from __future__ import annotations

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from app.preprocess import decode_and_resize
from src.utils import ROOT, load_params

AUTOTUNE = tf.data.AUTOTUNE


def augmenter(seed: int) -> keras.Sequential:
    """Random augmentation, applied to TRAINING batches only."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, seed=seed),
        keras.layers.RandomZoom(0.1, seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def make_dataset(split: str, params: dict, training: bool, augment: bool | None = None,
                 img_size: int | None = None) -> tf.data.Dataset:
    size = img_size or params["image"]["size"]
    bs = params["train"]["batch_size"]
    seed = params["seed"]
    augment = params["train"]["augment"] if augment is None else augment

    df = pd.read_csv(ROOT / params["data"]["splits_dir"] / f"{split}.csv")
    paths = [str(ROOT / p) for p in df.path]
    ds = tf.data.Dataset.from_tensor_slices((paths, df.label.astype("float32").values))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        return decode_and_resize(tf.io.read_file(path), size), label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(bs)
    if training and augment:
        aug = augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def save_augmented_batch_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = load_params()
    x, y = next(iter(make_dataset("train", p, training=True, augment=True)))
    n = min(16, x.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(tf.clip_by_value(x[i], 0, 255).numpy().astype("uint8"))
            ax.set_title("ai" if y[i] == 1 else "real")
        ax.axis("off")
    fig.suptitle("One augmented training batch")
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "augmented_batch.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    save_augmented_batch_figure()
