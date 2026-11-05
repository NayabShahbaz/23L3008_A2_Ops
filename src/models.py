"""Task 2.2 (baseline CNN) and Task 2.3 (transfer learning) model builders.

Both models take float32 images in [0, 255] and do their own normalisation as
the first layer, so the saved .keras file is self-contained for serving.

Print summaries:  python -m src.models
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]


def compile_model(model: keras.Model, lr: float) -> keras.Model:
    # Binary cross-entropy matches a single sigmoid output; Adam is a robust default.
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=METRICS)
    return model


def build_baseline_cnn(img_size: int, filters=(32, 64, 128), dropout: float = 0.4) -> keras.Model:
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inp)            # 0..255 -> 0..1
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)              # stabilises training
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)                  # halves spatial size
    x = layers.GlobalAveragePooling2D(name="gap")(x)                 # far fewer params than Flatten
    x = layers.Dropout(dropout, name="dropout")(x)                   # regularisation
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)      # P(image is AI-generated)
    return keras.Model(inp, out, name="baseline_cnn")


# backbone -> (constructor, preprocessing layer matching what ImageNet weights expect)
def _backbone(name: str, x, img_size: int):
    shape = (img_size, img_size, 3)
    if name == "MobileNetV2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)  # -> [-1, 1]
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_tensor=x, input_shape=shape)
    elif name == "EfficientNetB0":
        # EfficientNet has its own Rescaling/Normalization layers inside and expects 0..255.
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_tensor=x, input_shape=shape)
    else:
        raise ValueError(f"Unsupported backbone {name!r}; use MobileNetV2 or EfficientNetB0")
    return base


def build_transfer(img_size: int, backbone: str = "MobileNetV2", dropout: float = 0.3):
    """Returns (model, base). base.layers are the backbone layers (frozen at start)."""
    inp = keras.Input((img_size, img_size, 3), name="image")
    base = _backbone(backbone, inp, img_size)
    for layer in base.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inp, out, name=f"transfer_{backbone}"), base


def unfreeze_last(base: keras.Model, n: int) -> None:
    """Unfreeze the last n backbone layers. BatchNorm stays frozen so its running
    statistics are not destroyed by small fine-tuning batches."""
    for layer in base.layers[-n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


def trainable_layers(model: keras.Model) -> list[str]:
    return [layer.name for layer in model.layers if layer.trainable and layer.weights]


def last_conv_layer(model: keras.Model) -> str:
    """Name of the last layer that outputs a 4-D feature map (used by Grad-CAM)."""
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4 and not isinstance(layer, layers.InputLayer):
            return layer.name
    raise ValueError("no conv feature map found")


if __name__ == "__main__":
    build_baseline_cnn(224).summary()
    m, b = build_transfer(224)
    m.summary(show_trainable=True, line_length=120)
