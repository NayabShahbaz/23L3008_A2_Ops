"""Image preprocessing shared by training (src/data_pipeline.py) and serving
(app/inference.py, api/main.py). Keeping ONE function guarantees that training
and serving see identical pixels (Task 4.1).

Model-specific normalisation (e.g. /255 or MobileNetV2's [-1, 1] scaling) is
built INTO the saved Keras model, so callers only decode + resize here and feed
float32 values in [0, 255].
"""
import tensorflow as tf


def decode_and_resize(image_bytes, size: int) -> tf.Tensor:
    """bytes -> float32 tensor (size, size, 3), values 0..255.

    The image is stretched to a square: aspect ratio is NOT preserved.
    """
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size), method="bilinear")
    return tf.cast(img, tf.float32)
