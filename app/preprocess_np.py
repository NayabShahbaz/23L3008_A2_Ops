"""TensorFlow-free twin of app/preprocess.py, used by the TFLite backend (Task 7 on a
512 MB host). It must give the SAME pixels as training:
  - JPEG: TF decodes with libjpeg-turbo's FAST integer DCT + fancy upsampling. Pillow
    uses the accurate DCT (pixels differ by up to ~20 levels), so JPEGs are decoded
    with simplejpeg(fastdct=True, fastupsample=False), which is bit-identical to TF.
  - other formats: Pillow (PNG/BMP/GIF/WEBP decode identically).
  - resize: re-implements tf.image.resize(method="bilinear") exactly: half-pixel
    centres, no antialiasing, interpolate along x then y.
src/export_tflite.py checks the two pipelines agree on the test split.
"""
from __future__ import annotations

import io

import numpy as np
import simplejpeg
from PIL import Image


def _axis(in_len: int, out_len: int):
    scale = in_len / out_len
    pos = (np.arange(out_len, dtype=np.float64) + 0.5) * scale - 0.5
    floor = np.floor(pos)
    lerp = (pos - floor).astype(np.float32)
    lo = np.clip(floor, 0, in_len - 1).astype(np.int64)
    hi = np.clip(floor + 1, 0, in_len - 1).astype(np.int64)
    return lo, hi, lerp


def resize_bilinear(img: np.ndarray, size: int) -> np.ndarray:
    """(H, W, C) float32 -> (size, size, C), same maths as tf.image.resize bilinear."""
    h, w = img.shape[:2]
    y0, y1, ly = _axis(h, size)
    x0, x1, lx = _axis(w, size)
    lx = lx[None, :, None]
    top = img[y0][:, x0] + (img[y0][:, x1] - img[y0][:, x0]) * lx
    bottom = img[y1][:, x0] + (img[y1][:, x1] - img[y1][:, x0]) * lx
    return top + (bottom - top) * ly[:, None, None]


def decode_and_resize(image_bytes: bytes, size: int) -> np.ndarray:
    """bytes -> float32 array (size, size, 3), values 0..255 (aspect ratio not kept)."""
    if simplejpeg.is_jpeg(image_bytes):
        try:
            rgb = simplejpeg.decode_jpeg(image_bytes, colorspace="RGB", fastdct=True, fastupsample=False)
            return resize_bilinear(rgb.astype(np.float32), size)
        except ValueError:
            pass                                     # unusual JPEG (e.g. CMYK) -> Pillow below
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.seek(0)                                   # first frame of GIFs, like TF
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return resize_bilinear(arr, size)
