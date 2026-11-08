"""Task 4.1 / Task 8 - inference backends.

LocalPredictor  : loads the champion Keras model ONCE at start-up and predicts in-process.
RemotePredictor : sends the image to the cloud model API (Task 7) with a secret key.

Both return the same dict, so routes do not care which one is used.
Quick check:  python -m app.inference path/to/real.jpg path/to/ai.png
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass

from PIL import Image

TF_NATIVE = {"JPEG", "PNG", "BMP", "GIF"}


@dataclass
class Prediction:
    label: str            # "AI-generated" | "Real"
    prob_ai: float        # model output P(AI)
    confidence: float     # how far past the threshold, into the chosen side (0 = on threshold, 1 = certain)
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def to_tf_decodable(image_bytes: bytes) -> bytes:
    """WEBP etc. are not decodable by tf.io.decode_image -> re-encode as PNG in memory."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        if im.format in TF_NATIVE:
            return image_bytes
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


def confidence_for(prob_ai: float, threshold: float) -> float:
    """Normalised margin from the decision threshold.

    With a non-0.5 threshold, "probability of the chosen label" is misleading
    (p=0.78 with t=0.84 is labelled Real but P(real)=0.22). Instead measure how
    far the score lies into the chosen side: 0 = exactly on the threshold,
    1 = at the extreme (p=1 for AI, p=0 for Real).
    """
    if prob_ai >= threshold:
        return (prob_ai - threshold) / (1 - threshold) if threshold < 1 else 1.0
    return (threshold - prob_ai) / threshold if threshold > 0 else 1.0


def make_prediction(prob_ai: float, threshold: float, name: str, version: str, t0: float) -> Prediction:
    is_ai = prob_ai >= threshold
    return Prediction(
        label="AI-generated" if is_ai else "Real",
        prob_ai=round(prob_ai, 4),
        confidence=round(confidence_for(prob_ai, threshold), 4),
        threshold=threshold, model_name=name, model_version=version,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


class LocalPredictor:
    backend = "local"

    def __init__(self, model_path: str, img_size: int, threshold: float, name: str, version: str):
        from tensorflow import keras  # heavy import only when this backend is used

        from app.preprocess import decode_and_resize

        self._decode = decode_and_resize
        self.model = keras.models.load_model(model_path)       # once, not per request
        self.img_size, self.threshold, self.name, self.version = img_size, threshold, name, version
        self.model(self._decode(self._dummy_png(), img_size)[None], training=False)  # warm-up

    @staticmethod
    def _dummy_png() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    def predict(self, image_bytes: bytes) -> Prediction:
        t0 = time.perf_counter()
        x = self._decode(to_tf_decodable(image_bytes), self.img_size)[None]
        prob = float(self.model(x, training=False)[0, 0])
        return make_prediction(prob, self.threshold, self.name, self.version, t0)


class RemotePredictor:
    backend = "remote"

    def __init__(self, url: str, api_key: str, timeout: float = 30):
        import requests

        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.url, self.timeout = url, timeout

    def predict(self, image_bytes: bytes) -> Prediction:
        t0 = time.perf_counter()
        r = self.session.post(self.url, files={"image": ("upload", image_bytes)}, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return make_prediction(float(d["prob_ai"]), float(d["threshold"]), d["model_name"],
                               d["model_version"], t0)


def build_predictor(cfg) -> LocalPredictor | RemotePredictor:
    if cfg.INFERENCE_BACKEND == "remote":
        return RemotePredictor(cfg.MODEL_API_URL, cfg.MODEL_API_KEY, cfg.MODEL_API_TIMEOUT)
    return LocalPredictor(cfg.MODEL_PATH, cfg.IMG_SIZE, cfg.DECISION_THRESHOLD, cfg.MODEL_NAME, cfg.MODEL_VERSION)


if __name__ == "__main__":
    from app.config import Config

    pred = build_predictor(Config)
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(path, "->", pred.predict(f.read()).as_dict())
