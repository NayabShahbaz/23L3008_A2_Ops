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


class LitePredictor:
    """Same model converted to TFLite (src/export_tflite.py) - no TensorFlow needed,
    ~150 MB RAM instead of ~1 GB, so the API fits a 512 MB free host (Task 7)."""
    backend = "tflite"

    def __init__(self, model_path: str, img_size: int, threshold: float, name: str, version: str):
        import threading

        try:
            from ai_edge_litert.interpreter import Interpreter  # small runtime (Linux/macOS)
        except ImportError:
            import tensorflow as tf  # local fallback (e.g. Windows, no ai-edge-litert wheel)

            Interpreter = tf.lite.Interpreter

        from app.preprocess_np import decode_and_resize

        self._decode = decode_and_resize
        self.interp = Interpreter(model_path=model_path)             # once, not per request
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]["index"]
        self._out = self.interp.get_output_details()[0]["index"]
        self._lock = threading.Lock()                                # one interpreter, many threads
        self.img_size, self.threshold, self.name, self.version = img_size, threshold, name, version
        self._run(self._decode(LocalPredictor._dummy_png(), img_size))  # warm-up

    def _run(self, x) -> float:
        with self._lock:
            self.interp.set_tensor(self._in, x[None].astype("float32"))
            self.interp.invoke()
            return float(self.interp.get_tensor(self._out)[0, 0])

    def predict(self, image_bytes: bytes) -> Prediction:
        t0 = time.perf_counter()
        prob = self._run(self._decode(image_bytes, self.img_size))
        return make_prediction(prob, self.threshold, self.name, self.version, t0)


class ModelServiceError(Exception):
    """The remote model API failed; message is safe to show to the user."""


class RemotePredictor:
    """Task 8 - the Flask app sends the image to the cloud model API (Task 7)."""
    backend = "remote"

    def __init__(self, url: str, api_key: str, timeout: float = 90):
        import requests

        if not url or not api_key:
            raise ValueError("INFERENCE_BACKEND=remote needs MODEL_API_URL and MODEL_API_KEY")
        self._requests = requests
        self.session = requests.Session()                      # keeps the HTTPS connection alive
        self.session.headers.update({"X-API-Key": api_key, "User-Agent": "visionguard-web/1.0"})
        self.url, self.timeout = url, timeout
        self.health_url = url.rsplit("/predict", 1)[0] + "/health"

    def predict(self, image_bytes: bytes) -> Prediction:
        rq = self._requests
        t0 = time.perf_counter()
        try:
            # (connect, read) timeouts: a sleeping free-tier API can take ~1 min to wake up
            r = self.session.post(self.url, files={"image": ("upload", image_bytes)},
                                  timeout=(10, self.timeout))
        except rq.Timeout as e:
            raise ModelServiceError(f"The model API did not answer within {self.timeout:.0f}s "
                                    "(it may be waking up) - please try again.") from e
        except rq.ConnectionError as e:
            raise ModelServiceError("Cannot reach the model API - check MODEL_API_URL / internet.") from e
        if r.status_code == 401:
            raise ModelServiceError("The model API rejected our key (401) - check MODEL_API_KEY.")
        if r.status_code in (413, 422):
            raise ModelServiceError(f"The model API refused this image ({r.status_code}).")
        if r.status_code != 200:
            raise ModelServiceError(f"The model API returned HTTP {r.status_code} - try again later.")
        d = r.json()
        # label/confidence are recomputed here from the API's own threshold -> same logic as local
        return make_prediction(float(d["prob_ai"]), float(d["threshold"]), d["model_name"],
                               d["model_version"], t0)

    def health(self, timeout: float = 5) -> str:
        try:
            r = self.session.get(self.health_url, timeout=timeout)
            return "ok" if r.ok else f"HTTP {r.status_code}"
        except self._requests.RequestException as e:
            return f"unreachable ({type(e).__name__}) - may be waking up"


def load_model_predictor(model_path: str, img_size: int, threshold: float, name: str, version: str):
    """.tflite -> LitePredictor (no TensorFlow), anything else -> Keras LocalPredictor."""
    cls = LitePredictor if model_path.endswith(".tflite") else LocalPredictor
    return cls(model_path, img_size, threshold, name, version)


def build_predictor(cfg) -> LocalPredictor | LitePredictor | RemotePredictor:
    if cfg.INFERENCE_BACKEND == "remote":
        return RemotePredictor(cfg.MODEL_API_URL, cfg.MODEL_API_KEY, cfg.MODEL_API_TIMEOUT)
    return load_model_predictor(cfg.MODEL_PATH, cfg.IMG_SIZE, cfg.DECISION_THRESHOLD, cfg.MODEL_NAME,
                                cfg.MODEL_VERSION)


if __name__ == "__main__":
    from app.config import Config

    pred = build_predictor(Config)
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(path, "->", pred.predict(f.read()).as_dict())
