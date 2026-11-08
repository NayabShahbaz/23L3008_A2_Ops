"""Task 7 - public model API (deployed to the cloud). The local Flask UI calls it
when INFERENCE_BACKEND=remote (Task 8).

Endpoints
  GET  /health                   -> no auth, for uptime checks
  POST /predict  (multipart 'image', header X-API-Key) -> JSON prediction

Run locally:   python -m api.main          (port 8000)
Postman:       POST {URL}/predict, Headers: X-API-Key=<key>, Body: form-data image=<file>
"""
from __future__ import annotations

import hmac
import io
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s api: %(message)s")
log = logging.getLogger("api")

API_KEY = os.getenv("API_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
MODEL_NAME = os.getenv("MODEL_NAME", "visionguard")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
MAX_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))

api = Flask(__name__)
api.config["MAX_CONTENT_LENGTH"] = int(MAX_MB * 1024 * 1024)

predictor = None
try:
    from app.inference import load_model_predictor

    predictor = load_model_predictor(MODEL_PATH, IMG_SIZE, THRESHOLD, MODEL_NAME, MODEL_VERSION)  # once
    log.info("model loaded from %s (%s backend)", MODEL_PATH, predictor.backend)
except Exception as e:  # noqa: BLE001
    log.error("MODEL NOT LOADED: %s", e)


def client_ip() -> str:
    # behind a cloud proxy the real caller is the first X-Forwarded-For entry
    return request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()


def authorised() -> bool:
    supplied = request.headers.get("X-API-Key", "")
    return bool(API_KEY) and hmac.compare_digest(supplied, API_KEY)


@api.get("/")
def index():
    return jsonify(service="23L3008_A2_Ops VisionGuard model API",
                   endpoints={"GET /health": "no auth",
                              "POST /predict": "header X-API-Key + multipart field 'image'"})


@api.get("/health")
def health():
    ok = predictor is not None
    return jsonify(status="ok" if ok else "degraded", model_loaded=ok, model_name=MODEL_NAME,
                   model_version=MODEL_VERSION, threshold=THRESHOLD), (200 if ok else 503)


@api.post("/predict")
def predict():
    if not authorised():
        log.warning("401 rejected /predict from %s (missing or wrong key)", client_ip())
        return jsonify(error="unauthorised", message="missing or wrong X-API-Key"), 401
    if predictor is None:
        return jsonify(error="model_not_loaded"), 503
    f = request.files.get("image")
    data = f.read() if f else request.get_data()
    if not data:
        return jsonify(error="no_image", message="send multipart field 'image'"), 400
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return jsonify(error="undecodable_image"), 422
    pred = predictor.predict(data)
    log.info("200 predict from %s label=%s prob_ai=%.4f latency=%sms",
             client_ip(), pred.label, pred.prob_ai, pred.latency_ms)
    return jsonify(pred.as_dict())


@api.errorhandler(413)
def too_large(_):
    return jsonify(error="file_too_large", max_mb=MAX_MB), 413


if __name__ == "__main__":
    api.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
