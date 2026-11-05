"""Task 4.2 - routes and validation: upload, result, history, health.

Validation: extension allow-list, size limit (MAX_CONTENT_LENGTH -> 413),
real decode check with Pillow (catches renamed / truncated files), sanitized
file name. The upload is read into memory only; nothing is written to disk.
"""
from __future__ import annotations

import hashlib
import io

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

bp = Blueprint("main", __name__)


class UploadError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def validate_upload(file) -> tuple[str, bytes, tuple[int, int]]:
    if file is None or file.filename == "":
        raise UploadError("No file selected. Please choose an image.")
    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    if ext not in allowed:
        raise UploadError(f"Unsupported file type '.{ext or '?'}'. Allowed: {', '.join(sorted(allowed))}.", 415)
    data = file.read()
    if not data:
        raise UploadError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            size = im.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise UploadError(f"'{name}' looks like an image but cannot be decoded ({type(e).__name__}).", 422)
    return name, data, size


@bp.get("/")
def index():
    return render_template("index.html", max_mb=current_app.config["MAX_UPLOAD_MB"])


@bp.post("/predict")
def predict():
    name, data, (w, h) = validate_upload(request.files.get("image"))
    predictor = current_app.extensions["predictor"]
    if predictor is None:
        raise UploadError("The model is not loaded. Check /health and the server logs.", 503)
    try:
        pred = predictor.predict(data)
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("prediction failed")
        raise UploadError(f"Model service error: {type(e).__name__}", 502)

    row_id = current_app.extensions["db"].add(
        filename=name, image_sha256=hashlib.sha256(data).hexdigest(), width=w, height=h,
        label=pred.label, prob_ai=pred.prob_ai, confidence=pred.confidence, threshold=pred.threshold,
        model_name=pred.model_name, model_version=pred.model_version, backend=predictor.backend,
        latency_ms=pred.latency_ms, client_ip=request.headers.get("X-Forwarded-For", request.remote_addr),
    )
    current_app.logger.info("prediction id=%s file=%s label=%s conf=%.3f", row_id, name, pred.label, pred.confidence)
    if request.args.get("format") == "json" or request.accept_mimetypes.best == "application/json":
        return jsonify(id=row_id, filename=name, **pred.as_dict())
    return render_template("result.html", p=pred, filename=name, row_id=row_id, size=(w, h))


@bp.get("/history")
def history():
    rows = current_app.extensions["db"].recent(limit=int(request.args.get("limit", 100)))
    if request.args.get("format") == "json":
        return jsonify([{**r, "created_at": r["created_at"].isoformat()} for r in rows])
    return render_template("history.html", rows=rows)


@bp.get("/health")
def health():
    cfg = current_app.config
    predictor = current_app.extensions["predictor"]
    db_ok = current_app.extensions["db"].ping()
    body = {
        "status": "ok" if (predictor is not None and db_ok) else "degraded",
        "model_loaded": predictor is not None,
        "backend": getattr(predictor, "backend", cfg["INFERENCE_BACKEND"]),
        "model_name": cfg["MODEL_NAME"], "model_version": cfg["MODEL_VERSION"],
        "model_path": cfg["MODEL_PATH"] if cfg["INFERENCE_BACKEND"] == "local" else None,
        "model_api_url": cfg["MODEL_API_URL"] if cfg["INFERENCE_BACKEND"] == "remote" else None,
        "threshold": cfg["DECISION_THRESHOLD"], "database": "ok" if db_ok else "unreachable",
        "debug": cfg["DEBUG"],
    }
    return jsonify(body), (200 if body["status"] == "ok" else 503)


def register_error_handlers(app) -> None:
    def render_error(status: int, title: str, message: str):
        if request.path.startswith("/health") or request.args.get("format") == "json":
            return jsonify(error=title, message=message), status
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(UploadError)
    def _upload(e: UploadError):
        return render_error(e.status, "Upload rejected" if e.status < 500 else "Service problem", e.message)

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(e):
        return render_error(413, "File too large", f"Maximum upload size is {app.config['MAX_UPLOAD_MB']} MB.")

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        return render_error(e.code or 500, e.name, e.description or "")

    @app.errorhandler(Exception)
    def _other(e):
        app.logger.exception("unhandled error")
        return render_error(500, "Internal error", "Something went wrong. Please try again.")
