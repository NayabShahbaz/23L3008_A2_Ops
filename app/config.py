"""Task 4.4 - every setting comes from environment variables (12-factor style)."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env when running locally; in Docker, compose injects env vars


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-not-secret")
    DEBUG = _bool("FLASK_DEBUG")                     # must be off in production
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

    INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "local")   # local | remote
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    MODEL_NAME = os.getenv("MODEL_NAME", "visionguard")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
    IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
    DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    MODEL_API_URL = os.getenv("MODEL_API_URL", "")
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "90"))  # free hosts wake slowly

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
