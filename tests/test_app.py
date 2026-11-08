"""Flask app tests. A fake predictor is injected, so TensorFlow is not needed in CI."""
import io

import pytest
from PIL import Image

from app import create_app
from app.config import Config
from app.db import Database
from app.inference import make_prediction


class FakePredictor:
    backend = "fake"

    def predict(self, data: bytes):
        import time

        return make_prediction(0.9, 0.5, "visionguard", "test", time.perf_counter())


def png_bytes(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
    class TestConfig(Config):
        TESTING = True
        MAX_UPLOAD_MB = 0.5
        MAX_CONTENT_LENGTH = int(0.5 * 1024 * 1024)

    db = Database(f"sqlite:///{tmp_path / 'test.db'}")
    app = create_app(TestConfig, predictor=FakePredictor(), database=db)
    return app.test_client()


def upload(client, data: bytes, name: str):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_valid_upload_is_logged(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    rows = client.get("/history?format=json").get_json()
    assert len(rows) == 1 and rows[0]["filename"] == "cat.png" and rows[0]["confidence"] == 0.8


def test_wrong_extension_rejected(client):
    assert upload(client, b"hello", "notes.txt").status_code == 415


def test_undecodable_image_rejected(client):
    assert upload(client, b"\x89PNG\r\n\x1a\n not really a png", "fake.png").status_code == 422


def test_too_large_rejected(client):
    assert upload(client, b"0" * (1024 * 1024), "big.png").status_code == 413


def test_filename_sanitized(client):
    upload(client, png_bytes(), "../../etc/passwd.png")
    assert client.get("/history?format=json").get_json()[0]["filename"] == "etc_passwd.png"


# ---------- Task 8: remote backend (the cloud API is faked, no network) ----------
class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.ok = status, body or {}, status == 200

    def json(self):
        return self._body


def remote_with(response=None, exc=None):
    from app.inference import RemotePredictor

    rp = RemotePredictor("https://api.example/predict", "k")

    def post(*args, **kwargs):
        if exc:
            raise exc
        return response

    rp.session.post = post
    return rp


def test_remote_success_recomputes_label():
    body = {"prob_ai": 0.9, "threshold": 0.8, "model_name": "visionguard", "model_version": "champion"}
    p = remote_with(FakeResponse(200, body)).predict(b"img")
    assert p.label == "AI-generated" and p.confidence == 0.5 and p.threshold == 0.8


@pytest.mark.parametrize("status", [401, 500])
def test_remote_http_errors_become_clear_messages(status):
    from app.inference import ModelServiceError

    with pytest.raises(ModelServiceError, match=str(status)):
        remote_with(FakeResponse(status)).predict(b"img")


def test_remote_timeout_shows_friendly_502(tmp_path):
    import requests

    db = Database(f"sqlite:///{tmp_path / 'r.db'}")
    app = create_app(Config, predictor=remote_with(exc=requests.Timeout()), database=db)
    r = upload(app.test_client(), png_bytes(), "cat.png")
    assert r.status_code == 502 and b"waking up" in r.data
    assert db.recent() == []                     # failed calls are not logged as predictions
