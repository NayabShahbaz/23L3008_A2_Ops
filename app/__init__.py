"""VisionGuard Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask

from app.config import Config
from app.db import Database


def create_app(config: type = Config, predictor=None, database: Database | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app.extensions["db"] = database or Database(app.config["DATABASE_URL"])
    if predictor is None:
        from app.inference import build_predictor

        try:
            predictor = build_predictor(config)
            app.logger.info("Predictor ready (%s backend)", predictor.backend)
        except Exception as e:  # noqa: BLE001 - app still starts; /health reports the problem
            app.logger.error("MODEL NOT LOADED: %s", e)
            predictor = None
    app.extensions["predictor"] = predictor

    from app.routes import bp, register_error_handlers

    app.register_blueprint(bp)
    register_error_handlers(app)
    return app
