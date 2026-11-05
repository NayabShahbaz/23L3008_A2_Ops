# Task 6.2 - VisionGuard web app image.
# Order matters for the layer cache: base -> system settings -> dependencies
# (change rarely) -> source code (changes often). Editing code only rebuilds
# the last cheap layers; the ~500 MB TensorFlow layer is reused.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) dependencies first (cached). tensorflow-cpu: no CUDA libs -> much smaller than "tensorflow".
COPY requirements-app.txt .
RUN pip install tensorflow-cpu==2.20.* && pip install -r requirements-app.txt

# 2) non-root user
RUN useradd --create-home --uid 10001 appuser

# 3) source code last. No .env, data or models are copied (see .dockerignore).
COPY --chown=appuser:appuser app/ app/
RUN mkdir -p /app/models && chown appuser:appuser /app/models
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"

# Production WSGI server, debug off. 1 worker = model loaded once; threads for concurrency.
CMD ["gunicorn", "app.wsgi:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-"]
