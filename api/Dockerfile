# Task 7 - self-contained model API image (model is BUILT IN so a cloud host
# needs nothing else). Build from the repo root:
#   docker build -f api/Dockerfile -t 23l3008-a2-ops-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 PORT=8000

WORKDIR /srv
COPY requirements-app.txt .
RUN pip install tensorflow-cpu==2.20.* && pip install -r requirements-app.txt

RUN useradd --create-home --uid 10001 appuser
COPY app/ app/
COPY api/ api/
COPY models/champion.keras models/champion.keras
RUN chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
# 1 worker so the TF model is loaded once; threads give concurrency.
CMD ["sh", "-c", "gunicorn api.main:api --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile -"]
