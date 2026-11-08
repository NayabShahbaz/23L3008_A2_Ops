# VisionGuard — 23L3008_A2_Ops

![ci](https://github.com/NayabShahbaz/23L3008_A2_Ops/actions/workflows/ci.yml/badge.svg)

VisionGuard is a web service that says whether an uploaded image is **AI-generated** or **real**: a TensorFlow/Keras model, MLflow tracking on DagsHub, a Flask UI, a database log of every request, Docker Compose, and a cloud model API.

| Link | URL |
|---|---|
| GitHub | `https://github.com/NayabShahbaz/23L3008_A2_Ops` |
| DagsHub | `https://dagshub.com/NayabShahbaz/23L3008_A2_Ops` |
| Model API | `https://<your-api-host>/predict` |

## Layout

```
├── params.yaml              # every training setting (seed, image size, LRs, backbone...)
├── src/                     # data + training code
│   ├── env_check.py         # 1.1 versions + GPU check
│   ├── audit.py             # 1.2 dataset audit (size, dims, mode, format, SHA-256, corrupt, dups)
│   ├── eda.py               # 1.3 sample grid + histograms
│   ├── split.py             # stratified train/val/test lists -> data/splits/*.csv
│   ├── data_pipeline.py     # 2.1 tf.data pipeline + augmented batch figure
│   ├── models.py            # 2.2 baseline CNN, 2.3 transfer model
│   ├── train.py             # 2.4/2.5 + 3.3-3.5 training with MLflow (manual or autolog)
│   ├── evaluate.py          # 2.5 metrics/threshold, 2.6 misclassified grid
│   ├── tune.py              # 2.7 Keras Tuner on VisionGuard
│   ├── gradcam.py           # bonus Grad-CAM
│   ├── register.py          # 3.6 registry versions, alias, download champion
│   ├── predict_from_registry.py  # 3.6 load by alias, classify one image
│   ├── compare_runs.py      # 3.5 comparison table/plot of MLflow runs
│   └── show_db.py           # 5   query the predictions table directly
├── app/                     # Task 4/5 Flask app (upload, result, history, health) + DB
├── api/                     # Task 7 cloud model API (+ its own Dockerfile)
├── tests/                   # pytest for the app (no TensorFlow needed)
├── labs/                    # toy parts: DVC, MLflow, Keras Tuner, GitHub Actions
├── Dockerfile               # 6.2 web image
├── Dockerfile.before        # 6.4 deliberately naive image, only for the size comparison
├── docker-compose.yml       # 6.3 web + postgres (+ optional api profile)
├── .github/workflows/ci.yml # 6.6 lint, test, docker build
├── data/  (DVC)   models/  (DVC / registry)   reports/  (figures + metrics)
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate         # Windows
pip install -r requirements.txt
copy .env.example .env                                  # then fill in values
python -m src.audit && python -m src.split
python -m src.train --model transfer --run-name mnv2_ft
python -m src.register --download-champion
python -m app                                           # http://127.0.0.1:5000
docker compose up -d --build                            # full stack
```

## Model API (Task 7)

A small Flask service ([api/main.py](api/main.py)) that serves the champion model over HTTP. Its image ([api/Dockerfile](api/Dockerfile)) has the model **built in**, so a cloud host needs nothing else.

| Method | Path | Auth | Returns |
|---|---|---|---|
| `GET` | `/health` | none | `{"status": "ok", "model_loaded": true, "model_name": ..., "model_version": ..., "threshold": ...}` |
| `POST` | `/predict` | header `X-API-Key` | prediction JSON (below) |

`/predict` takes a multipart form field `image` (max 5 MB):

```bash
curl -H "X-API-Key: <key>" -F "image=@data/raw/AiArtData/images89.jpg" https://<your-api-host>/predict
```

```json
{"label": "AI-generated", "prob_ai": 0.9754, "confidence": 0.843, "threshold": 0.8432,
 "model_name": "visionguard", "model_version": "champion", "latency_ms": 180.2}
```

Errors: `401` missing/wrong key, `400` no image, `413` too large, `422` not a decodable image, `503` model not loaded.

Environment variables (set on the cloud host, never in the image):

| Variable | Example | Purpose |
|---|---|---|
| `API_KEY` | long random string | secret clients must send as `X-API-Key` |
| `DECISION_THRESHOLD` | `0.8432` | AI/real cut-off chosen on validation (default 0.5) |
| `PORT` | `8000` | most hosts set this themselves |
| `MODEL_NAME`, `MODEL_VERSION`, `IMG_SIZE`, `MAX_UPLOAD_MB` | `visionguard`, `champion`, `224`, `5` | optional |

Run it locally:

```bash
python -m api.main                                        # http://127.0.0.1:8000
docker build -f api/Dockerfile -t 23l3008-a2-ops-api .    # or as a container
docker compose --profile api up -d api                    # or next to the web app
```

**Flask → API (Task 8):** set `INFERENCE_BACKEND=remote`, `MODEL_API_URL=https://<your-api-host>/predict` and `MODEL_API_KEY=<key>` in `.env`; the web app then sends uploads to the API instead of loading the model itself, and still logs every prediction to its database.

## Notes

- Positive class = AI-generated (label 1). The model outputs P(AI).
- Training and serving use the same `app/preprocess.py`. Normalisation is part of the saved model.
- Secrets live only in `.env` (git- and docker-ignored). See `.env.example`.
