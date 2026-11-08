# VisionGuard — 23L3008_A2_Ops

![ci](https://github.com/NayabShahbaz/23L3008_A2_Ops/actions/workflows/ci.yml/badge.svg)

VisionGuard is a web service that says whether an uploaded image is **AI-generated** or **real**: a TensorFlow/Keras model, MLflow tracking on DagsHub, a Flask UI, a database log of every request, Docker Compose, and a cloud model API.

| Link | URL |
|---|---|
| GitHub | `https://github.com/NayabShahbaz/23L3008_A2_Ops` |
| DagsHub | `https://dagshub.com/NayabShahbaz/23L3008_A2_Ops` |
| Model API | `https://api-23l3008-a2-ops.onrender.com` (`/health`, `/predict`) - Render free tier, sleeps when idle (~1 min first request) |

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
│   ├── show_db.py           # 5   query the predictions table directly
│   ├── export_tflite.py     # 7   Keras -> TFLite + check predictions still match
│   └── latency_compare.py   # 8   same images: local model vs cloud API latency
├── app/                     # Task 4/5/8 Flask app (upload, result, history, health) + DB
│   ├── inference.py         # Local (Keras) / Lite (TFLite) / Remote (cloud API) predictors
│   ├── preprocess.py        # decode + resize shared by training and serving (TensorFlow)
│   └── preprocess_np.py     # bit-identical twin without TensorFlow (TFLite API)
├── api/                     # Task 7 cloud model API
│   ├── main.py              # /health, /predict (X-API-Key)
│   ├── Dockerfile(.lite)    # TensorFlow image / small TFLite image (deployed)
│   ├── demo_calls.ps1       # calls the deployed API from the terminal (key read from .env)
│   └── visionguard_api.postman_collection.json   # Postman requests for Figs 7.3-7.5
├── tests/                   # pytest for the app incl. remote backend (no TensorFlow, no network)
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
python -m src.export_tflite                             # TFLite model for the cloud API
```

The app has four pages: **/** upload, result, **/history** (every prediction from the database, also `?format=json`) and **/health** (model, database and, in remote mode, cloud-API status).

## Model API (Task 7)

A small Flask service ([api/main.py](api/main.py)) that serves the champion model over HTTP. The model is **built into** the image, so a cloud host needs nothing else. Two images, same API code:

| Image | Model / runtime | Size | RAM | Use |
|---|---|---|---|---|
| [api/Dockerfile](api/Dockerfile) | `champion.keras` + tensorflow-cpu | 2.2 GB | ~1 GB | hosts with ≥ 2 GB RAM |
| [api/Dockerfile.lite](api/Dockerfile.lite) | `champion.tflite` + ai-edge-litert | 0.5 GB | ~90 MB | **deployed** (free 512 MB tier) |

`python -m src.export_tflite` converts the model and checks it on the test split: identical labels on all 147 images, max P(AI) difference 1e-5 ([reports/metrics/tflite_check.json](reports/metrics/tflite_check.json)). JPEGs are decoded with `simplejpeg(fastdct=True)` so the pixels are bit-identical to `tf.io.decode_jpeg`.

| Method | Path | Auth | Returns |
|---|---|---|---|
| `GET` | `/health` | none | `{"status": "ok", "model_loaded": true, "model_name": ..., "model_version": ..., "threshold": ...}` |
| `POST` | `/predict` | header `X-API-Key` | prediction JSON (below) |

`/predict` takes a multipart form field `image` (max 5 MB):

```bash
curl -H "X-API-Key: <key>" -F "image=@data/raw/AiArtData/images89.jpg" https://api-23l3008-a2-ops.onrender.com/predict
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
docker build -f api/Dockerfile.lite -t 23l3008-a2-ops-api:lite .   # small TFLite image
docker compose --profile api up -d api                    # or next to the web app
```

### Deployment (Render)

Hugging Face Docker Spaces now need a paid plan (402) and Google Cloud Run needs a verified card, so the API runs on **Render's free web service** (512 MB RAM, no card) as the TFLite image:

```bash
python -m src.export_tflite                                                   # models/champion.tflite
docker build --provenance=false -f api/Dockerfile.lite -t <dockerhub-user>/23l3008-a2-ops-api:lite .
docker push <dockerhub-user>/23l3008-a2-ops-api:lite
```

Render → New → Web Service → Existing Image `docker.io/<dockerhub-user>/23l3008-a2-ops-api:lite`, instance type Free, env vars `API_KEY` (secret) and `DECISION_THRESHOLD=0.8432`, health check path `/health`. Render supplies `PORT` and HTTPS. Logs: Render → service → Logs. Check it from a terminal with `powershell -ExecutionPolicy Bypass -File .\api\demo_calls.ps1`.

## Flask → cloud API (Task 8)

Set in `.env` and restart the app (`MODEL_API_KEY` must equal the `API_KEY` on Render):

```
INFERENCE_BACKEND=remote
MODEL_API_URL=https://api-23l3008-a2-ops.onrender.com/predict
MODEL_API_KEY=<key>
MODEL_API_TIMEOUT=90
```

The upload route doesn't change; `RemotePredictor` ([app/inference.py](app/inference.py)) replaces the local model:

- multipart POST of the image to `MODEL_API_URL` over a reused `requests.Session`, header `X-API-Key`, user agent `visionguard-web/1.0` (visible in the API logs)
- timeouts: 10 s connect, 90 s read (a sleeping free-tier API needs ~1 min to wake)
- timeout / connection error / 401 / other non-200 → clear error page (HTTP 502), nothing written to the database
- label and confidence are recomputed from the API's threshold, so local and remote decide identically
- every prediction is still stored in the database with `backend = remote`; the result page shows "Served by: cloud model API (remote)"
- `/health` adds `"model_api": "ok"` after pinging the API

Latency on the same 10 test images (`python -m src.latency_compare`, [reports/metrics/latency_local_vs_remote.json](reports/metrics/latency_local_vs_remote.json)):

| Backend | Median latency | Notes |
|---|---|---|
| local (Keras in the app) | ~232 ms | needs TensorFlow + ~1 GB RAM in the web app |
| remote (Render API) | ~631 ms | network upload dominates; same label on 10/10 images; first call after idle up to ~1 min |

## Notes

- Positive class = AI-generated (label 1). The model outputs P(AI).
- Training and serving use the same `app/preprocess.py`. Normalisation is part of the saved model.
- Secrets live only in `.env` (git- and docker-ignored). See `.env.example`. On Render the API key is a service environment variable.
- `docker compose` publishes port 5000 too: stop it (`docker compose stop`) before running `python -m app` locally, or the browser reaches the container instead.
