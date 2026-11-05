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
│   └── predict_from_registry.py  # 3.6 load by alias, classify one image
├── app/                     # Task 4/5 Flask app (upload, result, history, health) + DB
├── api/                     # Task 7 cloud model API (+ its own Dockerfile)
├── tests/                   # pytest for the app (no TensorFlow needed)
├── labs/                    # toy parts: DVC, MLflow, Keras Tuner, GitHub Actions
├── Dockerfile               # 6.2 web image
├── docker-compose.yml       # 6.3 web + postgres
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

## Notes

- Positive class = AI-generated (label 1). The model outputs P(AI).
- Training and serving use the same `app/preprocess.py`. Normalisation is part of the saved model.
- Secrets live only in `.env` (git- and docker-ignored). See `.env.example`.
