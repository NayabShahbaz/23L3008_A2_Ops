"""Task 8 - latency of the SAME images through the local model vs the cloud API.

    python -m src.latency_compare            # 10 test images (5 AI + 5 real), settings from .env
    python -m src.latency_compare --n 20

local  = Keras model in this process (LocalPredictor, models/champion.keras)
remote = HTTPS call to MODEL_API_URL (RemotePredictor) = network + API inference
The first remote call is reported separately (it may include a free-tier cold start).
Writes reports/metrics/latency_local_vs_remote.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from app.config import Config  # noqa: E402  (loads .env)
from app.inference import LocalPredictor, RemotePredictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--url", default=Config.MODEL_API_URL)
    a = ap.parse_args()

    rows = pd.read_csv("data/splits/test.csv")
    per_class = max(1, a.n // 2)
    paths = pd.concat([rows[rows["class"] == c].head(per_class) for c in ("ai", "real")])["path"].tolist()

    local = LocalPredictor(Config.MODEL_PATH, Config.IMG_SIZE, Config.DECISION_THRESHOLD,
                           Config.MODEL_NAME, Config.MODEL_VERSION)
    remote = RemotePredictor(a.url, Config.MODEL_API_KEY, Config.MODEL_API_TIMEOUT)

    first = remote.predict(Path(paths[0]).read_bytes())          # wake-up / connection set-up
    print(f"first remote call (cold start / TLS handshake): {first.latency_ms:.0f} ms\n")

    print(f"{'image':<46} {'local ms':>9} {'remote ms':>10}  local label   remote label")
    t_local, t_remote, same = [], [], 0
    for p in paths:
        data = Path(p).read_bytes()
        lp, rp = local.predict(data), remote.predict(data)
        t_local.append(lp.latency_ms)
        t_remote.append(rp.latency_ms)
        same += lp.label == rp.label
        print(f"{Path(p).name[:45]:<46} {lp.latency_ms:>9.0f} {rp.latency_ms:>10.0f}  {lp.label:<13} {rp.label}")

    report = {
        "images": len(paths),
        "local_median_ms": round(statistics.median(t_local), 1),
        "remote_median_ms": round(statistics.median(t_remote), 1),
        "remote_first_call_ms": first.latency_ms,
        "same_label": f"{same}/{len(paths)}",
        "api_url": a.url,
    }
    print("\n" + json.dumps(report, indent=2))
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics/latency_local_vs_remote.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
