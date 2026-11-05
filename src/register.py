"""Task 3.6 - register a run's model and (optionally) set the 'champion' alias.

Run twice to create two versions, then alias the best one:
  python -m src.register --run-id <RUN_ID_A>
  python -m src.register --run-id <RUN_ID_B> --alias champion
Download the champion .keras file for serving (so the app never calls DagsHub per request):
  python -m src.register --download-champion
"""
import argparse
import shutil
from pathlib import Path

from src.utils import ROOT, load_params, setup_mlflow


def main() -> None:
    p = load_params()["registry"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--artifact", default="model", help="artifact path of the logged model")
    ap.add_argument("--name", default=p["model_name"])
    ap.add_argument("--alias", default=None)
    ap.add_argument("--download-champion", action="store_true")
    a = ap.parse_args()

    mlflow = setup_mlflow()
    client = mlflow.MlflowClient()

    if a.download_champion:
        mv = client.get_model_version_by_alias(a.name, p["alias"])
        local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file",
                                                    dst_path=str(ROOT / "models" / "_dl"))
        src = next(Path(local).glob("*.keras"))
        dst = ROOT / "models" / "champion.keras"
        shutil.copy(src, dst)
        print(f"Champion = {a.name} v{mv.version} (run {mv.run_id}) -> {dst}")
        return

    mv = mlflow.register_model(f"runs:/{a.run_id}/{a.artifact}", a.name)
    print(f"Registered {a.name} version {mv.version}")
    if a.alias:
        client.set_registered_model_alias(a.name, a.alias, mv.version)
        print(f"Alias '{a.alias}' -> version {mv.version}")


if __name__ == "__main__":
    main()
