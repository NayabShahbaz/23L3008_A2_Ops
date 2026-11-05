"""Task 1.2 - dataset audit.

Walks data/raw/<ClassFolder>/..., opens every file with Pillow and records:
class, file size, width, height, colour mode, format, SHA-256, readable flag.
Then reports per-class counts, min/median/max stats, corrupt files and exact
duplicates.

Run:  python -m src.audit
Outputs: reports/metrics/audit.csv, audit_summary.csv, duplicates.csv
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils import ROOT, load_params


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"width": None, "height": None, "mode": None, "format": None, "readable": True, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()                      # catches truncated / broken files
        with Image.open(path) as im:         # verify() invalidates the object; reopen
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as e:  # noqa: BLE001 - we want to record *any* failure
        row.update(readable=False, error=f"{type(e).__name__}: {e}"[:200])
    return row


def main() -> None:
    p = load_params()
    raw = ROOT / p["data"]["raw_dir"]
    class_map: dict = p["data"]["class_map"]
    out_dir = ROOT / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in raw.rglob("*") if f.is_file() and f.name != ".gitkeep"]
    if not files:
        raise SystemExit(f"No files found in {raw}. Download the Kaggle dataset there first.")

    rows = []
    for f in tqdm(files, desc="auditing"):
        top = f.relative_to(raw).parts[0]
        label = class_map.get(top)
        if label is None:
            continue  # ignore anything that is not a class folder
        rows.append({
            "path": f.relative_to(ROOT).as_posix(),
            "class": label,
            "file_size_kb": round(f.stat().st_size / 1024, 2),
            "sha256": sha256(f),
            **inspect(f),
        })

    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    df.to_csv(ROOT / p["data"]["audit_csv"], index=False)

    ok = df[df.readable]
    summary = ok.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    summary.insert(0, "count_readable", ok.groupby("class").size())
    summary.insert(0, "count_total", df.groupby("class").size())
    summary.to_csv(out_dir / "audit_summary.csv")

    dups = df[df.sha256.duplicated(keep=False)].sort_values("sha256")
    dups.to_csv(out_dir / "duplicates.csv", index=False)
    cross_class = dups.groupby("sha256")["class"].nunique().gt(1).sum()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== Counts per class ===")
    print(summary[["count_total", "count_readable"]].droplevel(1, axis=1).to_string())
    for col, unit in [("width", "px"), ("height", "px"), ("file_size_kb", "KB")]:
        print(f"\n=== {col} ({unit}): min / median / max ===")
        print(summary[col].to_string())

    ext = df.path.str.rsplit(".", n=1).str[-1].str.lower().replace({"jpg": "jpeg"})
    mismatch = df[df.readable & (ext != df.format.str.lower())]
    print("\n=== Formats / colour modes ===")
    print(pd.crosstab(ok["class"], ok["format"]))
    print(pd.crosstab(ok["class"], ok["mode"]))
    print(f"\nTotal files           : {len(df)}")
    print(f"Unreadable / corrupt  : {(~df.readable).sum()}")
    print(f"Exact duplicate files : {int(df.is_duplicate.sum())} (extra copies)")
    print(f"Hashes in BOTH classes: {int(cross_class)}")
    print(f"Extension != real fmt : {len(mismatch)}")
    for _, r in mismatch.iterrows():
        print(f"    {r.path}  (actually {r.format})")
    for _, r in df[df.is_duplicate].iterrows():
        twin = df[(df.sha256 == r.sha256) & ~df.is_duplicate].path.iloc[0]
        print(f"    duplicate: {r.path}  ==  {twin}")
    print(f"\nSaved: {p['data']['audit_csv']}, reports/metrics/audit_summary.csv, reports/metrics/duplicates.csv")


if __name__ == "__main__":
    main()
