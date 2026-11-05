"""Task 1.3 - visual EDA: sample grid per class + width/height/file-size histograms.

Run:  python -m src.eda     (after src.audit)
Outputs: reports/figures/sample_grid.png, hist_width.png, hist_height.png, hist_file_size.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.utils import ROOT, load_params  # noqa: E402

FIG = ROOT / "reports" / "figures"


def sample_grid(df: pd.DataFrame, per_class: int = 8, seed: int = 42) -> None:
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.2 * per_class, 2.4 * len(classes)))
    for r, cls in enumerate(classes):
        sample = df[df["class"] == cls].sample(per_class, random_state=seed)
        for c, (_, row) in enumerate(sample.iterrows()):
            ax = axes[r, c]
            with Image.open(ROOT / row.path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{cls}\n{row.width}x{row.height}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Random samples per class", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "sample_grid.png", dpi=150)
    plt.close(fig)


def hist(df: pd.DataFrame, col: str, fname: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, part in df.groupby("class"):
        ax.hist(part[col], bins=50, alpha=0.6, label=f"{cls} (n={len(part)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(f"{xlabel} per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    p = load_params()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / p["data"]["audit_csv"])
    df = df[df.readable & ~df.is_duplicate]
    sample_grid(df, seed=p["seed"])
    hist(df, "width", "hist_width.png", "Width (px)")
    hist(df, "height", "hist_height.png", "Height (px)")
    hist(df, "file_size_kb", "hist_file_size.png", "File size (KB)")
    print(f"Figures saved in {FIG}")


if __name__ == "__main__":
    main()
