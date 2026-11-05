"""Create stratified train/val/test split lists from the audit table.

Drops: unreadable files, duplicate copies (keeps first), hashes that appear in
both classes (label noise), and formats tf.io.decode_image cannot read.

Run:  python -m src.split
Outputs: data/splits/{train,val,test}.csv   (columns: path,label,class)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ROOT, load_params

TF_DECODABLE = {"JPEG", "PNG", "BMP", "GIF"}


def main() -> None:
    p = load_params()
    d = p["data"]
    df = pd.read_csv(ROOT / d["audit_csv"])
    n0 = len(df)

    df = df[df.readable]
    conflict = df.groupby("sha256")["class"].nunique()
    df = df[~df.sha256.isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("sha256", keep="first")
    df = df[df.format.isin(TF_DECODABLE)]
    df["label"] = df["class"].map(d["label_ids"])
    print(f"Kept {len(df)} / {n0} files after cleaning")

    seed = p["seed"]
    trainval, test = train_test_split(df, test_size=d["test_frac"], stratify=df.label, random_state=seed)
    rel_val = d["val_frac"] / (1 - d["test_frac"])
    train, val = train_test_split(trainval, test_size=rel_val, stratify=trainval.label, random_state=seed)

    out = ROOT / d["splits_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["path", "label", "class"]].to_csv(out / f"{name}.csv", index=False)
        print(f"{name:<5}: {len(part):>5}  {part['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
