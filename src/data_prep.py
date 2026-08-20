"""Combine HAM10000 image folders and build lesion-level train/val/test splits."""

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
IMAGES_DIR = RAW_DIR / "images"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"

MALIGNANT_DX = {"mel", "bcc", "akiec"}
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (~0.15) goes to test
RANDOM_STATE = 42


def combine_image_folders() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    part_dirs = [RAW_DIR / "HAM10000_images_part_1", RAW_DIR / "HAM10000_images_part_2"]
    n_linked = 0
    for part_dir in part_dirs:
        for img_path in part_dir.glob("*.jpg"):
            dest = IMAGES_DIR / img_path.name
            if not dest.exists():
                try:
                    dest.symlink_to(img_path)
                except OSError:
                    shutil.copy2(img_path, dest)
                n_linked += 1
    print(f"combine_image_folders: {n_linked} new images linked into {IMAGES_DIR}")


def build_labeled_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_PATH)
    df["label"] = df["dx"].apply(lambda dx: "malignant" if dx in MALIGNANT_DX else "benign")
    df["image_path"] = df["image_id"].apply(lambda iid: str(IMAGES_DIR / f"{iid}.jpg"))
    return df


def lesion_level_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split lesion_id groups first so no lesion appears in more than one split.
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=RANDOM_STATE)
    train_idx, rest_idx = next(gss1.split(df, groups=df["lesion_id"]))
    train_df = df.iloc[train_idx]
    rest_df = df.iloc[rest_idx]

    val_frac_of_rest = VAL_FRAC / (1 - TRAIN_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac_of_rest, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["lesion_id"]))
    val_df = rest_df.iloc[val_idx]
    test_df = rest_df.iloc[test_idx]

    return train_df, val_df, test_df


def assert_no_lesion_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_ids = set(train_df["lesion_id"])
    val_ids = set(val_df["lesion_id"])
    test_ids = set(test_df["lesion_id"])
    assert not (train_ids & val_ids), "lesion_id leakage between train and val"
    assert not (train_ids & test_ids), "lesion_id leakage between train and test"
    assert not (val_ids & test_ids), "lesion_id leakage between val and test"


def main() -> None:
    combine_image_folders()

    df = build_labeled_metadata()
    train_df, val_df, test_df = lesion_level_split(df)
    assert_no_lesion_leakage(train_df, val_df, test_df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "lesion_id", "dx", "label", "image_path"]
    train_df[cols].to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df[cols].to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df[cols].to_csv(SPLITS_DIR / "test.csv", index=False)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_images = len(split_df)
        n_lesions = split_df["lesion_id"].nunique()
        malignant_frac = (split_df["label"] == "malignant").mean()
        print(f"{name}: {n_images} images, {n_lesions} lesions, {malignant_frac:.1%} malignant")


if __name__ == "__main__":
    main()
