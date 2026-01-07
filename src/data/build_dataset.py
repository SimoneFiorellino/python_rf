# src/data/build_dataset.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from src.data.eda import run_eda
from src.data.load_uci import load_adult
from src.data.io import load_processed
from src.data.features import (
    FeatureConfig,
    binarize_target,
    one_hot_align_train_test,
)

TARGET_COL = "income"
CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

PROCESSED_DIR = Path("./data/processed")

X_TRAIN_OUT = PROCESSED_DIR / "X_train.parquet"
Y_TRAIN_OUT = PROCESSED_DIR / "y_train.parquet"
X_TEST_OUT = PROCESSED_DIR / "X_test.parquet"
Y_TEST_OUT = PROCESSED_DIR / "y_test.parquet"
META_OUT = PROCESSED_DIR / "metadata.json"


def _ensure_out_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _apply_common_steps(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    config: FeatureConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cat_cols = [c for c, t in train_df.schema.items() if t == pl.Utf8]

    train_df = train_df.drop(TARGET_COL)
    test_df = test_df.drop(TARGET_COL)
    train_df, test_df = one_hot_align_train_test(
        train_df, test_df, cat_cols, config=config
    )
    return train_df, test_df


def _align_test_to_train_columns(
    X_test: pl.DataFrame, train_columns: list[str]
) -> pl.DataFrame:
    # 1) aggiungi colonne mancanti (categorie viste in train ma non in test) come 0
    missing = [c for c in train_columns if c not in X_test.columns]
    if missing:
        X_test = X_test.with_columns(
            [pl.lit(0).cast(pl.Int8).alias(c) for c in missing]
        )

    # 2) rimuovi colonne extra (categorie apparse solo in test)
    extra = [c for c in X_test.columns if c not in train_columns]
    if extra:
        X_test = X_test.drop(extra)

    # 3) stesso ordine colonne
    return X_test.select(train_columns)


def _validate_no_strings(df: pl.DataFrame, name: str) -> None:
    string_cols = [c for c, t in df.schema.items() if t == pl.Utf8]
    if string_cols:
        raise ValueError(f"{name}: found Utf8 columns after processing: {string_cols}")


def build_dataset(
    *,
    config: FeatureConfig | None = None,
    target_col: str = TARGET_COL,
    min_count: int = 50,
    min_frac: float | None = None,
    other_value: int = -1,
) -> None:
    _ensure_out_dir()

    if config is None:
        config = FeatureConfig()

    raw_train_df, raw_test_df = load_adult()

    # --- remove rows with missing target ---
    raw_train_df = raw_train_df.filter(pl.col(target_col).is_not_null())
    raw_test_df = raw_test_df.filter(pl.col(target_col).is_not_null())

    # --- Target binarization ---
    y_train = binarize_target(raw_train_df, target_col, config)
    y_test = binarize_target(raw_test_df, target_col, config)

    # --- Train: common steps ---
    X_train, X_test = _apply_common_steps(raw_train_df, raw_test_df, config)

    # --- Save ---
    X_train.write_parquet(X_TRAIN_OUT)
    y_train.write_parquet(Y_TRAIN_OUT)
    X_test.write_parquet(X_TEST_OUT)
    y_test.write_parquet(Y_TEST_OUT)

    metadata = {
        "raw": {"train_rows": raw_train_df.height, "test_rows": raw_test_df.height},
        "processed": {"train_rows": X_train.height, "test_rows": X_test.height},
        "X_cols": X_train.columns,
        "target_col": target_col,
        "config": asdict(config),
        "outputs": {
            "X_train": str(X_TRAIN_OUT),
            "y_train": str(Y_TRAIN_OUT),
            "X_test": str(X_TEST_OUT),
            "y_test": str(Y_TEST_OUT),
        },
    }
    META_OUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=== BUILD DATASET ===")
    print(
        f"Saved X_train: {X_TRAIN_OUT}  (rows={X_train.height}, cols={X_train.width})"
    )
    print(
        f"Saved y_train: {Y_TRAIN_OUT}  (rows={y_train.height}, cols={y_train.width})"
    )
    print(f"Saved X_test : {X_TEST_OUT}   (rows={X_test.height}, cols={X_test.width})")
    print(f"Saved y_test : {Y_TEST_OUT}   (rows={y_test.height}, cols={y_test.width})")
    print(f"Saved meta   : {META_OUT}")


if __name__ == "__main__":
    # uv run -m src.data.build_dataset
    build_dataset()
    train_df, _ = load_processed()
    run_eda(train_df=train_df)
