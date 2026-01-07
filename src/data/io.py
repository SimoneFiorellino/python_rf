# src/data/io.py
from __future__ import annotations

import polars as pl
from pathlib import Path


PROCESSED_DIR = Path("./data/processed")


def load_processed() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load processed train/test and return joined dfs with target included."""
    X_train = pl.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pl.read_parquet(PROCESSED_DIR / "y_train.parquet")
    X_test = pl.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pl.read_parquet(PROCESSED_DIR / "y_test.parquet")

    train_df = pl.concat([X_train, y_train], how="horizontal")
    test_df = pl.concat([X_test, y_test], how="horizontal")
    return train_df, test_df
