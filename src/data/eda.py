# src/data/eda.py
from __future__ import annotations

import polars as pl

from src.data.load_uci import load_adult

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


TARGET_COL = "income"
POS_LABEL = ">50K"


def basic_info(df: pl.DataFrame) -> None:
    print("\n=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    for c, t in zip(df.columns, df.dtypes):
        print(f"  {c:<20} {t}")


def missing_values(df: pl.DataFrame) -> pl.DataFrame:
    print("\n=== MISSING VALUES ===")
    miss = df.select([pl.col(c).null_count().alias(c) for c in df.columns]).transpose(
        include_header=True
    )

    miss = miss.with_columns(
        (pl.col("column_0") / df.height).alias("missing_frac")
    ).rename({"column": "feature", "column_0": "missing_count"})

    print(
        miss.filter(pl.col("missing_count") > 0).sort("missing_frac", descending=True)
    )
    return miss


def target_distribution(df: pl.DataFrame) -> None:
    print("\n=== TARGET DISTRIBUTION ===")
    dist = (
        df.group_by(TARGET_COL)
        .len()
        .with_columns((pl.col("len") / df.height).alias("fraction"))
        .sort("len", descending=True)
    )
    print(dist)


def numeric_summary(df: pl.DataFrame, numeric_cols: list[str]) -> None:
    print("\n=== NUMERIC SUMMARY ===")
    summary = df.select(numeric_cols).describe()
    print(summary)


def categorical_cardinality(df: pl.DataFrame, categorical_cols: list[str]) -> None:
    print("\n=== CATEGORICAL CARDINALITY ===")
    card = pl.DataFrame(
        {
            "feature": categorical_cols,
            "n_unique": [df[c].n_unique() for c in categorical_cols],
        }
    ).sort("n_unique", descending=True)
    print(card)


def numeric_correlation(df: pl.DataFrame, numeric_cols: list[str]) -> None:
    print("\n=== NUMERIC CORRELATION (Pearson, NaN dropped) ===")

    df_num = df.select(numeric_cols).drop_nulls()

    dropped = df.height - df_num.height
    if dropped > 0:
        print(f"Dropped {dropped} row(s) with nulls for correlation analysis")

    corr = df_num.corr()
    print(corr)


def save_numeric_distributions(
    df: pl.DataFrame,
    numeric_cols: Iterable[str],
    out_dir: str | Path = r"./reports/figures/numeric_distributions",
    bins: int = 50,
    log1p_cols: Iterable[str] = ("capital-gain", "capital-loss"),
) -> None:
    """
    Save histogram plots for all numeric features in `numeric_cols`.

    - Saves: <col>.png
    - Optionally saves log1p version for skewed cols: <col>__log1p.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Work on a null-free numeric view (per-column)
    for col in numeric_cols:
        if col not in df.columns:
            continue

        s = df.select(pl.col(col).drop_nulls()).to_series()
        if s.len() == 0:
            continue

        x = s.to_numpy()

        # --- linear scale ---
        plt.figure()
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.hist(x, bins=bins)
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}.png", dpi=160)
        plt.close()

        # --- log1p scale for selected skewed features (only if non-negative) ---
        if col in set(log1p_cols) and np.all(x >= 0):
            x_log = np.log1p(x)

            plt.figure()
            plt.title(f"Distribution: {col} (log1p)")
            plt.xlabel(f"log1p({col})")
            plt.ylabel("count")
            plt.hist(x_log, bins=bins)
            plt.tight_layout()
            plt.savefig(out_dir / f"{col}__log1p.png", dpi=160)
            plt.close()


def run_eda() -> None:
    train_df, _ = load_adult()

    basic_info(train_df)

    _ = missing_values(train_df)

    target_distribution(train_df)

    numeric_cols = [
        c
        for c, t in zip(train_df.columns, train_df.dtypes)
        if t in (pl.Int64, pl.Float64) and c != TARGET_COL
    ]

    categorical_cols = [
        c
        for c, t in zip(train_df.columns, train_df.dtypes)
        if t == pl.Utf8 and c != TARGET_COL
    ]

    save_numeric_distributions(train_df, numeric_cols)
    print("Saved plots to reports/figures/numeric_distributions/")

    numeric_summary(train_df, numeric_cols)
    categorical_cardinality(train_df, categorical_cols)
    numeric_correlation(train_df, numeric_cols)


if __name__ == "__main__":
    # uv run -m src.data.eda
    run_eda()
