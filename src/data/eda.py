# src/data/eda.py
from __future__ import annotations

import polars as pl

from src.data.load_uci import load_adult

from pathlib import Path
from typing import Iterable

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

    # check high correlations
    high_corrs = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i >= j:
                continue
            val = corr[i, j]
            if abs(val) >= 0.5:
                high_corrs.append((col1, col2, val))

    if not high_corrs:
        print("No high correlations found (|corr| >= 0.5)")
    else:
        for col1, col2, val in high_corrs:
            print(f"  {col1} <-> {col2}: {val:.3f}")

    print(corr)


def detect_rare_categories(
    df: pl.DataFrame,
    cols: Iterable[str],
    *,
    min_count: int = 50,
    min_frac: float | None = None,
    include_null: bool = False,
    top_k: int | None = None,
) -> pl.DataFrame:
    """Detect rare categories/values for both string and integer columns.

    Parameters
    ----------
    df:
        Input dataframe.
    cols:
        Columns to analyze (categorical-like), can be Utf8, Int, etc.
    min_count:
        Mark a value as rare if count < min_count.
    min_frac:
        Optional: mark a value as rare if fraction < min_frac.
        If provided, rarity is (count < min_count) OR (fraction < min_frac).
    include_null:
        If True, consider null as its own category; otherwise ignore nulls.
    top_k:
        Optional: keep only top_k most frequent values per feature in the output
        (rare flag still computed before slicing). Useful for very high-cardinality cols.

    Returns
    -------
    pl.DataFrame
        Long-form table with:
        - feature
        - value (as string)
        - count
        - fraction
        - is_rare
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pl.DataFrame(
            {"feature": [], "value": [], "count": [], "fraction": [], "is_rare": []}
        )

    # Denominator for fractions: null-free rows per column if include_null=False,
    # otherwise total rows.
    n_total = df.height

    out_frames: list[pl.DataFrame] = []

    for col in cols:
        s = pl.col(col)
        if include_null:
            base = df
            denom = n_total
            val_expr = s
        else:
            base = df.filter(s.is_not_null())
            denom = base.height
            val_expr = s

        if denom == 0:
            continue

        counts = (
            base.group_by(val_expr)
            .len()
            .rename({col: "value", "len": "count"})
            .with_columns((pl.col("count") / denom).alias("fraction"))
            .with_columns(pl.lit(col).alias("feature"))
        )

        # Cast value to string for uniform printing across dtypes
        counts = counts.with_columns(pl.col("value").cast(pl.Utf8))

        # rare condition
        rare_expr = pl.col("count") < min_count
        if min_frac is not None:
            rare_expr = rare_expr | (pl.col("fraction") < min_frac)

        counts = (
            counts.with_columns(rare_expr.alias("is_rare"))
            .select(["feature", "value", "count", "fraction", "is_rare"])
            .sort(["feature", "count"], descending=[False, True])
        )

        if top_k is not None:
            counts = (
                counts.with_columns(
                    pl.col("count")
                    .rank("dense", descending=True)
                    .over("feature")
                    .alias("_rk")
                )
                .filter(pl.col("_rk") <= top_k)
                .drop("_rk")
            )

        out_frames.append(counts)

    if not out_frames:
        return pl.DataFrame(
            {"feature": [], "value": [], "count": [], "fraction": [], "is_rare": []}
        )

    res = pl.concat(out_frames, how="vertical")

    # Pretty print (optional, but consistent with your EDA style)
    print("\n=== RARE CATEGORIES ===")
    print(
        res.filter(pl.col("is_rare")).sort(
            ["feature", "count"], descending=[False, False]
        )
    )

    return res


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
        plt.show()
        plt.savefig(out_dir / f"{col}.png", dpi=80)
        plt.close()


def run_eda(
    train_df: pl.DataFrame | None = None,
) -> None:
    if train_df is None:
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

    detect_rare_categories(
        train_df,
        categorical_cols,
        min_count=40,
    )

    numeric_summary(train_df, numeric_cols)
    categorical_cardinality(train_df, categorical_cols)
    numeric_correlation(train_df, numeric_cols)


if __name__ == "__main__":
    # uv run -m src.data.eda
    run_eda()
