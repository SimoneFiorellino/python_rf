# src/data/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl
import numpy as np


""" Feature engineering utilities.
Includes:
- Converting String into numeric features
- Fix missing values
- Rare category bucketing
- Binarizing target variable
"""


_UNSIGNED_INT_DTYPES = {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}


@dataclass(frozen=True)
class FeatureConfig:
    convert_strings_to_numeric: bool = False
    fix_missing_values: bool = False
    remove_string_cols: list[str] = None  # default -> []
    rare_category_bucketing: bool = False
    binarize_target: bool = True
    to_dummies_all_strings: bool = True


def binarize_target(
    df: pl.DataFrame,
    target_col: str,
    config: FeatureConfig,
) -> pl.DataFrame:
    if not config.binarize_target:
        return df

    return df.select(
        pl.when(pl.col(target_col).cast(pl.Utf8).str.contains(">50K"))
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias(target_col)
    )


def one_hot_align_train_test(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    cat_cols: list[str],
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-hot encode train and test using Polars, then align columns:
    - create dummies on train and test separately
    - add missing columns (filled with 0)
    - reorder columns identically
    """
    if not config.to_dummies_all_strings:
        return X_train.to_numpy(), X_test.to_numpy()

    X_train_ohe = X_train.to_dummies(columns=cat_cols)
    X_test_ohe = X_test.to_dummies(columns=cat_cols)

    train_cols = set(X_train_ohe.columns)
    test_cols = set(X_test_ohe.columns)
    all_cols = sorted(train_cols | test_cols)

    # add missing cols with zeros
    missing_in_train = [c for c in all_cols if c not in train_cols]
    missing_in_test = [c for c in all_cols if c not in test_cols]

    if missing_in_train:
        X_train_ohe = X_train_ohe.with_columns(
            [pl.lit(0).cast(pl.UInt8).alias(c) for c in missing_in_train]
        )
    if missing_in_test:
        X_test_ohe = X_test_ohe.with_columns(
            [pl.lit(0).cast(pl.UInt8).alias(c) for c in missing_in_test]
        )

    # same order
    X_train_ohe = X_train_ohe.select(all_cols)
    X_test_ohe = X_test_ohe.select(all_cols)

    return X_train_ohe, X_test_ohe


def _infer_categorical_numeric_cols(
    df: pl.DataFrame,
    *,
    exclude: Iterable[str] = (),
    max_unique: int = 50,
    max_unique_frac: float = 0.05,
) -> list[str]:
    """Heuristic: numeric columns with relatively few unique values are treated as categorical-like."""
    exclude_set = set(exclude)  # exclude these columns from consideration

    # Identify numeric columns
    numeric_cols: list[str] = [
        c
        for c, t in df.schema.items()
        if c not in exclude_set
        and t
        in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
    ]

    if df.height == 0:  # if no rows, return empty list
        return []

    out: list[str] = []  # init: inferred categorical-like numeric columns
    n = df.height  # total number of rows

    for c in numeric_cols:  # check each numeric column
        nu = df.select(pl.col(c).n_unique()).item()  # number of unique values
        if nu <= max_unique and (nu / n) <= max_unique_frac:  # check thresholds
            out.append(c)  # add to output list

    return out  # return inferred categorical-like numeric columns


def fit_rare_category_bucketing(
    df: pl.DataFrame,
    *,
    categorical_cols: list[str] | None = None,
    exclude: Iterable[str] = (),
    min_count: int = 50,
    min_frac: float | None = None,
    include_null: bool = False,
) -> dict[str, set[int]]:
    """Fit rare-category bucketing on a dataframe.

    Returns a mapping: {column -> set(rare_values)}.
    Intended to be fit on TRAIN only.
    """
    if df.height == 0:  # no rows, return empty map
        return {}

    if categorical_cols is None:  # infer categorical-like numeric columns
        categorical_cols = _infer_categorical_numeric_cols(df, exclude=exclude)

    rare_map: dict[str, set[int]] = {}  # init: rare category mapping

    for col in categorical_cols:
        if col not in df.columns:  # skip missing columns
            continue

        s = pl.col(col)  # s is the specific column for each iteration

        if include_null:  # consider nulls as a category
            base = df
            denom = df.height  # total rows
            val_expr = s
        else:  # ignore nulls
            base = df.filter(s.is_not_null())
            denom = base.height
            val_expr = s

        if denom == 0:  # no valid rows,
            continue

        counts = (
            base.group_by(val_expr)
            .len()
            .rename({col: "value", "len": "count"})
            .with_columns((pl.col("count") / denom).alias("fraction"))
        )  # compute counts and fractions

        rare_cond = pl.col("count") < min_count  # base condition for rarity
        if min_frac is not None:  # add fraction condition if provided
            rare_cond = rare_cond | (pl.col("fraction") < min_frac)

        rare_values = (
            counts.filter(rare_cond)
            .select(pl.col("value").cast(pl.Int64))
            .to_series()
            .to_list()
        )  # list of rare values

        if rare_values:  # if there are rare values, add to map
            rare_map[col] = set(int(v) for v in rare_values if v is not None)

    return rare_map  # return the rare category mapping


def apply_rare_category_bucketing(
    df: pl.DataFrame,
    rare_map: dict[str, set[int]],
    *,
    other_value: int = -1,
) -> pl.DataFrame:
    """Apply a previously-fitted rare_map to a dataframe (train/val/test).

    NOTE: Polars categorical physical codes can be unsigned (e.g. u32).
    If a column is unsigned, cast it to Int64 so we can safely use -1 as OTHER.
    """
    if not rare_map:
        return df

    exprs: list[pl.Expr] = []

    for c in df.columns:
        if c in rare_map and rare_map[c]:
            vals = list(rare_map[c])
            out_dtype = df.schema[c]

            # unsigned ints cannot represent -1
            if out_dtype in _UNSIGNED_INT_DTYPES:
                col_expr = pl.col(c).cast(pl.Int64)
                exprs.append(
                    pl.when(col_expr.is_in(vals))
                    .then(pl.lit(other_value).cast(pl.Int64))
                    .otherwise(col_expr)
                    .alias(c)
                )
            else:
                exprs.append(
                    pl.when(pl.col(c).cast(pl.Int64).is_in(vals))
                    .then(pl.lit(other_value).cast(out_dtype))
                    .otherwise(pl.col(c))
                    .alias(c)
                )
        else:
            exprs.append(pl.col(c))

    return df.select(exprs)


def rare_category_bucketing(
    df: pl.DataFrame,
    config: "FeatureConfig",
    *,
    rare_map: dict[str, set[int]] | None = None,
    categorical_cols: list[str] | None = None,
    exclude: Iterable[str] = (),
    min_count: int = 100,
    min_frac: float | None = None,
    other_value: int = -1,
) -> tuple[pl.DataFrame, dict[str, set[int]]]:
    """Convenience wrapper.

    - If rare_map is None -> FIT on df (use ONLY on train).
    - Always APPLY returned map to df.

    Returns: (transformed_df, fitted_rare_map)
    """
    if not getattr(config, "rare_category_bucketing", False):  # skip if disabled
        return df, (rare_map or {})

    if rare_map is None:
        rare_map = fit_rare_category_bucketing(
            df,
            categorical_cols=categorical_cols,
            exclude=exclude,
            min_count=min_count,
            min_frac=min_frac,
            include_null=False,
        )  # fit on df

    df_out = apply_rare_category_bucketing(
        df, rare_map, other_value=other_value
    )  # apply to df
    return df_out, rare_map


def add_numeric_features(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Convert string columns to numeric (categorical) features.

    - String columns are cast to categorical and then to their physical (integer) representation.
    - Numeric columns are left untouched.
    """
    if not config.convert_strings_to_numeric:
        return df

    exprs: list[pl.Expr] = []

    for col, dtype in df.schema.items():
        if dtype == pl.Utf8:
            exprs.append(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        else:
            exprs.append(pl.col(col))

    return df.select(exprs)


def fix_missing_values(
    df: pl.DataFrame,
    config: FeatureConfig,
    *,
    max_missing_row_frac: float = 0.05,
) -> pl.DataFrame:
    """Fix missing values in the dataframe.

    Strategy:
    - If the fraction of rows containing at least one null is <= max_missing_row_frac,
      drop those rows.
    - Otherwise, raise an error to force explicit handling.

    Designed for clean baselines with tree-based models.
    """
    if not config.fix_missing_values:
        return df

    n_rows = df.height
    if n_rows == 0:
        return df

    # Boolean mask: rows with at least one null
    has_null = pl.any_horizontal(pl.all().is_null())

    n_missing_rows = df.select(has_null.sum()).item()
    frac_missing = n_missing_rows / n_rows

    if frac_missing <= max_missing_row_frac:
        return df.drop_nulls()

    raise ValueError(
        f"Too many rows with missing values: "
        f"{frac_missing:.2%} (> {max_missing_row_frac:.2%}). "
        "Explicit missing-value handling required."
    )


def remove_string_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Remove all string (Utf8) columns from the dataframe.

    Useful as a safety step before model training to ensure
    only numeric features are present.
    """
    numeric_cols = [col for col, dtype in df.schema.items() if dtype != pl.Utf8]

    return df.select(numeric_cols)
