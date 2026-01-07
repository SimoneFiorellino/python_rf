# tests/test_rf_with_processed_data.py
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)


PROCESSED_DIR = Path("data/processed")
X_TRAIN_PATH = PROCESSED_DIR / "X_train.parquet"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.parquet"
X_TEST_PATH = PROCESSED_DIR / "X_test.parquet"
Y_TEST_PATH = PROCESSED_DIR / "y_test.parquet"


def _load_processed() -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    missing = [
        p
        for p in (X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH)
        if not p.exists()
    ]
    if missing:
        pytest.skip(
            "Processed dataset not found. Run: uv run -m src.data.build_dataset "
            f"(missing: {[str(p) for p in missing]})"
        )

    X_train = pl.read_parquet(X_TRAIN_PATH)
    y_train_df = pl.read_parquet(Y_TRAIN_PATH)
    X_test = pl.read_parquet(X_TEST_PATH)
    y_test_df = pl.read_parquet(Y_TEST_PATH)

    # y_* parquet are saved as 1-col DataFrames
    y_train = y_train_df.to_series()
    y_test = y_test_df.to_series()

    return X_train, y_train, X_test, y_test


def test_processed_data_sanity() -> None:
    X_train, y_train, X_test, y_test = _load_processed()

    # Basic shape checks
    assert X_train.height > 0 and X_test.height > 0
    assert X_train.width > 0
    assert y_train.len() == X_train.height
    assert y_test.len() == X_test.height

    # No strings
    assert all(dt != pl.Utf8 for dt in X_train.dtypes), "Found Utf8 columns in X_train"
    assert all(dt != pl.Utf8 for dt in X_test.dtypes), "Found Utf8 columns in X_test"

    # No nulls
    assert X_train.null_count().to_numpy().sum() == 0, "Nulls found in X_train"
    assert X_test.null_count().to_numpy().sum() == 0, "Nulls found in X_test"
    assert y_train.null_count() == 0, "Nulls found in y_train"
    assert y_test.null_count() == 0, "Nulls found in y_test"

    # Train/test schema match
    assert X_train.schema == X_test.schema, "Train/Test schema mismatch"


def test_random_forest_trains_and_scores_reasonably() -> None:
    X_train_pl, y_train_pl, X_test_pl, y_test_pl = _load_processed()

    # Convert to numpy for sklearn
    X_train = X_train_pl.to_numpy()
    X_test = X_test_pl.to_numpy()

    # Ensure y is binary 1d array
    y_train = y_train_pl.cast(pl.Int64).to_numpy().ravel()
    y_test = y_test_pl.cast(pl.Int64).to_numpy().ravel()

    # Binary sanity check
    assert set(y_train) <= {0, 1}, "Target is not binary {0,1}"
    assert len(set(y_train)) == 2, "Only one class present in y_train"

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    # Probabilities for AUC
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # --- Assertions (loose but meaningful) ---
    assert auc >= 0.70, f"ROC-AUC too low ({auc:.3f})"
    assert acc >= 0.75, f"Accuracy too low ({acc:.3f})"
    assert f1 >= 0.60, f"F1-score too low ({f1:.3f})"
