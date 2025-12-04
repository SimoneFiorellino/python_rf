import numpy as np

from src.models.rf import (
    DecisionTreeClassifierNumpy,
    RandomForestClassifierNumpy,
)


def make_simple_dataset(n_samples: int = 200, random_state: int = 0):
    """
    Create a simple, almost linearly separable 2D classification dataset.

    Class 0: centered at (-1, -1)
    Class 1: centered at (+1, +1)
    """
    rng = np.random.RandomState(random_state)

    n0 = n_samples // 2
    n1 = n_samples - n0

    x0 = rng.randn(n0, 2) * 0.5 + np.array([-1.0, -1.0])
    x1 = rng.randn(n1, 2) * 0.5 + np.array([1.0, 1.0])

    X = np.vstack([x0, x1])
    y = np.concatenate(
        [
            np.zeros(n0, dtype=int),
            np.ones(n1, dtype=int),
        ]
    )

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    return X, y


def test_decision_tree_numpy_basic_fit_predict():
    X, y = make_simple_dataset(n_samples=200, random_state=42)

    tree = DecisionTreeClassifierNumpy(
        max_depth=4,
        min_samples_split=2,
        max_features="sqrt",
        random_state=123,
    )

    tree.fit(X, y)
    preds = tree.predict(X)

    assert preds.shape == y.shape
    acc = (preds == y).mean()
    # Dataset is easy; we expect high training accuracy
    assert acc > 0.9


def test_random_forest_numpy_basic_fit_predict():
    X, y = make_simple_dataset(n_samples=300, random_state=123)

    model = RandomForestClassifierNumpy(
        n_estimators=10,
        max_depth=3,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    acc = (preds == y).mean()
    # On this simple dataset, RF should perform well
    assert acc > 0.9


def test_random_forest_numpy_reproducibility():
    X, y = make_simple_dataset(n_samples=250, random_state=999)

    # Same random_state -> same predictions
    model1 = RandomForestClassifierNumpy(
        n_estimators=2,
        max_depth=2,
        random_state=123,
    )
    model2 = RandomForestClassifierNumpy(
        n_estimators=2,
        max_depth=2,
        random_state=123,
    )

    model1.fit(X, y)
    model2.fit(X, y)

    preds1 = model1.predict(X)
    preds2 = model2.predict(X)

    assert np.array_equal(preds1, preds2)


def test_random_forest_numpy_sklearn_comparison():
    """
    Optional integration-style test:
    Compare NumPy RF and sklearn RF on the same real dataset
    (breast cancer), checking that both reach decent accuracy
    and agree reasonably often.
    """
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        import pytest

        pytest.skip("sklearn is not available")

    data = load_breast_cancer()
    X_np = data.data
    y_np = data.target

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=0.3,
        random_state=42,
        stratify=y_np,
    )

    # NumPy RF
    model_np = RandomForestClassifierNumpy(
        n_estimators=1,
        max_depth=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    model_np.fit(X_train_np, y_train_np)
    preds_np = model_np.predict(X_test_np)
    acc_np = (preds_np == y_test_np).mean()

    # sklearn RF
    model_skl = RandomForestClassifier(
        n_estimators=1,
        max_depth=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    model_skl.fit(X_train_np, y_train_np)
    preds_skl = model_skl.predict(X_test_np)
    acc_skl = (preds_skl == y_test_np).mean()

    # Both should be reasonably accurate
    assert acc_np > 0.85
    assert acc_skl > 0.85

    # Their predictions should agree often
    agreement = (preds_np == preds_skl).mean()
    assert agreement > 0.8
