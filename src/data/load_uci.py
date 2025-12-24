# src/data/load.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import hashlib
import urllib.request

import polars as pl


SPLIT = Literal["train", "test"]

_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
_UCI_FILES = {
    "train": "adult.data",
    "test": "adult.test",
    "names": "adult.names",
}

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",  # target
]


@dataclass(frozen=True)
class AdultPaths:
    repo_root: Path
    raw_dir: Path
    dataset_dir: Path

    @staticmethod
    def from_repo_layout() -> "AdultPaths":
        repo_root = Path(__file__).resolve().parents[2]
        raw_dir = repo_root / "data" / "raw"
        dataset_dir = raw_dir / "adult"
        return AdultPaths(repo_root, raw_dir, dataset_dir)


# --------------------------
# Download + reproducibility
# --------------------------


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "python-urllib/AdultPolarsLoader"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def ensure_adult_raw_files(
    paths: Optional[AdultPaths] = None,
    force: bool = False,
) -> AdultPaths:
    paths = paths or AdultPaths.from_repo_layout()
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for fname in _UCI_FILES.values():
        url = _UCI_BASE + fname
        dest = paths.dataset_dir / fname
        if force or not dest.exists():
            _download(url, dest)
        manifest[fname] = _sha256_file(dest)

    manifest_path = paths.dataset_dir / "SHA256SUMS.txt"
    manifest_path.write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(manifest.items())) + "\n",
        encoding="utf-8",
    )

    return paths


# -------------------------
# Loading
# -------------------------


def _read_split(path: Path, split: SPLIT) -> pl.DataFrame:
    skip_rows = 1 if split == "test" else 0

    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=COLUMNS,
        separator=",",
        skip_rows=skip_rows,
        null_values=["?"],
        try_parse_dates=False,
    )

    # 1) strip su tutte le stringhe
    string_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
    if string_cols:
        df = df.with_columns(pl.col(string_cols).str.strip_chars())

    # 2) (opzionale ma robusto) stringhe vuote -> null
    if string_cols:
        df = df.with_columns(
            [
                pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
                for c in string_cols
            ]
        )

    # 3) cast esplicito numeriche (anche se arrivano come stringhe)
    numeric_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    df = df.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False).alias(c) for c in numeric_cols]
    )

    # 4) normalizza label target (adult.test a volte ha il ".")
    df = df.with_columns(
        pl.col("income")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace(r"\.$", "")
        .alias("income")
    )

    return df


def load_adult_split(
    split: SPLIT,
    paths: Optional[AdultPaths] = None,
    download_if_missing: bool = True,
) -> pl.DataFrame:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    paths = paths or AdultPaths.from_repo_layout()

    if download_if_missing:
        ensure_adult_raw_files(paths)

    file_path = paths.dataset_dir / _UCI_FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return _read_split(file_path, split)


def load_adult(
    paths: Optional[AdultPaths] = None,
    download_if_missing: bool = True,
    add_split_column: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
    train = load_adult_split("train", paths, download_if_missing)
    test = load_adult_split("test", paths, download_if_missing)

    if not add_split_column:
        return train, test

    return pl.concat(
        [
            train.with_columns(pl.lit("train").alias("split")),
            test.with_columns(pl.lit("test").alias("split")),
        ],
        how="vertical",
    )


# -------------------------
# Sanity check
# -------------------------
if __name__ == "__main__":
    paths = ensure_adult_raw_files()
    tr, te = load_adult(paths)

    print(f"Dataset dir: {paths.dataset_dir}")
    print(f"Train shape: {tr.shape}")
    print(f"Test shape:  {te.shape}")
    print(tr.head())
