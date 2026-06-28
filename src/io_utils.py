"""I/O helpers used by build_features.py and rank.py.

Pure I/O: Parquet read/write and chunked sha256. No business logic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def sha256_file(path: Path) -> str:
    """Return hex sha256 of `path`, streamed in 1 MiB chunks (memory-bounded)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file via the pyarrow engine."""
    return pd.read_parquet(path, engine="pyarrow")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet via pyarrow, snappy-compressed, no index column."""
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
