"""JSONL → Parquet ingestion.

Phase-1 contract: parse the raw candidates dataset into a single source of truth
(`candidates.parquet`). Nested structures (`career_history`, `education`, `skills`,
`redrob_signals`) are preserved as list-of-struct or struct columns — date strings
stay strings here; date math happens in Phase-2 feature builders that need it.

No date math, no regex, no business logic in this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_jsonl(path: Path) -> Iterator[dict]:
    """Stream candidate dicts from a JSONL file, one per non-empty line.

    Yields each record without mutation. Caller is responsible for memory if
    they materialize into a list.
    """
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def candidates_to_parquet(records: Iterable[dict], out_path: Path) -> int:
    """Persist candidate dicts as a Parquet file. Returns row count written.

    Uses `pa.Table.from_pylist` so pyarrow infers the schema (preserving nested
    lists-of-structs). Snappy-compressed. Caller controls how many records are
    materialized into memory before this call.
    """
    records_list = records if isinstance(records, list) else list(records)
    table = pa.Table.from_pylist(records_list)
    pq.write_table(table, out_path, compression="snappy")
    return len(records_list)
