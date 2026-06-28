#!/usr/bin/env python3
"""One-off: download BGE-small-en-v1.5 from HuggingFace and save it locally
under ``artifacts/model/`` so subsequent runs (and Stage-3 Docker reproduction)
work with NO network access.

Idempotent — if a model dir already exists with the expected files, skip.

Usage:
    python tools/vendor_model.py [--model BAAI/bge-small-en-v1.5] [--out artifacts/model]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def looks_vendored(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    expected_one_of = ["config.json", "pytorch_model.bin", "model.safetensors", "modules.json"]
    return any((model_dir / name).exists() for name in expected_one_of)


def vendor(model_name: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if looks_vendored(out_dir):
        print(f"[vendor_model] already vendored at {out_dir} — skipping")
        return 0

    print(f"[vendor_model] downloading {model_name} → {out_dir}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.save(str(out_dir))

    n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"[vendor_model] wrote {n_files} files, {total / 1e6:.1f} MB total")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id to vendor")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "model",
        help="Target directory (default: artifacts/model)",
    )
    args = p.parse_args(argv)
    return vendor(args.model, args.out)


if __name__ == "__main__":
    sys.exit(main())
