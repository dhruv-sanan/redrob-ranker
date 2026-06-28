"""Static AST check: ``rank.py`` must only import from the allow-list.

This enforces the problem.md §0 / lld.md §5 invariant that ``rank.py`` can
never accidentally pull in `sentence_transformers`, `torch`, `requests`, etc.
We parse ``rank.py`` itself and walk every top-level Import / ImportFrom node.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RANK_PY = Path(__file__).resolve().parent.parent / "rank.py"

ALLOWED_TOPLEVEL: frozenset[str] = frozenset(
    {
        "numpy",
        "pandas",
        "pyarrow",
        "json",
        "sys",
        "pathlib",
        "yaml",
        "__future__",
    }
)

ALLOWED_PROJECT_PREFIXES: tuple[str, ...] = ("src",)


def _toplevel_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                out.append(n.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


@pytest.fixture(scope="session")
def rank_tree() -> ast.Module:
    return ast.parse(RANK_PY.read_text(encoding="utf-8"))


def test_rank_py_imports_in_allow_list(rank_tree: ast.Module) -> None:
    violations: list[str] = []
    for mod in _toplevel_imports(rank_tree):
        head = mod.split(".")[0]
        if head in ALLOWED_TOPLEVEL:
            continue
        if head in ALLOWED_PROJECT_PREFIXES:
            continue
        violations.append(mod)
    assert not violations, f"rank.py imports not on allow-list: {violations}"


def test_rank_py_no_forbidden_imports(rank_tree: ast.Module) -> None:
    forbidden = {
        "sentence_transformers",
        "torch",
        "tensorflow",
        "transformers",
        "huggingface_hub",
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "cohere",
        "google",  # google.generativeai
    }
    for mod in _toplevel_imports(rank_tree):
        head = mod.split(".")[0]
        assert head not in forbidden, f"rank.py imports forbidden module: {mod}"


def test_rank_py_has_runtime_guard() -> None:
    """Belt-and-braces: rank.py main() must call ``assert_no_forbidden_modules``."""
    text = RANK_PY.read_text(encoding="utf-8")
    assert "assert_no_forbidden_modules()" in text


def test_no_datetime_now_or_today_in_rank_py() -> None:
    """REFERENCE_DATE invariant: rank.py must not call now()/today()."""
    text = RANK_PY.read_text(encoding="utf-8")
    assert "datetime.now()" not in text
    assert "date.today()" not in text
