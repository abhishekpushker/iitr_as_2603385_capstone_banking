"""Static repository rules checked by walking the AST (§ARCH 4 layering, §ARCH 5 config)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "credagent"

# §ARCH 4: api → pipeline → {crew, review, guardrails, memory, cache, governance}
#          → {tools, rag, llm} → {kb, text, schemas, settings}. Same rank may import each other.
# `observability` (counters, later the JSON log formatter) is a leaf every layer may use (D-024).
# `cache` and `governance` share the tools rank: RagTool wraps the ResponseCache (§7.9), the ACL
# guard in tools reads governance's TOOL_ACL (§7.14.1), and llm charges the BudgetMeter (D-029).
LAYER_RANK: dict[str, int] = {
    "settings": 0, "schemas": 0, "text": 0, "kb": 0, "observability": 0,
    "tools": 1, "rag": 1, "llm": 1, "cache": 1, "governance": 1,
    "crew": 2, "review": 2, "guardrails": 2, "memory": 2,
    "pipeline": 3,
    "api": 4,
}  # fmt: skip


def _modules() -> Iterator[Path]:
    yield from sorted(SRC.rglob("*.py"))


def _component(path: Path) -> str | None:
    rel = path.relative_to(SRC)
    if rel == Path("__init__.py"):
        return None  # the package root
    return rel.parts[0].removesuffix(".py")


def _credagent_imports(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "credagent":
                    yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "credagent":
                yield node.module
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            yield "<relative>"


def test_source_tree_is_not_empty() -> None:
    assert any(_modules())


@pytest.mark.parametrize("path", list(_modules()), ids=lambda p: str(p.relative_to(SRC)))
def test_module_belongs_to_a_known_layer(path: Path) -> None:
    component = _component(path)
    assert component is None or component in LAYER_RANK, (
        f"{component!r} has no layer in §ARCH 4 — add it to LAYER_RANK deliberately"
    )


@pytest.mark.parametrize("path", list(_modules()), ids=lambda p: str(p.relative_to(SRC)))
def test_no_upward_or_relative_imports(path: Path) -> None:
    component = _component(path)
    for target in _credagent_imports(ast.parse(path.read_text(encoding="utf-8"))):
        assert target != "<relative>", f"{path}: use absolute credagent.* imports"
        parts = target.split(".")
        if component is None:
            pytest.fail(f"package root must not import {target}")
        if len(parts) == 1:
            pytest.fail(f"{path}: import a component, not the bare package")
        assert parts[1] in LAYER_RANK, f"{path}: imports unknown component {target}"
        assert LAYER_RANK[parts[1]] <= LAYER_RANK[component], (
            f"{path} ({component}) imports upper layer {target}"
        )


def test_dataset_imports_nothing_from_credagent() -> None:
    tree = ast.parse((REPO / "dataset.py").read_text(encoding="utf-8"))
    assert list(_credagent_imports(tree)) == []


ENV_ACCESS_ALLOWED = {SRC / "__init__.py", SRC / "settings.py"}


def _env_accesses(tree: ast.AST) -> Iterator[int]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"getenv", "environ", "putenv"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and {a.name for a in node.names} & {"getenv", "environ", "putenv"}
        ):
            yield node.lineno


@pytest.mark.parametrize("path", list(_modules()), ids=lambda p: str(p.relative_to(SRC)))
def test_environment_is_read_only_through_settings(path: Path) -> None:
    """§ARCH 5: no os.getenv anywhere except credagent/__init__.py and settings.py."""
    if path in ENV_ACCESS_ALLOWED:
        return
    lines = list(_env_accesses(ast.parse(path.read_text(encoding="utf-8"))))
    assert lines == [], f"{path} reads the environment directly at lines {lines}"


# Stricter than rank order: rag shares a rank with llm, yet retrieval must never reach the LLM.
FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "kb": {"crew", "api", "pipeline", "llm", "rag"},
    "rag": {"crew", "api", "pipeline", "llm"},
    # D-029: re-ranked beside tools, but they stay leaves of that rank (no tool, LLM or retrieval).
    "cache": {"tools", "rag", "llm", "governance"},
    "governance": {"tools", "rag", "llm", "cache"},
}
_GUARDED = [p for p in _modules() if _component(p) in FORBIDDEN_IMPORTS]


def test_guarded_modules_exist() -> None:
    assert {_component(p) for p in _GUARDED} == set(FORBIDDEN_IMPORTS)


@pytest.mark.parametrize("path", _GUARDED, ids=lambda p: str(p.relative_to(SRC)))
def test_guarded_components_avoid_their_forbidden_imports(path: Path) -> None:
    forbidden = FORBIDDEN_IMPORTS[_component(path) or ""]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = [
        t for t in _credagent_imports(tree) if t.split(".")[1:2] and t.split(".")[1] in forbidden
    ]
    assert hits == [], f"{path} imports {hits}"


SCRIPT_DIR = REPO / "scripts"


@pytest.mark.parametrize("path", sorted(SCRIPT_DIR.glob("*.py")), ids=lambda p: p.name)
def test_scripts_read_the_environment_only_through_settings(path: Path) -> None:
    lines = list(_env_accesses(ast.parse(path.read_text(encoding="utf-8"))))
    assert lines == [], f"{path} reads the environment directly at lines {lines}"
