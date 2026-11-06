"""Suite-wide wiring: every test belongs to exactly one layer (§ARCH 11.1).

A test gets its layer marker from its top-level directory (tests/<layer>/...) unless it
declares a layer marker itself (e.g. an integration-marked check living next to the contract
tests it belongs to). A test outside a layer directory without a marker fails collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import settings

# The repo is mounted read-only (no example database on disk). A wall-clock deadline would make
# property tests depend on machine load, so it is off; runs are seeded by pytest-randomly.
settings.register_profile("credagent", database=None, deadline=None, max_examples=200)
settings.load_profile("credagent")

TESTS_DIR = Path(__file__).resolve().parent
LAYERS = ("unit", "contract", "integration", "e2e", "security", "golden")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        declared = [m for m in LAYERS if item.get_closest_marker(m) is not None]
        if len(declared) > 1:
            raise pytest.UsageError(f"{item.nodeid}: more than one layer marker {declared}")
        if declared:
            continue
        layer = Path(item.path).resolve().relative_to(TESTS_DIR).parts[0]
        if layer not in LAYERS:
            raise pytest.UsageError(f"{item.nodeid}: not under a layer directory {LAYERS}")
        item.add_marker(getattr(pytest.mark, layer))
