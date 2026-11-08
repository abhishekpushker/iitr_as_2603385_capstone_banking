"""The evaluation test set (§ARCH 7.18): exactly 15 items, every required topic, the edge cases,
and no query shared with any other query set in the repository."""

from __future__ import annotations

import pytest

from credagent.cache.response_cache import normalize_query
from credagent.guardrails.pii import PiiMasker
from credagent.settings import get_settings
from scripts.run_eval import load_testset
from tests.unit.test_kb import REQUIRED_TOPICS
from tests.unit.test_query_sets import other_query_sets

pytestmark = pytest.mark.unit

ITEMS = load_testset(get_settings().eval_testset_path)


def test_exactly_15_items_with_unique_ids_and_queries() -> None:
    assert len(ITEMS) == 15
    assert len({i.id for i in ITEMS}) == 15
    assert len({normalize_query(i.query) for i in ITEMS}) == 15


def test_topic_union_covers_all_12_required_topics() -> None:
    covered = {t for i in ITEMS for t in i.topics}
    assert covered >= set(REQUIRED_TOPICS), sorted(set(REQUIRED_TOPICS) - covered)


def test_edge_items_out_of_scope_injection_and_pii() -> None:
    by_type = {t: [i for i in ITEMS if i.type == t] for t in ("out_of_scope", "injection", "pii")}
    assert sum(len(v) for v in by_type.values()) >= 2
    assert [i.expected_behaviour for i in by_type["out_of_scope"]] == ["refuse"]
    assert [i.expected_behaviour for i in by_type["injection"]] == ["refuse"]
    (pii,) = by_type["pii"]
    assert pii.expected_behaviour == "answer"
    assert PiiMasker().mask(pii.query).total >= 1, "the PII item must carry PII"


def test_no_query_overlaps_another_query_set() -> None:
    ours = {normalize_query(i.query) for i in ITEMS}
    for name, queries in other_query_sets().items():
        if name != "eval test set (§ARCH 7.18)":
            overlap = ours & {normalize_query(q) for q in queries}
            assert not overlap, f"test set overlaps {name}: {sorted(overlap)}"
