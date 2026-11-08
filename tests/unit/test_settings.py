"""`credagent.settings` is the single, frozen source of configuration (§ARCH 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from credagent.settings import BUDGET_CONFIG_KEYS, Settings, get_settings

MEASURED = json.loads((Path(__file__).resolve().parents[2] / "config" / "budget.json").read_text())

ENV_KEYS = [
    "MOCK_LLM", "LLM_PROVIDER", "CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED",
    "CREWAI_TRACING_ENABLED", "ANONYMIZED_TELEMETRY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "EMBED_MODEL_PATH",
    "CHROMA_HOST", "CHROMA_PORT", "COLLECTION_FIXED", "COLLECTION_SENTENCE", "DEPLOYED_STRATEGY",
    "TOP_K", "API_KEY", "ADMIN_API_KEY", "LOG_PATH", "MAX_MESSAGE_CHARS", "BUDGET_MAX_TOKENS",
    "BUDGET_MAX_COST_USD", "PRICE_PER_1K_TOKENS_USD", "CACHE_MAX_ENTRIES", "CACHE_TTL_SECONDS",
    "REQUEST_TIMEOUT_S", "CREWAI_STORAGE_DIR", "WEB_CONCURRENCY", "GIT_SHA", "IMAGE_ID",
    "EMBED_BATCH_SIZE", "KB_DIR", "CHROMA_CONNECT_ATTEMPTS", "CHROMA_BACKOFF_CAP_S",
    "COLLECTION_META", "ESCALATION_CONFIG_PATH", "BUDGET_CONFIG_PATH", "PROMPT_OVERHEAD_TOKENS",
    "MAX_OUTPUT_TOKENS", "MAX_BODY_BYTES",
]  # fmt: skip


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees only the variables it sets (the image ENV must not mask defaults)."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


DEFAULTS: dict[str, Any] = {
    "mock_llm": True,
    "llm_provider": "mock",
    "crewai_disable_telemetry": True,
    "otel_sdk_disabled": True,
    "crewai_tracing_enabled": False,
    "anonymized_telemetry": False,
    "hf_hub_offline": True,
    "transformers_offline": True,
    "hf_hub_disable_telemetry": True,
    "tokenizers_parallelism": False,
    "omp_num_threads": 1,
    "embed_model_path": Path("/models/all-MiniLM-L6-v2"),
    "embed_batch_size": 32,
    "kb_dir": Path("/app/kb"),
    "chroma_host": "chroma",
    "chroma_port": 8000,
    "chroma_connect_attempts": 10,
    "chroma_backoff_cap_s": 5.0,
    "collection_fixed": "kb_fixed_v1",
    "collection_sentence": "kb_sentence_v1",
    "collection_meta": "kb_meta",
    "deployed_strategy": None,
    "top_k": 3,
    "retrieval_config_path": Path("/app/config/retrieval.json"),
    "chunking_report_path": Path("/app/eval/results/chunking_pr.md"),
    "escalation_config_path": Path("/app/config/escalation.json"),
    "log_path": Path("/data/logs/requests.jsonl"),
    "max_message_chars": 16000,
    "max_body_bytes": 65536,
    "budget_config_path": Path("/app/config/budget.json"),
    **{k: MEASURED[k] for k in BUDGET_CONFIG_KEYS},  # generated, not the fallbacks
    "price_per_1k_tokens_usd": 0.002,
    "cache_max_entries": 512,
    "cache_ttl_seconds": 3600,
    "request_timeout_s": 30.0,
    "session_max": 1000,
    "session_ttl_s": 3600.0,
    "injection_corpus_path": Path("/app/eval/injection_corpus.yaml"),
    "eval_testset_path": Path("/app/eval/testset.yaml"),
    "judge_prompt_path": Path("/app/eval/judge_prompt.md"),
    "eval_results_dir": Path("/app/eval/results"),
    "crewai_storage_dir": Path("/tmp/crewai"),
    "web_concurrency": None,
    "git_sha": "unknown",
    "image_id": "unknown",
}


@pytest.mark.parametrize(("field", "expected"), sorted(DEFAULTS.items()))
def test_default_matches_architecture_table(field: str, expected: Any) -> None:
    assert getattr(Settings(), field) == expected


def test_every_field_is_covered_by_the_defaults_table() -> None:
    assert set(Settings.model_fields) == set(DEFAULTS) | {"api_key", "admin_api_key"}


def test_api_keys_are_secret_and_distinct() -> None:
    s = Settings()
    assert "dev" in s.api_key.get_secret_value()
    assert s.api_key.get_secret_value() != s.admin_api_key.get_secret_value()
    assert s.api_key.get_secret_value() not in repr(s)


def test_env_overrides_a_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("DEPLOYED_STRATEGY", "sentence")
    s = Settings()
    assert (s.top_k, s.deployed_strategy) == (5, "sentence")


def test_settings_are_frozen() -> None:
    with pytest.raises(ValidationError):
        Settings().top_k = 9  # type: ignore[misc]


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_mock_llm_must_be_true(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MOCK_LLM", value)
    with pytest.raises(ValidationError, match="MOCK_LLM"):
        Settings()


def test_only_the_mock_provider_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("key", "unsafe"),
    [
        ("CREWAI_DISABLE_TELEMETRY", "false"),
        ("OTEL_SDK_DISABLED", "false"),
        ("CREWAI_TRACING_ENABLED", "true"),
        ("ANONYMIZED_TELEMETRY", "True"),
        ("HF_HUB_OFFLINE", "0"),
        ("TRANSFORMERS_OFFLINE", "0"),
        ("HF_HUB_DISABLE_TELEMETRY", "0"),
    ],
)
def test_egress_flags_must_stay_off(monkeypatch: pytest.MonkeyPatch, key: str, unsafe: str) -> None:
    monkeypatch.setenv(key, unsafe)
    with pytest.raises(ValidationError, match=key):
        Settings()


@pytest.mark.parametrize("value", ["2", "4"])
def test_multi_worker_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """P14: in-process sessions and cache would silently split across workers (§ARCH 8)."""
    monkeypatch.setenv("WEB_CONCURRENCY", value)
    with pytest.raises(ValidationError, match="WEB_CONCURRENCY"):
        Settings()


def test_single_worker_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    assert Settings().web_concurrency == "1"


@pytest.mark.parametrize(
    ("key", "bad"),
    [("TOP_K", "0"), ("CHROMA_PORT", "70000"), ("DEPLOYED_STRATEGY", "semantic")],
)
def test_out_of_range_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, key: str, bad: str
) -> None:
    monkeypatch.setenv(key, bad)
    with pytest.raises(ValidationError):
        Settings()


def test_unrelated_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_OTHER_TOOL_SETTING", "x")
    Settings()


def test_get_settings_is_a_cached_singleton() -> None:
    assert get_settings() is get_settings()
