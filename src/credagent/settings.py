"""Single, frozen source of configuration (§ARCH 5). No other module reads the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# flag → the only value allowed at runtime (N2: zero external network)
_EGRESS_OFF: dict[str, bool] = {
    "crewai_disable_telemetry": True,
    "otel_sdk_disabled": True,
    "crewai_tracing_enabled": False,
    "anonymized_telemetry": False,
    "hf_hub_offline": True,
    "transformers_offline": True,
    "hf_hub_disable_telemetry": True,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="ignore", case_sensitive=False)

    mock_llm: bool = True
    llm_provider: Literal["mock"] = "mock"

    crewai_disable_telemetry: bool = True
    otel_sdk_disabled: bool = True
    crewai_tracing_enabled: bool = False
    anonymized_telemetry: bool = False
    hf_hub_offline: bool = True
    transformers_offline: bool = True
    hf_hub_disable_telemetry: bool = True
    tokenizers_parallelism: bool = False
    omp_num_threads: int = Field(default=1, ge=1)

    embed_model_path: Path = Path("/models/all-MiniLM-L6-v2")
    embed_batch_size: int = Field(default=32, ge=1)
    kb_dir: Path = Path("/app/kb")
    chroma_host: str = "chroma"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_connect_attempts: int = Field(default=10, ge=1)  # §ARCH 3.5 bounded retry
    chroma_backoff_cap_s: float = Field(default=5.0, gt=0)
    collection_fixed: str = "kb_fixed_v1"
    collection_sentence: str = "kb_sentence_v1"
    collection_meta: str = "kb_meta"  # §ARCH says `_meta`; Chroma rejects it (D-019)
    deployed_strategy: Literal["fixed", "sentence"] | None = None  # set in Phase 4
    top_k: int = Field(default=3, ge=1, le=20)
    retrieval_config_path: Path = Path("/app/config/retrieval.json")  # generated, Phase 3
    chunking_report_path: Path = Path("/app/eval/results/chunking_pr.md")  # generated, Phase 4
    escalation_config_path: Path = Path("/app/config/escalation.json")  # generated, Phase 5

    api_key: SecretStr = SecretStr("dev-api-key-not-a-secret")
    admin_api_key: SecretStr = SecretStr("dev-admin-key-not-a-secret")

    log_path: Path = Path("/data/logs/requests.jsonl")
    max_message_chars: int = Field(default=20000, ge=1)
    budget_max_tokens: int = Field(default=8000, ge=1)  # calibrated in Phase 9
    budget_max_cost_usd: float = Field(default=0.016, gt=0)
    price_per_1k_tokens_usd: float = Field(default=0.002, gt=0)
    cache_max_entries: int = Field(default=512, ge=1)
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)
    crewai_storage_dir: Path = Path("/tmp/crewai")  # noqa: S108  (tmpfs under a read-only root)
    web_concurrency: str | None = None

    # Provenance for transcript headers (§ARCH 7.19); passed in by `make transcripts`.
    git_sha: str = "unknown"
    image_id: str = "unknown"

    @model_validator(mode="after")
    def _enforce_runtime_invariants(self) -> Self:
        if not self.mock_llm:
            raise ValueError("MOCK_LLM must be true: only the mock LLM is implemented (§ARCH 5)")
        for field, required in _EGRESS_OFF.items():
            if getattr(self, field) is not required:
                raise ValueError(f"{field.upper()} must be {required} (egress off, §ARCH 5)")
        if self.web_concurrency not in (None, "1"):
            raise ValueError("WEB_CONCURRENCY must be unset or 1: state is in-process (§ARCH 8)")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
