"""Cred (Banking & FinTech) loan-support agent.

Importing this package sets the egress-off defaults (§ARCH 5) *before* anything can import
crewai, chromadb or the HF stack, so even a script that imports a library right after us is
covered. `setdefault` keeps explicit values; `credagent.settings` rejects unsafe ones.
"""

import os
from types import MappingProxyType

EGRESS_OFF_DEFAULTS = MappingProxyType(
    {
        "CREWAI_DISABLE_TELEMETRY": "true",
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "ANONYMIZED_TELEMETRY": "False",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
)

for _key, _value in EGRESS_OFF_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
