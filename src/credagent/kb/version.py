"""`kb_version` (§ARCH 3.5): sha256 over the KB files and the chunk parameters.

Any edit to a KB file, an added or removed file, or a change to the chunking parameters or the
splitter's abbreviation list yields a new version. Ingests it into chunk metadata and `kb_meta`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from credagent.kb.chunking import chunk_params
from credagent.kb.loader import kb_files


def kb_version(kb_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in kb_files(kb_dir):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(json.dumps(chunk_params(), sort_keys=True).encode())
    return digest.hexdigest()
