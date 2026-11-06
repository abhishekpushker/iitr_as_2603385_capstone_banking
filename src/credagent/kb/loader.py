"""Knowledge-base loader (§ARCH 7.2): YAML front matter + a short prose body per Markdown file.

Front matter is parsed with `yaml.safe_load` and validated into a frozen `KbDocument`
(`extra="forbid"`). Every string is NFKC-normalised and whitespace-collapsed (`text.clean`).
The `doc_id` must equal the file stem and be unique across the directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from credagent.text import clean

DISCLAIMER = "Fictional policy written for a capstone; not Cred's actual policy."
_FRONT_MATTER = re.compile(r"\A---\n(?P<front>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)


class KbError(ValueError):
    """A KB file is malformed or the directory is inconsistent."""


class KbDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    doc_id: str = Field(pattern=r"^kb-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    version: Literal[1]
    disclaimer: Literal["Fictional policy written for a capstone; not Cred's actual policy."]
    body: str = Field(min_length=1)


def kb_files(kb_dir: Path) -> list[Path]:
    """The KB's Markdown files in a stable (name) order."""
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise KbError(f"{kb_dir}: no .md files")
    return files


def split_front_matter(raw: str, *, source: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER.match(raw.replace("\r\n", "\n"))
    if m is None:
        raise KbError(f"{source}: expected '---' front matter, a closing '---', then the body")
    try:
        front = yaml.safe_load(m["front"])
    except yaml.YAMLError as exc:
        raise KbError(f"{source}: invalid YAML front matter: {exc}") from exc
    if not isinstance(front, dict):
        raise KbError(f"{source}: front matter must be a mapping")
    return front, m["body"]


def _clean(value: object) -> object:
    return clean(value) if isinstance(value, str) else value


def load_document(path: Path) -> KbDocument:
    front, body = split_front_matter(path.read_text(encoding="utf-8"), source=path.name)
    if "body" in front:
        raise KbError(f"{path.name}: 'body' is not a front-matter field")
    return KbDocument(**{str(k): _clean(v) for k, v in front.items()}, body=clean(body))


def load_kb(kb_dir: Path) -> list[KbDocument]:
    """All documents in `kb_dir`, sorted by doc_id. Raises on any invalid file or duplicate id."""
    docs: dict[str, KbDocument] = {}
    for path in kb_files(kb_dir):
        doc = load_document(path)
        if doc.doc_id in docs:
            raise KbError(f"{path.name}: duplicate doc_id {doc.doc_id!r}")
        if doc.doc_id != path.stem:
            raise KbError(f"{path.name}: doc_id {doc.doc_id!r} differs from the file name")
        docs[doc.doc_id] = doc
    return [docs[k] for k in sorted(docs)]
