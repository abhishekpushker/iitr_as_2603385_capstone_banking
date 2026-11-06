"""Knowledge base and loader (§ARCH 7.2, Task 2): 13 docs, the content contract, the loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from credagent import text
from credagent.kb import loader
from credagent.kb.loader import DISCLAIMER, KbDocument, KbError, load_kb

KB_DIR = Path(__file__).resolve().parents[2] / "kb"

# The brief's 12 required topics (Task 2) plus the status glossary the Composer needs.
REQUIRED_TOPICS = {
    "Loan eligibility by type": "kb-01-loan-eligibility",
    "EMI rules": "kb-02-emi-calculation",
    "Credit-card fee structure": "kb-03-credit-card-fees",
    "KYC requirements": "kb-04-kyc-documents",
    "Fraud-dispute resolution": "kb-05-fraud-dispute",
    "Account closure": "kb-06-account-closure",
    "Interest-rate slabs": "kb-07-interest-rate-slabs",
    "Prepayment penalties": "kb-08-prepayment-penalty",
    "Minimum balance": "kb-09-minimum-balance",
    "Credit-score impact": "kb-10-credit-score-factors",
    "Joint-account rules": "kb-11-joint-accounts",
    "NRI eligibility": "kb-12-nri-eligibility",
}
GLOSSARY = ("Status meanings & SLAs", "kb-13-application-status-glossary")

# §ARCH 7.2 content contract: (phrases in the normalised body, canonical numbers in the body).
CONTRACT: dict[str, tuple[list[str], list[str]]] = {
    "kb-01-loan-eligibility": (
        ["personal", "credit score", "home", "co-applicant", "auto", "education",
         "admission letter", "parent or guardian", "co-borrower", "business", "vintage", "itr",
         "turnover"],
        ["21", "58", "25000", "700", "725", "70", "680", "20000", "3", "2", "4000000"],
    ),
    "kb-02-emi-calculation": (
        ["reducing", "emi = p·r·(1+r)^n / ((1+r)^n − 1)", "r = annual rate/12/100",
         "5th of the month", "after disbursal", "broken-period interest",
         "rounded to the nearest rupee"],
        ["5", "12", "100"],
    ),
    "kb-03-credit-card-fees": (
        ["joining fee", "waived", "annual fee", "forex markup", "cash advance", "min ₹500",
         "late", "slab", "outstanding"],
        ["500", "10000", "60", "999", "150000", "3.5%", "2.5%"],
    ),
    "kb-04-kyc-documents": (
        ["pan", "mandatory", "address proof", "aadhaar", "passport", "voter id",
         "driving licence", "video kyc", "re-kyc", "high", "medium", "low"],
        ["2", "8", "10"],
    ),
    "kb-05-fraud-dispute": (
        ["3 working days", "zero liability", "4–7 working days", "capped", "provisional",
         "10 working days", "final resolution", "90 days"],
        ["3", "4", "7", "10000", "10", "90"],
    ),
    "kb-06-account-closure": (
        ["in the app", "in writing", "all dues", "cleared", "7 working days", "credit bureau",
         "30 days", "first year"],
        ["7", "30", "500"],
    ),
    "kb-07-interest-rate-slabs": (
        ["personal", "p.a.", "credit score band", "home", "floating", "auto", "education",
         "business"],
        ["10.5", "24%", "800", "12%", "750", "799", "16%", "700", "749", "8.4", "9.9%", "8.9",
         "12.5%", "9.5", "13%", "14", "22%"],
    ),
    "kb-08-prepayment-penalty": (
        ["floating-rate home loans", "individuals", "no prepayment penalty",
         "fixed-rate personal loans", "prepaid principal", "first 12 emis", "foreclose",
         "6 emis", "auto loans", "part-prepayment", "free"],
        ["4%", "12", "2%", "6", "25%"],
    ),
    "kb-09-minimum-balance": (
        ["savings", "average monthly balance", "metro", "semi-urban", "shortfall", "capped",
         "salary", "basic", "exempt"],
        ["10000", "5000", "6%", "500"],
    ),
    "kb-10-credit-score-factors": (
        ["payment history", "weighs most", "utilisation", "hard enquiry", "24 months",
         "credit mix", "account age", "rejection", "does not lower"],
        ["30%", "24"],
    ),
    "kb-11-joint-accounts": (
        ["up to 4 holders", "either-or-survivor", "former-or-survivor", "jointly",
         "consent of all holders", "nri", "resident close relative"],
        ["4"],
    ),
    "kb-12-nri-eligibility": (
        ["nre", "nro", "valid passport", "visa", "oci", "home loans", "ltv",
         "not for personal loans", "repaid from an nre or nro account", "joint holder",
         "former-or-survivor"],
        ["80%"],
    ),
    "kb-13-application-status-glossary": (
        ["submitted", "2 working days", "under review", "7 working days", "fraud review",
         "5 additional working days", "disbursed", "48 hours", "approval", "rejected",
         "reapply", "90 days"],
        ["2", "7", "5", "48", "90"],
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def docs() -> list[KbDocument]:
    return load_kb(KB_DIR)


def _by_id(docs: list[KbDocument]) -> dict[str, KbDocument]:
    return {d.doc_id: d for d in docs}


# ------------------------------------------------------------------ inventory


def test_exactly_13_documents(docs: list[KbDocument]) -> None:
    assert len(docs) == 13
    assert len(list(KB_DIR.glob("*.md"))) == 13


def test_documents_are_sorted_by_doc_id(docs: list[KbDocument]) -> None:
    assert [d.doc_id for d in docs] == sorted(d.doc_id for d in docs)


def test_all_12_required_topics_map_to_one_doc(docs: list[KbDocument]) -> None:
    topic_to_doc = {d.topic: d.doc_id for d in docs}
    assert len(topic_to_doc) == len(docs), "two docs share a topic"
    for topic, doc_id in REQUIRED_TOPICS.items():
        assert topic_to_doc.get(topic) == doc_id, topic
    assert topic_to_doc[GLOSSARY[0]] == GLOSSARY[1]


def test_doc_ids_are_unique_and_match_file_names(docs: list[KbDocument]) -> None:
    assert len({d.doc_id for d in docs}) == len(docs)
    assert sorted(p.stem for p in KB_DIR.glob("*.md")) == [d.doc_id for d in docs]


@pytest.mark.parametrize("doc_id", sorted(CONTRACT))
def test_body_has_2_to_5_sentences(docs: list[KbDocument], doc_id: str) -> None:
    sentences = text.split_sentences(_by_id(docs)[doc_id].body)
    assert 2 <= len(sentences) <= 5, sentences


@pytest.mark.parametrize("doc_id", sorted(CONTRACT))
def test_front_matter_schema(docs: list[KbDocument], doc_id: str) -> None:
    doc = _by_id(docs)[doc_id]
    assert doc.version == 1
    assert (
        doc.disclaimer
        == DISCLAIMER
        == ("Fictional policy written for a capstone; not Cred's actual policy.")
    )
    assert doc.title and doc.title == doc.title.strip()
    raw = (KB_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
    front = loader.split_front_matter(raw, source=doc_id)[0]
    assert set(front) == {"doc_id", "title", "topic", "version", "disclaimer"}


@pytest.mark.parametrize("doc_id", sorted(CONTRACT))
def test_content_contract_facts_present(docs: list[KbDocument], doc_id: str) -> None:
    body = _by_id(docs)[doc_id].body
    normalised = text.normalize(body)
    phrases, numbers = CONTRACT[doc_id]
    missing = [p for p in phrases if p not in normalised]
    assert missing == [], f"{doc_id} lacks {missing}"
    found = text.extract_numbers(body)
    assert [n for n in numbers if n not in found] == [], found


def test_deliberate_overlaps_are_kept(docs: list[KbDocument]) -> None:
    """§ARCH 7.2: NRI joint holders in kb-11 and kb-12; personal-loan rates near penalties."""
    by_id = {d.doc_id: text.normalize(d.body) for d in docs}
    for doc_id in ("kb-11-joint-accounts", "kb-12-nri-eligibility"):
        assert "nri" in by_id[doc_id] and "joint holder" in by_id[doc_id]
        assert "resident close relative" in by_id[doc_id]
    for doc_id in ("kb-07-interest-rate-slabs", "kb-08-prepayment-penalty"):
        assert "personal loan" in by_id[doc_id] and "home loans" in by_id[doc_id]


def test_bodies_are_single_line_normalised_prose(docs: list[KbDocument]) -> None:
    for doc in docs:
        assert doc.body == " ".join(doc.body.split())
        assert not doc.body.startswith("#")


# ------------------------------------------------------------------ loader behaviour


def _write(path: Path, front: str, body: str = "One sentence here. Another one here.") -> Path:
    path.write_text(f"---\n{front}---\n{body}\n", encoding="utf-8")
    return path


_FRONT = (
    'doc_id: kb-99-test\ntitle: "Test"\ntopic: "Test topic"\nversion: 1\n'
    'disclaimer: "Fictional policy written for a capstone; not Cred\'s actual policy."\n'
)


def test_loader_parses_a_valid_file(tmp_path: Path) -> None:
    _write(tmp_path / "kb-99-test.md", _FRONT, "First  line.\nSecond\u200b line.")
    (doc,) = load_kb(tmp_path)
    assert doc.doc_id == "kb-99-test"
    assert doc.body == "First line. Second line."


def test_loader_applies_nfkc(tmp_path: Path) -> None:
    _write(tmp_path / "kb-99-test.md", _FRONT.replace('"Test"', '"Ｔｅｓｔ"'), "Ｆｅｅ ok. Yes.")
    (doc,) = load_kb(tmp_path)
    assert doc.title == "Test"
    assert doc.body == "Fee ok. Yes."


def test_loader_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    _write(tmp_path / "kb-99-test.md", _FRONT)
    _write(tmp_path / "kb-99-zzz.md", _FRONT)
    with pytest.raises(KbError, match="duplicate doc_id 'kb-99-test'"):
        load_kb(tmp_path)


def test_loader_rejects_doc_id_that_differs_from_the_file_name(tmp_path: Path) -> None:
    _write(tmp_path / "kb-98-other.md", _FRONT)
    with pytest.raises(KbError, match="file name"):
        load_kb(tmp_path)


@pytest.mark.parametrize(
    "front",
    [
        _FRONT + "extra: 1\n",  # extra="forbid"
        _FRONT.replace("version: 1", "version: 2"),
        _FRONT.replace("Cred's actual", "the actual"),  # wrong disclaimer
        _FRONT.replace("doc_id: kb-99-test", "doc_id: KB 99"),
        _FRONT.replace('title: "Test"\n', ""),
    ],
)
def test_loader_rejects_invalid_front_matter(tmp_path: Path, front: str) -> None:
    _write(tmp_path / "kb-99-test.md", front)
    with pytest.raises((KbError, ValidationError)):
        load_kb(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        "no front matter at all\n",
        "---\ndoc_id: x\n",  # unterminated
        "---\n- a\n- b\n---\nBody. Body.\n",  # not a mapping
        "---\n!!python/object/apply:os.system ['true']\n---\nBody.\n",  # safe_load only
    ],
)
def test_loader_rejects_malformed_files(tmp_path: Path, raw: str) -> None:
    (tmp_path / "kb-99-test.md").write_text(raw, encoding="utf-8")
    with pytest.raises(KbError):
        load_kb(tmp_path)


def test_loader_rejects_an_empty_body(tmp_path: Path) -> None:
    _write(tmp_path / "kb-99-test.md", _FRONT, "   ")
    with pytest.raises(ValidationError):
        load_kb(tmp_path)


def test_loader_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(KbError, match="no .md files"):
        load_kb(tmp_path)


def test_kb_document_is_frozen(docs: list[KbDocument]) -> None:
    with pytest.raises(ValidationError):
        docs[0].body = "changed"  # type: ignore[misc]
