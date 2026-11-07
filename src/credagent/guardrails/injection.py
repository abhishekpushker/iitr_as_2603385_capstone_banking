"""Prompt-injection detection (§ARCH 7.12.2) — pipeline stage 2, on the PII-masked text.

Detection works on a **copy** (`normalize_for_detection`): invisible characters stripped and
NFKC (`text.strip_invisible`), casefold, spaced-out letters joined ("i g n o r e" → "ignore"),
whitespace collapsed, then de-leet (`0→o 1→i 3→e 4→a 5→s 7→t @→a $→s`). The text that flows on
is never changed.

Each rule is one regex family with a weight; a rule counts once however often it matches, and
the score is the sum of the weights that fired. `blocked = score ≥ BLOCK_SCORE`. Strong families
(instruction override, exfiltration, bulk tool coercion, developer mode) weigh 1.0 alone; words
that also occur in ordinary questions ("act as", "you are now", "Dan") weigh less, so only a
combination blocks. **Our own delimiters** (`<<<…>>>`, the mock-action marker `⟦…⟧`) and chat
role tags weigh `BLOCK_SCORE`: they always block (P17), because a forged `<<<RECORD_ID>>>` block
could otherwise steer the mock. Long base64 runs are a weak signal; each is also decoded and
re-scanned, and a decoded payload that would block on its own adds `BLOCK_SCORE`.

The result carries the score and the names of the rules that fired, never any text.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from credagent.text import strip_invisible

BLOCK_SCORE: Final = 1.0
DECODED_SIGNAL: Final = "decoded_payload_injection"
_LEET: Final = str.maketrans("013457@$", "oieastas")
_SPACED_LETTERS: Final = re.compile(r"(?<!\S)\S(?: \S){2,}(?!\S)")
_WS: Final = re.compile(r"\s+")
_BASE64_RUN: Final = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_W = r"[^.?!\n]"  # "within the same sentence"


def normalize_for_detection(text: str) -> str:
    folded = strip_invisible(text).casefold()
    joined = _SPACED_LETTERS.sub(lambda m: m.group().replace(" ", ""), folded)
    return _WS.sub(" ", joined).strip().translate(_LEET)


@dataclass(frozen=True)
class Rule:
    name: str
    weight: float
    pattern: re.Pattern[str]
    source: Literal["normalized", "stripped"] = "normalized"


def _rule(name: str, weight: float, pattern: str) -> Rule:
    return Rule(name, weight, re.compile(pattern))


_TARGETS = (
    r"(?:instructions?|rules|prompts?|directions|guidelines|guardrails?|polic(?:y|ies)"
    r"|restrictions|constraints|filters?)"
)
_BULK_OBJECTS = r"(?:applications?|records?|customers?|applicants?|accounts|users|ids?)"

RULES: Final[tuple[Rule, ...]] = (
    _rule(
        "instruction_override",
        1.0,
        rf"\b(?:ignore|disregard|forget|override|bypass)\b{_W}{{0,40}}?"
        rf"\b(?:previous|prior|above|earlier|preceding|all|any|your|the|these|those|system"
        rf"|safety|original)\b{_W}{{0,30}}?\b{_TARGETS}\b",
    ),
    _rule(
        "instructions_voided",
        1.0,
        rf"\b{_TARGETS}\b{_W}{{0,20}}?\b(?:no longer apply|(?:are|is) (?:now )?(?:void|cancell?ed"
        rf"|disabled|lifted|suspended|off)|do(?:es)? not apply any ?more)",
    ),
    _rule("new_instructions", 0.6, r"\bnew instructions\b|\bfrom now on\b"),
    _rule(
        "no_restrictions",
        0.6,
        r"\b(?:no|without(?: any)?(?: \w+)?) (?:restrictions|filters?|guardrails|limitations)\b"
        r"|\bunrestricted\b",
    ),
    _rule("role_you_are_now", 0.6, r"\byou are now\b"),
    _rule("role_act_as", 0.5, r"\bact as\b"),
    _rule("role_pretend", 0.6, r"\bpretend (?:to be|you are|that you)\b"),
    _rule("role_dan_name", 0.4, r"\bdan\b"),
    _rule(
        "role_jailbreak",
        1.0,
        r"\b(?:developer|god|dan|admin|debug) mode\b|\bjailbreak\w*|\bdo anything now\b",
    ),
    _rule(
        "prompt_exfiltration",
        1.0,
        rf"\b(?:reveal|print|show|display|repeat|output|tell me|give me|leak|expose|dump"
        rf"|write out|what (?:is|are|were))\b{_W}{{0,60}}?\b(?:system|hidden|initial|original"
        rf"|secret|developer|internal)\s+(?:prompts?|instructions?|messages?|rules|configuration)\b"
        rf"|\brepeat (?:everything|all|the text|the words) (?:above|before)\b",
    ),
    _rule("system_prompt_mention", 0.6, r"\b(?:system|hidden|initial) prompt\b"),
    _rule("tool_name_mention", 0.6, r"\b(?:check_loan_application_status|rag_lookup)\b"),
    _rule(
        "bulk_tool_coercion",
        1.0,
        rf"\b(?:dump|list|export|print|show|give me|download|extract|fetch)\b{_W}{{0,20}}?"
        rf"\b(?:all|every|each)\b{_W}{{0,25}}?\b{_BULK_OBJECTS}\b"
        rf"|\b(?:dump|export|download)\b{_W}{{0,30}}?\bdatabase\b",
    ),
    _rule("for_every_record", 0.6, rf"\bfor (?:every|each|all)\b{_W}{{0,20}}?\b{_BULK_OBJECTS}\b"),
    _rule(
        "delimiter_smuggling",
        BLOCK_SCORE,
        r"<<<|>>>|⟦|⟧|</?system>|<\|im_(?:start|end)\|>|\[/?inst\]|#{2,}\s*system\b",
    ),
    _rule("react_forgery", 0.6, r"\b(?:final answer|action input|observation|thought)\s*:"),
    _rule("decode_and_follow", 0.5, r"\bdecode\b.{0,40}\b(?:follow|execute|run|obey|do)\b"),
    Rule("encoded_payload", 0.6, _BASE64_RUN, source="stripped"),
)


class InjectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    blocked: bool
    score: float = Field(ge=0)
    signals: tuple[str, ...]


class CorpusItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str = Field(min_length=1)


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attacks: tuple[CorpusItem, ...]
    benign: tuple[CorpusItem, ...]


def load_corpus(path: Path) -> Corpus:
    return Corpus.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int
    missed: tuple[str, ...]  # attack ids that passed
    flagged: tuple[str, ...]  # benign ids that were blocked

    @property
    def false_negative_rate(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.false_negatives / total if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        total = self.true_negatives + self.false_positives
        return self.false_positives / total if total else 0.0


def _decoded_payloads(stripped: str) -> list[str]:
    out = []
    for run in _BASE64_RUN.findall(stripped):
        padded = run + "=" * (-len(run) % 4)
        try:
            out.append(base64.b64decode(padded, validate=True).decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError):
            continue
    return out


class InjectionDetector:
    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self._rules = rules

    def _signals(self, text: str) -> list[Rule]:
        views = {"normalized": normalize_for_detection(text), "stripped": strip_invisible(text)}
        return [r for r in self._rules if r.pattern.search(views[r.source])]

    def detect(self, text: str) -> InjectionResult:
        fired = self._signals(text)
        names = [r.name for r in fired]
        score = sum(r.weight for r in fired)
        if any(r.name == "encoded_payload" for r in fired) and any(
            sum(r.weight for r in self._signals(p)) >= BLOCK_SCORE
            for p in _decoded_payloads(strip_invisible(text))
        ):
            names.append(DECODED_SIGNAL)
            score += BLOCK_SCORE
        score = round(score, 6)
        return InjectionResult(blocked=score >= BLOCK_SCORE, score=score, signals=tuple(names))

    def evaluate(
        self, corpus: Corpus, preprocess: Callable[[str], str] = lambda t: t
    ) -> ConfusionMatrix:
        """Run every corpus item through `preprocess` (the pipeline passes the PII masker) and
        `detect`."""
        missed = tuple(i.id for i in corpus.attacks if not self.detect(preprocess(i.text)).blocked)
        flagged = tuple(i.id for i in corpus.benign if self.detect(preprocess(i.text)).blocked)
        return ConfusionMatrix(
            true_positives=len(corpus.attacks) - len(missed), false_negatives=len(missed),
            true_negatives=len(corpus.benign) - len(flagged), false_positives=len(flagged),
            missed=missed, flagged=flagged,
        )  # fmt: skip
