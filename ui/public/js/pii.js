// A port of credagent.guardrails.pii.PiiMasker for the chat UI (D-058).
//
// The server masks every message before the pipeline sees it; this copy exists only so the
// history the browser keeps in localStorage never holds a raw PAN, Aadhaar or bank-account
// number. Same rules, same precedence, same tokens; ui/tests/pii_cases.json pins both sides.
//
// Python's `\d` and `[^\W_]` are Unicode-aware, so the ports use `\p{Nd}` and `[\p{L}\p{N}]`.

const TOKENS = {
  PAN: "[PAN_REDACTED]",
  AADHAAR: "[AADHAAR_REDACTED]",
  BANK_ACCOUNT: "[BANK_ACCOUNT_REDACTED]",
};

// Zero-width (U+200B–U+200D, U+2060, U+FEFF, soft hyphen) and bidi controls, as in text.py.
const INVISIBLE = /[​‌‍⁠﻿­‎‏؜‪-‮⁦-⁩]/gu;

const NOT_ALNUM_BEFORE = String.raw`(?<![\p{L}\p{N}])`;
const NOT_ALNUM_AFTER = String.raw`(?![\p{L}\p{N}])`;
const D = String.raw`\p{Nd}`;

// [type, pattern, capture group that is masked]; strict precedence order.
const RULES = [
  [
    "PAN",
    NOT_ALNUM_BEFORE +
      String.raw`(?:[a-zA-Z]{3}[abcfghljptABCFGHLJPT][a-zA-Z][0-9]{4}[a-zA-Z]` +
      String.raw`|[A-Z]{3}[ABCFGHLJPT][A-Z][ -]?[0-9]{4}[ -]?[A-Z])` +
      NOT_ALNUM_AFTER,
    0,
  ],
  ["AADHAAR", `(?<!${D})[2-9]${D}{3}[\\s-]${D}{4}[\\s-]${D}{4}(?!${D})`, 0],
  [
    "BANK_ACCOUNT",
    NOT_ALNUM_BEFORE +
      String.raw`(?:[aA]\/?[cC]|[aA][cC][cC][tT]|[aA][cC][cC][oO][uU][nN][tT])` +
      String.raw`(?:\s*(?:[nN][oO]|[nN][uU][mM][bB][eE][rR]|#))?\.?\s*[:\-#]?\s*` +
      `(${D}{9,18})(?!${D})`,
    1,
  ],
  ["AADHAAR", `(?<!${D})[2-9]${D}{11}(?!${D})`, 0],
  ["BANK_ACCOUNT", `(?<!${D})${D}{9,}`, 0],
].map(([type, source, group]) => ({ type, pattern: new RegExp(source, "gud"), group }));

export function stripInvisible(text) {
  return String(text).replace(INVISIBLE, "").normalize("NFKC");
}

function overlaps([a, b], taken) {
  return taken.some(([start, end]) => a < end && start < b);
}

/** Masks `text`; returns `{ text, counts: { PAN, AADHAAR, BANK_ACCOUNT } }`. Never the values. */
export function maskPii(text) {
  const stripped = stripInvisible(text);
  const taken = [];
  for (const rule of RULES) {
    const spans = [...stripped.matchAll(rule.pattern)].map((m) => m.indices[rule.group]);
    spans.sort((x, y) => x[0] - x[1] - (y[0] - y[1]) || x[0] - y[0]); // longest first
    for (const span of spans) {
      if (!overlaps(span, taken)) taken.push([span[0], span[1], rule.type]);
    }
  }
  const counts = { PAN: 0, AADHAAR: 0, BANK_ACCOUNT: 0 };
  let out = stripped;
  for (const [start, end, type] of taken.sort((x, y) => y[0] - x[0])) {
    out = out.slice(0, start) + TOKENS[type] + out.slice(end);
    counts[type] += 1;
  }
  return { text: out, counts };
}
