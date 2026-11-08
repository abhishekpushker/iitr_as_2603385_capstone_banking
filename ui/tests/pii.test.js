// The browser's masker against the case list the server's PiiMasker is pinned to
// (tests/unit/test_ui_pii_parity.py). Run by `make test-ui`.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { maskPii } from "../public/js/pii.js";

const { cases } = JSON.parse(readFileSync(new URL("./pii_cases.json", import.meta.url), "utf-8"));

for (const c of cases) {
  test(`maskPii: ${c.name}`, () => {
    assert.equal(maskPii(c.input).text, c.masked);
  });
}

test("maskPii counts each type", () => {
  const all = cases.find((c) => c.name === "everything at once");
  assert.deepEqual(maskPii(all.input).counts, { PAN: 1, AADHAAR: 2, BANK_ACCOUNT: 1 });
});

test("maskPii counts nothing in clean text", () => {
  assert.deepEqual(maskPii("What is the status of LA-0037?").counts, {
    PAN: 0,
    AADHAAR: 0,
    BANK_ACCOUNT: 0,
  });
});
