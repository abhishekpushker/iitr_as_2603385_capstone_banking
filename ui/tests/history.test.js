// Conversation history and its summarisation policy (D-058). Run by `make test-ui`.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  POLICY,
  addAnswer,
  addError,
  addUserMessage,
  loadConversations,
  newConversation,
  newSessionId,
  prune,
  saveConversations,
  serverMemoryStale,
  splitForDisplay,
  summarize,
  topicName,
} from "../public/js/history.js";

const T0 = Date.UTC(2026, 10, 8, 9, 0, 0);
const RAW = "My PAN is ABCPE1234F, account no 123456789012345. Status of LA-0037?";

function reply(overrides = {}) {
  return {
    trace_id: "0f1e2d3c-aaaa-bbbb-cccc-000000000001",
    session_id: "ui-abcdefghijklmnopqrst",
    answer: "Home loans need a credit score of at least 725.",
    intent: "policy",
    grounded: true,
    refused: false,
    refusal_reason: null,
    sources: [{ doc_id: "kb-01-loan-eligibility", chunk_id: "kb-01-loan-eligibility#s0", similarity: 0.839 }],
    application: null,
    escalation: null,
    review: { approved: true, final_answer: "Home loans need a credit score of at least 725.", reason: "ok" },
    tools_invoked: [{ agent: "retrieval_agent", tool: "rag_lookup", args: { query: "q" }, cache_hit: false }],
    cache_hit: false,
    usage: { estimated_tokens: 3933, budget_tokens: 17000, estimated_cost_usd: 0.0079 },
    timings_ms: { total: 812.5 },
    ...overrides,
  };
}

const STATUS_REPLY = reply({
  answer: "Application LA-0037 (Home Loan) is currently Under Review.",
  intent: "status",
  sources: [],
  application: {
    record_id: "LA-0037",
    category: "Home Loan",
    status: "Under Review",
    loan_amount_inr: 6380000,
    days_since_created: 5,
    flagged_for_fraud_review: true,
  },
  escalation: { score: 0.5, threshold: 0.5, recommended: true },
  tools_invoked: [
    { agent: "lookup_agent", tool: "check_loan_application_status", args: { record_id: "LA-0037" }, cache_hit: false },
  ],
});

const REFUSAL = reply({
  answer: "I can't answer that from the policy documents I have. Please contact support.",
  intent: "refused",
  grounded: false,
  refused: true,
  refusal_reason: "ungrounded",
  sources: [],
});

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
    data,
  };
}

// ------------------------------------------------------------------ session ids
test("a session id matches the API pattern and has no digits, so the server masker never alters it", () => {
  const ids = new Set(Array.from({ length: 200 }, () => newSessionId()));
  assert.equal(ids.size, 200);
  for (const id of ids) {
    assert.match(id, /^[A-Za-z0-9_-]{8,64}$/);
    assert.doesNotMatch(id, /[0-9]/);
  }
});

// ------------------------------------------------------------------ masking before storage
test("the stored user message is masked and records what was masked", () => {
  const conv = addUserMessage(newConversation(T0), RAW, T0 + 1);
  const [msg] = conv.messages;
  assert.equal(msg.role, "user");
  assert.equal(msg.text, "My PAN is [PAN_REDACTED], account no [BANK_ACCOUNT_REDACTED]. Status of LA-0037?");
  assert.deepEqual(msg.masked, { PAN: 1, AADHAAR: 0, BANK_ACCOUNT: 1 });
  const stored = JSON.stringify(conv);
  assert.ok(!stored.includes("ABCPE1234F"));
  assert.ok(!stored.includes("123456789012345"));
});

test("the title is the first masked question, cut to the policy length", () => {
  let conv = addUserMessage(newConversation(T0), RAW, T0 + 1);
  conv = addUserMessage(conv, "Second question", T0 + 2);
  assert.ok(conv.title.startsWith("My PAN is [PAN_REDACTED]"));
  assert.ok(conv.title.length <= POLICY.titleChars);
  assert.ok(conv.title.endsWith("…"));
  assert.equal(newConversation(T0).title, "New conversation");
});

test("only the answer fields the UI shows are kept from a response", () => {
  let conv = addUserMessage(newConversation(T0), "q", T0);
  conv = addAnswer(conv, { ...reply(), unexpected: "x" }, T0 + 5);
  const msg = conv.messages[1];
  assert.equal(msg.role, "assistant");
  assert.equal(msg.unexpected, undefined);
  assert.equal(msg.answer, "Home loans need a credit score of at least 725.");
  assert.equal(conv.updatedAt, T0 + 5);
});

test("an error reply is kept with its code and trace id only", () => {
  let conv = addUserMessage(newConversation(T0), "q", T0);
  conv = addError(conv, { code: "BUDGET_EXCEEDED", message: "the request exceeds the token budget", trace_id: "t1" }, T0 + 1);
  assert.deepEqual(conv.messages[1], {
    role: "error",
    code: "BUDGET_EXCEEDED",
    message: "the request exceeds the token budget",
    trace_id: "t1",
    at: T0 + 1,
  });
});

// ------------------------------------------------------------------ summarisation
test("a topic name comes from the document id", () => {
  assert.equal(topicName("kb-08-prepayment-penalty"), "Prepayment penalty");
  assert.equal(topicName("kb-13-application-status-glossary"), "Application status glossary");
  assert.equal(topicName("kb-lunar-rover"), "Lunar rover");
});

test("the summary lists applications, topics, questions and declines", () => {
  let conv = newConversation(T0);
  conv = addAnswer(addUserMessage(conv, "Status of LA-0037?", T0), STATUS_REPLY, T0 + 1);
  conv = addAnswer(addUserMessage(conv, "What credit score does it need?", T0 + 2), reply(), T0 + 3);
  conv = addAnswer(
    addUserMessage(conv, "And a prepayment penalty?", T0 + 4),
    reply({ sources: [{ doc_id: "kb-08-prepayment-penalty", chunk_id: "c", similarity: 0.7 }] }),
    T0 + 5,
  );
  conv = addAnswer(addUserMessage(conv, "Gold loan penalty?", T0 + 6), REFUSAL, T0 + 7);
  const s = summarize(conv.messages);
  assert.deepEqual(s.applications, [{ id: "LA-0037", category: "Home Loan", status: "Under Review", escalated: true }]);
  assert.deepEqual(s.topics, ["Loan eligibility", "Prepayment penalty"]);
  assert.equal(s.questions, 4);
  assert.equal(s.declined, 1);
  assert.equal(
    s.line,
    "LA-0037 (Home Loan, Under Review) · Loan eligibility, Prepayment penalty · 4 questions, 1 declined",
  );
});

test("a refused answer adds no topic, and an application keeps its latest status", () => {
  let conv = newConversation(T0);
  conv = addAnswer(addUserMessage(conv, "a", T0), STATUS_REPLY, T0);
  const later = reply({
    intent: "status",
    sources: [],
    application: { ...STATUS_REPLY.application, status: "Approved" },
    escalation: { score: 0.2, threshold: 0.5, recommended: false },
  });
  conv = addAnswer(addUserMessage(conv, "b", T0), later, T0);
  conv = addAnswer(
    addUserMessage(conv, "c", T0),
    { ...REFUSAL, sources: [{ doc_id: "kb-03-credit-card-fees", chunk_id: "c", similarity: 0.5 }] },
    T0,
  );
  const s = summarize(conv.messages);
  assert.deepEqual(s.applications, [{ id: "LA-0037", category: "Home Loan", status: "Approved", escalated: false }]);
  assert.deepEqual(s.topics, []);
  assert.equal(s.line, "LA-0037 (Home Loan, Approved) · 3 questions, 1 declined");
});

test("the summary of an empty or single-question conversation", () => {
  assert.equal(summarize([]).line, "No questions yet");
  const conv = addUserMessage(newConversation(T0), "q", T0);
  assert.equal(summarize(conv.messages).line, "1 question");
});

test("errors count as declined", () => {
  let conv = addUserMessage(newConversation(T0), "q", T0);
  conv = addError(conv, { code: "PIPELINE_TIMEOUT", message: "m", trace_id: "t" }, T0);
  assert.equal(summarize(conv.messages).declined, 1);
});

// ------------------------------------------------------------------ display policy
function conversationWithTurns(n) {
  let conv = newConversation(T0);
  for (let i = 0; i < n; i += 1) {
    conv = addAnswer(addUserMessage(conv, `question ${i}`, T0 + i), reply(), T0 + i);
  }
  return conv;
}

test("up to the visible-turn limit nothing is folded", () => {
  const conv = conversationWithTurns(POLICY.visibleTurns);
  const { older, recent } = splitForDisplay(conv.messages);
  assert.equal(older.length, 0);
  assert.equal(recent.length, conv.messages.length);
});

test("older turns fold into a summary and the latest turns stay whole", () => {
  const conv = conversationWithTurns(POLICY.visibleTurns + 3);
  const { older, recent } = splitForDisplay(conv.messages);
  assert.equal(older.length, 6); // 3 folded turns × (question + answer)
  assert.equal(recent.length, POLICY.visibleTurns * 2);
  assert.equal(recent[0].role, "user");
  assert.equal(recent[0].text, "question 3");
});

test("a turn whose reply is pending still counts as a turn", () => {
  let conv = conversationWithTurns(POLICY.visibleTurns);
  conv = addUserMessage(conv, "pending", T0 + 99);
  const { older, recent } = splitForDisplay(conv.messages);
  assert.equal(older.length, 2);
  assert.equal(recent.at(-1).text, "pending");
});

// ------------------------------------------------------------------ retention
test("prune keeps the most recently updated conversations, newest first", () => {
  const list = Array.from({ length: POLICY.maxConversations + 5 }, (_, i) => ({
    ...newConversation(T0 + i),
    updatedAt: T0 + i,
  }));
  const kept = prune(list);
  assert.equal(kept.length, POLICY.maxConversations);
  assert.equal(kept[0].updatedAt, T0 + POLICY.maxConversations + 4);
  assert.equal(kept.at(-1).updatedAt, T0 + 5);
});

test("prune caps the messages of a conversation and keeps the latest", () => {
  const conv = conversationWithTurns(POLICY.maxMessages); // 2 × maxMessages messages
  const [kept] = prune([conv]);
  assert.equal(kept.messages.length, POLICY.maxMessages);
  assert.equal(kept.messages.at(-1).role, "assistant");
  assert.equal(kept.messages[0].role, "user");
  assert.equal(kept.title, conv.title);
});

test("server memory is treated as gone before the API's idle TTL", () => {
  const conv = addUserMessage(newConversation(T0), "q", T0);
  assert.ok(POLICY.serverMemoryTtlMs < 3600 * 1000);
  assert.equal(serverMemoryStale(conv, T0 + POLICY.serverMemoryTtlMs - 1), false);
  assert.equal(serverMemoryStale(conv, T0 + POLICY.serverMemoryTtlMs + 1), true);
  assert.equal(serverMemoryStale(newConversation(T0), T0 + 10 * POLICY.serverMemoryTtlMs), false); // no messages yet
});

// ------------------------------------------------------------------ storage
test("save then load round-trips", () => {
  const storage = memoryStorage();
  const conv = conversationWithTurns(2);
  saveConversations([conv], storage);
  assert.deepEqual(loadConversations(storage), [conv]);
});

test("load survives broken or blocked storage", () => {
  const throwing = {
    getItem() {
      throw new Error("blocked");
    },
    setItem() {
      throw new Error("blocked");
    },
  };
  assert.deepEqual(loadConversations(throwing), []);
  assert.doesNotThrow(() => saveConversations([newConversation(T0)], throwing));
  assert.deepEqual(loadConversations(memoryStorage({ [POLICY.storageKey]: "{not json" })), []);
  assert.deepEqual(loadConversations(memoryStorage({ [POLICY.storageKey]: '{"a":1}' })), []);
});

test("load drops malformed conversations and re-masks stored text", () => {
  const good = conversationWithTurns(1);
  const tampered = {
    ...good,
    id: "c2",
    messages: [{ role: "user", text: "PAN ABCPE1234F", masked: {}, at: T0 }],
  };
  const storage = memoryStorage({
    [POLICY.storageKey]: JSON.stringify([good, { id: 5 }, null, tampered]),
  });
  const loaded = loadConversations(storage);
  assert.equal(loaded.length, 2);
  assert.equal(loaded[1].messages[0].text, "PAN [PAN_REDACTED]");
});
