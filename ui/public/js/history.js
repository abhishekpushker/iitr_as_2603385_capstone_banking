// Conversation history for the chat UI and its summarisation policy (D-058).
//
// * Storage: conversations live in this browser's localStorage only. Every user message is
//   masked with the server's rules before it is stored, so no raw PAN/Aadhaar/bank-account
//   number is ever written to disk by the browser either.
// * Summary: each conversation is summarised from the structured replies, not from free text:
//   the applications looked up (latest status, escalation), the policy topics that answered it
//   (from the cited documents), and how many questions were asked and declined. Deterministic,
//   no model call, nothing invented.
// * Display: the latest `visibleTurns` turns are shown in full; earlier turns fold into a
//   summary card that can be expanded.
// * Retention: at most `maxConversations` conversations (least recently updated dropped first)
//   and `maxMessages` messages each (oldest dropped first; the title is kept).
// * Server memory: the API forgets a session after 3600 s idle, so after `serverMemoryTtlMs`
//   the UI warns that follow-ups ("it", "its") can no longer be resolved.

import { maskPii } from "./pii.js";

export const POLICY = Object.freeze({
  storageKey: "credagent.chat.v1",
  maxConversations: 30,
  maxMessages: 200,
  visibleTurns: 6,
  titleChars: 60,
  serverMemoryTtlMs: 55 * 60 * 1000,
});

const LETTERS = "abcdefghijklmnopqrstuvwxyz";

/** `ui-` + 20 random letters: matches the API's session-id pattern and, having no digits,
 * can never look like PII to the server's masker (which would reject it). */
export function newSessionId() {
  const bytes = new Uint8Array(20);
  globalThis.crypto.getRandomValues(bytes);
  return "ui-" + Array.from(bytes, (b) => LETTERS[b % LETTERS.length]).join("");
}

export function newConversation(now = Date.now()) {
  return {
    id: newSessionId().slice(3),
    sessionId: newSessionId(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function cut(text, max) {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length <= max ? flat : flat.slice(0, max - 1).trimEnd() + "…";
}

function withMessage(conv, message, now) {
  return { ...conv, updatedAt: now, messages: [...conv.messages, message] };
}

export function addUserMessage(conv, rawText, now = Date.now()) {
  const { text, counts } = maskPii(rawText);
  const next = withMessage(conv, { role: "user", text, masked: counts, at: now }, now);
  const first = conv.messages.every((m) => m.role !== "user");
  return first ? { ...next, title: cut(text, POLICY.titleChars) } : next;
}

const ANSWER_FIELDS = [
  "trace_id", "answer", "intent", "grounded", "refused", "refusal_reason", "sources",
  "application", "escalation", "review", "tools_invoked", "cache_hit", "usage", "timings_ms",
  "elapsed_ms", // measured by the browser: the round trip as the user waited for it
];

export function addAnswer(conv, response, now = Date.now()) {
  const kept = Object.fromEntries(ANSWER_FIELDS.map((f) => [f, response[f] ?? null]));
  kept.answer = maskPii(String(kept.answer ?? "")).text;
  return withMessage(conv, { role: "assistant", ...kept, at: now }, now);
}

export function addError(conv, error, now = Date.now()) {
  const message = { role: "error", code: error.code, message: error.message, trace_id: error.trace_id ?? null, at: now };
  return withMessage(conv, message, now);
}

// ------------------------------------------------------------------ summarisation
export function topicName(docId) {
  const words = String(docId).replace(/^kb-(?:\d+-)?/, "").split("-").filter(Boolean).join(" ");
  return words ? words[0].toUpperCase() + words.slice(1) : String(docId);
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Summary of a list of messages: `{ applications, topics, questions, declined, line }`. */
export function summarize(messages) {
  const apps = new Map();
  const topics = [];
  let questions = 0;
  let declined = 0;
  for (const m of messages) {
    if (m.role === "user") questions += 1;
    else if (m.role === "error") declined += 1;
    else if (m.role === "assistant") {
      if (m.refused) {
        declined += 1;
        continue;
      }
      if (m.application) {
        apps.delete(m.application.record_id); // re-insert: latest status, latest position
        apps.set(m.application.record_id, {
          id: m.application.record_id,
          category: m.application.category,
          status: m.application.status,
          escalated: Boolean(m.escalation?.recommended),
        });
      }
      for (const s of m.sources ?? []) {
        const name = topicName(s.doc_id);
        if (!topics.includes(name)) topics.push(name);
      }
    }
  }
  const applications = [...apps.values()];
  const parts = [];
  if (applications.length) parts.push(applications.map((a) => `${a.id} (${a.category}, ${a.status})`).join(", "));
  if (topics.length) parts.push(topics.join(", "));
  if (questions) parts.push(plural(questions, "question") + (declined ? `, ${declined} declined` : ""));
  return { applications, topics, questions, declined, line: parts.length ? parts.join(" · ") : "No questions yet" };
}

/** Splits messages into turns (a user message and the replies after it). */
function turns(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === "user" || out.length === 0) out.push([m]);
    else out.at(-1).push(m);
  }
  return out;
}

export function splitForDisplay(messages, visibleTurns = POLICY.visibleTurns) {
  const all = turns(messages);
  const fold = Math.max(0, all.length - visibleTurns);
  return { older: all.slice(0, fold).flat(), recent: all.slice(fold).flat() };
}

// ------------------------------------------------------------------ retention
export function prune(conversations) {
  return [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, POLICY.maxConversations)
    .map((c) => (c.messages.length > POLICY.maxMessages ? { ...c, messages: c.messages.slice(-POLICY.maxMessages) } : c));
}

export function serverMemoryStale(conv, now = Date.now()) {
  return conv.messages.length > 0 && now - conv.updatedAt > POLICY.serverMemoryTtlMs;
}

// ------------------------------------------------------------------ storage
function isConversation(c) {
  return (
    c !== null &&
    typeof c === "object" &&
    typeof c.id === "string" &&
    typeof c.sessionId === "string" &&
    typeof c.title === "string" &&
    Number.isFinite(c.createdAt) &&
    Number.isFinite(c.updatedAt) &&
    Array.isArray(c.messages)
  );
}

function remask(c) {
  const messages = c.messages
    .filter((m) => m && typeof m === "object" && typeof m.role === "string")
    .map((m) => (m.role === "user" ? { ...m, text: maskPii(String(m.text ?? "")).text } : m));
  return { ...c, title: maskPii(c.title).text, messages };
}

export function loadConversations(storage = globalThis.localStorage) {
  try {
    const parsed = JSON.parse(storage.getItem(POLICY.storageKey) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isConversation).map(remask) : [];
  } catch {
    return [];
  }
}

export function saveConversations(conversations, storage = globalThis.localStorage) {
  try {
    storage.setItem(POLICY.storageKey, JSON.stringify(prune(conversations)));
  } catch {
    // Storage may be full, blocked or unavailable (private window); the chat still works.
  }
}
