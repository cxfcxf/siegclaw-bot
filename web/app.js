const $ = (s) => document.querySelector(s);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };

// --- Persistence (last-used provider + per-provider model) ------------------
// Stored in localStorage so the sidebar restores to where you left off across
// reloads. The model is keyed per provider so switching providers doesn't leak
// a stale model id from one into another.
const PROVIDER_KEY = "harness.provider";
const modelKey = (p) => `harness.model.${p}`;
function savePref(k, v) { try { localStorage.setItem(k, v); } catch {} }
function loadPref(k) { try { return localStorage.getItem(k); } catch { return null; } }

const state = {
  providers: [],
  models: [],
  modelContext: {},    // {providerId: {modelId: maxContextLength}}
  promptTokens: null,  // last-reported prompt-token count for the open conversation
  conversationId: null,
  streaming: false,
  streamConvId: null,   // conversation id with an active background stream (for sidebar indicator)
  attachments: [],   // [{ url }]
  think: true,
  abort: null,       // AbortController for the in-flight stream (stop button)
};

// --- Thinking toggle --------------------------------------------------------
$("#think-toggle").onclick = () => {
  state.think = !state.think;
  $("#think-toggle").classList.toggle("on", state.think);
};

// --- Providers --------------------------------------------------------------
async function loadProviders() {
  const data = await fetch("/api/providers").then((r) => r.json());
  state.providers = data.providers;
  // Build per-provider {modelId: contextLength} lookup for the ctx meter.
  state.modelContext = {};
  state.providers.forEach((p) => {
    if (p.model_context) state.modelContext[p.id] = p.model_context;
  });
  const psel = $("#provider");
  psel.innerHTML = "";
  if (!state.providers.length) {
    psel.innerHTML = '<option>No providers (set keys in .env)</option>';
    $("#model").value = "";
    $("#model").disabled = true;
    renderCtxMeter();
    return;
  }
  state.providers.forEach((p) => {
    const o = el("option");
    o.value = p.id;
    o.textContent = p.name;
    psel.appendChild(o);
  });
  // Restore the last-used provider if it's still available.
  const savedProvider = loadPref(PROVIDER_KEY);
  if (savedProvider && state.providers.some((p) => p.id === savedProvider)) {
    psel.value = savedProvider;
  }
  populateModels();
}

function populateModels() {
  const pid = $("#provider").value;
  const p = state.providers.find((x) => x.id === pid);
  const msel = $("#model");
  const list = $("#model-list");
  list.innerHTML = "";
  list.hidden = true;
  state.models = (p && p.models) || [];
  if (!state.models.length) {
    msel.value = "";
    msel.placeholder = "(type a model name)";
    msel.disabled = true;
    return;
  }
  msel.disabled = false;
  msel.placeholder = "Pick or type a model…";
  // Restore this provider's last-used model (if any).
  msel.value = loadPref(modelKey(pid)) || "";
  renderCtxMeter();
}

// --- Context-window meter ---------------------------------------------------
// K = 1000 tokens (integer), M kicks in past 1024K (1M = 1024K).
function fmtCtx(n) {
  if (n == null) return "?";
  if (n >= 1000) {
    const k = n / 1000;
    if (k >= 1024) {
      const m = k / 1024;
      return (m >= 10 ? Math.round(m) : m.toFixed(1).replace(/\.0$/, "")) + "M";
    }
    return Math.round(k) + "K";
  }
  return String(n);
}

function currentMaxContext() {
  const p = $("#provider").value;
  const m = $("#model").value.trim();
  const ctx = state.modelContext[p];
  return (ctx && m && ctx[m]) || null;
}

function renderCtxMeter() {
  const meter = $("#ctx-meter");
  if (!meter) return;
  const max = currentMaxContext();
  const used = state.promptTokens;
  if (max) {
    const pct = used ? Math.min(100, Math.round((used / max) * 100)) : 0;
    meter.textContent = `${pct}% · ${fmtCtx(max)} ctx`;
    meter.classList.toggle("high", pct >= 80);
  } else if (used) {
    meter.textContent = `${fmtCtx(used)} used`;
    meter.classList.remove("high");
  } else {
    meter.textContent = "";
    meter.classList.remove("high");
  }
}

// Filter the model dropdown by the input text (case-insensitive substring).
const MODEL_LIST_CAP = 200;
function renderModelList() {
  const list = $("#model-list");
  const input = $("#model");
  const filter = input.value.trim().toLowerCase();
  const matches = filter
    ? state.models.filter((m) => m.toLowerCase().includes(filter))
    : state.models;
  list.innerHTML = "";
  matches.slice(0, MODEL_LIST_CAP).forEach((m) => {
    const li = el("li");
    li.textContent = m;
    li.title = m;
    li.dataset.value = m;
    if (m === input.value) li.classList.add("selected");
    list.appendChild(li);
  });
  if (matches.length > MODEL_LIST_CAP) {
    const more = el("li", "more");
    more.textContent = `…${matches.length - MODEL_LIST_CAP} more — keep typing`;
    list.appendChild(more);
  }
  list.hidden = matches.length === 0;
  markActive(0);
}

function markActive(i) {
  const items = $("#model-list").querySelectorAll("li:not(.more)");
  items.forEach((li, j) => li.classList.toggle("active", i === j));
}

function pickModel(value) {
  const input = $("#model");
  input.value = value;
  const list = $("#model-list");
  list.hidden = true;
  savePref(modelKey($("#provider").value), value);
  renderCtxMeter();
}

$("#provider").addEventListener("change", () => {
  savePref(PROVIDER_KEY, $("#provider").value);
  populateModels();
});

// Combobox interactions on the model input.
(() => {
  const input = $("#model");
  const list = $("#model-list");
  input.addEventListener("focus", () => {
    if (state.models.length) renderModelList();
  });
  input.addEventListener("input", renderModelList);
  input.addEventListener("blur", () => {
    // Delay so a click on an item registers before the list disappears.
    setTimeout(() => { list.hidden = true; }, 150);
    // Persist free-typed model ids (picks are saved in pickModel).
    const v = input.value.trim();
    if (v) savePref(modelKey($("#provider").value), v);
    renderCtxMeter();
  });
  input.addEventListener("keydown", (e) => {
    if (list.hidden) return;
    const items = [...list.querySelectorAll("li:not(.more)")];
    if (!items.length) return;
    const cur = list.querySelector("li.active");
    let idx = cur ? items.indexOf(cur) : -1;
    if (e.key === "ArrowDown") { e.preventDefault(); markActive((idx + 1) % items.length); items[(idx + 1) % items.length].scrollIntoView({ block: "nearest" }); }
    else if (e.key === "ArrowUp") { e.preventDefault(); markActive((idx - 1 + items.length) % items.length); items[(idx - 1 + items.length) % items.length].scrollIntoView({ block: "nearest" }); }
    else if (e.key === "Enter" && idx >= 0) { e.preventDefault(); pickModel(items[idx].dataset.value); }
    else if (e.key === "Escape") { list.hidden = true; }
  });
  list.addEventListener("mousedown", (e) => {
    // mousedown fires before the input's blur; use it so we can pick + keep focus.
    const li = e.target.closest("li:not(.more)");
    if (li) { e.preventDefault(); pickModel(li.dataset.value); input.focus(); }
  });
})();

// --- Conversations ----------------------------------------------------------
function convGroup(ts) {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86400000;
  const t = (ts || 0) * 1000;
  if (t >= startToday) return "Today";
  if (t >= startToday - day) return "Yesterday";
  if (t >= startToday - 7 * day) return "Previous 7 days";
  if (t >= startToday - 30 * day) return "Previous 30 days";
  const d = new Date(t);
  return d.getFullYear() === now.getFullYear()
    ? d.toLocaleString(undefined, { month: "long" })
    : d.toLocaleString(undefined, { month: "long", year: "numeric" });
}

async function loadConversations() {
  const data = await fetch("/api/conversations").then((r) => r.json());
  const box = $("#conversations");
  box.innerHTML = "";
  let lastGroup = null;
  // API returns newest-first, so groups emerge in order.
  data.conversations.forEach((c) => {
    const group = convGroup(c.updated_at || c.created_at);
    if (group !== lastGroup) {
      const h = el("div", "conv-group");
      h.textContent = group;
      box.appendChild(h);
      lastGroup = group;
    }
    const item = el("div", "conv");
    if (c.id === state.conversationId) item.classList.add("active");
    if (state.streaming && c.id === state.streamConvId) item.classList.add("streaming");
    const title = el("span", "title");
    title.textContent = c.title || "Untitled";
    const del = el("button", "del");
    del.type = "button";
    del.textContent = "✕";
    del.title = "Delete conversation";
    del.onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (state.conversationId === c.id) newChat();
      loadConversations();
    };
    item.append(title, del);
    item.onclick = () => openConversation(c.id);
    box.appendChild(item);
  });
}

// --- Collapsible sidebar ----------------------------------------------------
function setSidebarCollapsed(collapsed) {
  $("#app").classList.toggle("sidebar-collapsed", collapsed);
  try { localStorage.setItem("harness.sidebar", collapsed ? "1" : "0"); } catch (_) {}
}
$("#collapse-sidebar").onclick = () => setSidebarCollapsed(true);
$("#expand-sidebar").onclick = () => setSidebarCollapsed(false);
try { if (localStorage.getItem("harness.sidebar") === "1") setSidebarCollapsed(true); } catch (_) {}

async function openConversation(id) {
  state.conversationId = id;
  const data = await fetch(`/api/conversations/${id}`).then((r) => r.json());
  state.promptTokens = (data.conversation && data.conversation.prompt_tokens) || null;
  $("#messages").innerHTML = "";
  renderHistory(data.messages);
  setMainEmpty(data.messages.length === 0);
  updateScrollJump();
  renderCtxMeter();
  loadConversations();
}

const EMPTY_STATE = `
  <div class="empty">
    <h2>Ready when you are</h2>
  </div>`;

function newChat() {
  state.conversationId = null;
  state.promptTokens = null;
  $("#messages").innerHTML = EMPTY_STATE;
  setMainEmpty(true);
  updateScrollJump();
  renderCtxMeter();
  loadConversations();
}
$("#new-chat").onclick = newChat;

// --- Rendering --------------------------------------------------------------
const COPY_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
const RETRY_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>';
const EDIT_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>';
const METRICS_ICON = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';
const SEND_ICON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h13M12 5l7 7-7 7"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" stroke="none"><rect x="6.5" y="6.5" width="11" height="11" rx="2"/></svg>';

// Human-friendly duration: "3.2s" under a minute, else "1m 04s".
function fmtDur(ms) {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  return m + "m " + String(Math.round(s % 60)).padStart(2, "0") + "s";
}

// Parse markdown, first converting runs of tab-separated rows into GFM pipe
// tables. Local models sometimes emit raw tabs to "draw" a table; marked would
// otherwise render those as plain paragraphs.
function renderMarkdown(text) {
  return marked.parse(convertTabsToTables(text));
}

function convertTabsToTables(text) {
  const lines = text.split("\n");
  const out = [];
  let inCode = false;
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) inCode = !inCode;
    if (!inCode && hasTabColumns(line) && i + 1 < lines.length && hasTabColumns(lines[i + 1])) {
      // Collect the run of consecutive tab-separated rows.
      const rows = [];
      while (i < lines.length && hasTabColumns(lines[i])) rows.push(lines[i++]);
      out.push(toMarkdownTable(rows));
    } else {
      out.push(line);
      i++;
    }
  }
  return out.join("\n");
}

function hasTabColumns(line) {
  // A non-empty line with at least one tab between non-whitespace cells.
  return /^\S[^\n]*\t[^\n]*\S/.test(line);
}

function toMarkdownTable(rows) {
  const cells = rows.map((r) => r.split(/\t+/).map((s) => s.trim()));
  const ncols = Math.max(...cells.map((r) => r.length));
  cells.forEach((r) => { while (r.length < ncols) r.push(""); });
  const header = cells[0];
  const sep = header.map(() => "---");
  const body = cells.slice(1);
  return [
    `| ${header.join(" | ")} |`,
    `| ${sep.join(" | ")} |`,
    ...body.map((r) => `| ${r.join(" | ")} |`),
  ].join("\n");
}

function addMessageBubble(role, markdown, images, msgId) {
  const empty = $("#messages .empty");
  if (empty) empty.remove();
  const wrap = el("div", `msg ${role}`);
  if (msgId) wrap.dataset.msgId = msgId;
  wrap.dataset.rawText = markdown || "";
  if (images && images.length) wrap.dataset.images = JSON.stringify(images);
  const r = el("div", "role");
  r.textContent = role === "user" ? "you" : "model";
  const bubble = el("div", "bubble");
  bubble.innerHTML = markdown ? renderMarkdown(markdown) : "";
  if (images && images.length) {
    const box = el("div", "msg-images");
    images.forEach((u) => {
      const im = el("img");
      im.src = u;
      im.onclick = () => window.open(u, "_blank");
      box.appendChild(im);
    });
    bubble.appendChild(box);
  }

  // Hover-revealed action toolbar. Edit/retry only on user messages; edit is
  // further limited (via the .editable class) to the most recent one.
  const actions = el("div", "actions");
  if (role === "user") {
    const edit = el("button", "action edit");
    edit.type = "button";
    edit.title = "Edit & resend";
    edit.innerHTML = EDIT_ICON;
    edit.onclick = (e) => { e.stopPropagation(); editFromMessage(wrap); };
    actions.appendChild(edit);

    const retry = el("button", "action");
    retry.type = "button";
    retry.title = "Retry from here";
    retry.innerHTML = RETRY_ICON;
    retry.onclick = (e) => { e.stopPropagation(); retryFromMessage(wrap); };
    actions.appendChild(retry);
  }
  const copy = el("button", "action copy");
  copy.type = "button";
  copy.title = "Copy";
  copy.innerHTML = COPY_ICON;
  copy.onclick = (e) => { e.stopPropagation(); copyMessage(wrap, copy); };
  actions.appendChild(copy);

  wrap.append(r, bubble, actions);
  $("#messages").appendChild(wrap);
  if (role === "user") refreshEditAffordance();
  scroll();
  return bubble;
}

async function copyMessage(wrap, btn) {
  const text = wrap.dataset.rawText || "";
  try {
    await navigator.clipboard.writeText(text);
    btn.classList.add("copied");
    setTimeout(() => btn.classList.remove("copied"), 1400);
  } catch (err) {
    setStatus("Copy failed: " + err.message);
  }
}

// Rewind a turn: delete this user message and everything after it on the
// server (if it was persisted), then remove its bubble and all following
// blocks from the DOM. Returns false if the server rewind failed.
async function rewindToMessage(userWrap, failLabel) {
  const msgId = userWrap.dataset.msgId;
  if (state.conversationId && msgId) {
    try {
      await fetch(`/api/conversations/${state.conversationId}/rewind/${msgId}`, { method: "DELETE" });
    } catch (err) {
      setStatus(`${failLabel} failed: ` + err.message);
      return false;
    }
  }
  let next = userWrap.nextElementSibling;
  while (next) { const cur = next; next = next.nextElementSibling; cur.remove(); }
  userWrap.remove();
  return true;
}

async function retryFromMessage(userWrap) {
  if (state.streaming) { setStatus("Wait for the current turn to finish…"); return; }
  const text = userWrap.dataset.rawText || "";
  const imagesJson = userWrap.dataset.images;
  if (!userWrap.dataset.msgId) { setStatus("Can't retry — message id missing."); return; }
  if (!state.conversationId) return;

  if (!(await rewindToMessage(userWrap, "Retry"))) return;

  // Re-send via the composer path (handles the empty-state transition).
  const images = imagesJson ? JSON.parse(imagesJson) : [];
  await send(text, images);
}

// Edit the last message you sent: rewind from it and drop its text + images
// back into the composer so you can change it and send again.
async function editFromMessage(userWrap) {
  if (state.streaming) { setStatus("Stop the current turn before editing."); return; }
  const text = userWrap.dataset.rawText || "";
  const imagesJson = userWrap.dataset.images;

  if (!(await rewindToMessage(userWrap, "Edit"))) return;

  // Restore the message into the composer for editing.
  const input = $("#input");
  input.value = text;
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
  state.attachments = (imagesJson ? JSON.parse(imagesJson) : []).map((url) => ({ url }));
  renderAttachments();

  // If the conversation is now empty, return to the centered landing layout.
  if (!$("#messages").querySelector(".msg")) {
    $("#messages").innerHTML = EMPTY_STATE;
    setMainEmpty(true);
  }
  refreshEditAffordance();
  input.focus();
  loadConversations();
}

// Show the edit button only on the most recent message you sent.
function refreshEditAffordance() {
  const users = document.querySelectorAll("#messages > .msg.user");
  users.forEach((u, i) => u.classList.toggle("editable", i === users.length - 1));
}

// A collapsible <details> with the shared chevron summary. Returns { d, summary };
// callers fill in the summary's inner markup and append their own body.
function detailsBlock(cls, summaryHtml, open = false) {
  const d = el("details", cls);
  d.open = open;
  const summary = el("summary");
  summary.innerHTML = summaryHtml;
  return { d, summary };
}

function addToolBlock(name, args) {
  const { d, summary } = detailsBlock(
    "tool",
    `<span class="tname">${name}</span><span class="args"></span><span class="spinner">●</span>`
  );
  summary.querySelector(".args").textContent = truncate(args, 80);
  const body = el("div", "body");
  const pre = el("pre");
  pre.textContent = "running…";
  body.appendChild(pre);
  d.append(summary, body);
  if (stepTools) {
    // Nest under the reasoning step that triggered it.
    stepTools.appendChild(d);
    stepCount += 1;
    if (stepCountEl) stepCountEl.textContent = `· ${stepCount} tool${stepCount === 1 ? "" : "s"}`;
  } else {
    ensureTrace().appendChild(d);  // no preceding thinking — keep it top-level
  }
  bumpTrace("tool");
  scroll();
  return { details: d, pre, spinner: summary.querySelector(".spinner") };
}

function addThinkingBlock(initial, open = false) {
  const { d, summary } = detailsBlock(
    "thinking",
    '<span>Thinking</span><span class="step-count"></span><span class="live"></span>',
    open
  );
  const reason = el("div", "reason");
  reason.textContent = initial || "";
  const tools = el("div", "step-tools");  // tools from this reasoning step nest here
  d.append(summary, reason, tools);
  ensureTrace().appendChild(d);
  bumpTrace("thinking");
  // This block becomes the active step; subsequent tool calls nest into it.
  stepTools = tools;
  stepCount = 0;
  stepCountEl = summary.querySelector(".step-count");
  scroll();
  return { details: d, reason, live: summary.querySelector(".live") };
}

const MEM_GLYPH = { ADD: "+", UPDATE: "~", DELETE: "−", NOOP: "·" };
function addMemoryChip(changes) {
  const chip = el("div", "mem-chip");
  const label = el("div", "chip-label");
  label.textContent = `memory updated · ${changes.length} change${changes.length === 1 ? "" : "s"}`;
  const list = el("ul");
  changes.forEach((c) => {
    const li = el("li");
    const ev = (c.event || "ADD").toUpperCase();
    li.classList.add(ev.toLowerCase());
    const g = el("span", "g"); g.textContent = MEM_GLYPH[ev] || "·";
    const t = el("span", "t"); t.textContent = c.memory;
    li.append(g, t);
    list.appendChild(li);
  });
  chip.append(label, list);
  ensureTrace().appendChild(chip);
  bumpTrace("memory");
  scroll();
}

// --- Trace container: one collapsed wrapper per turn for all process events ---
let currentTrace = null;        // the outer <details class="trace">
let currentTraceBody = null;    // its .trace-body div
const currentTraceCounts = { tool: 0, thinking: 0, memory: 0 };

// The active "step": tools called after a reasoning phase nest inside that
// thinking block, so collapsing the thinking hides its tools too. Null when no
// thinking preceded the tools (e.g. thinking off) — those stay top-level.
let stepTools = null;     // the active thinking block's .step-tools container
let stepCount = 0;        // tools nested in the active step
let stepCountEl = null;   // the active step's "· N tools" badge

function resetTrace() {
  currentTrace = null;
  currentTraceBody = null;
  currentTraceCounts.tool = 0;
  currentTraceCounts.thinking = 0;
  currentTraceCounts.memory = 0;
  stepTools = null;
  stepCount = 0;
  stepCountEl = null;
}

function ensureTrace() {
  if (currentTraceBody) return currentTraceBody;
  const empty = $("#messages .empty");
  if (empty) empty.remove();
  const { d, summary } = detailsBlock(
    "trace",
    '<span class="tlabel">process</span><span class="tactivity"></span><span class="tmeta"></span><span class="tclock"></span>'
  );
  const body = el("div", "trace-body");
  d.append(summary, body);
  $("#messages").appendChild(d);
  currentTrace = d;
  currentTraceBody = body;
  updateTraceSummary();
  return body;
}

function bumpTrace(kind) {
  currentTraceCounts[kind] = (currentTraceCounts[kind] || 0) + 1;
  updateTraceSummary();
}

function updateTraceSummary() {
  if (!currentTrace) return;
  const c = currentTraceCounts;
  const parts = [];
  if (c.tool) parts.push(`${c.tool} tool${c.tool === 1 ? "" : "s"}`);
  if (c.thinking) parts.push("thinking");
  if (c.memory) parts.push(`${c.memory} memor${c.memory === 1 ? "y" : "ies"}`);
  currentTrace.querySelector(".tmeta").textContent = parts.join(" · ") || "—";
}

function setTraceLive(on) {
  if (currentTrace) currentTrace.classList.toggle("live", on);
}

// Toggle the trace activity light's mode ("thinking" patina dot / "tool" brass dot).
function setTraceActivity(kind, on) {
  if (!currentTrace) return;
  const act = currentTrace.querySelector(".tactivity");
  if (act) act.classList.toggle(kind, on);
}
const setTraceThinking = (on) => setTraceActivity("thinking", on);
const setTraceTool = (on) => setTraceActivity("tool", on);

function renderHistory(messages) {
  const toolBlocks = {};
  let lastAssistant = null;
  for (const m of messages) {
    if (m.role === "user") {
      resetTrace();
      addMessageBubble("user", m.content || "", m.images, m.id);
    }
    else if (m.role === "assistant") {
      if (m.reasoning) addThinkingBlock(m.reasoning, false);
      if (m.content) lastAssistant = addMessageBubble("assistant", m.content, null, m.id);
      (m.tool_calls || []).forEach((tc) => {
        const block = addToolBlock(tc.function.name, tc.function.arguments || "");
        toolBlocks[tc.id] = block;
      });
    } else if (m.role === "tool") {
      const block = toolBlocks[m.tool_call_id];
      if (block) { block.pre.textContent = m.content || ""; block.spinner.remove(); }
    }
  }
  if (!messages.length) $("#messages").innerHTML = EMPTY_STATE;
}

const truncate = (s, n) => (s && s.length > n ? s.slice(0, n) + "…" : s || "");
const scroll = () => { const m = $("#messages"); m.scrollTop = m.scrollHeight; updateScrollJump(); };

// --- Scroll-jump button -----------------------------------------------------
// At the bottom it offers to jump to the first message; once scrolled up it
// flips to jump back to the newest message.
function updateScrollJump() {
  const m = $("#messages");
  const btn = $("#scroll-jump");
  if (!m || !btn) return;
  const scrollable = m.scrollHeight - m.clientHeight > 60;
  if (!scrollable) { btn.classList.remove("show"); return; }
  btn.classList.add("show");
  const atBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 80;
  btn.classList.toggle("down", !atBottom);
  btn.title = atBottom ? "Jump to first message" : "Jump to newest message";
}
$("#scroll-jump").addEventListener("click", () => {
  const m = $("#messages");
  const toBottom = $("#scroll-jump").classList.contains("down");
  m.scrollTo({ top: toBottom ? m.scrollHeight : 0, behavior: "smooth" });
});
$("#messages").addEventListener("scroll", updateScrollJump);
window.addEventListener("resize", updateScrollJump);

// Toggle the centered "landing" layout (greeting + composer in the middle)
// vs the active layout (messages flow, composer pinned to bottom).
// Uses FLIP to animate the composer sliding from the middle to the bottom.
function setMainEmpty(empty) {
  const main = $("#main");
  const wasEmpty = main.classList.contains("is-empty");
  if (empty === wasEmpty) return;

  if (!empty && wasEmpty) {
    // FLIP: capture "first" positions, apply the layout change, then animate
    // from the old position to the new one. We track the composer plus any
    // children of #messages so the user bubble slides up in sync.
    const composer = $("#composer");
    const msgs = $("#messages");
    const tracked = [composer, ...msgs.children];
    const firstTops = tracked.map((el) => el.getBoundingClientRect().top);

    main.classList.remove("is-empty");

    const lastTops = tracked.map((el) => el.getBoundingClientRect().top);
    const duration = "0.45s";
    const easing = "cubic-bezier(0.2, 0.7, 0.2, 1)";

    tracked.forEach((el, i) => {
      const dy = firstTops[i] - lastTops[i];
      if (Math.abs(dy) < 1) return;
      el.style.transition = "none";
      el.style.transform = `translateY(${dy}px)`;
    });

    document.body.offsetHeight;  // force reflow so the inverse transform lands

    tracked.forEach((el) => {
      if (el.style.transform) {
        el.style.transition = `transform ${duration} ${easing}`;
        el.style.transform = "";
      }
    });

    const cleanup = () => {
      tracked.forEach((el) => {
        el.style.transition = "";
        el.style.transform = "";
      });
    };
    composer.addEventListener("transitionend", cleanup, { once: true });
    setTimeout(cleanup, 600);  // safety net
  } else {
    main.classList.add("is-empty");
  }
}

// --- Attachments ------------------------------------------------------------
$("#attach").onclick = () => $("#file-input").click();
$("#file-input").addEventListener("change", async (e) => {
  for (const file of e.target.files) await uploadAttachment(file);
  e.target.value = "";
});
// Paste an image directly into the composer.
$("#input").addEventListener("paste", async (e) => {
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      await uploadAttachment(item.getAsFile());
    }
  }
});

async function uploadAttachment(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const att = { url: null, uploading: true };
  state.attachments.push(att);
  renderAttachments();
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json());
    if (res.url) { att.url = res.url; att.uploading = false; }
    else { state.attachments = state.attachments.filter((a) => a !== att); setStatus("Upload failed."); }
  } catch (err) {
    state.attachments = state.attachments.filter((a) => a !== att);
    setStatus("Upload error: " + err.message);
  }
  renderAttachments();
}

function renderAttachments() {
  const box = $("#attachments");
  box.innerHTML = "";
  state.attachments.forEach((att) => {
    const thumb = el("div", "thumb" + (att.uploading ? " uploading" : ""));
    const im = el("img");
    im.src = att.url || "";
    const rm = el("button", "rm"); rm.type = "button"; rm.textContent = "✕";
    rm.onclick = () => { state.attachments = state.attachments.filter((a) => a !== att); renderAttachments(); };
    thumb.append(im, rm);
    box.appendChild(thumb);
  });
}

// --- Sending / streaming ----------------------------------------------------
// While streaming the send button is a stop control — submitting aborts the turn.
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.streaming) stopStream(); else send();
});
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!state.streaming) send(); }
});

// Abort the in-flight stream regardless of where it is (thinking, tools, tokens).
function stopStream() {
  if (state.abort) { state.abort.abort(); state.abort = null; }
}
$("#input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
});

async function send(providedText, providedImages) {
  if (state.streaming) return;
  // Retry passes an explicit text/images; otherwise read from the composer.
  const isRetry = providedText !== undefined;
  if (!isRetry && state.attachments.some((a) => a.uploading)) {
    setStatus("Wait for image upload to finish…"); return;
  }
  const text = (isRetry ? providedText : $("#input").value).trim();
  const provider = $("#provider").value;
  const model = $("#model").value;
  const images = isRetry ? (providedImages || []) : state.attachments.map((a) => a.url).filter(Boolean);
  if (!text && !images.length) return;
  if (!provider || !model) { setStatus("Pick a provider and model first."); return; }

  addMessageBubble("user", text, images);
  if (!isRetry) {
    $("#input").value = "";
    $("#input").style.height = "auto";
    state.attachments = [];
    renderAttachments();
  }
  setStreaming(true);
  setMainEmpty(false);
  resetTrace();
  setTraceLive(true);

  // Track which conversation this stream belongs to. If the user clicks another
  // chat in the sidebar, we mark the stream "backgrounded" — events keep being
  // consumed (so the backend finishes cleanly) but stop touching the DOM.
  let streamConvId = state.conversationId;
  state.streamConvId = streamConvId;
  let streamBackgrounded = false;

  const toolBlocks = {};
  let assistantBubble = null;
  let assistantText = "";
  let thinkingBlock = null;

  // --- Turn metrics: live elapsed timer + thinking time + tok/s -------------
  // Wall-clock from the moment we start the turn (00) until the stream closes.
  // While there's no answer bubble yet (thinking/tool phase) the live time shows
  // in the status pill; once the answer starts it lives in a chip after the copy
  // icon under the response, then settles to total · thought · tok/s on done.
  const turnStart = performance.now();
  let firstTokenAt = null;
  let thinkingMs = 0, thinkStart = null;
  let completionTokens = 0;
  let metricsEl = null;
  const liveTimer = setInterval(renderLiveMetrics, 100);

  function thinkTick(active) {
    if (active) { if (thinkStart === null) thinkStart = performance.now(); }
    else if (thinkStart !== null) { thinkingMs += performance.now() - thinkStart; thinkStart = null; }
  }
  function ensureMetricsEl(bubble) {
    if (!metricsEl) {
      metricsEl = el("span", "metrics live");
      metricsEl.innerHTML = METRICS_ICON + '<span class="mtext"></span>';
    }
    // Follow the latest answer bubble (a tool call starts a fresh bubble).
    const actions = bubble.closest(".msg").querySelector(".actions");
    if (actions && metricsEl.parentElement !== actions) actions.appendChild(metricsEl);
  }
  function setMetricsText(text, title) {
    if (metricsEl) {
      metricsEl.querySelector(".mtext").textContent = text;
      if (title) metricsEl.title = title;
    }
    // Before the answer bubble exists the live time shows in the process trace
    // clock (set in renderLiveMetrics); nothing goes to the status pill.
  }
  function renderLiveMetrics() {
    const t = fmtDur(performance.now() - turnStart);
    setMetricsText(t);  // chip if the answer has started, else the status pill
    // Also tick the process trace, so the timer is visible during thinking/tools.
    const clock = currentTrace && currentTrace.querySelector(".tclock");
    if (clock) clock.textContent = t;
  }
  function finalizeMetrics() {
    clearInterval(liveTimer);
    thinkTick(false);
    const total = performance.now() - turnStart;
    // Leave the final total on the (collapsed) process trace.
    const clock = currentTrace && currentTrace.querySelector(".tclock");
    if (clock) clock.textContent = fmtDur(total);
    if (!metricsEl) return;  // tool-only or errored turn with no answer bubble
    metricsEl.classList.remove("live");
    const genMs = firstTokenAt ? performance.now() - firstTokenAt : 0;
    const parts = [fmtDur(total)];
    const title = ["total " + fmtDur(total)];
    if (thinkingMs > 200) { parts.push("thought " + fmtDur(thinkingMs)); title.push("thinking " + fmtDur(thinkingMs)); }
    if (completionTokens > 0 && genMs > 250) {
      parts.push((completionTokens / (genMs / 1000)).toFixed(1) + " tok/s");
      title.push(completionTokens + " tokens in " + fmtDur(genMs));
    }
    setMetricsText(parts.join(" · "), title.join(" · "));
  }

  const controller = new AbortController();
  state.abort = controller;
  let aborted = false;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: state.conversationId, provider, model, message: text, images: images.length ? images : null, think: state.think }),
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();
      for (const frame of frames) {
        const line = frame.replace(/^data: /, "").trim();
        if (!line) continue;
        const ev = JSON.parse(line);
        handleEvent(ev);
      }
    }
  } catch (err) {
    if (err.name === "AbortError") aborted = true;
    else setStatus("Error: " + err.message);
  } finally {
    state.abort = null;
    finalizeMetrics();
    setTraceLive(false);
    setStreaming(false);
    if (aborted) setStatus("stopped");
    state.streamConvId = null;
    loadConversations();
  }

  function endThinking() {
    if (thinkingBlock) { thinkingBlock.live.textContent = ""; thinkingBlock = null; }
    setTraceThinking(false);
  }

  function handleEvent(ev) {
    // The conversation event must always be processed (it tells us the cid for
    // brand-new chats) and shouldn't be affected by backgrounding.
    if (ev.type === "conversation") {
      if (!state.conversationId) state.conversationId = ev.id;
      if (streamConvId !== state.conversationId) {
        streamConvId = state.conversationId;
        state.streamConvId = streamConvId;
        loadConversations();  // refresh sidebar so the streaming dot shows
      }
      return;
    }

    // The user_message event carries the stored id of the message we just sent
    // — attach it to the most recent user bubble so retry works post-send.
    if (ev.type === "user_message") {
      const userWraps = document.querySelectorAll("#messages > .msg.user");
      if (userWraps.length) userWraps[userWraps.length - 1].dataset.msgId = ev.id;
      return;
    }

    // If the user has switched to another conversation, stop rendering. The
    // backend stream continues; we just detach from the DOM. Once backgrounded,
    // we stay backgrounded to avoid duplicates with the reloaded history.
    if (!streamBackgrounded && state.conversationId !== streamConvId) {
      streamBackgrounded = true;
    }
    if (streamBackgrounded) {
      // When the stream completes, auto-refresh if the user is back on this chat
      // so they see the final persisted state instead of a stale snapshot.
      if (ev.type === "done" && state.conversationId === streamConvId) {
        openConversation(streamConvId);
      }
      return;
    }

    switch (ev.type) {
      case "reasoning":
        thinkTick(true);
        if (!thinkingBlock) {
          thinkingBlock = addThinkingBlock("", false);
          setTraceThinking(true);
        }
        thinkingBlock.live.textContent = "…";
        thinkingBlock.reason.textContent += ev.text;
        if (thinkingBlock.details.open) scroll();
        break;
      case "token":
        thinkTick(false);
        if (firstTokenAt === null) firstTokenAt = performance.now();
        endThinking();  // reasoning for this step is done once the answer starts
        if (!assistantBubble) assistantBubble = addMessageBubble("assistant", "");
        ensureMetricsEl(assistantBubble);
        assistantText += ev.text;
        assistantBubble.innerHTML = renderMarkdown(assistantText);
        scroll();
        break;
      case "tool_call":
        thinkTick(false);
        endThinking();
        setTraceTool(true);
        toolBlocks[ev.id] = addToolBlock(ev.name, ev.arguments || "");
        assistantBubble = null; assistantText = "";  // text after a tool is a new bubble
        break;
      case "tool_result": {
        const b = toolBlocks[ev.id];
        if (b) { b.pre.textContent = ev.result; b.spinner.remove(); }
        setTraceTool(false);
        break;
      }
      case "error":
        endThinking();
        setTraceTool(false);
        setTraceLive(false);
        setStatus("⚠ " + ev.message);
        addMessageBubble("assistant", "_Error: " + ev.message + "_");
        break;
      case "memory_saving":
        endThinking();
        setStatus("saving memory…");
        break;
      case "memory_updated":
        addMemoryChip(ev.changes || []);
        break;
      case "done":
        if (ev.tokens) completionTokens = ev.tokens;
        if (ev.prompt_tokens) state.promptTokens = ev.prompt_tokens;
        endThinking();
        setTraceTool(false);
        setTraceLive(false);
        setStatus("");
        renderCtxMeter();
        break;
    }
  }
}

function setStreaming(on) {
  state.streaming = on;
  // The button stays enabled and becomes a stop control while streaming.
  const send = $("#send");
  send.classList.toggle("stop", on);
  send.innerHTML = on ? STOP_ICON : SEND_ICON;
  send.setAttribute("aria-label", on ? "Stop" : "Send");
  send.title = on ? "Stop" : "";
  // Activity is shown by the live timer now — clear any stale notice on start.
  if (on) setStatus("");
}
// The status pill is only for one-off notices (errors, "stopped"); it hides when
// empty. Routine working/ready state is no longer shown.
function setStatus(s) { $("#status").textContent = s; }

// --- soul.md & skills dialogs ----------------------------------------------
$("#edit-soul").onclick = async () => {
  const data = await fetch("/api/soul").then((r) => r.json());
  $("#soul-text").value = data.content;
  $("#soul-dialog").showModal();
};
$("#soul-save").onclick = async (e) => {
  e.preventDefault();
  await fetch("/api/soul", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: $("#soul-text").value }),
  });
  $("#soul-dialog").close();
  setStatus("soul.md saved — applies to the next turn.");
};
// --- Memory -----------------------------------------------------------------
async function renderMemories() {
  const data = await fetch("/api/memories").then((r) => r.json());
  const list = $("#memory-list");
  if (!data.memories.length) {
    list.innerHTML = '<div class="empty">No memories yet.</div>';
    return;
  }
  list.innerHTML = "";
  data.memories.forEach((m) => {
    const row = el("div", "mem");
    const id = el("span", "id"); id.textContent = "•";
    const text = el("span", "text"); text.textContent = m.content;
    const forget = el("button", "forget"); forget.type = "button"; forget.textContent = "✕"; forget.title = "Forget";
    forget.onclick = async () => {
      await fetch(`/api/memories/${m.id}`, { method: "DELETE" });
      renderMemories();
    };
    row.append(id, text, forget);
    list.appendChild(row);
  });
}

$("#show-memory").onclick = async () => {
  await renderMemories();
  $("#memory-dialog").showModal();
};
$("#memory-add").onclick = async (e) => {
  e.preventDefault();
  const input = $("#memory-input");
  const content = input.value.trim();
  if (!content) return;
  await fetch("/api/memories", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  input.value = "";
  renderMemories();
};

$("#show-skills").onclick = async () => {
  const data = await fetch("/api/skills").then((r) => r.json());
  const list = $("#skills-list");
  list.innerHTML = data.skills.length
    ? data.skills.map((s) => `<div class="skill"><b>${s.name}</b><br>${s.description}</div>`).join("")
    : '<div class="empty">No skills in ./skills</div>';
  $("#skills-dialog").showModal();
};

// --- Boot -------------------------------------------------------------------
(async function init() {
  await loadProviders();
  await loadConversations();
  newChat();
})();
