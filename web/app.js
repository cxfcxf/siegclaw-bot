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
  effortLevels: {},   // {providerId: [effort level, ...]} — empty when unsupported
  effort: null,       // currently-selected reasoning effort (null when n/a)
  defaultModel: null,  // {provider, model, effort} a new conversation starts on (server-resolved)
  promptTokens: null,  // last-reported prompt-token count for the open conversation
  conversationId: null,
  streaming: false,
  streamConvId: null,   // conversation id with an active background stream (for sidebar indicator)
  attachments: [],   // [{ url }]
  think: true,
  abort: null,       // AbortController for the in-flight stream (stop button)
  stick: true,       // follow the bottom as content streams; false once you scroll up
};

// --- Thinking toggle --------------------------------------------------------
const effortKey = (p) => `harness.effort.${p}`;
$("#think-toggle").onclick = () => {
  state.think = !state.think;
  $("#think-toggle").classList.toggle("on", state.think);
  renderEffort();
};

// Reasoning-effort selector: only shown for providers that support it, and only
// when thinking is on. Persisted per provider alongside the model choice.
function renderEffort() {
  const sel = $("#effort");
  const pid = $("#provider").value;
  const levels = state.effortLevels[pid] || [];
  if (!levels.length || !state.think) {
    sel.hidden = true;
    state.effort = null;
    return;
  }
  sel.hidden = false;
  sel.innerHTML = "";
  levels.forEach((lvl) => {
    const o = el("option");
    o.value = lvl;
    o.textContent = lvl;
    sel.appendChild(o);
  });
  const saved = loadPref(effortKey(pid));
  sel.value = levels.includes(saved) ? saved : levels[0];
  state.effort = sel.value;
}
$("#effort").addEventListener("change", () => {
  state.effort = $("#effort").value;
  savePref(effortKey($("#provider").value), state.effort);
});

// --- Providers --------------------------------------------------------------
async function loadProviders() {
  const data = await fetch("/api/providers").then((r) => r.json());
  state.providers = data.providers;
  state.defaultModel = data.default || null;
  // Build per-provider {modelId: contextLength} + effort-level lookups.
  state.modelContext = {};
  state.effortLevels = {};
  state.providers.forEach((p) => {
    if (p.model_context) state.modelContext[p.id] = p.model_context;
    if (p.effort_levels && p.effort_levels.length) state.effortLevels[p.id] = p.effort_levels;
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
  renderEffort();
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
  renderEffort();
}

// Point the provider/model/effort picker at a specific selection — used to make
// the picker follow the active conversation (resume → its stored model; new chat
// → the server default). Falls back gracefully if the provider isn't available.
function applyModelSelection(provider, model, effort) {
  const psel = $("#provider");
  if (provider && state.providers.some((p) => p.id === provider)) {
    psel.value = provider;
    savePref(PROVIDER_KEY, provider);
  }
  populateModels();              // rebuilds the model list for the chosen provider
  if (model) {
    $("#model").value = model;
    savePref(modelKey($("#provider").value), model);
  }
  renderEffort();
  if (effort) {
    const es = $("#effort");
    if (!es.hidden && [...es.options].some((o) => o.value === effort)) {
      es.value = effort;
      state.effort = effort;
      savePref(effortKey($("#provider").value), effort);
    }
  }
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
  updateModelChip();  // renderCtxMeter runs on every selection change — the chip rides along
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
    li.textContent = m;   // items wrap in CSS, so the full id is readable inline
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
  // Picking a model is the terminal action of the menu — close it and hand
  // focus back to the composer, like the Claude/Gemini pickers.
  setModelMenu(false);
  $("#input").focus();
}

$("#provider").addEventListener("change", () => {
  savePref(PROVIDER_KEY, $("#provider").value);
  populateModels();
});

// --- Model chip + popover ----------------------------------------------------
// The provider/model controls live in a popover anchored to a compact
// "provider / model" chip in the composer bar (next to send).
function updateModelChip() {
  const psel = $("#provider");
  const opt = psel.selectedIndex >= 0 ? psel.options[psel.selectedIndex] : null;
  const pname = state.providers.length && opt ? opt.textContent : "";
  const m = $("#model").value.trim();
  $("#chip-provider").textContent = pname;
  $("#chip-provider").hidden = !pname;
  $("#chip-model").textContent = m || (state.providers.length ? "choose model" : "no providers");
  $("#model-chip").classList.toggle("unset", !m);
  // Full untruncated readout inside the popover — the chip ellipsizes long ids,
  // and a hover tooltip would cover the popover's own controls.
  $("#model-current").textContent = m ? `${pname ? pname + " / " : ""}${m}` : "no model selected";
}

function setModelMenu(open) {
  const menu = $("#model-menu");
  if (menu.hidden === !open) return;
  menu.hidden = !open;
  $("#model-chip").setAttribute("aria-expanded", String(open));
  if (open) {
    const mi = $("#model");
    if (!mi.disabled) { mi.focus(); mi.select(); }  // focus opens the filtered list
  } else {
    $("#model-list").hidden = true;
  }
}
$("#model-chip").onclick = () => setModelMenu($("#model-menu").hidden);
document.addEventListener("mousedown", (e) => {
  if (!$("#model-menu").hidden && !$("#model-menu-wrap").contains(e.target)) setModelMenu(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#model-menu").hidden) setModelMenu(false);
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
    if (e.key === "Enter") {
      // The input lives inside the composer form — never let Enter submit it.
      e.preventDefault();
      const active = list.hidden ? null : list.querySelector("li.active");
      if (active) { pickModel(active.dataset.value); return; }
      // No highlighted suggestion: accept the free-typed model id.
      const v = input.value.trim();
      if (v) savePref(modelKey($("#provider").value), v);
      renderCtxMeter();
      setModelMenu(false);
      $("#input").focus();
      return;
    }
    if (list.hidden) return;
    const items = [...list.querySelectorAll("li:not(.more)")];
    if (!items.length) return;
    const cur = list.querySelector("li.active");
    let idx = cur ? items.indexOf(cur) : -1;
    if (e.key === "ArrowDown") { e.preventDefault(); markActive((idx + 1) % items.length); items[(idx + 1) % items.length].scrollIntoView({ block: "nearest" }); }
    else if (e.key === "ArrowUp") { e.preventDefault(); markActive((idx - 1 + items.length) % items.length); items[(idx - 1 + items.length) % items.length].scrollIntoView({ block: "nearest" }); }
    // Two-stage Escape: first closes the suggestion list, a second closes the menu.
    else if (e.key === "Escape") { e.stopPropagation(); list.hidden = true; }
  });
  list.addEventListener("mousedown", (e) => {
    // mousedown fires before the input's blur, so the pick registers first.
    const li = e.target.closest("li:not(.more)");
    if (li) { e.preventDefault(); pickModel(li.dataset.value); }
  });
})();

// --- Conversations ----------------------------------------------------------
// Today / Yesterday / Previous 7 days; anything older is headed by its exact
// date (year added once it's not the current year).
function convGroup(ts) {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86400000;
  const t = (ts || 0) * 1000;
  if (t >= startToday) return "Today";
  if (t >= startToday - day) return "Yesterday";
  if (t >= startToday - 7 * day) return "Previous 7 days";
  const d = new Date(t);
  return d.toLocaleDateString(undefined, d.getFullYear() === now.getFullYear()
    ? { month: "long", day: "numeric" }
    : { month: "long", day: "numeric", year: "numeric" });
}

let convCache = [];  // newest-first; feeds both the sidebar and chat search

async function loadConversations() {
  const data = await fetch("/api/conversations").then((r) => r.json());
  convCache = data.conversations || [];
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

// --- Chat search --------------------------------------------------------------
// A Gemini-style overlay: pill search input, "Recent"/"Results" section label,
// rows of title + date. Client-side title filter over the cached conversation
// list; arrow keys + Enter to pick, click to open.
function convDateLabel(ts) {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = (ts || 0) * 1000;
  if (t >= startToday) return "Today";
  if (t >= startToday - 86400000) return "Yesterday";
  const d = new Date(t);
  return d.toLocaleDateString(undefined, d.getFullYear() === now.getFullYear()
    ? { month: "short", day: "numeric" }
    : { month: "short", day: "numeric", year: "numeric" });
}

function renderSearchRows(items, label) {
  $("#search-section").textContent = label;
  const box = $("#search-results");
  box.innerHTML = "";
  items.forEach((c, i) => {
    const row = el("div", "search-result" + (i === 0 ? " active" : ""));
    const title = el("span", "title");
    title.textContent = c.title || "Untitled";
    const date = el("span", "date");
    date.textContent = convDateLabel(c.updated_at || c.created_at);
    row.append(title, date);
    if (c.snippet) {
      // Matched terms arrive delimited by \x01…\x02 — render them highlighted.
      const sn = el("div", "snippet");
      c.snippet.split("\x01").forEach((part, j) => {
        if (j === 0) { sn.append(part); return; }
        const [hit, rest] = part.split("\x02");
        const mark = document.createElement("mark");
        mark.textContent = hit;
        sn.append(mark, rest ?? "");
      });
      row.appendChild(sn);
    }
    row.onclick = () => openConversation(c.id);  // openConversation exits search
    box.appendChild(row);
  });
}

// Two-pass search: instant title matches from the cached list, then the
// server's full-text results (FTS5 over message bodies) merged in with
// snippets. A sequence counter drops stale fetches on fast typing.
let searchSeq = 0;
async function renderSearchResults() {
  const q = $("#chat-search-input").value.trim();
  const seq = ++searchSeq;
  if (!q) { renderSearchRows(convCache, "Recent"); return; }
  const ql = q.toLowerCase();
  const titleHits = convCache.filter((c) => (c.title || "Untitled").toLowerCase().includes(ql));
  renderSearchRows(titleHits, titleHits.length ? "Results" : "Searching…");
  let results = [];
  try {
    const data = await fetch(`/api/search?q=${encodeURIComponent(q)}`).then((r) => r.json());
    results = data.results || [];
  } catch (_) {}
  if (seq !== searchSeq) return;  // a newer keystroke superseded this fetch
  const snippets = new Map(results.map((r) => [r.id, r.snippet]));
  const merged = [
    ...titleHits.map((c) => ({ ...c, snippet: snippets.get(c.id) })),
    ...results.filter((r) => !titleHits.some((c) => c.id === r.id)),
  ];
  renderSearchRows(merged, merged.length ? "Results" : "No chats match");
}

// --- Page router --------------------------------------------------------------
// Search/wiki/cron are full pages in the main column (not popups):
// each swaps in for the chat view; opening a chat, starting a new one, or Escape
// swaps back. The sidebar nav button of the open page gets an active marker.
const PAGES = {
  search: "#search-page",
  wiki: "#wiki-page",
  jobs: "#jobs-page",
};
let currentPage = null;  // null = chat view

function showPage(name) {
  currentPage = name || null;
  Object.entries(PAGES).forEach(([k, sel]) => { $(sel).hidden = k !== currentPage; });
  $("#chat").hidden = !!currentPage;
  document.querySelectorAll("#sidebar .nav-btn[data-page]").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === currentPage);
  });
  updateScrollJump();
  if (currentPage === "search") $("#chat-search-input").focus();
}

async function openChatSearch() {
  $("#chat-search-input").value = "";
  await loadConversations();  // fresh list (also refreshes the sidebar)
  renderSearchResults();
  showPage("search");
}
$("#search-chats").onclick = openChatSearch;
$("#search-chats-collapsed").onclick = openChatSearch;
$("#new-chat-collapsed").onclick = newChat;
$("#search-close").onclick = () => showPage(null);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && currentPage) showPage(null);
});

$("#chat-search-input").addEventListener("input", renderSearchResults);
$("#chat-search-input").addEventListener("keydown", (e) => {
  const rows = [...$("#search-results").querySelectorAll(".search-result")];
  if (!rows.length) return;
  const idx = rows.findIndex((r) => r.classList.contains("active"));
  const move = (to) => {
    rows.forEach((r) => r.classList.remove("active"));
    rows[to].classList.add("active");
    rows[to].scrollIntoView({ block: "nearest" });
  };
  if (e.key === "ArrowDown") { e.preventDefault(); move((idx + 1) % rows.length); }
  else if (e.key === "ArrowUp") { e.preventDefault(); move((idx - 1 + rows.length) % rows.length); }
  else if (e.key === "Enter") { e.preventDefault(); if (idx >= 0) rows[idx].click(); }
});

// --- Collapsible sidebar ----------------------------------------------------
function setSidebarCollapsed(collapsed) {
  $("#app").classList.toggle("sidebar-collapsed", collapsed);
  try { localStorage.setItem("harness.sidebar", collapsed ? "1" : "0"); } catch (_) {}
}
$("#collapse-sidebar").onclick = () => setSidebarCollapsed(true);
$("#expand-sidebar").onclick = () => setSidebarCollapsed(false);
try { if (localStorage.getItem("harness.sidebar") === "1") setSidebarCollapsed(true); } catch (_) {}

async function openConversation(id) {
  showPage(null);
  state.conversationId = id;
  const data = await fetch(`/api/conversations/${id}`).then((r) => r.json());
  state.promptTokens = (data.conversation && data.conversation.prompt_tokens) || null;
  // Snap the picker to this conversation's stored model — model is per-conversation.
  if (data.conversation) {
    const cp = data.conversation.provider;
    if (cp && state.providers.some((p) => p.id === cp)) {
      applyModelSelection(cp, data.conversation.model, null);
    } else if (state.defaultModel) {
      // Conversation's provider isn't available right now (e.g. the local engine
      // is down). Snap to the default model so the picker stays consistent
      // (provider + model from the same source) — the backend persists the
      // fallback on the next turn.
      applyModelSelection(state.defaultModel.provider, state.defaultModel.model, state.defaultModel.effort);
    }
  }
  $("#messages").innerHTML = "";
  state.stick = true;  // opening a chat lands at the latest message
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
  showPage(null);
  state.conversationId = null;
  state.promptTokens = null;
  state.stick = true;
  // A new conversation starts on the server-resolved default model.
  if (state.defaultModel) {
    applyModelSelection(state.defaultModel.provider, state.defaultModel.model, state.defaultModel.effort);
  }
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
// tables (local models sometimes emit raw tabs to "draw" a table) and turning
// bare image URLs on their own line into markdown images — the Discord
// convention (bare URL auto-embeds there), so DM turns viewed here render the
// same picture instead of a raw link.
function renderMarkdown(text) {
  return marked.parse(convertTabsToTables(embedBareImageUrls(text)));
}

const BARE_IMG_URL_RE = /^https?:\/\/\S+\.(png|jpe?g|gif|webp|avif|bmp)(\?\S*)?$/i;
function embedBareImageUrls(text) {
  const out = [];
  let inCode = false;
  for (const line of text.split("\n")) {
    if (/^\s*```/.test(line)) inCode = !inCode;
    const t = line.trim();
    out.push(!inCode && BARE_IMG_URL_RE.test(t) ? `![](${t})` : line);
  }
  return out.join("\n");
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

// navigator.clipboard only exists in a secure context (HTTPS or localhost).
// Over plain http on a LAN IP (typical on Windows) it's undefined, so fall back
// to the legacy execCommand path via an off-screen textarea.
async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (_) {}
  document.body.removeChild(ta);
  if (!ok) throw new Error("copy command was blocked");
}

async function copyMessage(wrap, btn) {
  const text = wrap.dataset.rawText || "";
  try {
    await writeClipboard(text);
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

// A tool step on the timeline rail: brass diamond node, one-line summary
// (name + args + duration), result pre expands inline.
function addToolBlock(name, args) {
  const { d, summary } = detailsBlock(
    "step tool",
    '<span class="node"></span><span class="tname"></span><span class="args"></span><span class="sdur spinner">●</span>'
  );
  summary.querySelector(".tname").textContent = name;
  summary.querySelector(".args").textContent = truncate(args, 80);
  const body = el("div", "body");
  const pre = el("pre");
  pre.textContent = "running…";
  body.appendChild(pre);
  d.append(summary, body);
  ensureTrace().appendChild(d);
  bumpTrace("tool");
  scroll();
  return { details: d, pre, dur: summary.querySelector(".sdur"), started: performance.now() };
}

// A thought step on the rail: patina ring node, "Thinking… / Thought for Xs"
// label, the reasoning text expands inline.
function addThinkingBlock(initial, open = false) {
  const { d, summary } = detailsBlock(
    "step think",
    '<span class="node"></span><span class="slabel">Thought</span><span class="sdur"></span>',
    open
  );
  const reason = el("div", "reason");
  reason.textContent = initial || "";
  d.append(summary, reason);
  ensureTrace().appendChild(d);
  bumpTrace("thinking");
  scroll();
  return { details: d, reason, label: summary.querySelector(".slabel"), started: performance.now() };
}

// --- Trace: one quiet collapsible timeline per turn --------------------------
// Collapsed, its header narrates the live step ("thinking…", "web_search · …")
// and settles to "2 thoughts · 3 tools"; open, it shows the flat step rail.
let currentTrace = null;        // the outer <details class="trace">
let currentTraceBody = null;    // its .trace-body div
const currentTraceCounts = { tool: 0, thinking: 0 };

function resetTrace() {
  currentTrace = null;
  currentTraceBody = null;
  currentTraceCounts.tool = 0;
  currentTraceCounts.thinking = 0;
}

function ensureTrace() {
  if (currentTraceBody) return currentTraceBody;
  const empty = $("#messages .empty");
  if (empty) empty.remove();
  const { d, summary } = detailsBlock(
    "trace",
    '<span class="tactivity"></span><span class="tstatus"></span><span class="tclock"></span>'
  );
  const body = el("div", "trace-body");
  d.append(summary, body);
  $("#messages").appendChild(d);
  currentTrace = d;
  currentTraceBody = body;
  updateTraceSummary();
  return body;
}

function traceCountsText() {
  const c = currentTraceCounts;
  const parts = [];
  if (c.thinking) parts.push(`${c.thinking} thought${c.thinking === 1 ? "" : "s"}`);
  if (c.tool) parts.push(`${c.tool} tool${c.tool === 1 ? "" : "s"}`);
  return parts.join(" · ") || "process";
}

function bumpTrace(kind) {
  currentTraceCounts[kind] = (currentTraceCounts[kind] || 0) + 1;
  updateTraceSummary();
}

// Refresh the header counts, but never overwrite a live narration.
function updateTraceSummary() {
  if (!currentTrace) return;
  const act = currentTrace.querySelector(".tactivity");
  if (act.className === "tactivity") {
    currentTrace.querySelector(".tstatus").textContent = traceCountsText();
  }
}

function setTraceLive(on) {
  if (currentTrace) currentTrace.classList.toggle("live", on);
}

// Drive the header narration: a colored activity dot (patina = thinking,
// brass = tool) plus a label of what's happening right now; off → counts.
function setTraceActivity(kind, on, label) {
  if (!currentTrace) return;
  currentTrace.querySelector(".tactivity").className = "tactivity" + (on ? " " + kind : "");
  currentTrace.querySelector(".tstatus").textContent = (on && label) || traceCountsText();
}
const setTraceThinking = (on) => setTraceActivity("thinking", on, "thinking…");
const setTraceTool = (on, label) => setTraceActivity("tool", on, label);

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
      // Replayed history has no timings — clear the spinner, leave no duration.
      if (block) { block.pre.textContent = m.content || ""; block.dur.textContent = ""; block.dur.classList.remove("spinner"); }
    }
  }
  if (!messages.length) $("#messages").innerHTML = EMPTY_STATE;
}

const truncate = (s, n) => (s && s.length > n ? s.slice(0, n) + "…" : s || "");
// Auto-scroll only while "stuck" to the bottom. Once you scroll up to read the
// start of a streaming answer we leave you there; scrolling back down (or the
// jump-to-newest button) re-sticks. updateScrollJump() recomputes state.stick
// from the scroll position, so a programmatic snap to the bottom keeps it stuck.
const scroll = () => { const m = $("#messages"); if (state.stick) m.scrollTop = m.scrollHeight; updateScrollJump(); };

// --- Scroll-jump button -----------------------------------------------------
// At the bottom it offers to jump to the first message; once scrolled up it
// flips to jump back to the newest message.
function updateScrollJump() {
  const m = $("#messages");
  const btn = $("#scroll-jump");
  if (!m || !btn) return;
  const atBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 80;
  // Re-stick when the view is back at the bottom; unstick when scrolled up. This
  // runs on every scroll, so reading up during a stream stops the auto-follow.
  state.stick = atBottom;
  const scrollable = m.scrollHeight - m.clientHeight > 60;
  if (!scrollable) { btn.classList.remove("show"); return; }
  btn.classList.add("show");
  btn.classList.toggle("down", !atBottom);
  btn.title = atBottom ? "Jump to first message" : "Jump to newest message";
}
$("#scroll-jump").addEventListener("click", () => {
  const m = $("#messages");
  const toBottom = $("#scroll-jump").classList.contains("down");
  // Following the stream again re-sticks; jumping to the top intentionally
  // detaches so streamed tokens don't yank you back down.
  state.stick = toBottom;
  m.scrollTo({ top: toBottom ? m.scrollHeight : 0, behavior: "smooth" });
});
$("#messages").addEventListener("scroll", updateScrollJump);
// Markdown images in responses (image_search results) open full-size in a new
// tab. Attachment thumbnails (.msg-images) already have their own handler.
$("#messages").addEventListener("click", (e) => {
  const img = e.target.closest(".bubble img");
  if (img && img.src && !img.closest(".msg-images")) window.open(img.src, "_blank");
});
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
  if (!provider || !model) { setStatus("Pick a provider and model first."); setModelMenu(true); return; }

  state.stick = true;  // a turn you just started should follow the new output
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
      body: JSON.stringify({ conversation_id: state.conversationId, provider, model, message: text, images: images.length ? images : null, think: state.think, effort: state.think ? (state.effort || null) : null }),
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
    if (thinkingBlock) {
      thinkingBlock.label.textContent = "Thought for " + fmtDur(performance.now() - thinkingBlock.started);
      thinkingBlock.details.classList.remove("live");
      thinkingBlock = null;
    }
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
      case "model":
        // Server switched the conversation to a fallback model (the chosen
        // provider wasn't serving). Sync the picker so it reflects reality.
        applyModelSelection(ev.provider, ev.model, ev.effort);
        setStatus(`↳ fell back to ${ev.provider}/${ev.model}`);
        break;
      case "reasoning":
        thinkTick(true);
        if (!thinkingBlock) {
          thinkingBlock = addThinkingBlock("", false);
          thinkingBlock.label.textContent = "Thinking…";
          thinkingBlock.details.classList.add("live");
          setTraceThinking(true);
        }
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
        toolBlocks[ev.id] = addToolBlock(ev.name, ev.arguments || "");
        setTraceTool(true, `${ev.name} · ${truncate(ev.arguments || "", 48)}`);
        assistantBubble = null; assistantText = "";  // text after a tool is a new bubble
        break;
      case "tool_result": {
        const b = toolBlocks[ev.id];
        if (b) {
          b.pre.textContent = ev.result;
          b.dur.classList.remove("spinner");
          b.dur.textContent = fmtDur(performance.now() - b.started);
        }
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

// --- LLM-Wiki -----------------------------------------------------------------
// One page, two views: the page list (cards) and the editor (name + summary +
// body). `home` is the system prompt — it's editable but not deletable.
let wikiEditing = null;  // page name being edited, or null for a new page

function fmtWikiDate(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
}

function setWikiView(editor) {
  $("#wiki-list-view").hidden = editor;
  $("#wiki-editor-view").hidden = !editor;
}

async function renderWikiList() {
  const data = await fetch("/api/wiki").then((r) => r.json());
  const list = $("#wiki-list");
  list.innerHTML = "";
  (data.pages || []).forEach((p) => {
    const card = el("div", "wiki-card" + (p.name === "home" ? " home" : ""));
    const head = el("div", "wiki-card-head");
    const name = el("span", "name"); name.textContent = p.name;
    const meta = el("span", "meta");
    meta.textContent = `${p.chars} chars · ${fmtWikiDate(p.updated_at)}`;
    head.append(name, meta);
    if (p.name !== "home") {
      const del = el("button", "del"); del.type = "button"; del.textContent = "✕"; del.title = "Delete page";
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete wiki page “${p.name}”?`)) return;
        await fetch(`/api/wiki/${encodeURIComponent(p.name)}`, { method: "DELETE" });
        renderWikiList();
      };
      head.appendChild(del);
    }
    const desc = el("div", "desc");
    desc.textContent = p.summary || "(no summary)";
    card.append(head, desc);
    card.onclick = () => openWikiEditor(p.name);
    list.appendChild(card);
  });
  if (!list.children.length) list.innerHTML = '<div class="empty">No wiki pages yet.</div>';
}

async function openWikiEditor(name) {
  wikiEditing = name;
  $("#wiki-status").textContent = "";
  if (name) {
    const data = await fetch(`/api/wiki/${encodeURIComponent(name)}`).then((r) => r.json());
    $("#wiki-editor-title").textContent = name === "home" ? "home — the system prompt" : name;
    $("#wiki-name").value = name;
    $("#wiki-name").disabled = true;
    $("#wiki-summary").value = data.summary || "";
    $("#wiki-content").value = data.content || "";
  } else {
    $("#wiki-editor-title").textContent = "New page";
    $("#wiki-name").value = "";
    $("#wiki-name").disabled = false;
    $("#wiki-summary").value = "";
    $("#wiki-content").value = "";
  }
  setWikiView(true);
  (name ? $("#wiki-content") : $("#wiki-name")).focus();
}

$("#show-wiki").onclick = async () => {
  showPage("wiki");
  setWikiView(false);
  await renderWikiList();
};
$("#wiki-new").onclick = () => openWikiEditor(null);
$("#wiki-back").onclick = async () => { setWikiView(false); await renderWikiList(); };

let wikiStatusTimer = null;
$("#wiki-save").onclick = async () => {
  const name = ($("#wiki-name").value || "").trim();
  if (!name) { $("#wiki-status").textContent = "page name required"; return; }
  const res = await fetch(`/api/wiki/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary: $("#wiki-summary").value, content: $("#wiki-content").value }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#wiki-status").textContent = err.detail || "save failed";
    return;
  }
  const saved = await res.json();
  wikiEditing = saved.name;
  $("#wiki-name").value = saved.name;
  $("#wiki-name").disabled = true;
  $("#wiki-editor-title").textContent = saved.name === "home" ? "home — the system prompt" : saved.name;
  $("#wiki-status").textContent = "saved — applies to the next turn";
  clearTimeout(wikiStatusTimer);
  wikiStatusTimer = setTimeout(() => { $("#wiki-status").textContent = ""; }, 4000);
};

// --- Scheduled jobs ---------------------------------------------------------
let jobsChannels = [];        // [{id, label}] the bot can post to
let jobsOwner = null;         // {id, name} the bot owner (DM recipient)
let editingJobId = null;

function fmtJobTime(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}
function channelLabel(id) {
  const c = jobsChannels.find((x) => x.id === String(id));
  return c ? c.label : null;
}

async function loadJobChannels() {
  const data = await fetch("/api/discord/channels").then((r) => r.json());
  jobsChannels = data.channels || [];
  jobsOwner = data.owner || null;
  const note = $("#job-target-dm-note");
  if (note) note.textContent = jobsOwner ? `→ DMs you (${jobsOwner.name})` : "→ DMs you (the bot owner)";
  const sel = $("#job-target-channel");
  sel.innerHTML = "";
  if (!data.connected) {
    sel.innerHTML = '<option value="">(bot not connected)</option>';
    return;
  }
  if (!jobsChannels.length) {
    sel.innerHTML = '<option value="">(no channels available)</option>';
    return;
  }
  jobsChannels.forEach((c) => {
    const o = el("option");
    o.value = c.id;
    o.textContent = c.label;
    sel.appendChild(o);
  });
}

function updateJobTargetInputs() {
  const type = $("#job-target-type").value;
  $("#job-target-channel").hidden = type !== "channel";
  $("#job-target-dm-note").hidden = type !== "dm";
}
$("#job-target-type").addEventListener("change", updateJobTargetInputs);

function resetJobForm() {
  editingJobId = null;
  $("#job-name").value = "";
  $("#job-prompt").value = "";
  $("#job-cron").value = "";
  $("#job-target-type").value = "channel";
  $("#job-save").textContent = "Add job";
  $("#job-cancel").hidden = true;
  $("#job-form-hint").textContent = "";
  updateJobTargetInputs();
}

function fillJobForm(j) {
  editingJobId = j.id;
  $("#job-name").value = j.name || "";
  $("#job-prompt").value = j.prompt || "";
  $("#job-cron").value = j.cron || "";
  $("#job-target-type").value = j.target_type || "channel";
  if (j.target_type === "channel") {
    updateJobTargetInputs();
    $("#job-target-channel").value = j.target_id || "";
  }
  $("#job-save").textContent = "Update job";
  $("#job-cancel").hidden = false;
  $("#job-form-hint").textContent = `editing “${j.name}”`;
  updateJobTargetInputs();
}

async function renderJobs() {
  const data = await fetch("/api/jobs").then((r) => r.json());
  const list = $("#jobs-list");
  list.innerHTML = "";
  if (!data.jobs.length) {
    list.innerHTML = '<div class="empty">No scheduled jobs yet.</div>';
    return;
  }
  data.jobs.forEach((j) => {
    const row = el("div", "job");
    if (!j.enabled) row.classList.add("disabled");

    const head = el("div", "job-head");
    const name = el("span", "job-name"); name.textContent = j.name || "Job";
    const toggle = el("label", "job-toggle");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = j.enabled;
    cb.onchange = async () => {
      await fetch(`/api/jobs/${j.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: cb.checked }),
      });
      renderJobs();
    };
    toggle.append(cb, document.createTextNode(" on"));
    head.append(name, toggle);

    const tgt = j.target_type === "dm"
      ? (j.target_id === "owner" ? "DM (you)" : `DM ${j.target_id}`)
      : (channelLabel(j.target_id) || `#${j.target_id}`);
    const meta = el("div", "job-meta");
    meta.textContent = `${j.cron} · ${j.cron_desc || ""} · → ${tgt}`;

    const sched = el("div", "job-sched");
    let s = `next: ${fmtJobTime(j.next_run)}`;
    if (j.last_run) s += ` · last: ${fmtJobTime(j.last_run)} (${j.last_status || "?"})`;
    sched.textContent = s;

    const prm = el("div", "job-prompt-preview"); prm.textContent = j.prompt;

    const actions = el("div", "job-actions");
    const runBtn = el("button"); runBtn.type = "button"; runBtn.textContent = "Run now";
    runBtn.onclick = async () => {
      runBtn.disabled = true; runBtn.textContent = "Running…";
      try { await fetch(`/api/jobs/${j.id}/run`, { method: "POST" }); }
      finally { renderJobs(); }
    };
    const editBtn = el("button"); editBtn.type = "button"; editBtn.textContent = "Edit";
    editBtn.onclick = () => fillJobForm(j);
    const delBtn = el("button", "job-del"); delBtn.type = "button"; delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm(`Delete job “${j.name}”?`)) return;
      await fetch(`/api/jobs/${j.id}`, { method: "DELETE" });
      if (editingJobId === j.id) resetJobForm();
      renderJobs();
    };
    actions.append(runBtn, editBtn, delBtn);

    row.append(head, meta, sched, prm);
    if (j.last_status === "error" && j.last_result) {
      const e = el("div", "job-error"); e.textContent = j.last_result;
      row.append(e);
    }
    row.append(actions);
    list.appendChild(row);
  });
}

$("#job-cancel").onclick = resetJobForm;

$("#job-save").onclick = async () => {
  const type = $("#job-target-type").value;
  const target_id = type === "dm" ? "owner" : $("#job-target-channel").value;
  const body = {
    name: $("#job-name").value.trim(),
    prompt: $("#job-prompt").value.trim(),
    cron: $("#job-cron").value.trim(),
    target_type: type,
    target_id,
  };
  if (!body.prompt || !body.cron) { $("#job-form-hint").textContent = "Prompt and cron are required."; return; }
  if (type === "channel" && !body.target_id) { $("#job-form-hint").textContent = "Pick a channel."; return; }
  const url = editingJobId ? `/api/jobs/${editingJobId}` : "/api/jobs";
  const method = editingJobId ? "PUT" : "POST";
  const res = await fetch(url, {
    method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#job-form-hint").textContent = err.detail || "Save failed.";
    return;
  }
  resetJobForm();
  renderJobs();
};

$("#show-jobs").onclick = async () => {
  showPage("jobs");
  resetJobForm();
  await loadJobChannels();
  await renderJobs();
};

// --- Boot -------------------------------------------------------------------
(async function init() {
  // The sidebar history doesn't depend on provider detection, so load it in
  // parallel — a slow provider probe must never hold back the chat list.
  const convos = loadConversations();
  await loadProviders();   // newChat() needs the resolved default model
  await convos;
  newChat();
})();
