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
  research: false,   // composer research toggle (scope, then deep_research)
  abort: null,       // AbortController for the in-flight stream (stop button)
  stick: true,       // follow the bottom as content streams; false once you scroll up
};

// --- Theme (system / light / dark) -------------------------------------------
// An explicit pick sets html[data-theme]; "system" removes it so the CSS
// prefers-color-scheme media query decides. The pick is also applied pre-paint
// by an inline <head> script, so this only handles the button + later changes.
const THEME_KEY = "harness.theme";
const THEME_ICONS = {
  system: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  light: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  dark: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
};
function applyTheme(mode) {
  if (mode === "light" || mode === "dark") {
    document.documentElement.dataset.theme = mode;
  } else {
    mode = "system";
    delete document.documentElement.dataset.theme;
  }
  const btn = $("#theme-btn");
  btn.innerHTML = THEME_ICONS[mode];
  btn.title = `Theme: ${mode} (click to switch)`;
}
$("#theme-btn").onclick = () => {
  const order = ["system", "light", "dark"];
  const cur = loadPref(THEME_KEY) || "system";
  const next = order[(order.indexOf(cur) + 1) % order.length];
  savePref(THEME_KEY, next);
  applyTheme(next);
};
applyTheme(loadPref(THEME_KEY) || "system");

// --- Thinking toggle --------------------------------------------------------
const effortKey = (p) => `harness.effort.${p}`;
// Research mode: the next message is scoped (the model may ask clarifying
// questions back) and handed to deep_research. Stays on across the clarifying
// round-trip; auto-disarms once the model actually launches the research.
$("#research-toggle").onclick = () => {
  state.research = !state.research;
  $("#research-toggle").classList.toggle("on", state.research);
  $("#research-toggle").setAttribute("aria-pressed", String(state.research));
  setStatus(state.research ? "Research mode: the model will scope your question, then run a deep research." : "");
};

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

let convCache = [];   // newest-first; feeds both the sidebar and chat search
let groupCache = [];  // group names (empties included) for the assign dropdown

const FOLDER_ICON = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';
const PENCIL_ICON = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>';
const KEBAB_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" stroke="none"><circle cx="12" cy="5" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="19" r="1.6"/></svg>';
const TRASH_ICON = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';

// Row action menu: everything a chat/group row can do lives behind one ⋮
// button. The menu is appended to <body> and fixed-positioned at the button
// (sidebar rows clip overflow, so an in-row dropdown would be cut off).
let _actionMenu = null;
function closeActionMenu() {
  if (!_actionMenu) return;
  _actionMenu.remove();
  _actionMenu = null;
  document.removeEventListener("mousedown", _menuOutside, true);
  document.removeEventListener("keydown", _menuKey, true);
}
function _menuOutside(e) { if (_actionMenu && !_actionMenu.contains(e.target)) closeActionMenu(); }
function _menuKey(e) { if (e.key === "Escape") { e.stopPropagation(); closeActionMenu(); } }
function showActionMenu(anchor, items) {
  const wasOpen = _actionMenu && _actionMenu._anchor === anchor;
  closeActionMenu();
  if (wasOpen) return;  // clicking the same ⋮ again toggles the menu shut
  const menu = el("div", "action-menu");
  menu._anchor = anchor;
  items.forEach((it) => {
    const b = el("button", "action-menu-item" + (it.danger ? " danger" : ""));
    b.type = "button";
    b.innerHTML = it.icon || "";
    const label = el("span");
    label.textContent = it.label;
    b.appendChild(label);
    b.onclick = (e) => { e.stopPropagation(); closeActionMenu(); it.onClick(e); };
    menu.appendChild(b);
  });
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const mr = menu.getBoundingClientRect();
  menu.style.left = Math.min(r.left, window.innerWidth - mr.width - 8) + "px";
  const below = r.bottom + 4;
  menu.style.top = (below + mr.height > window.innerHeight - 8 ? r.top - mr.height - 4 : below) + "px";
  _actionMenu = menu;
  document.addEventListener("mousedown", _menuOutside, true);
  document.addEventListener("keydown", _menuKey, true);
}
// The menu is fixed-positioned: any scroll under it would leave it floating
// over the wrong row, so just dismiss.
document.addEventListener("scroll", closeActionMenu, true);
window.addEventListener("resize", closeActionMenu);

async function patchConversation(id, body) {
  await fetch(`/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// Swap a conversation row's content for an inline input editing its title or
// group. Enter saves, Escape/blur cancels. For groups, a dropdown of every
// existing group opens immediately (click to pick); the input filters it.
function editConvInline(item, c, kind) {
  const input = el("input", "conv-edit");
  input.value = kind === "title" ? (c.title || "") : (c.grp || "");
  input.placeholder = kind === "title" ? "Chat title" : "Pick or type a group…";
  item.classList.add("editing");  // lift the row's overflow so the dropdown shows
  item.replaceChildren(input);

  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    if (save) {
      const v = input.value.trim();
      if (kind === "title") { if (v) await patchConversation(c.id, { title: v }); }
      else await patchConversation(c.id, { group: v });
    }
    loadConversations();
  };

  if (kind === "grp") {
    const list = el("ul", "conv-edit-list");
    const render = () => {
      const f = input.value.trim().toLowerCase();
      list.innerHTML = "";
      if (c.grp) {
        const li = el("li", "clear");
        li.textContent = "✕ remove from group";
        li.dataset.value = "";
        list.appendChild(li);
      }
      groupCache
        .filter((g) => !f || g.toLowerCase().includes(f))
        .forEach((g) => {
          const li = el("li");
          li.textContent = g;
          li.dataset.value = g;
          if (g === c.grp) li.classList.add("selected");
          list.appendChild(li);
        });
      const t = input.value.trim();
      if (t && !groupCache.some((g) => g.toLowerCase() === t.toLowerCase())) {
        const li = el("li", "create");
        li.textContent = `＋ create “${t}”`;
        li.dataset.value = t;
        list.appendChild(li);
      }
      list.hidden = !list.children.length;
    };
    // mousedown fires before the input's blur, so the pick lands first.
    list.onmousedown = async (e) => {
      const li = e.target.closest("li");
      if (!li) return;
      e.preventDefault();
      e.stopPropagation();
      if (done) return;
      done = true;
      await patchConversation(c.id, { group: li.dataset.value });
      loadConversations();
    };
    input.addEventListener("input", render);
    item.appendChild(list);
    render();
  }

  input.focus();
  input.select();
  input.onclick = (e) => e.stopPropagation();
  input.onkeydown = (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") finish(false);
  };
  input.onblur = () => finish(false);
}

// In-app replacement for window.confirm(): a small centered dialog.
// Resolves true on confirm; Escape, Cancel, or clicking the backdrop resolve false.
function confirmDialog(message, okLabel = "Delete") {
  return new Promise((resolve) => {
    const overlay = $("#confirm-overlay");
    $("#confirm-msg").textContent = message;
    $("#confirm-ok").textContent = okLabel;
    const prevFocus = document.activeElement;
    const done = (v) => {
      overlay.hidden = true;
      document.removeEventListener("keydown", onKey, true);
      if (prevFocus && prevFocus.focus) prevFocus.focus();
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); done(false); }
    };
    document.addEventListener("keydown", onKey, true);
    $("#confirm-ok").onclick = () => done(true);
    $("#confirm-cancel").onclick = () => done(false);
    overlay.onclick = (e) => { if (e.target === overlay) done(false); };
    overlay.hidden = false;
    $("#confirm-cancel").focus();
  });
}

function convRow(c) {
  const item = el("div", "conv");
  if (c.id === state.conversationId) item.classList.add("active");
  if (state.streaming && c.id === state.streamConvId) item.classList.add("streaming");
  const title = el("span", "title");
  const full = c.title || "Untitled";
  title.textContent = full;
  // The narrow sidebar ellipsizes long titles; hovering reveals the full text
  // (native tooltip), itself capped — a pasted-question title doesn't need to
  // be readable in its entirety to identify the chat. Same 90-char threshold
  // as long-link display shortening (LINK_TEXT_MAX).
  item.title = full.length > LINK_TEXT_MAX ? full.slice(0, LINK_TEXT_MAX) + "…" : full;

  const kebab = el("button", "cact kebab");
  kebab.type = "button";
  kebab.innerHTML = KEBAB_ICON;
  kebab.title = "Options";
  kebab.onclick = (e) => {
    e.stopPropagation();
    showActionMenu(kebab, [
      { icon: PENCIL_ICON, label: "Rename", onClick: () => editConvInline(item, c, "title") },
      { icon: FOLDER_ICON, label: c.grp ? `Group: ${c.grp}` : "Add to group", onClick: () => editConvInline(item, c, "grp") },
      { icon: TRASH_ICON, label: "Delete", danger: true, onClick: async () => {
        if (!(await confirmDialog(`Delete “${c.title || "Untitled"}”? The conversation and its history are removed permanently.`))) return;
        await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
        if (state.conversationId === c.id) newChat();
        loadConversations();
      } },
    ]);
  };
  item.append(title, kebab);
  item.onclick = () => openConversation(c.id);
  return item;
}

// Collapsed-group names, persisted across reloads.
function loadGrpCollapsed() {
  try { return JSON.parse(localStorage.getItem("harness.groupsCollapsed")) || {}; }
  catch { return {}; }
}
function saveGrpCollapsed(o) {
  try { localStorage.setItem("harness.groupsCollapsed", JSON.stringify(o)); } catch (_) {}
}

async function loadConversations() {
  const data = await fetch("/api/conversations").then((r) => r.json());
  convCache = data.conversations || [];
  const groups = data.groups || [];  // first-class: includes empty groups
  groupCache = groups.map((g) => g.name);
  const box = $("#conversations");
  box.innerHTML = "";

  // Group sections first (server orders by recent activity, empties last).
  const collapsed = loadGrpCollapsed();
  const byGroup = new Map();
  convCache.forEach((c) => {
    if (!c.grp) return;
    if (!byGroup.has(c.grp)) byGroup.set(c.grp, []);
    byGroup.get(c.grp).push(c);
  });
  const renderGroup = (g) => {
    const items = byGroup.get(g.name) || [];
    const isCollapsed = !!collapsed[g.name];
    const h = el("div", "conv-group grp" + (isCollapsed ? " closed" : ""));
    h.innerHTML = `<span class="tri">▾</span>${FOLDER_ICON}<span class="gname"></span><span class="gcount"></span><button class="gact kebab" type="button" title="Options">${KEBAB_ICON}</button>`;
    h.querySelector(".gname").textContent = g.name;
    h.querySelector(".gcount").textContent = items.length || "empty";
    h.onclick = () => {
      if (isCollapsed) delete collapsed[g.name]; else collapsed[g.name] = 1;
      saveGrpCollapsed(collapsed);
      loadConversations();
    };
    const renameGroup = () => {
      const input = el("input", "conv-edit");
      input.value = g.name;
      h.replaceChildren(input);
      h.onclick = null;
      input.focus();
      input.select();
      let done = false;
      const finish = async (save) => {
        if (done) return;
        done = true;
        const v = input.value.trim();
        if (save && v && v !== g.name) {
          await fetch(`/api/groups/${encodeURIComponent(g.name)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: v }),
          });
          if (collapsed[g.name]) { delete collapsed[g.name]; collapsed[v] = 1; saveGrpCollapsed(collapsed); }
        }
        loadConversations();
      };
      input.onclick = (ev) => ev.stopPropagation();
      input.onkeydown = (ev) => {
        ev.stopPropagation();
        if (ev.key === "Enter") { ev.preventDefault(); finish(true); }
        else if (ev.key === "Escape") finish(false);
      };
      input.onblur = () => finish(false);
    };
    const deleteGroup = async () => {
      const warn = items.length
        ? `Delete group “${g.name}”? Its ${items.length} chat${items.length === 1 ? "" : "s"} move back to the main list (nothing is deleted).`
        : `Delete empty group “${g.name}”?`;
      if (!(await confirmDialog(warn))) return;
      await fetch(`/api/groups/${encodeURIComponent(g.name)}`, { method: "DELETE" });
      loadConversations();
    };
    // System groups (Research, Cron: *) are feature-owned: no rename/delete —
    // renaming would break the feature's group-name linkage (cron pruning),
    // and the group would just be recreated anyway. Chats inside can still be
    // moved out individually.
    if (g.system) {
      h.querySelector(".gact").remove();
    } else {
      h.querySelector(".gact").onclick = (e) => {
        e.stopPropagation();
        showActionMenu(e.currentTarget, [
          { icon: PENCIL_ICON, label: "Rename group", onClick: renameGroup },
          { icon: TRASH_ICON, label: "Delete group", danger: true, onClick: deleteGroup },
        ]);
      };
    }
    box.appendChild(h);
    if (!isCollapsed) items.forEach((c) => box.appendChild(convRow(c)));
  };

  // Two leagues, stable order: automatic groups (feature-owned — Research,
  // Cron: *) always on top under their own label, then the user's groups.
  const sysGroups = groups.filter((g) => g.system);
  const customGroups = groups.filter((g) => !g.system);
  if (sysGroups.length) {
    const lbl = el("div", "conv-group");
    lbl.textContent = "automatic";
    box.appendChild(lbl);
    sysGroups.forEach(renderGroup);
  }
  const clbl = el("div", "conv-group");
  clbl.textContent = "your groups";
  box.appendChild(clbl);
  customGroups.forEach(renderGroup);

  // "+ new group": creates an empty group (chats can be moved in later).
  const add = el("div", "conv-group grp new");
  add.innerHTML = `<span class="tri">＋</span><span class="gname">new group</span>`;
  add.onclick = () => {
    const input = el("input", "conv-edit");
    input.placeholder = "Group name";
    add.replaceChildren(input);
    input.focus();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const v = input.value.trim();
      if (save && v) {
        await fetch("/api/groups", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: v }),
        });
      }
      loadConversations();
    };
    input.onclick = (e) => e.stopPropagation();
    input.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") finish(false);
    };
    input.onblur = () => finish(false);
  };
  box.appendChild(add);

  // Ungrouped conversations in the usual date sections.
  let lastGroup = null;
  convCache.filter((c) => !c.grp).forEach((c) => {
    const group = convGroup(c.updated_at || c.created_at);
    if (group !== lastGroup) {
      const h = el("div", "conv-group");
      h.textContent = group;
      box.appendChild(h);
      lastGroup = group;
    }
    box.appendChild(convRow(c));
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
  status: "#status-page",
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
  // Resumed chats always start on the current default model, not the model the
  // chat last used (that stays logged server-side, it's just not restored) —
  // e.g. when the local engine comes up, old chats should pick it up too.
  // loadProviders() re-detects availability, so the default is fresh: local
  // engine if it's up, the fallback otherwise.
  const [data] = await Promise.all([
    fetch(`/api/conversations/${id}`).then((r) => r.json()),
    loadProviders(),
  ]);
  state.promptTokens = (data.conversation && data.conversation.prompt_tokens) || null;
  if (state.defaultModel) {
    applyModelSelection(state.defaultModel.provider, state.defaultModel.model, state.defaultModel.effort);
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
  // A new conversation starts on the server-resolved default model. Re-detect
  // providers first so a local engine that (dis)appeared since page load counts.
  loadProviders().then(() => {
    if (state.defaultModel) {
      applyModelSelection(state.defaultModel.provider, state.defaultModel.model, state.defaultModel.effort);
    }
  });
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
const SPEAKER_ICON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.8 5.8a9 9 0 0 1 0 12.4"/></svg>';
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

// Bare pasted URLs render with the whole URL as the link text; a 200-char eBay
// link is noise. Shorten the *display text* of such links (href untouched, so
// the click still goes to the real place; full URL on hover via title). Only
// links whose text is the URL itself are touched — authored link text stays.
// ~90 chars is one line of bubble at this font size: a URL that fits on one
// line is left whole, and anything shortened saves at least a third of its
// length (never elide a handful of chars — that destroys info for no gain).
const LINK_TEXT_MAX = 90;
function shortenLongLinks(root) {
  for (const a of root.querySelectorAll("a")) {
    const t = a.textContent;
    if (t.length > LINK_TEXT_MAX && /^https?:\/\//.test(t) && !a.title) {
      a.title = t;
      // Bracketed elision marker: "[…]" can't occur in a valid URL, so it
      // reads unambiguously as "characters omitted" rather than URL content.
      a.textContent = t.slice(0, 42) + "[…]" + t.slice(-14);
    }
  }
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

// A small playback control for a message's voice clip (the user's recording,
// or the TTS reading of a reply). preload=none: clips load on first play.
function attachAudio(bubble, url) {
  const player = el("audio", "msg-audio");
  player.controls = true;
  player.preload = "none";
  player.src = url;
  bubble.appendChild(player);
  return player;
}

// Read-aloud button: first click asks the server to synthesize the reply
// (persisted on the message, so it's once per message ever), attaches the
// player, and plays; with a player already there it just toggles play/pause.
async function speakMessage(wrap, btn) {
  const bubble = wrap.querySelector(".bubble");
  let player = bubble.querySelector(".msg-audio");
  if (player) {
    if (player.paused) player.play(); else player.pause();
    return;
  }
  const mid = wrap.dataset.msgId;
  if (!mid) { setStatus("Reply not saved yet — try again in a moment."); return; }
  btn.disabled = true;
  setStatus("Synthesizing speech…");
  try {
    const res = await fetch(`/api/tts/${mid}`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const { url } = await res.json();
    player = attachAudio(bubble, url);
    player.play();
    setStatus("");
  } catch (e) {
    setStatus("⚠ TTS failed: " + e.message);
  } finally {
    btn.disabled = false;
  }
}

function addMessageBubble(role, markdown, images, msgId, model, audio, docs) {
  const empty = $("#messages .empty");
  if (empty) empty.remove();
  const wrap = el("div", `msg ${role}`);
  if (msgId) wrap.dataset.msgId = msgId;
  wrap.dataset.rawText = markdown || "";
  if (images && images.length) wrap.dataset.images = JSON.stringify(images);
  if (docs && docs.length) wrap.dataset.docs = JSON.stringify(docs);
  const r = el("div", "role");
  r.textContent = role === "user" ? "you" : "model";
  // Per-reply provenance: which provider/model produced this answer, shown
  // right after the channel label ("▸ MODEL · deepseek/deepseek-v4-flash").
  if (role !== "user" && model) {
    const tag = el("span", "role-model");
    tag.textContent = " · " + model;
    r.appendChild(tag);
  }
  const bubble = el("div", "bubble");
  bubble.innerHTML = markdown ? renderMarkdown(markdown) : "";
  shortenLongLinks(bubble);
  // Document attachments render as chips above the question text; each links
  // to the stored file. The model reads the extracted text server-side.
  if (docs && docs.length) {
    const box = el("div", "msg-docs");
    docs.forEach((d) => {
      const chip = el("a", "doc-chip");
      chip.href = d.url;
      chip.target = "_blank";
      chip.textContent = "📄 " + (d.name || "document");
      box.appendChild(chip);
    });
    bubble.prepend(box);
  }
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
  // Voice clips: the user's recording sits above their transcript, the TTS
  // reading of a reply sits below the text (text stays skimmable either way).
  if (audio) {
    const player = attachAudio(bubble, audio);
    if (role === "user") bubble.prepend(player);
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
  } else {
    // Read aloud: TTS is on demand only — nothing is synthesized until this
    // is clicked (first click synthesizes + plays; after that it's play/pause).
    const speak = el("button", "action speak");
    speak.type = "button";
    speak.title = "Read aloud";
    speak.innerHTML = SPEAKER_ICON;
    speak.onclick = (e) => { e.stopPropagation(); speakMessage(wrap, speak); };
    actions.appendChild(speak);
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
  const docs = userWrap.dataset.docs ? JSON.parse(userWrap.dataset.docs) : [];
  await send(text, images, docs);
}

// Edit the last message you sent: rewind from it and drop its text + images
// back into the composer so you can change it and send again.
async function editFromMessage(userWrap) {
  if (state.streaming) { setStatus("Stop the current turn before editing."); return; }
  const text = userWrap.dataset.rawText || "";
  const imagesJson = userWrap.dataset.images;
  const docsJson = userWrap.dataset.docs;

  if (!(await rewindToMessage(userWrap, "Edit"))) return;

  // Restore the message into the composer for editing.
  const input = $("#input");
  input.value = text;
  autosizeInput();
  state.attachments = (imagesJson ? JSON.parse(imagesJson) : []).map((url) => ({ url }))
    .concat((docsJson ? JSON.parse(docsJson) : []).map((d) => ({ url: d.url, kind: "doc", name: d.name })));
  renderAttachments();
  updateSendVisibility();

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
      addMessageBubble("user", m.content || "", m.images, m.id, null, m.audio, m.docs);
    }
    else if (m.role === "assistant") {
      if (m.reasoning) addThinkingBlock(m.reasoning, false);
      if (m.content) lastAssistant = addMessageBubble("assistant", m.content, null, m.id, m.model, m.audio);
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
  // "#convo:<id>" links (e.g. a finished deep-research note) open that
  // conversation in place instead of navigating.
  const a = e.target.closest('.bubble a[href^="#convo:"]');
  if (a) {
    e.preventDefault();
    openConversation(a.getAttribute("href").slice(7));
    return;
  }
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
// Drag a file anywhere onto the page to attach it (image or document).
// dragover must be cancelled continuously or the browser navigates to the file.
document.addEventListener("dragover", (e) => {
  if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) {
    e.preventDefault();
    document.body.classList.add("dragging");
  }
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("dragging");
});
document.addEventListener("drop", async (e) => {
  document.body.classList.remove("dragging");
  if (!e.dataTransfer || !e.dataTransfer.files.length) return;
  e.preventDefault();
  for (const file of e.dataTransfer.files) await uploadAttachment(file);
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

// Document formats the composer accepts alongside images (must mirror the
// server's docs.DOC_EXT — the server is the real gate).
const DOC_EXT_RE = /\.(pdf|txt|md|markdown|rst|csv|tsv|json|jsonl|log|yaml|yml|toml|ini|xml|html|py|js|ts|sh|sql|c|h|cpp|go|rs)$/i;

async function uploadAttachment(file) {
  if (!file) return;
  const isImage = file.type.startsWith("image/");
  const isDoc = !isImage && DOC_EXT_RE.test(file.name || "");
  if (!isImage && !isDoc) { setStatus(`Can't attach '${file.name}' — images and text/PDF documents only.`); return; }
  const att = { url: null, uploading: true, kind: isDoc ? "doc" : "image", name: file.name };
  state.attachments.push(att);
  renderAttachments();
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (res.ok && data.url) {
      att.url = data.url;
      att.uploading = false;
      if (data.name) att.name = data.name;
    } else {
      state.attachments = state.attachments.filter((a) => a !== att);
      setStatus(data.detail || "Upload failed.");
    }
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
    const rm = el("button", "rm"); rm.type = "button"; rm.textContent = "✕";
    rm.onclick = () => { state.attachments = state.attachments.filter((a) => a !== att); renderAttachments(); };
    if (att.kind === "doc") {
      const chip = el("div", "thumb doc" + (att.uploading ? " uploading" : ""));
      const label = el("span", "doc-name");
      label.textContent = "📄 " + (att.name || "document");
      chip.append(label, rm);
      box.appendChild(chip);
    } else {
      const thumb = el("div", "thumb" + (att.uploading ? " uploading" : ""));
      const im = el("img");
      im.src = att.url || "";
      thumb.append(im, rm);
      box.appendChild(thumb);
    }
  });
  updateSendVisibility();
}

// --- Voice input (mic button → /api/transcribe, local faster-whisper) --------
// Click to record, click again to stop; the transcript lands in the composer.
let micRec = null;   // active MediaRecorder while recording
(() => {
  const btn = $("#mic");

  async function startRecording() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setStatus("Mic needs HTTPS or localhost (browser security rule).");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatus("Mic access denied: " + err.message);
      return;
    }
    const chunks = [];
    micRec = new MediaRecorder(stream);
    micRec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    micRec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.classList.remove("recording");
      btn.classList.add("busy");
      try {
        const mime = micRec.mimeType || "audio/webm";
        const ext = mime.includes("mp4") ? "mp4" : mime.includes("ogg") ? "ogg" : "webm";
        const fd = new FormData();
        fd.append("file", new Blob(chunks, { type: mime }), `voice.${ext}`);
        const res = await fetch("/api/transcribe", { method: "POST", body: fd });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.text) {
          const input = $("#input");
          input.value = (input.value.trim() ? input.value.replace(/\s*$/, " ") : "") + data.text;
          autosizeInput();
          updateSendVisibility();
          input.focus();
        } else {
          setStatus("Heard nothing — try again closer to the mic.");
        }
      } catch (err) {
        setStatus("Transcription failed: " + err.message);
      } finally {
        btn.classList.remove("busy");
        micRec = null;
      }
    };
    micRec.start();
    btn.classList.add("recording");
    btn.title = "Stop recording";
  }

  btn.onclick = () => {
    if (btn.classList.contains("busy")) return;
    if (micRec && micRec.state === "recording") {
      btn.title = "Voice input";
      micRec.stop();
    } else {
      startRecording();
    }
  };
})();

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
// Grow the composer with its content, up to ~10 lines (matches the CSS
// max-height); past that it scrolls internally. Once the text overflows the
// cap, an expand toggle appears in the composer's corner — expanded mode
// raises the cap to most of the viewport.
const COMPOSER_CAP = 240;
let composerExpanded = false;
const EXPAND_SVG = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg>';
const COLLAPSE_SVG = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M3 21l7-7"/></svg>';
function autosizeInput() {
  const input = $("#input");
  const cap = composerExpanded ? Math.round(window.innerHeight * 0.7) : COMPOSER_CAP;
  input.style.height = "auto";
  const overflows = input.scrollHeight > COMPOSER_CAP;
  input.style.height = Math.min(input.scrollHeight, cap) + "px";
  // Content shrank back under the cap → nothing to expand; drop the mode too
  // so the next overflow starts collapsed.
  if (!overflows) composerExpanded = false;
  const btn = $("#expand-input");
  btn.hidden = !overflows;
  btn.innerHTML = composerExpanded ? COLLAPSE_SVG : EXPAND_SVG;
  btn.title = btn.ariaLabel = composerExpanded ? "Collapse input" : "Expand input";
  const box = input.closest(".composer-box");
  box.classList.toggle("expandable", overflows);
  box.classList.toggle("expanded", composerExpanded);
}
$("#expand-input").addEventListener("click", () => {
  composerExpanded = !composerExpanded;
  autosizeInput();
  $("#input").focus();
});
window.addEventListener("resize", () => { if (composerExpanded) autosizeInput(); });
$("#input").addEventListener("input", () => {
  autosizeInput();
  updateSendVisibility();
});

// The send button only exists once there's something to send (text or an
// attachment); while streaming it stays visible as the stop control.
function updateSendVisibility() {
  const show = state.streaming || $("#input").value.trim() ||
    state.attachments.length;
  $("#send").hidden = !show;
}

async function send(providedText, providedImages, providedDocs) {
  if (state.streaming) return;
  // Retry passes an explicit text/images/docs; otherwise read from the composer.
  const isRetry = providedText !== undefined;
  if (!isRetry && state.attachments.some((a) => a.uploading)) {
    setStatus("Wait for the upload to finish…"); return;
  }
  const text = (isRetry ? providedText : $("#input").value).trim();
  const provider = $("#provider").value;
  const model = $("#model").value;
  const images = isRetry
    ? (providedImages || [])
    : state.attachments.filter((a) => a.kind !== "doc").map((a) => a.url).filter(Boolean);
  const docs = isRetry
    ? (providedDocs || [])
    : state.attachments.filter((a) => a.kind === "doc" && a.url).map((a) => ({ url: a.url, name: a.name }));
  if (!text && !images.length && !docs.length) return;
  if (!provider || !model) { setStatus("Pick a provider and model first."); setModelMenu(true); return; }
  // What actually answers this turn — updated if the server falls back.
  let turnModel = `${provider}/${model}`;

  state.stick = true;  // a turn you just started should follow the new output
  addMessageBubble("user", text, images, null, null, null, docs);
  if (!isRetry) {
    $("#input").value = "";
    autosizeInput();
    state.attachments = [];
    renderAttachments();  // also refreshes send-button visibility
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
      body: JSON.stringify({ conversation_id: state.conversationId, provider, model, message: text, images: images.length ? images : null, docs: docs.length ? docs : null, think: state.think, effort: state.think ? (state.effort || null) : null, research: state.research }),
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
        turnModel = `${ev.provider}/${ev.model}`;
        setStatus(`↳ fell back to ${turnModel}`);
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
        if (!assistantBubble) assistantBubble = addMessageBubble("assistant", "", null, null, turnModel);
        ensureMetricsEl(assistantBubble);
        assistantText += ev.text;
        assistantBubble.innerHTML = renderMarkdown(assistantText);
        shortenLongLinks(assistantBubble);
        scroll();
        break;
      case "tool_call":
        thinkTick(false);
        endThinking();
        // Research launched: disarm the composer toggle so follow-up messages
        // in this chat are normal turns, not another research kickoff.
        if (ev.name === "deep_research" && state.research) {
          state.research = false;
          $("#research-toggle").classList.remove("on");
          $("#research-toggle").setAttribute("aria-pressed", "false");
        }
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
        // Tag the finished bubble with its stored id so read-aloud (which
        // synthesizes server-side from the saved message) works immediately.
        if (ev.message_id && assistantBubble) {
          const w = assistantBubble.closest(".msg");
          if (w && !w.dataset.msgId) w.dataset.msgId = ev.message_id;
        }
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
  updateSendVisibility();
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
        if (!(await confirmDialog(`Delete wiki page “${p.name}”? The model loses this knowledge.`))) return;
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
      if (!(await confirmDialog(`Delete job “${j.name}”? It stops running on its schedule.`))) return;
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

// --- Status page --------------------------------------------------------------
const humanCount = (n) => {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + "K";
  return String(n);
};

const humanBytes = (n) => {
  if (n < 1024) return n + " B";
  const units = ["KB", "MB", "GB"];
  let u = -1;
  do { n /= 1024; u++; } while (n >= 1024 && u < units.length - 1);
  return n.toFixed(n >= 10 ? 0 : 1) + " " + units[u];
};

function statusSection(title) {
  const sec = el("div", "status-section");
  const h = el("h3");
  h.textContent = title;
  sec.appendChild(h);
  return sec;
}

// A label/value rows table (plain divs; keeps markup out of innerHTML).
function statusRows(pairs) {
  const box = el("div", "status-rows");
  pairs.forEach(([label, value, cls]) => {
    const row = el("div", "status-row");
    const l = el("span", "slabel"); l.textContent = label;
    const v = el("span", "svalue" + (cls ? " " + cls : "")); v.textContent = value;
    row.append(l, v);
    box.appendChild(row);
  });
  return box;
}

function renderStatus(data) {
  const body = $("#status-body");
  body.innerHTML = "";

  const prov = statusSection("Providers");
  prov.appendChild(statusRows(data.providers.map((p) => [
    `${p.name} · ${p.models} model${p.models === 1 ? "" : "s"}`,
    p.serving ? "● serving" : "○ down",
    p.serving ? "ok" : "bad",
  ])));
  body.appendChild(prov);

  const search = statusSection("Search API quota");
  if (!data.search) {
    search.appendChild(statusRows([["searchmw", "not configured (IMAGE_SEARCH_URL)", ""]]));
  } else if (data.search.error) {
    search.appendChild(statusRows([["searchmw", data.search.error, "bad"]]));
  } else {
    const rows = (data.search.slots || []).map((s) => {
      const q = s.quota || {};
      let val = "?";
      if (q.limit) {
        const left = Math.max(0, q.limit - (q.used || 0));
        val = `${left} left of ${q.limit}` + (q.source === "local-count" ? " (self-counted)" : "");
      } else if (q.used != null) {
        val = `${q.used} used`;
      }
      if (s.cooling_down) val += " · ⏸ cooling down";
      return [s.slot, val, s.cooling_down ? "warn" : ""];
    });
    const cache = data.search.cache;
    if (cache && cache.hit_rate != null) {
      rows.push(["query cache", `${Math.round(cache.hit_rate * 100)}% hit rate (${cache.size} cached)`, ""]);
    }
    search.appendChild(statusRows(rows));
  }
  body.appendChild(search);

  const usage = statusSection("Model usage (replies)");
  const models = (data.stats.models || []).slice(0, 12);
  usage.appendChild(statusRows(
    models.length
      ? models.map((m) => [m.model, `${m.replies_7d} this week · ${m.replies} all-time`, ""])
      : [["—", "no replies recorded yet", ""]]
  ));
  body.appendChild(usage);

  const jobs = statusSection("Scheduled jobs");
  const jrows = (data.jobs || []).map((j) => [
    `${j.enabled ? "●" : "○"} ${j.name}`,
    `${j.cron_desc}${j.last_status ? " · last: " + j.last_status : ""}`,
    j.last_status === "error" ? "bad" : "",
  ]);
  jobs.appendChild(statusRows(jrows.length ? jrows : [["—", "none", ""]]));
  body.appendChild(jobs);

  const tok = data.stats.tokens || {};
  const tp = (tok.tokens_prompt || {}).value || 0;
  const tc = (tok.tokens_completion || {}).value || 0;
  const since = (tok.tokens_prompt || tok.tokens_completion || {}).since;
  const tokLabel = "Tokens" + (since ? ` (since ${new Date(since * 1000).toLocaleDateString()})` : "");

  const sys = statusSection("System");
  sys.appendChild(statusRows([
    ["Discord", data.discord.connected ? `● connected as ${data.discord.user}` : "○ not connected",
     data.discord.connected ? "ok" : "bad"],
    [tokLabel, tp || tc ? `${humanCount(tp)} in · ${humanCount(tc)} out` : "counting starts now", ""],
    ["Conversations", `${data.stats.conversations} chats · ${data.stats.messages} messages`, ""],
    ["Database", humanBytes(data.storage.db_bytes), ""],
    ["Uploads", humanBytes(data.storage.uploads_bytes), ""],
  ]));
  body.appendChild(sys);
}

$("#show-status").onclick = async () => {
  showPage("status");
  $("#status-body").innerHTML = '<p class="hint">Loading…</p>';
  try {
    const res = await fetch("/api/status");
    renderStatus(await res.json());
  } catch (err) {
    $("#status-body").innerHTML = "";
    $("#status-body").appendChild(statusRows([["error", err.message, "bad"]]));
  }
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
