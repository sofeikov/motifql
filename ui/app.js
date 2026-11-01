// MotifQL demo UI: compile English, animate the witness hop by hop.
// Pure helpers are exported for the node harness; the rest is DOM glue.

'use strict';

const HOP_DUR = 1.2;   // seconds per hop (chain witnesses)
const MEMBER_DUR = 0.5; // seconds per member (group witnesses have many)
const INTRO = 0.5;     // pause before the trace starts
const OUTRO = 0.6;     // hold on the final frame

// -- pure helpers -----------------------------------------------------------

function fmtMoney(x) {
  const s = Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return '$' + s;
}

function fmtGap(sec) {
  if (sec < 90) return Math.round(sec) + 's';
  if (sec < 90 * 60) return (sec / 60).toFixed(sec < 600 ? 1 : 0) + 'm';
  if (sec < 48 * 3600) return (sec / 3600).toFixed(1) + 'h';
  return (sec / 86400).toFixed(1) + 'd';
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function fmtTime(epoch) {
  const d = new Date(epoch * 1000);
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${hh}:${mm}`;
}

function acctKey(hop, end) {
  return end === 'from'
    ? `${hop.from_bank}/${hop.from_account}`
    : `${hop.to_bank}/${hop.to_account}`;
}

/** True when every hop continues the previous account (a real chain). */
function isChain(hops) {
  if (!hops.length) return true;
  for (let i = 1; i < hops.length; i++) {
    if (acctKey(hops[i], 'from') !== acctKey(hops[i - 1], 'to')) return false;
  }
  return true;
}

/**
 * Node positions. Chains: circle when closed, arc otherwise. Anything
 * else (fan/scatter/gather witnesses) lays out in source/transit/sink
 * columns by each node's in/out signature. Returns {nodes, closed,
 * kind, unit, mids}.
 */
function layout(hops, W, H) {
  const order = [];
  const seen = new Set();
  const push = (k) => { if (!seen.has(k)) { seen.add(k); order.push(k); } };
  hops.forEach((h) => { push(acctKey(h, 'from')); push(acctKey(h, 'to')); });
  const closed = hops.length > 1 &&
    acctKey(hops[0], 'from') === acctKey(hops[hops.length - 1], 'to');
  const cx = W / 2, cy = H / 2 - 8;
  if (!isChain(hops)) {
    // Layered group layout: sources | transit | sinks.
    const hasIn = new Set(), hasOut = new Set();
    hops.forEach((h) => {
      hasOut.add(acctKey(h, 'from'));
      hasIn.add(acctKey(h, 'to'));
    });
    const cols = [[], [], []];
    order.forEach((key) => {
      const src = hasOut.has(key) && !hasIn.has(key);
      const snk = hasIn.has(key) && !hasOut.has(key);
      cols[src ? 0 : snk ? 2 : 1].push(key);
    });
    const nodes = [];
    cols.forEach((col, c) => {
      if (!col.length) return; // e.g. all-transit witnesses center alone
      const x = W * (0.14 + 0.36 * c);
      col.forEach((key, j) => {
        const y = col.length === 1 ? cy
          : H * 0.12 + ((H * 0.72) * j) / (col.length - 1);
        // Middle-column tags stay short (m1..mN); the caption and the
        // tooltip carry the full keys as members reveal.
        const tag = c === 1 && col.length > 2 ? `m${j + 1}` : null;
        nodes.push({ key, x, y, col: c, tag });
      });
    });
    return { nodes, closed: false, kind: 'group', unit: MEMBER_DUR, mids: cols[1].length };
  }
  const nodes = order.map((key, i) => {
    let x, y;
    if (closed) {
      const R = Math.min(W, H) * 0.30;
      const a = -Math.PI / 2 + (i / order.length) * 2 * Math.PI;
      x = cx + R * Math.cos(a);
      y = cy + R * Math.sin(a);
    } else {
      const n = Math.max(order.length, 2);
      x = W * 0.12 + ((W * 0.76) * i) / (n - 1);
      y = cy - Math.sin((i / (n - 1)) * Math.PI) * H * 0.10;
    }
    return { key, x, y };
  });
  return { nodes, closed, kind: 'chain', unit: HOP_DUR, mids: 0 };
}

function hopStart(i, unit) { return INTRO + i * (unit || HOP_DUR); }
function totalDur(n, unit) { return INTRO + n * (unit || HOP_DUR) + OUTRO; }

const EVASION_SEED = 12345; // pinned: canned curve + live path share the draws

/** Boot deep-links: ?sentence=, ?anchors=tid,.. ?at=sec, ?noise=pct, ?gap=factor, ?autoplay=0, ?autorun=0, ?landing=0. */
function parseBootParams(search) {
  const q = new URLSearchParams(search);
  const anchors = (q.get('anchors') || '').split(',')
    .map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isInteger(n));
  const at = q.has('at') ? parseFloat(q.get('at')) : null;
  const noise = q.has('noise') ? parseFloat(q.get('noise')) : 0;
  const gap = q.has('gap') ? parseFloat(q.get('gap')) : 1.0;
  return {
    sentence: q.get('sentence'),
    anchors,
    at: Number.isFinite(at) ? at : null,
    noise: Number.isFinite(noise) ? Math.max(0, Math.min(30, noise)) : 0,
    gap: Number.isFinite(gap) ? Math.max(1, Math.min(2, gap)) : 1.0,
    autoplay: q.get('autoplay') !== '0',
    autorun: q.get('autorun') !== '0',
    landing: q.get('landing') !== '0',
    loop: q.get('loop') === '1',
  };
}

/** Chip display text: single sentence, or queries joined by a blank line. */
function chipText(c) {
  return Array.isArray(c.english) ? c.english.join('\n\n') : c.english;
}

/** Exact chip match for submitted text (single or composite). */
function matchChip(text) {
  const k = (text || '').trim();
  return (window.__chips || []).find((c) => chipText(c).trim() === k) || null;
}

/** Grow a textarea to fit its content (no scrollbar for demo text). */
function autosize(box) {
  if (!box || !Number.isFinite(box.scrollHeight)) return;
  box.style.height = 'auto';
  box.style.height = `${box.scrollHeight}px`;
}

/** Pinned truth anchors for known demo sentences (path motifs find
 * ~nothing on the random sample, so their chips carry anchors). */
function chipAnchorsFor(text) {
  const hit = matchChip(text);
  return hit ? (hit.anchor_tids || []) : [];
}

/** Render the landing "Try one" chips from /demo-config. */
function renderChips(chips) {
  const list = $('examples-list');
  list.innerHTML = '';
  (chips || []).forEach((c) => {
    const b = document.createElement('button');
    b.setAttribute('data-ex', c.key);
    const label = document.createElement('b');
    label.textContent = c.label;
    b.appendChild(label);
    b.appendChild(document.createTextNode(c.blurb));
    b.onclick = () => {
      $('landing-english').value = chipText(c);
      autosize($('landing-english'));
      $('landing-english').focus();
    };
    list.appendChild(b);
  });
  $('examples-sec').hidden = list.children.length === 0;
}

/** Evasion caption: "+15% noise ×1.50 gaps: fuzzy 0.718→0.357, exact 1→0". */
function formatEvasionCaption(noisePct, gap, baseF, pertF, baseC, pertC) {
  const f = (x) => x.toFixed(3);
  const c = (x) => String(parseFloat(x.toFixed(3)));
  const head = noisePct === 0 && gap === 1
    ? 'Clean witness'
    : `+${noisePct}% noise ×${Number(gap).toFixed(2)} gaps`;
  return `<span class="dim">${head}: fuzzy </span>` +
    `<span class="amt">${f(baseF)}→${f(pertF)}</span>` +
    `<span class="dim">, exact </span>` +
    `<span class="amt">${c(baseC)}→${c(pertC)}</span>`;
}

/** Nearest canned-curve point (exact match wins with distance 0). */
function lookupEvasionPoint(curve, noisePct, gap) {
  const pts = (curve && curve.points) || [];
  if (!pts.length) return null;
  let best = pts[0], bd = Infinity;
  for (const p of pts) {
    const d = Math.abs(p.noise_pct - noisePct) / 30 + Math.abs(p.gap - gap);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

// -- app --------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';

const state = {
  witnesses: [], wi: 0, motif: '', compiled: false,
  composite: null, // {unionHops, shared, groups:[{label,motif,top_score,found,explanation,verdict,witness}]}
  compositeEvGroup: 0, // which shape the evasion sliders attack
  loop: false, // replay the trace on end (demo hold + ?loop=1)
  playing: false, t: 0, raf: 0, lastTick: 0,
  geom: null, // {nodes: Map key->{x,y}, edges: [{path, len, label}], nodeEls}
  evasion: { noise: 0, gap: 1.0, timer: 0, curve: null, seq: 0 },
};

/** True when the stage shows a unified multi-shape trace. */
function isComposite() { return !!state.composite; }

/**
 * Union witness hop-lists into one time-ordered trace. Returns
 * {hops, shared} where shared holds transfer_ids present in 2+
 * lists. Pure: node-harnessable.
 */
function unionWitnesses(lists) {
  const seen = new Map(); // tid -> {hop, lists}
  (lists || []).forEach((hops) => {
    const seenHere = new Set(); // group witnesses repeat tids across roles
    (hops || []).forEach((h) => {
      if (seenHere.has(h.transfer_id)) return;
      seenHere.add(h.transfer_id);
      const e = seen.get(h.transfer_id);
      if (e) e.lists += 1;
      else seen.set(h.transfer_id, { hop: h, lists: 1 });
    });
  });
  const hops = [...seen.values()].map((e) => e.hop).sort(
    (a, b) => (a.timestamp - b.timestamp) || (a.transfer_id - b.transfer_id));
  const shared = new Set([...seen.entries()]
    .filter(([, e]) => e.lists > 1).map(([tid]) => tid));
  return { hops, shared };
}

function el(name, attrs, parent) {
  const e = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

function setBusy(busy, msg) {
  $('compile-btn').disabled = busy;
  $('scripted-btn').disabled = busy;
  $('landing-go').disabled = busy;
  $('landing-scripted').disabled = busy;
  $('caption').innerHTML = busy
    ? `<span class="spin"></span><span class="dim">${esc(msg)}</span>`
    : '';
}

/** Landing → workspace transition (instant under reduced motion). */
function enterWorkspace() {
  if (!document.body.classList.contains('landing')) return Promise.resolve();
  const landing = $('landing');
  const swap = () => {
    document.body.classList.remove('landing');
    document.body.classList.add('workspace-enter');
    landing.classList.remove('leaving');
  };
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    swap();
    return Promise.resolve();
  }
  landing.classList.add('leaving');
  return new Promise((resolve) => setTimeout(() => { swap(); resolve(); }, 330));
}

function backToLanding() {
  stopPlayer();
  $('landing-english').value = $('english').value;
  autosize($('landing-english'));
  document.body.classList.remove('workspace-enter');
  document.body.classList.add('landing');
  setTimeout(() => $('landing-english').focus(), 60);
}

/** Route submitted text: composite chips run all queries at once. */
async function submitText(text, anchors) {
  const match = matchChip(text);
  const tids = (anchors && anchors.length)
    ? anchors : (match ? (match.anchor_tids || []) : []);
  if (match && Array.isArray(match.english)) {
    const labels = match.labels && match.labels.length === match.english.length
      ? match.labels
      : match.english.map((_, i) => `Shape ${i + 1}`);
    return runComposite(match.english, labels, tids);
  }
  return runCompile(text, tids);
}

/** Submit the landing chat box: dive into the workspace, then compile. */
async function landingSubmit(english, anchors) {
  const text = (english !== undefined ? english : $('landing-english').value).trim();
  if (!text) return;
  $('english').value = text;
  $('landing-english').value = text;
  autosize($('english'));
  autosize($('landing-english'));
  await enterWorkspace();
  await submitText(text, anchors || []);
}

function showError(where, err) {
  $(where).innerHTML = `<div class="error-box">${esc(String(err.message || err))}</div>`;
}

function sourceLabel(source) {
  const labels = {
    model: ['live · Muse Spark', 'live'],
    cache: ['cached · Muse Spark', 'cached'],
    canned: ['scripted · Muse Spark output', 'cached'],
  };
  return labels[source] || [source, ''];
}

function renderMotif(motif, source, attempts) {
  $('motif-box').innerHTML = `<pre class="motif">${esc(motif)}</pre>`;
  const badge = $('source-badge');
  badge.hidden = false;
  const [text, cls] = sourceLabel(source);
  badge.textContent = `${text} · ${attempts} attempt${attempts === 1 ? '' : 's'}`;
  badge.className = `badge ${cls}`;
}

/** Composite motif panel: one labeled block per query group. */
function renderMotifs(groups, labels) {
  const box = $('motif-box');
  box.innerHTML = '';
  groups.forEach((g, gi) => {
    const head = document.createElement('div');
    head.className = 'motif-head';
    const [text] = sourceLabel(g.source);
    head.textContent = `${labels[gi] || `Shape ${gi + 1}`} · ${text}`;
    box.appendChild(head);
    if (g.compiled) {
      const pre = document.createElement('pre');
      pre.className = 'motif';
      pre.textContent = g.motif;
      box.appendChild(pre);
    } else {
      const err = document.createElement('div');
      err.className = 'error-box';
      err.textContent = `Still not compiling after ${g.attempts} attempt(s):\n${g.errors.join('\n')}`;
      box.appendChild(err);
    }
  });
  const badge = $('source-badge');
  badge.hidden = false;
  const attempts = groups.reduce((n, g) => n + g.attempts, 0);
  badge.textContent = `${groups.length} queries · ${attempts} attempt${attempts === 1 ? '' : 's'}`;
  badge.className = `badge ${sourceLabel(groups[0].source)[1]}`;
}

function renderTabs() {
  const tabs = $('witness-tabs');
  tabs.innerHTML = '';
  state.witnesses.forEach((w, i) => {
    const b = document.createElement('button');
    b.textContent = `W${i + 1} · ${w.top_score.toFixed(3)}`;
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', i === state.wi ? 'true' : 'false');
    b.onclick = () => selectWitness(i, true);
    tabs.appendChild(b);
  });
  $('witness-sec').hidden = state.witnesses.length === 0;
}

/** Composite strip: shapes firing + transfers shared across shapes. */
function renderCompositeLine() {
  const box = $('composite-line');
  const c = state.composite;
  if (!c) {
    box.hidden = true;
    box.textContent = '';
    return;
  }
  const firing = c.groups.filter((g) => g.found).length;
  const n = c.shared.size;
  const firstW = c.groups.map((g) => g.witness).find((w) => w && w.length);
  const anchor = firstW ? firstW[firstW.length - 1].transfer_id : null;
  box.hidden = false;
  box.innerHTML = `<span class="dim">${firing}/${c.groups.length} shapes fire` +
    (anchor !== null ? ` on transfer #${anchor}` : '') +
    ` · </span><span class="amt">${n} shared transfer${n === 1 ? '' : 's'}</span>` +
    `<span class="dim"> link${n === 1 ? 's' : ''} the shapes</span>`;
}

function degRow(e) {
  const row = document.createElement('div');
  row.className = 'deg-row';
  const pct = Math.max(0, Math.min(1, e.degree)) * 100;
  row.innerHTML = `<div class="label" title="${esc(e.predicate)}">${esc(e.predicate)}</div>
    <div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
    <div class="sub">${e.degree.toFixed(3)}${e.step ? ` · weakest at hop ${e.step}` : ''}</div>`;
  return row;
}

function renderDegrees(w) {
  const box = $('degrees');
  box.innerHTML = '';
  (w.explanation || []).forEach((e) => box.appendChild(degRow(e)));
  $('degrees-sec').hidden = (w.explanation || []).length === 0;
}

/** Composite degrees: each shape's bars under its own subhead. */
function renderDegreesComposite(overrides) {
  const box = $('degrees');
  box.innerHTML = '';
  let rows = 0;
  state.composite.groups.forEach((g, gi) => {
    const expl = (overrides && overrides[gi]) || g.explanation || [];
    if (!expl.length) return;
    const head = document.createElement('div');
    head.className = 'deg-group';
    head.textContent = `${g.label} · ${g.top_score.toFixed(3)}`;
    box.appendChild(head);
    expl.forEach((e) => box.appendChild(degRow(e)));
    rows += expl.length;
  });
  $('degrees-sec').hidden = rows === 0;
}

/** Verdict lede + narrative paragraph HTML (pure: node-harnessable). */
function narrativeHTML(narrative, weakestStep, verdict) {
  if (!narrative && !verdict) return '';
  let html = verdict ? `<p class="lede">${esc(verdict)}</p>` : '';
  if (narrative) html += `<p>${esc(narrative)}</p>`;
  if (Number.isInteger(weakestStep)) {
    html += `<span class="weakest">Weakest step: hop ${weakestStep}</span>`;
  }
  return html;
}

function renderNarrative(w) {
  const box = $('narrative');
  const html = narrativeHTML(w.narrative, w.narrative_weakest_step, w.verdict);
  box.hidden = !html;
  box.innerHTML = html;
}

function renderScore(w) {
  $('score-line').hidden = false;
  $('score-val').textContent = w.top_score.toFixed(3);
  const v = $('verdict');
  v.textContent = w.found ? 'fires' : 'below emit';
  v.className = `verdict ${w.found ? 'fires' : 'quiet'}`;
  const hops = w.witness || [];
  const unitWord = isChain(hops) ? 'hop' : 'member';
  const span = hops.length > 1
    ? fmtGap(hops[hops.length - 1].timestamp - hops[0].timestamp) : '–';
  // Engine proof: the search work behind this witness, not the LLM.
  const exp = Number.isInteger(w.expansions) && w.expansions > 0
    ? ` · ${w.expansions.toLocaleString('en-US')} expansions` : '';
  const trunc = w.truncated ? ' · truncated' : '';
  $('score-meta').textContent =
    `${hops.length} ${unitWord}${hops.length === 1 ? '' : 's'} · span ${span} · anchor #${hops.length ? hops[hops.length - 1].transfer_id : '–'}${exp}${trunc}`;
}

/** Composite score line: firing count big, per-shape scores in meta. */
function renderScoreComposite() {
  const c = state.composite;
  $('score-line').hidden = false;
  const firing = c.groups.filter((g) => g.found).length;
  $('score-val').textContent = `${firing}/${c.groups.length}`;
  const v = $('verdict');
  v.textContent = 'composite';
  v.className = `verdict ${firing === c.groups.length && c.groups.length ? 'fires' : 'quiet'}`;
  const perShape = c.groups.map((g) => `${g.label} ${g.top_score.toFixed(3)}`).join(' · ');
  const firstW = c.groups.map((g) => g.witness).find((w) => w && w.length);
  const anchor = firstW ? firstW[firstW.length - 1].transfer_id : null;
  $('score-meta').textContent =
    `${perShape} · ${c.unionHops.length} transfers` +
    (anchor !== null ? ` · anchor #${anchor}` : '');
}

/** Composite narrative: unified diagnosis first, shape verdicts after. */
function renderNarrativeComposite() {
  const box = $('narrative');
  const c = state.composite;
  let html = c.diagnosis ? `<p class="lede">${esc(c.diagnosis)}</p>` : '';
  html += c.groups
    .filter((g) => g.verdict)
    .map((g) => `<p><b>${esc(g.label)}:</b> ${esc(g.verdict)}</p>`).join('');
  box.hidden = !html;
  box.innerHTML = html;
}

function edgeCurve(x1, y1, x2, y2, cx, cy, closed) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  let dx = mx - cx, dy = my - cy;
  const len = Math.hypot(dx, dy) || 1;
  const chord = Math.hypot(x2 - x1, y2 - y1);
  if (!closed) { dx = 0; dy = -1; } // bow upward on open paths
  if (chord < 1) { dx = 1; dy = -1; } // self-loop: loop out sideways
  const k = chord < 1 ? 46 : Math.min(60, chord * 0.18);
  return `M ${x1} ${y1} Q ${mx + (dx / len) * k} ${my + (dy / len) * k} ${x2} ${y2}`;
}

function shortenEdge(x1, y1, x2, y2, r) {
  // Pull endpoints back to node rims so arrows don't hide in nodes.
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  r = r || 30;
  if (len < 2 * r + 4) return [x1, y1, x2, y2];
  const ux = dx / len, uy = dy / len;
  return [x1 + ux * r, y1 + uy * r, x2 - ux * (r + 3), y2 - uy * (r + 3)];
}

function buildSVG(w) {
  const svg = $('graph');
  svg.innerHTML = '';
  const W = 960, H = 600;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const hops = w.witness || [];
  const defs = el('defs', {}, svg);
  const colors = { todo: '#3a4658', done: '#3ddc84', active: '#f0b429' };
  for (const [name, color] of Object.entries(colors)) {
    const m = el('marker', {
      id: `arr-${name}`, viewBox: '0 0 10 10', refX: 8, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse',
    }, defs);
    el('path', { d: 'M 0 1 L 9 5 L 0 9 z', fill: color }, m);
  }
  // faint dotted grid backdrop
  const grid = el('g', { opacity: 0.5 }, svg);
  for (let gx = 40; gx < W; gx += 56) {
    for (let gy = 40; gy < H; gy += 56) {
      el('circle', { cx: gx, cy: gy, r: 1.1, fill: '#1b2330' }, grid);
    }
  }
  if (!hops.length) return null;
  const lay = layout(hops, W, H);
  const { nodes, closed } = lay;
  const group = lay.kind === 'group';
  const NR = group ? 24 : 30;
  const at = new Map(nodes.map((n) => [n.key, n]));
  const cx = W / 2, cy = H / 2 - 8;
  const gE = el('g', {}, svg);
  const gL = el('g', {}, svg);
  const gN = el('g', {}, svg);
  const shared = (state.composite && state.composite.shared) || new Set();
  // Same-pair census: duplicate transfers would draw identical curves
  // with stacked labels, so parallels bow apart and labels stagger.
  const pairTotal = new Map();
  hops.forEach((h) => {
    const k = acctKey(h, 'from') + '>' + acctKey(h, 'to');
    pairTotal.set(k, (pairTotal.get(k) || 0) + 1);
  });
  // Column slots: same-column edges skipping nodes must bow around them.
  const slotOf = new Map();
  if (group) {
    const byCol = new Map();
    nodes.forEach((n) => {
      if (!byCol.has(n.col)) byCol.set(n.col, []);
      byCol.get(n.col).push(n);
    });
    byCol.forEach((col) => {
      col.sort((p, q) => p.y - q.y).forEach((n, s) => slotOf.set(n.key, s));
    });
  }
  const pairSeen = new Map();
  const edges = hops.map((h, i) => {
    const a = at.get(acctKey(h, 'from')), b = at.get(acctKey(h, 'to'));
    const [x1, y1, x2, y2] = shortenEdge(a.x, a.y, b.x, b.y, NR);
    const pk = acctKey(h, 'from') + '>' + acctKey(h, 'to');
    const pi = pairSeen.get(pk) || 0;
    pairSeen.set(pk, pi + 1);
    const spread = group ? pi - ((pairTotal.get(pk) || 1) - 1) / 2 : 0;
    let nx = -(y2 - y1), ny = (x2 - x1);
    const nl = Math.hypot(nx, ny);
    if (nl < 1) { nx = 1; ny = 0; } else { nx /= nl; ny /= nl; }
    let d;
    if (!group) {
      d = edgeCurve(x1, y1, x2, y2, cx, cy, closed);
    } else if (Math.hypot(x2 - x1, y2 - y1) < 1) {
      // Self-loop, rim-anchored (shortenEdge no-ops at the center):
      // exits top, bulges sideways, re-enters upper rim.
      const s = spread === 0 ? 1 : Math.sign(spread);
      const r = 30 * (1 + Math.abs(spread) * 0.35);
      const x0 = x1 + s * NR * 0.55, y0 = y1 - NR * 0.85;
      const x3 = x1 + s * NR * 0.95, y3 = y1 - NR * 0.15;
      d = `M ${x0} ${y0} C ${x0 + s * r} ${y0 - 22}, ` +
        `${x3 + s * (r + 8)} ${y3 - 12}, ${x3} ${y3}`;
    } else {
      const bow = spread * 14;
      let sideX = 0;
      const fk = acctKey(h, 'from'), tk = acctKey(h, 'to');
      const fa = at.get(fk), ta = at.get(tk);
      if (fa && ta && fa.col === ta.col &&
          Math.abs((slotOf.get(fk) || 0) - (slotOf.get(tk) || 0)) > 1) {
        sideX = -(30 + 25 * Math.abs(slotOf.get(fk) - slotOf.get(tk)));
      }
      const c1x = x1 + (x2 - x1) * 0.45 + nx * bow + sideX;
      const c1y = y1 + ny * bow;
      const c2x = x2 - (x2 - x1) * 0.45 + nx * bow + sideX;
      const c2y = y2 + ny * bow;
      d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
    }
    const stack = `translate(${nx * spread * 18} ${ny * spread * 18})`;
    const isShared = shared.has(h.transfer_id);
    const path = el('path', {
      d, fill: 'none', stroke: colors.todo,
      'stroke-width': isShared ? 4.5 : 2.5,
      'stroke-dasharray': '5 6', 'marker-end': 'url(#arr-todo)',
      'stroke-linecap': 'round',
    }, gE);
    const len = path.getTotalLength();
    // Chain labels: amount over gap at the curve midpoint. Group
    // witnesses are dense: amount only (12px), and the hop badge rides
    // off-curve along the normal so the two never stack.
    const mid = path.getPointAtLength(len / 2);
    const gap = i === 0 ? null : h.timestamp - hops[i - 1].timestamp;
    const sub = group ? null : gap === null ? null : `Δ${fmtGap(gap)}`;
    let bx = mid.x, by = mid.y - 38;
    if (group) {
      const ahead = path.getPointAtLength(Math.min(len, len / 2 + 3));
      const back = path.getPointAtLength(Math.max(0, len / 2 - 3));
      let nx = -(ahead.y - back.y), ny = ahead.x - back.x;
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl; ny /= nl;
      if (ny > 0) { nx = -nx; ny = -ny; }
      bx = mid.x + nx * 24;
      by = mid.y + ny * 24;
    }
    const lab = el('g', { opacity: 0, transform: stack }, gL);
    const halo = {
      'text-anchor': 'middle', 'font-family': 'ui-monospace, Menlo, monospace',
      stroke: '#0b0e13', 'stroke-width': 4, 'paint-order': 'stroke',
    };
    el('text', Object.assign({
      x: mid.x, y: mid.y - 6, fill: '#e9eef5',
      'font-size': group ? 12 : 14, 'font-weight': 700,
    }, halo), lab).textContent = fmtMoney(h.amount);
    if (sub !== null) {
      el('text', Object.assign({
        x: mid.x, y: mid.y + 11, fill: '#9fadbf', 'font-size': 11,
      }, halo), lab).textContent = sub;
    }
    // Dense group witnesses skip badges (the caption narrates the
    // order); the dots would land on neighbouring labels.
    let badge = null;
    if (!group || hops.length <= 8) {
      badge = el('g', { opacity: 0, transform: stack }, gL);
      el('circle', { cx: bx, cy: by, r: 10, fill: '#f0b429' }, badge);
      el('text', {
        x: bx, y: by + 4, 'text-anchor': 'middle',
        fill: '#241a02', 'font-size': 11, 'font-weight': 700,
      }, badge).textContent = String(i + 1);
    }
    return { path, len, lab, badge };
  });
  const nodeEls = nodes.map((n, j) => {
    // Labels sit clear of the edges: radially outward on closed
    // chains, below the node otherwise.
    const outward = closed && n.y < cy - 4 ? -1 : 1;
    const g = el('g', { opacity: 0 }, gN);
    el('circle', {
      cx: n.x, cy: n.y, r: NR, fill: '#141c27',
      stroke: '#3a4658', 'stroke-width': 2,
    }, g);
    if (!group) {
      el('text', { // visit order, so the flow reads at a glance
        x: n.x, y: n.y + 5, 'text-anchor': 'middle',
        fill: '#667080', 'font-size': 14, 'font-weight': 700,
        'font-family': 'ui-monospace, Menlo, monospace',
      }, g).textContent = String(j + 1);
    }
    if (group) {
      const shown = n.tag || (n.key.length > 12 ? '…' + n.key.slice(-11) : n.key);
      const leftSide = n.col === 0; // source labels escape the fan origin
      el('text', {
        x: leftSide ? n.x - NR - 8 : n.x + NR + 8, y: n.y + 4,
        'text-anchor': leftSide ? 'end' : 'start',
        fill: '#9fadbf', 'font-size': 11.5,
        'font-family': 'ui-monospace, Menlo, monospace',
        stroke: '#0b0e13', 'stroke-width': 3, 'paint-order': 'stroke',
      }, g).textContent = shown;
    } else {
      const shown = n.key.length > 14 ? '…' + n.key.slice(-13) : n.key;
      el('text', {
        x: n.x, y: n.y + outward * (NR + 20), 'text-anchor': 'middle',
        fill: '#9fadbf', 'font-size': 11.5,
        'font-family': 'ui-monospace, Menlo, monospace',
        stroke: '#0b0e13', 'stroke-width': 3, 'paint-order': 'stroke',
      }, g).textContent = shown;
    }
    el('title', {}, g).textContent = n.key;
    return g;
  });
  // member ticks on the progress bar
  const track = $('track');
  track.querySelectorAll('.tick').forEach((e) => e.remove());
  const dur = totalDur(hops.length, lay.unit);
  hops.forEach((_, i) => {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${(hopStart(i, lay.unit) / dur) * 100}%`;
    track.appendChild(tick);
  });
  return { nodes, at, edges, nodeEls, dur, kind: lay.kind, unit: lay.unit, mids: lay.mids };
}

function renderFrame(t) {
  const g = state.geom;
  const w = state.witnesses[state.wi];
  if (!g || !w) return;
  const hops = w.witness || [];
  const group = g.kind === 'group';
  const unitWord = group ? 'Member' : 'Hop';
  const doneC = '#3ddc84', activeC = '#f0b429', todoC = '#3a4658';
  hops.forEach((h, i) => {
    const e = g.edges[i];
    const p = Math.max(0, Math.min(1, (t - hopStart(i, g.unit)) / g.unit));
    const setBadge = (v) => { if (e.badge) e.badge.setAttribute('opacity', v); };
    if (p <= 0) {
      e.path.setAttribute('stroke', todoC);
      e.path.setAttribute('stroke-dasharray', '5 6');
      e.path.setAttribute('marker-end', 'url(#arr-todo)');
      e.lab.setAttribute('opacity', 0);
      setBadge(0);
    } else if (p >= 1) {
      e.path.setAttribute('stroke', doneC);
      e.path.setAttribute('stroke-dasharray', 'none');
      e.path.setAttribute('marker-end', 'url(#arr-done)');
      e.path.style.filter = '';
      e.lab.setAttribute('opacity', 1);
      setBadge(1);
    } else {
      const draw = Math.min(1, p / 0.6);
      e.path.setAttribute('stroke', activeC);
      e.path.setAttribute('stroke-dasharray', `${e.len}`);
      e.path.setAttribute('stroke-dashoffset', `${e.len * (1 - draw)}`);
      e.path.setAttribute('marker-end', 'url(#arr-active)');
      e.path.style.filter = 'drop-shadow(0 0 6px rgba(240,180,41,0.8))';
      e.lab.setAttribute('opacity', Math.max(0, Math.min(1, (p - 0.55) / 0.3)));
      setBadge(Math.max(0, Math.min(1, (p - 0.4) / 0.3)));
    }
  });
  // nodes light as their first incident edge reveals (chains keep
  // the old visit-order timing: node 0 at intro, then per hop)
  const firstTouch = new Map();
  hops.forEach((h, i) => {
    [acctKey(h, 'from'), acctKey(h, 'to')].forEach((k) => {
      if (!firstTouch.has(k)) firstTouch.set(k, i);
    });
  });
  g.nodes.forEach((n, j) => {
    const gEl = g.nodeEls[j];
    const touch = firstTouch.has(n.key) ? firstTouch.get(n.key) : 0;
    const litAt = !group && j === 0 ? INTRO * 0.5
      : hopStart(touch, g.unit) + g.unit * 0.6;
    const lit = t >= litAt;
    gEl.setAttribute('opacity', lit ? 1 : 0.25);
    gEl.firstChild.setAttribute('stroke', lit ? doneC : todoC);
  });
  // caption: current member, or the summary at the ends
  const cap = $('caption');
  const shared = (state.composite && state.composite.shared) || new Set();
  const cur = hops.findIndex((_, i) => t < hopStart(i, g.unit) + g.unit);
  if (t < INTRO || !hops.length) {
    cap.innerHTML = `<span class="dim">${hops.length ? 'Tracing the money…' : 'No members in this witness.'}</span>`;
  } else if (cur === -1) {
    const span = hops.length > 1
      ? fmtGap(hops[hops.length - 1].timestamp - hops[0].timestamp) : '–';
    const mids = group && g.mids > 0 ? `, ${g.mids} intermediaries` : '';
    const units = group ? 'members' : 'hops';
    const link = shared.size
      ? `, <span class="amt">${shared.size}</span> in both shapes` : '';
    const scoreBit = isComposite() ? '' :
      `, score <span class="amt">${w.top_score.toFixed(3)}</span>`;
    cap.innerHTML = `<span class="dim">Trace complete — ${hops.length} ${units}${mids}, span ${span}${link}${scoreBit}</span>`;
  } else {
    const h = hops[cur];
    const gap = !group && cur > 0 ? h.timestamp - hops[cur - 1].timestamp : null;
    const link = shared.has(h.transfer_id)
      ? ` · <span class="amt">in both shapes</span>` : '';
    cap.innerHTML = `<span class="hop-no">${unitWord} ${cur + 1}/${hops.length}</span> ` +
      `<span class="dim">${esc(acctKey(h, 'from'))} → ${esc(acctKey(h, 'to'))}</span> · ` +
      `<span class="amt">${fmtMoney(h.amount)}</span> ` +
      `<span class="dim">${fmtTime(h.timestamp)}${gap !== null ? ` · Δ${fmtGap(gap)} since last` : ''}</span>${link}`;
  }
  $('fill').style.width = `${Math.max(0, Math.min(1, t / g.dur)) * 100}%`;
  $('clock').textContent = `${t.toFixed(1)}s / ${g.dur.toFixed(1)}s`;
}

function showPlayIcon(playing) {
  // style.display: the `hidden` attribute does not hide SVG children
  // reliably across browsers.
  $('icon-play').style.display = playing ? 'none' : '';
  $('icon-pause').style.display = playing ? '' : 'none';
}

function stopPlayer() {
  state.playing = false;
  cancelAnimationFrame(state.raf);
  showPlayIcon(false);
}

function tick(now) {
  if (!state.playing) return;
  const dt = (now - state.lastTick) / 1000;
  state.lastTick = now;
  state.t += dt;
  if (state.t >= state.geom.dur) {
    if (state.loop) {
      state.t = 0; // hold the demo on a living trace
    } else {
      state.t = state.geom.dur;
      renderFrame(state.t);
      stopPlayer();
      return;
    }
  }
  renderFrame(state.t);
  state.raf = requestAnimationFrame(tick);
}

/** Loop toggle (landing + transport checkboxes stay in sync). */
function setLoop(on) {
  state.loop = !!on;
  $('landing-loop').checked = state.loop;
  $('transport-loop').checked = state.loop;
  const q = new URLSearchParams(location.search);
  if (state.loop) q.set('loop', '1');
  else q.delete('loop');
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

function play(fromStart) {
  if (!state.geom) return;
  if (fromStart || state.t >= state.geom.dur) state.t = 0;
  state.playing = true;
  state.lastTick = performance.now();
  showPlayIcon(true);
  state.raf = requestAnimationFrame(tick);
}

function evasionLabels() {
  $('ev-noise-val').textContent = `±${state.evasion.noise}%`;
  $('ev-gap-val').textContent = `×${Number(state.evasion.gap).toFixed(2)}`;
}

function syncEvasionURL() {
  const q = new URLSearchParams(location.search);
  q.set('noise', String(state.evasion.noise));
  q.set('gap', String(state.evasion.gap));
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

function renderEvasion(d, source) {
  $('ev-fuzzy-val').textContent = d.fuzzy_top.toFixed(3);
  $('ev-crisp-val').textContent = String(parseFloat(d.crisp_top.toFixed(3)));
  $('ev-fuzzy-fill').style.width =
    `${Math.max(0, Math.min(1, d.fuzzy_top)) * 100}%`;
  $('ev-crisp-fill').style.width =
    `${Math.max(0, Math.min(1, d.crisp_top)) * 100}%`;
  if (isComposite()) {
    renderDegreesComposite({ [state.compositeEvGroup]: d.explanation || [] });
  } else {
    renderDegrees({ explanation: d.explanation || [] });
  }
  $('evasion-caption').innerHTML = formatEvasionCaption(
    state.evasion.noise, state.evasion.gap,
    d.baseline_fuzzy, d.fuzzy_top, d.baseline_crisp, d.crisp_top);
  const badge = $('ev-source');
  badge.hidden = false;
  badge.textContent = source;
  badge.className = `badge ${source === 'live' ? 'live' : 'cached'}`;
}

async function updateEvasion() {
  // Composite mode attacks the toggled shape's own witness and motif.
  const compGroup = isComposite()
    ? state.composite.groups[state.compositeEvGroup] : null;
  const w = compGroup || state.witnesses[state.wi];
  const motif = compGroup ? compGroup.motif : state.motif;
  if (!w || !(w.witness || []).length || !motif) return;
  const anchor = w.witness[w.witness.length - 1].transfer_id;
  const my = ++state.evasion.seq;
  evasionLabels();
  let data = null, source = 'live';
  try {
    const res = await api('/perturb', {
      motif_text: motif, anchor_tids: [anchor],
      amount_noise_pct: state.evasion.noise / 100,
      gap_scale: state.evasion.gap, seed: EVASION_SEED,
    });
    data = res.witnesses[0];
  } catch (err) {
    // Offline: the canned curve covers the scripted anchor exactly;
    // anything else fails honestly in the evasion caption.
    const curve = state.evasion.curve;
    const pt = curve && anchor === curve.anchor_tid
      ? lookupEvasionPoint(curve, state.evasion.noise, state.evasion.gap)
      : null;
    if (my !== state.evasion.seq) return;
    if (pt) {
      data = {
        fuzzy_top: pt.fuzzy_top, crisp_top: pt.crisp_top,
        baseline_fuzzy: curve.baseline.fuzzy_top,
        baseline_crisp: curve.baseline.crisp_top,
        explanation: pt.explanation,
      };
      source = 'scripted curve';
    } else {
      showError('evasion-caption', err);
      return;
    }
  }
  if (my !== state.evasion.seq || !data) return;
  renderEvasion(data, source);
}

function selectWitness(i, autoplay) {
  stopPlayer();
  state.wi = i;
  const w = state.witnesses[i];
  $('legend-shared').hidden = true;
  renderEvShapeToggle(); // hides itself outside composite mode
  renderTabs();
  renderDegrees(w);
  renderScore(w);
  renderNarrative(w); // before the still/empty returns: ?at= keeps it
  state.geom = buildSVG(w);
  state.t = 0;
  $('legend-done').textContent =
    state.geom && state.geom.kind === 'group' ? 'traced member' : 'traced hop';
  $('evasion').hidden = !(w.witness || []).length || !state.motif;
  if (!$('evasion').hidden) updateEvasion();
  if (!state.geom) {
    $('caption').innerHTML = '<span class="dim">No hops in this witness.</span>';
    return;
  }
  const params = parseBootParams(location.search);
  if (params.at !== null) {
    state.t = Math.max(0, Math.min(state.geom.dur, params.at));
    renderFrame(state.t);
    return;
  }
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    state.t = state.geom.dur;
    renderFrame(state.t);
    return;
  }
  renderFrame(0);
  if (autoplay && params.autoplay) play(true);
}

/** Composite selection: one unified trace over the union of shapes. */
function selectComposite(autoplay) {
  stopPlayer();
  const c = state.composite;
  state.witnesses = [{ witness: c.unionHops, top_score: 0, explanation: [] }];
  state.wi = 0;
  state.motif = (c.groups[state.compositeEvGroup] || {}).motif || '';
  $('witness-sec').hidden = true;
  $('legend-shared').hidden = c.shared.size === 0;
  renderScoreComposite();
  renderDegreesComposite({});
  renderNarrativeComposite();
  renderCompositeLine();
  renderEvShapeToggle();
  state.geom = buildSVG(state.witnesses[0]);
  state.t = 0;
  $('legend-done').textContent =
    state.geom && state.geom.kind === 'group' ? 'traced member' : 'traced hop';
  $('evasion').hidden = !c.unionHops.length || !state.motif;
  if (!$('evasion').hidden) updateEvasion();
  if (!state.geom) {
    $('caption').innerHTML = '<span class="dim">No hops in this witness.</span>';
    return;
  }
  const params = parseBootParams(location.search);
  if (params.at !== null) {
    state.t = Math.max(0, Math.min(state.geom.dur, params.at));
    renderFrame(state.t);
    return;
  }
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    state.t = state.geom.dur;
    renderFrame(state.t);
    return;
  }
  renderFrame(0);
  if (autoplay && params.autoplay) play(true);
}

/** Evasion shape toggle (composite only): which shape the sliders attack. */
function renderEvShapeToggle() {
  const box = $('ev-shape');
  box.innerHTML = '';
  if (!isComposite()) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  state.composite.groups.forEach((g, gi) => {
    const b = document.createElement('button');
    b.textContent = g.label;
    b.setAttribute('aria-selected', gi === state.compositeEvGroup ? 'true' : 'false');
    b.onclick = () => {
      state.compositeEvGroup = gi;
      state.motif = g.motif || '';
      renderEvShapeToggle();
      updateEvasion();
    };
    box.appendChild(b);
  });
}

async function runCompile(english, anchorTids) {
  if (!anchorTids || !anchorTids.length) {
    anchorTids = chipAnchorsFor(english); // demo sentences pin truth anchors
  }
  state.composite = null;
  $('composite-line').hidden = true;
  setBusy(true, 'Consulting Muse Spark…');
  try {
    const res = await api('/compile', {
      english, max_attempts: 3, n_witnesses: 3, anchor_tids: anchorTids || [],
    });
    renderMotif(res.motif, res.source, res.attempts);
    if (!res.compiled) {
      showError('motif-box', new Error(
        `Still not compiling after ${res.attempts} attempt(s):\n${res.errors.join('\n')}`));
      const pre = document.createElement('pre');
      pre.className = 'motif';
      pre.textContent = res.motif;
      $('motif-box').appendChild(pre);
      setBusy(false);
      $('narrative').hidden = true;
      $('caption').innerHTML = '<span class="dim">The model output did not compile — see errors.</span>';
      return;
    }
    state.motif = res.motif;
    state.witnesses = res.witnesses || [];
    setBusy(false);
    if (!state.witnesses.length) {
      $('caption').innerHTML = '<span class="dim">Compiled — but matches nothing on the sample.</span>';
      $('witness-sec').hidden = true;
      $('narrative').hidden = true;
      $('evasion').hidden = true;
      return;
    }
    selectWitness(0, true);
  } catch (err) {
    setBusy(false);
    showError('motif-box', err);
    $('witness-sec').hidden = true;
    $('degrees-sec').hidden = true;
    $('narrative').hidden = true;
    $('evasion').hidden = true;
    $('composite-line').hidden = true;
    $('caption').innerHTML = '<span class="dim">Press Scripted demo — or describe a pattern and compile it.</span>';
  }
}

/** Composite compile: several queries, same anchors, one unified trace. */
async function runComposite(englishList, labels, anchorTids) {
  state.composite = null;
  $('composite-line').hidden = true;
  setBusy(true, 'Consulting Muse Spark…');
  try {
    const res = await api('/compile', {
      english: englishList[0], english2: englishList[1],
      max_attempts: 3, n_witnesses: 3, anchor_tids: anchorTids || [],
    });
    const groups = res.groups || [];
    renderMotifs(groups, labels);
    const shaped = groups.map((g, gi) => {
      const w = (g.witnesses || [])[0] || null;
      return {
        label: labels[gi] || `Shape ${gi + 1}`,
        motif: g.compiled ? g.motif : '',
        top_score: w ? w.top_score : 0,
        found: !!(w && w.found),
        explanation: w ? (w.explanation || []) : [],
        verdict: w ? w.verdict : null,
        witness: w ? (w.witness || []) : [],
      };
    });
    const { hops, shared } = unionWitnesses(shaped.map((g) => g.witness));
    state.composite = {
      unionHops: hops, shared, groups: shaped,
      diagnosis: res.diagnosis || null,
    };
    state.compositeEvGroup = 0;
    setBusy(false);
    if (!hops.length) {
      const allFailed = groups.length && groups.every((g) => !g.compiled);
      state.composite = null;
      $('caption').innerHTML = allFailed
        ? '<span class="dim">The model output did not compile — see errors.</span>'
        : '<span class="dim">Compiled — but matches nothing on the sample.</span>';
      $('witness-sec').hidden = true;
      $('narrative').hidden = true;
      $('evasion').hidden = true;
      return;
    }
    selectComposite(true);
  } catch (err) {
    state.composite = null;
    setBusy(false);
    showError('motif-box', err);
    $('witness-sec').hidden = true;
    $('degrees-sec').hidden = true;
    $('narrative').hidden = true;
    $('evasion').hidden = true;
    $('composite-line').hidden = true;
    $('caption').innerHTML = '<span class="dim">Press Scripted demo — or describe a pattern and compile it.</span>';
  }
}

async function boot() {
  showPlayIcon(false);
  $('btn-play').onclick = () => (state.playing ? stopPlayer() : play(false));
  $('btn-back').onclick = () => {
    if (!state.geom) return;
    stopPlayer();
    const hops = state.witnesses[state.wi].witness || [];
    const u = state.geom.unit;
    const prev = [...hops.keys()].reverse().find((i) => hopStart(i, u) < state.t - 0.05);
    state.t = prev === undefined ? 0 : hopStart(prev, u);
    renderFrame(state.t);
  };
  $('btn-fwd').onclick = () => {
    if (!state.geom) return;
    stopPlayer();
    const hops = state.witnesses[state.wi].witness || [];
    const u = state.geom.unit;
    const next = hops.findIndex((_, i) => hopStart(i, u) > state.t + 0.05);
    state.t = next === -1 ? state.geom.dur : hopStart(next, u);
    renderFrame(state.t);
  };
  $('progress').onclick = (ev) => {
    if (!state.geom) return;
    const r = $('track').getBoundingClientRect();
    state.t = Math.max(0, Math.min(state.geom.dur,
      ((ev.clientX - r.left) / r.width) * state.geom.dur));
    renderFrame(state.t);
  };
  $('compile-btn').onclick = () => submitText($('english').value, []);
  $('scripted-btn').onclick = () => runCompile(window.__demoEnglish, window.__demoAnchors);
  $('new-search-btn').onclick = backToLanding;
  $('landing-go').onclick = () => landingSubmit();
  $('landing-scripted').onclick = () =>
    landingSubmit(window.__demoEnglish, window.__demoAnchors);
  $('landing-english').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      landingSubmit();
    }
  });
  $('landing-english').addEventListener('input', (ev) => autosize(ev.target));
  $('english').addEventListener('input', (ev) => autosize(ev.target));
  $('landing-loop').addEventListener('change', (ev) => setLoop(ev.target.checked));
  $('transport-loop').addEventListener('change', (ev) => setLoop(ev.target.checked));
  window.__chips = [];
  renderChips([]);
  // Evasion sliders: labels + URL update instantly, re-scoring debounced.
  const bootParams = parseBootParams(location.search);
  state.evasion.noise = Math.round(bootParams.noise);
  state.evasion.gap = bootParams.gap;
  $('ev-noise').value = state.evasion.noise;
  $('ev-gap').value = state.evasion.gap;
  evasionLabels();
  const scheduleEvasion = () => {
    state.evasion.noise = parseInt($('ev-noise').value, 10);
    state.evasion.gap = parseFloat($('ev-gap').value);
    evasionLabels();
    syncEvasionURL();
    clearTimeout(state.evasion.timer);
    state.evasion.timer = setTimeout(updateEvasion, 150);
  };
  $('ev-noise').oninput = scheduleEvasion;
  $('ev-gap').oninput = scheduleEvasion;
  try {
    const [cfg, health] = await Promise.all([api('/demo-config'), api('/healthz')]);
    window.__demoEnglish = cfg.english;
    window.__demoAnchors = cfg.anchor_tids || [];
    window.__chips = cfg.chips || [];
    renderChips(window.__chips);
    state.evasion.curve = cfg.evasion_curve || null;
    $('english').value = cfg.english || '';
    $('landing-english').value = cfg.english || '';
    autosize($('english'));
    autosize($('landing-english'));
    const dataLine = `${health.edges.toLocaleString('en-US')} transfers · ${health.motifs.length} motifs`;
    const db = $('data-badge');
    db.hidden = false;
    db.textContent = dataLine;
    $('landing-data').textContent = dataLine;
    const params = parseBootParams(location.search);
    state.loop = params.loop;
    $('landing-loop').checked = params.loop;
    $('transport-loop').checked = params.loop;
    // ?sentence= + ?anchors= deep-link any demo (rehearsal links per pattern).
    // Stills (?at=) and ?landing=0 skip the landing; otherwise the scripted
    // demo auto-runs through it (skipped with ?autorun=0).
    const english = params.sentence || cfg.english;
    if (params.sentence) {
      $('english').value = params.sentence;
      $('landing-english').value = params.sentence;
      autosize($('english'));
      autosize($('landing-english'));
    }
    const skipLanding = !params.landing || params.at !== null;
    if (skipLanding) document.body.classList.remove('landing');
    if (english && params.autorun) {
      const anchors = params.anchors.length ? params.anchors
        : window.__demoAnchors;
      if (skipLanding) {
        await submitText(english, anchors);
      } else {
        // Let the landing read on camera, then dive into the workspace.
        await new Promise((r) => setTimeout(r, 1100));
        await enterWorkspace();
        await submitText(english, anchors);
      }
    } else if (!skipLanding) {
      $('landing-english').focus();
    }
  } catch (err) {
    setBusy(false);
    showError('motif-box', err);
    if (document.body.classList.contains('landing')) {
      $('landing-data').textContent = `demo unavailable: ${err.message || err}`;
    }
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', boot);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    fmtMoney, fmtGap, fmtTime, acctKey, layout, hopStart, totalDur,
    parseBootParams, isChain, narrativeHTML, unionWitnesses,
    formatEvasionCaption, lookupEvasionPoint,
    HOP_DUR, MEMBER_DUR, INTRO, OUTRO, EVASION_SEED,
  };
}
