// MotifQL demo UI: compile English, animate the witness hop by hop.
// Pure helpers are exported for the node harness; the rest is DOM glue.

'use strict';

const HOP_DUR = 1.5;   // seconds per hop (chain witnesses)
const MEMBER_DUR = 0.6; // seconds per member (group witnesses have many)
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
  };
}

/** Pinned truth anchors for known demo sentences (path motifs find
 * ~nothing on the random sample, so their chips carry anchors). */
function chipAnchorsFor(text) {
  const chips = window.__chips || [];
  const k = (text || '').trim();
  const hit = chips.find((c) => c.english.trim() === k);
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
      $('landing-english').value = c.english;
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
  playing: false, t: 0, raf: 0, lastTick: 0,
  geom: null, // {nodes: Map key->{x,y}, edges: [{path, len, label}], nodeEls}
  evasion: { noise: 0, gap: 1.0, timer: 0, curve: null, seq: 0 },
};

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
  document.body.classList.remove('workspace-enter');
  document.body.classList.add('landing');
  setTimeout(() => $('landing-english').focus(), 60);
}

/** Submit the landing chat box: dive into the workspace, then compile. */
async function landingSubmit(english, anchors) {
  const text = (english !== undefined ? english : $('landing-english').value).trim();
  if (!text) return;
  $('english').value = text;
  $('landing-english').value = text;
  await enterWorkspace();
  await runCompile(text, anchors || []);
}

function showError(where, err) {
  $(where).innerHTML = `<div class="error-box">${esc(String(err.message || err))}</div>`;
}

function renderMotif(motif, source, attempts) {
  $('motif-box').innerHTML = `<pre class="motif">${esc(motif)}</pre>`;
  const badge = $('source-badge');
  badge.hidden = false;
  const labels = {
    model: ['live · Muse Spark', 'live'],
    cache: ['cached · Muse Spark', 'cached'],
    canned: ['scripted · Muse Spark output', 'cached'],
  };
  const [text, cls] = labels[source] || [source, ''];
  badge.textContent = `${text} · ${attempts} attempt${attempts === 1 ? '' : 's'}`;
  badge.className = `badge ${cls}`;
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

function renderDegrees(w) {
  const box = $('degrees');
  box.innerHTML = '';
  (w.explanation || []).forEach((e) => {
    const row = document.createElement('div');
    row.className = 'deg-row';
    const pct = Math.max(0, Math.min(1, e.degree)) * 100;
    row.innerHTML = `<div class="label" title="${esc(e.predicate)}">${esc(e.predicate)}</div>
      <div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
      <div class="sub">${e.degree.toFixed(3)}${e.step ? ` · weakest at hop ${e.step}` : ''}</div>`;
    box.appendChild(row);
  });
  $('degrees-sec').hidden = (w.explanation || []).length === 0;
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
  const edges = hops.map((h, i) => {
    const a = at.get(acctKey(h, 'from')), b = at.get(acctKey(h, 'to'));
    const [x1, y1, x2, y2] = shortenEdge(a.x, a.y, b.x, b.y, NR);
    const d = group
      ? `M ${x1} ${y1} C ${x1 + (x2 - x1) * 0.45} ${y1}, ${x2 - (x2 - x1) * 0.45} ${y2}, ${x2} ${y2}`
      : edgeCurve(x1, y1, x2, y2, cx, cy, closed);
    const path = el('path', {
      d, fill: 'none', stroke: colors.todo, 'stroke-width': 2.5,
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
    const lab = el('g', { opacity: 0 }, gL);
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
      badge = el('g', { opacity: 0 }, gL);
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
  const cur = hops.findIndex((_, i) => t < hopStart(i, g.unit) + g.unit);
  if (t < INTRO || !hops.length) {
    cap.innerHTML = `<span class="dim">${hops.length ? 'Tracing the money…' : 'No members in this witness.'}</span>`;
  } else if (cur === -1) {
    const span = hops.length > 1
      ? fmtGap(hops[hops.length - 1].timestamp - hops[0].timestamp) : '–';
    const mids = group && g.mids > 0 ? `, ${g.mids} intermediaries` : '';
    const units = group ? 'members' : 'hops';
    cap.innerHTML = `<span class="dim">Trace complete — ${hops.length} ${units}${mids}, span ${span}, ` +
      `score <span class="amt">${w.top_score.toFixed(3)}</span></span>`;
  } else {
    const h = hops[cur];
    const gap = !group && cur > 0 ? h.timestamp - hops[cur - 1].timestamp : null;
    cap.innerHTML = `<span class="hop-no">${unitWord} ${cur + 1}/${hops.length}</span> ` +
      `<span class="dim">${esc(acctKey(h, 'from'))} → ${esc(acctKey(h, 'to'))}</span> · ` +
      `<span class="amt">${fmtMoney(h.amount)}</span> ` +
      `<span class="dim">${fmtTime(h.timestamp)}${gap !== null ? ` · Δ${fmtGap(gap)} since last` : ''}</span>`;
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
    state.t = state.geom.dur;
    renderFrame(state.t);
    stopPlayer();
    return;
  }
  renderFrame(state.t);
  state.raf = requestAnimationFrame(tick);
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
  renderDegrees({ explanation: d.explanation || [] });
  $('evasion-caption').innerHTML = formatEvasionCaption(
    state.evasion.noise, state.evasion.gap,
    d.baseline_fuzzy, d.fuzzy_top, d.baseline_crisp, d.crisp_top);
  const badge = $('ev-source');
  badge.hidden = false;
  badge.textContent = source;
  badge.className = `badge ${source === 'live' ? 'live' : 'cached'}`;
}

async function updateEvasion() {
  const w = state.witnesses[state.wi];
  if (!w || !(w.witness || []).length || !state.motif) return;
  const anchor = w.witness[w.witness.length - 1].transfer_id;
  const my = ++state.evasion.seq;
  evasionLabels();
  let data = null, source = 'live';
  try {
    const res = await api('/perturb', {
      motif_text: state.motif, anchor_tids: [anchor],
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

async function runCompile(english, anchorTids) {
  if (!anchorTids || !anchorTids.length) {
    anchorTids = chipAnchorsFor(english); // demo sentences pin truth anchors
  }
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
  $('compile-btn').onclick = () => runCompile($('english').value, []);
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
    const dataLine = `${health.edges.toLocaleString('en-US')} transfers · ${health.motifs.length} motifs`;
    const db = $('data-badge');
    db.hidden = false;
    db.textContent = dataLine;
    $('landing-data').textContent = dataLine;
    const params = parseBootParams(location.search);
    // ?sentence= + ?anchors= deep-link any demo (rehearsal links per pattern).
    // Stills (?at=) and ?landing=0 skip the landing; otherwise the scripted
    // demo auto-runs through it (skipped with ?autorun=0).
    const english = params.sentence || cfg.english;
    if (params.sentence) {
      $('english').value = params.sentence;
      $('landing-english').value = params.sentence;
    }
    const skipLanding = !params.landing || params.at !== null;
    if (skipLanding) document.body.classList.remove('landing');
    if (english && params.autorun) {
      const anchors = params.anchors.length ? params.anchors
        : window.__demoAnchors;
      if (skipLanding) {
        await runCompile(english, anchors);
      } else {
        // Let the landing read on camera, then dive into the workspace.
        await new Promise((r) => setTimeout(r, 1100));
        await enterWorkspace();
        await runCompile(english, anchors);
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
    parseBootParams, isChain, narrativeHTML,
    formatEvasionCaption, lookupEvasionPoint,
    HOP_DUR, MEMBER_DUR, INTRO, OUTRO, EVASION_SEED,
  };
}
