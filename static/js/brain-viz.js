/* brain-viz.js — canvas scatter for /brain, the embedding-space "brain" view.
 * No dependencies (no CDN, no bundler) — plain Canvas 2D, consistent with
 * this app's self-hosted/offline-friendly posture. Data comes from
 * /api/brain-viz/points (PCA -> t-SNE projection) and /api/brain-viz/search
 * (cosine-similarity highlight against the same point set).
 */
(function () {
  'use strict';

  // Fixed category -> categorical-slot order. Never reorder or generate a
  // slot per new value — that's the CVD-safety mechanism (dataviz skill).
  var CATEGORY_ORDER = ['memory', 'personal_document', 'imported:chatgpt', 'imported:claude', 'imported', 'skill'];
  var CATEGORY_LABELS = {
    'memory': 'Memory',
    'personal_document': 'Document',
    'imported:chatgpt': 'ChatGPT import',
    'imported:claude': 'Claude import',
    'imported': 'Imported chat',
    'skill': 'Skill',
    'other': 'Other',
  };
  var SLOT_COUNT = 7; // --series-1..7 in brain.html

  var canvas = document.getElementById('viz-canvas');
  var ctx = canvas.getContext('2d');
  var tooltip = document.getElementById('tooltip');
  var ttValue = document.getElementById('tt-value');
  var ttLabel = document.getElementById('tt-label');
  var statusEl = document.getElementById('status');
  var countEl = document.getElementById('count');
  var legendEl = document.getElementById('legend');
  var searchEl = document.getElementById('search');

  var points = [];         // {id,x,y,preview,title,category,source}
  var highlighted = null;  // Set<id> | null
  var view = { scale: 1, tx: 0, ty: 0 };
  var dpr = Math.max(1, window.devicePixelRatio || 1);
  var hoverPoint = null;
  var dragging = false;
  var dragStart = null;
  var animFrame = null;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function slotColor(category) {
    var idx = CATEGORY_ORDER.indexOf(category);
    var slot = (idx === -1 || idx >= SLOT_COUNT - 1) ? SLOT_COUNT : idx + 1;
    return cssVar('--series-' + slot);
  }

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    var rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    draw();
  }

  function fitToPoints() {
    if (!points.length) return;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    var rect = canvas.getBoundingClientRect();
    var pad = 60;
    var w = Math.max(1e-6, maxX - minX);
    var h = Math.max(1e-6, maxY - minY);
    var scale = Math.min((rect.width - pad * 2) / w, (rect.height - pad * 2) / h);
    scale = Math.max(0.01, Math.min(scale, 40));
    view.scale = scale;
    view.tx = rect.width / 2 - scale * (minX + maxX) / 2;
    view.ty = rect.height / 2 - scale * (minY + maxY) / 2;
  }

  function toScreen(p) {
    return { x: p.x * view.scale + view.tx, y: p.y * view.scale + view.ty };
  }

  function toWorld(sx, sy) {
    return { x: (sx - view.tx) / view.scale, y: (sy - view.ty) / view.scale };
  }

  function draw() {
    var rect = canvas.getBoundingClientRect();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    var ring = cssVar('--surface-1') || '#111';
    var r = 4.5; // marker radius, spec: r >= 4
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      var s = toScreen(p);
      if (s.x < -20 || s.x > rect.width + 20 || s.y < -20 || s.y > rect.height + 20) continue;
      var isHi = highlighted && highlighted.has(p.id);
      var isHover = hoverPoint === p;
      var rad = isHi || isHover ? r + 2.5 : r;

      if (isHi) {
        // Highlight halo — a soft outer ring, not a repaint of survivors
        // (identity color never changes; the halo signals "matched").
        ctx.beginPath();
        ctx.arc(s.x, s.y, rad + 6, 0, Math.PI * 2);
        ctx.strokeStyle = slotColor(p.category);
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      ctx.beginPath();
      ctx.arc(s.x, s.y, rad, 0, Math.PI * 2);
      ctx.fillStyle = slotColor(p.category);
      ctx.globalAlpha = highlighted && !isHi ? 0.28 : 1;
      ctx.fill();
      // 2px surface ring so overlapping points stay legible.
      ctx.lineWidth = 2;
      ctx.strokeStyle = ring;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  function nearestPoint(sx, sy, maxDist) {
    var best = null, bestD = maxDist * maxDist;
    for (var i = 0; i < points.length; i++) {
      var s = toScreen(points[i]);
      var dx = s.x - sx, dy = s.y - sy;
      var d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = points[i]; }
    }
    return best;
  }

  function showTooltip(p, sx, sy) {
    ttValue.textContent = p.title || p.preview || '(untitled)';
    var label = (CATEGORY_LABELS[p.category] || p.category || 'Other');
    ttLabel.textContent = label + (p.preview && p.title ? ' — ' + p.preview.slice(0, 140) : '');
    tooltip.style.display = 'block';
    var rect = canvas.getBoundingClientRect();
    var left = Math.min(sx + 14, rect.width - 336);
    var top = Math.min(sy + 14, rect.height - 90);
    tooltip.style.left = Math.max(8, left) + 'px';
    tooltip.style.top = Math.max(8, top) + 'px';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  function renderLegend() {
    var present = [];
    var seen = {};
    for (var i = 0; i < points.length; i++) {
      var c = points[i].category;
      if (!CATEGORY_LABELS[c]) c = 'other';
      if (!seen[c]) { seen[c] = true; present.push(c); }
    }
    present.sort(function (a, b) {
      var ia = CATEGORY_ORDER.indexOf(a); if (ia === -1) ia = 999;
      var ib = CATEGORY_ORDER.indexOf(b); if (ib === -1) ib = 999;
      return ia - ib;
    });
    legendEl.textContent = '';
    if (present.length < 2) { legendEl.style.display = 'none'; return; } // single series needs no legend box
    legendEl.style.display = 'block';
    present.forEach(function (c) {
      var row = document.createElement('div');
      row.className = 'legend-row';
      var sw = document.createElement('span');
      sw.className = 'legend-swatch';
      sw.style.background = slotColor(c);
      var label = document.createElement('span');
      label.className = 'legend-label';
      label.textContent = CATEGORY_LABELS[c] || c;
      row.appendChild(sw);
      row.appendChild(label);
      legendEl.appendChild(row);
    });
  }

  function setStatus(html) {
    if (!html) { statusEl.classList.add('hidden'); statusEl.textContent = ''; return; }
    statusEl.classList.remove('hidden');
    statusEl.textContent = html;
  }

  async function loadPoints() {
    setStatus('Loading your brain…');
    try {
      var res = await fetch('/api/brain-viz/points?dims=2&limit=1500');
      if (res.status === 503) {
        var body = await res.json().catch(function () { return {}; });
        setStatus(body.detail || 'Brain visualizer is not available yet.');
        return;
      }
      if (!res.ok) { setStatus('Could not load your brain (HTTP ' + res.status + ').'); return; }
      var data = await res.json();
      points = data.points || [];
      countEl.textContent = points.length + (points.length === 1 ? ' point' : ' points');
      if (!points.length) {
        setStatus('Nothing here yet — chat with Zeus, import history, or add documents to grow your brain.');
        return;
      }
      setStatus(null);
      fitToPoints();
      renderLegend();
      draw();
    } catch (e) {
      setStatus('Could not reach the server.');
    }
  }

  var searchTimer = null;
  function scheduleSearch(q) {
    clearTimeout(searchTimer);
    if (!q) { highlighted = null; draw(); return; }
    searchTimer = setTimeout(function () { runSearch(q); }, 250);
  }

  async function runSearch(q) {
    try {
      var res = await fetch('/api/brain-viz/search?q=' + encodeURIComponent(q) + '&k=25');
      if (!res.ok) return;
      var data = await res.json();
      var ids = (data.matches || []).map(function (m) { return m.id; });
      highlighted = new Set(ids);
      // Pan/zoom to fit the matched cluster so search results are actually visible.
      var matched = points.filter(function (p) { return highlighted.has(p.id); });
      if (matched.length) {
        var save = points;
        points = matched;
        fitToPoints();
        points = save;
        view.scale *= 0.85; // a little breathing room around the matches
      }
      draw();
    } catch (e) { /* non-fatal */ }
  }

  // --- interaction ---
  canvas.addEventListener('pointerdown', function (e) {
    dragging = true;
    canvas.classList.add('dragging');
    dragStart = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', function () {
    dragging = false;
    canvas.classList.remove('dragging');
  });
  canvas.addEventListener('pointermove', function (e) {
    var rect = canvas.getBoundingClientRect();
    var sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (dragging && dragStart) {
      view.tx = dragStart.tx + (e.clientX - dragStart.x);
      view.ty = dragStart.ty + (e.clientY - dragStart.y);
      hideTooltip();
      draw();
      return;
    }
    // Nearest-point hit test with a generous radius (spec: hit target
    // bigger than the mark — ~24px — for dense scatter).
    var p = nearestPoint(sx, sy, 16);
    if (p !== hoverPoint) {
      hoverPoint = p;
      draw();
    }
    if (p) {
      var s = toScreen(p);
      showTooltip(p, s.x, s.y);
    } else {
      hideTooltip();
    }
  });
  canvas.addEventListener('pointerleave', function () {
    hoverPoint = null;
    hideTooltip();
    draw();
  });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var rect = canvas.getBoundingClientRect();
    var sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    var before = toWorld(sx, sy);
    var factor = Math.exp(-e.deltaY * 0.001);
    view.scale = Math.max(0.005, Math.min(view.scale * factor, 200));
    var after = toScreen(before);
    view.tx += sx - after.x;
    view.ty += sy - after.y;
    draw();
  }, { passive: false });

  searchEl.addEventListener('input', function () {
    scheduleSearch(searchEl.value.trim());
  });

  window.addEventListener('resize', resize);
  resize();
  loadPoints();
})();
