// SmartPoli: Digital Twin tab for the doctor dashboard (demo copy).
//
// The nav item stays hidden unless GET /twin/info answers (it is a 404 on the live app, where the twin is off).
// The twin only informs: every output is a risk flag for the doctor, never a diagnosis or a treatment change.
// The screen is built to be read at a glance: one short headline, one picture, two small meters, and a guided
// tour (a moving pointer with short captions) that shows a first-time viewer where to press.
// Doctor actions go to the server (AuditLog / clinical notes), so they show on the patient timeline and report.

(function () {
  const T = { enabled: false, loaded: false, list: [], others: [], sel: null, d: null, at: null, why: false, curve: false, playing: false, timer: null, seq: 0, wait: null, tour: null };
  const COL = { alert: '#D3402A', watch: '#B4791F', calm: '#12876F', none: '#B4BDB8' };
  const SEG = { alert: '#D3402A', watch: '#E2A93B', calm: '#58B79B', high: '#6C9BEA' };
  const STEP_MS = 15 * 60000;
  const DATA_CREDIT = { ShanghaiT2DM: 'Data: ShanghaiT2DM (Zhao et al. 2023, CC BY 4.0), anonymised and replayed.', CGMacros: 'Data: CGMacros (PhysioNet), anonymised and replayed.' };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const hhmm = (iso) => iso.slice(11, 16);
  const pad = (n) => String(n).padStart(2, '0');
  const toIso = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00`;
  const dayLabel = (iso, year = true) => new Date(iso).toLocaleDateString([], year ? { day: 'numeric', month: 'long', year: 'numeric' } : { day: 'numeric', month: 'long' });
  const calmMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  async function init() {
    try {
      const r = await fetch('/twin/info');
      if (!r.ok) return;
      T.enabled = true;
      const btn = document.querySelector('nav.pill-nav button[data-tab="twin"]');
      if (btn) btn.style.display = '';
    } catch (e) { /* offline: leave the tab hidden */ }
  }

  function toast(msg) {
    let el = $('twnToast');
    if (!el) { el = document.createElement('div'); el.id = 'twnToast'; el.setAttribute('role', 'status'); document.body.appendChild(el); }
    el.textContent = msg; el.style.opacity = 1;
    clearTimeout(T.toastTimer); T.toastTimer = setTimeout(() => { el.style.opacity = 0; }, 2600);
  }

  function tierOf(p, d) {
    if (p.mg_dl <= d.low.alert_at_or_under) return 'alert';
    if (p.mg_dl <= d.low.watch_at_or_under) return 'watch';
    return p.mg_dl >= 180 ? 'high' : 'calm';
  }

  function reasonsHtml(list) {
    if (!list || !list.length) return '';
    return list.slice(0, 3).map((r) => `<div class="twn-reason"><span>${esc(r.text)}</span><span class="twn-bar" title="${r.up ? 'raises' : 'lowers'} the risk"><i style="width:${Math.max(8, Math.min(100, r.weight))}%;background:${r.up ? '#D3402A' : '#12876F'}"></i></span></div>`).join('') +
      '<p class="twn-quiet" style="margin:4px 0 0">Red raises the risk, green lowers it.</p>';
  }

  // ---- pictures -------------------------------------------------------------------------------------------------
  function curveSvg(d, width, full) {
    const W = Math.max(300, Math.min(900, width)), L = full ? 36 : 6, R = 8, top = full ? 10 : 8, ch = full ? 190 : 104, H = full ? 250 : ch + top * 2;
    const hist = d.history.map((p) => [new Date(p[0]).getTime(), p[1]]);
    const now = new Date(d.as_of).getTime(), span = full ? 6 * 3600000 : 3 * 3600000, t0 = now - span, t1 = now + 125 * 60000;
    const X = (t) => L + (t - t0) / (t1 - t0) * (W - L - R);
    const Y = (v) => top + ch - (Math.max(40, Math.min(240, v)) - 40) / 200 * ch;
    let s = `<rect x="${L}" y="${Y(70)}" width="${W - L - R}" height="${Y(40) - Y(70)}" fill="#FBE2DD" opacity=".7"/><rect x="${L}" y="${Y(240)}" width="${W - L - R}" height="${Y(180) - Y(240)}" fill="#FBEBD3" opacity=".7"/>`;
    if (full) {
      [70, 120, 180].forEach((g) => { s += `<line x1="${L}" x2="${W - R}" y1="${Y(g)}" y2="${Y(g)}" stroke="#8B9A94" stroke-dasharray="4 4"/><text x="${L - 6}" y="${Y(g) + 4}" text-anchor="end" font-size="12" fill="#5C6E68">${g}</text>`; });
      for (let t = Math.ceil(t0 / 3600000) * 3600000; t <= t1; t += (W < 600 ? 2 : 1) * 3600000) s += `<text x="${X(t)}" y="${top + ch + 18}" text-anchor="middle" font-size="12" fill="#5C6E68">${pad(new Date(t).getHours())}:00</text>`;
    } else {
      [70, 180].forEach((g) => { s += `<line x1="${L}" x2="${W - R}" y1="${Y(g)}" y2="${Y(g)}" stroke="#8B9A94" stroke-dasharray="3 5" opacity=".6"/>`; });
    }
    const f = d.forecast, up = f.map((p) => `${X(new Date(p.time))},${Y(p.high)}`), dn = f.slice().reverse().map((p) => `${X(new Date(p.time))},${Y(p.low)}`);
    s += `<polygon points="${X(now)},${Y(d.current)} ${up.join(' ')} ${dn.join(' ')}" fill="#9FD9C5" opacity=".75"/><polyline fill="none" stroke="#0C6455" stroke-width="2.5" stroke-dasharray="6 5" points="${X(now)},${Y(d.current)} ${f.map((p) => `${X(new Date(p.time))},${Y(p.mg_dl)}`).join(' ')}"/>`;
    if (full && d.what_really_happened.length) s += `<polyline fill="none" stroke="#16241F" stroke-width="2" stroke-dasharray="2 4" opacity=".6" points="${X(now)},${Y(d.current)} ${d.what_really_happened.map((p) => `${X(new Date(p[0]))},${Y(p[1])}`).join(' ')}"/>`;
    s += `<polyline fill="none" stroke="#16241F" stroke-width="3" stroke-linejoin="round" points="${hist.filter((p) => p[0] >= t0).map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}"/><line x1="${X(now)}" x2="${X(now)}" y1="${top}" y2="${top + ch}" stroke="#0C6455" stroke-width="1.5"/><circle cx="${X(now)}" cy="${Y(d.current)}" r="5" fill="#16241F"/>`;
    if (!full) s += `<text x="${X(now) - 6}" y="${top + 12}" text-anchor="end" font-size="12" font-weight="700" fill="#0C6455">now</text>`;
    if (full) {
      const lane = top + ch + 40;
      d.events.forEach((e) => { const x = X(new Date(e.time)); if (x < L || x > W - R) return; const o = new Date(e.time).getTime() <= now ? 1 : .35; s += e.kind === 'insulin' ? `<path d="M${x} ${lane - 9} l8 14 h-16z" fill="#2E6FE0" opacity="${o}"><title>${esc(e.text)}</title></path>` : `<circle cx="${x}" cy="${lane + 14}" r="6" fill="#B4791F" opacity="${o}"><title>${esc(e.text)}</title></circle>`; });
    }
    return `<svg viewBox="0 0 ${W} ${full ? H + 20 : H}" role="img" aria-label="Sugar so far and the forecast with its likely range">${s}</svg>`;
  }

  function hrSvg(d, W) {
    if (!d.heart_rate || d.heart_rate.length < 3) return '';
    const v = d.heart_rate.map((p) => p[1]), lo = Math.min(...v, 50), hi = Math.max(...v, 100), H = 60, L = 36, R = 8;
    const now = new Date(d.as_of).getTime(), t0 = now - 6 * 3600000, t1 = now + 125 * 60000;
    const X = (t) => L + (t - t0) / (t1 - t0) * (W - L - R), Y = (b) => 6 + (hi - b) / (hi - lo) * (H - 12);
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Heart rate" style="margin-top:4px"><text x="4" y="12" font-size="12" fill="#5C6E68">Heart</text><polyline fill="none" stroke="#D3402A" stroke-width="2" points="${d.heart_rate.map((p) => `${X(new Date(p[0]))},${Y(p[1])}`).join(' ')}"/><text x="${W - 8}" y="12" text-anchor="end" font-size="12" fill="#5C6E68">${Math.round(lo)} to ${Math.round(hi)} bpm</text></svg>`;
  }

  function detailHtml() {
    const d = T.d;
    if (!d) return '<div class="card"><p class="twn-quiet">Loading...</p></div>';
    const rv = d.review.state !== 'open', tier = d.low.tier, col = rv ? 'var(--ink)' : COL[tier];
    const w = ($('twnDetail') ? $('twnDetail').clientWidth : 700) - 40;
    let h = `<div class="card"><div class="twn-head"><div><b style="font-size:18px">${esc(d.label)}</b><div class="twn-quiet">${esc(d.summary || '')}</div></div>`;
    if (d.review.state === 'reviewed') h += `<span class="twn-chip reviewed">Reviewed by ${esc(d.review.info.by)}</span>`;
    if (d.review.state === 'snoozed') h += '<span class="twn-chip">Snoozed for 1 hour</span>';
    h += `</div><div class="twn-big" style="color:${col}">${esc(d.headline)}</div>`;
    h += `<div class="twn-mini">${curveSvg(d, w, false)}</div>`;
    h += `<div class="twn-bar8" role="img" aria-label="Next 2 hours, in 15-minute blocks">${d.forecast.map((p) => `<i style="background:${SEG[tierOf(p, d)]}" title="${hhmm(p.time)}: about ${Math.round(p.mg_dl)} mg/dL"></i>`).join('')}</div><div class="twn-bar8-l"><span>now</span><span>+1 h</span><span>+2 h</span></div>`;
    const hp = d.high.already_high ? 100 : d.high.chance_percent;
    h += `<div class="twn-meters"><div class="twn-meter"><span>Low sugar</span><b class="twn-chip ${tier}">${{ calm: 'Calm', watch: 'Watch', alert: 'Alert' }[tier]}</b></div>` +
      `<div class="twn-meter"><span>High sugar</span><div class="twn-track" title="${d.high.already_high ? 'Already above 180' : 'Chance of going above 180 in the next 2 hours'}"><i style="width:${hp}%;background:${d.high.already_high ? SEG.high : (d.high.tier === 'calm' ? SEG.calm : d.high.tier === 'watch' ? SEG.watch : SEG.alert)}"></i></div><b>${d.high.already_high ? 'now' : d.high.chance_percent + ' in 100'}</b></div></div>`;
    if (d.confidence !== 'ok') h += '<div class="twn-heart">Lower confidence: another dataset and sensor.</div>';
    if (d.heart_note && !rv) h += '<div class="twn-heart">Heart disease: lows matter more for this patient.</div>';
    if (d.review.state === 'reviewed') h += '<p class="twn-sub" style="margin-top:12px"><button class="twn-link" data-twn="reopen">Reopen</button></p>';
    else if (d.review.state === 'snoozed') h += '<p class="twn-sub" style="margin-top:12px"><button class="twn-link" data-twn="reopen">Reopen</button></p>';
    else if (tier !== 'calm') h += '<div class="twn-acts" id="twnActs"><button class="primary" data-twn="note">Write a note</button><button class="ghost" data-twn="reviewed">Mark reviewed</button><button class="ghost" data-twn="snoozed">Snooze 1 h</button></div>';
    h += `<div class="twn-more"><button class="twn-link" data-twn="why">${T.why ? 'Hide' : 'Why?'}</button><button class="twn-link" data-twn="curve">${T.curve ? 'Hide curve' : 'Full curve'}</button><button class="twn-link" data-twn="how">How it works</button><button class="twn-link" data-twn="tour">Quick tour</button></div>`;
    if (T.why) h += `<div class="twn-panel">${reasonsHtml(d.low.reasons)}${d.checks.length ? `<ul class="twn-checks">${d.checks.slice(0, 3).map((c) => `<li>${esc(c)}</li>`).join('')}</ul>` : ''}</div>`;
    if (T.curve) h += `<div class="twn-panel">${curveSvg(d, w, true)}${hrSvg(d, Math.max(300, Math.min(900, w)))}<p class="twn-quiet" style="margin:4px 0 0">Black: so far. Green: forecast and likely range. Dotted: what really happened.</p></div>`;
    h += '</div>';
    h += `<div class="card"><details${d.activity.length ? ' open' : ''}><summary style="cursor:pointer;font-weight:600;min-height:32px;display:flex;align-items:center">Activity${d.activity.length ? ' (' + d.activity.length + ')' : ''}</summary>` +
      (d.activity.length ? d.activity.map((a) => `<div class="twn-log"><b>${a.at ? new Date(a.at + 'Z').toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</b><span>${esc(a.text)} <span class="twn-quiet">(${esc(a.by)})</span></span></div>`).join('') : '<p class="twn-quiet">Nothing yet.</p>') + '</details></div>';
    h += `<p class="twn-quiet" style="font-size:12.5px">Risk flag for the doctor, not a diagnosis. ${esc(DATA_CREDIT[d.data_source] || '')}</p>`;
    return h;
  }

  function listHtml() {
    const rowTier = (p) => (p.review && p.review !== 'open' ? 'reviewed' : p.tier);
    const lab = { alert: 'Alert', watch: 'Watch', calm: 'Calm', reviewed: 'Reviewed' };
    const col = { alert: COL.alert, watch: COL.watch, calm: COL.calm, reviewed: COL.calm };
    let h = T.list.map((p) => { const t = rowTier(p); return `<button class="twn-item${T.sel === p.patient_id ? ' sel' : ''}" data-twn-sel="${p.patient_id}"><span class="twn-dot" style="background:${col[t]}"></span><span><b>${esc(p.label)}</b></span><span class="twn-chip ${t}">${lab[t]}</span></button>`; }).join('');
    if (T.others.length) h += '<div class="twn-sep">No sugar sensor</div>' + T.others.map((n) => `<div class="twn-item twn-off"><span class="twn-dot" style="background:${COL.none}"></span><span><b>${esc(n)}</b></span><span></span></div>`).join('');
    return h;
  }

  function draw() {
    const v = $('view-twin'); if (!v) return;
    const d = T.d;
    let ctl = '';
    if (d) {
      const total = Math.round((new Date(d.range.end) - new Date(d.range.start)) / STEP_MS), idx = Math.round((new Date(d.as_of) - new Date(d.range.start)) / STEP_MS);
      ctl = `<div class="twn-controls"><button class="primary small" data-twn="play" aria-label="Play or pause the replay">${T.playing ? 'Pause' : 'Play'}</button><input type="range" id="twnScrub" min="0" max="${total}" step="1" value="${idx}" aria-label="Replay time"><b style="font-size:14px;min-width:92px">${dayLabel(d.as_of, false)}, ${hhmm(d.as_of)}</b></div>`;
    }
    v.innerHTML = `<div class="twn-top"><h2 style="margin:0">Digital Twin</h2>${ctl}</div>` +
      `<div class="twn-layout"><div class="card twn-list" id="twnList">${listHtml()}</div><div id="twnDetail">${detailHtml()}</div></div>`;
  }

  async function loadDetail(pid, at) {
    const my = ++T.seq;
    try {
      const d = await apiFetch('GET', `/patients/${pid}/ml${at ? '?at=' + encodeURIComponent(at) : ''}`);
      if (my !== T.seq) return;
      T.d = d; T.at = d.as_of;
      const row = T.list.find((p) => p.patient_id === pid);
      if (row) { row.headline = d.headline; row.tier = d.low.tier; row.review = d.review.state; row.as_of = d.as_of; }
      draw();
      if (T.tour && T.tour.waiting) { T.tour.waiting = false; tourShow(); }
    } catch (e) { toast(e.message); }
  }

  async function load() {
    try {
      const w = await apiFetch('GET', '/twin/worklist');
      T.list = w.patients;
      const all = await apiFetch('GET', '/doctor/patients').catch(() => []);
      const ids = new Set(T.list.map((p) => p.patient_id));
      T.others = all.filter((p) => !ids.has(p.id)).map((p) => p.name);
      T.loaded = true;
      if (!T.sel && T.list.length) T.sel = T.list[0].patient_id;
      draw();
      if (T.sel) await loadDetail(T.sel, null);
      let seen = false; try { seen = localStorage.getItem('twn_tour_done') === '1'; } catch (e) { /* storage blocked */ }
      if (!seen && T.d) setTimeout(() => tourStart(), 900);
    } catch (e) { $('view-twin').innerHTML = `<h2>Digital Twin</h2><div class="card"><p class="twn-quiet">${esc(e.message)}</p></div>`; }
  }

  function setPlaying(on) {
    T.playing = on; clearInterval(T.timer);
    if (on) T.timer = setInterval(() => { if (!T.d) return; const nxt = new Date(new Date(T.d.as_of).getTime() + STEP_MS); if (nxt > new Date(T.d.range.end)) { setPlaying(false); draw(); return; } loadDetail(T.sel, toIso(nxt)); }, 1300);
  }

  async function act(action) {
    if (!T.d) return;
    try {
      const r = await apiFetch('POST', `/patients/${T.sel}/ml/actions`, { action, at: T.d.as_of });
      T.d.review = r.review; T.d.activity = r.activity;
      const row = T.list.find((p) => p.patient_id === T.sel); if (row) row.review = r.review.state;
      toast({ reviewed: 'Marked as reviewed.', snoozed: 'Snoozed for 1 hour.', reopened: 'Back to open.' }[action]);
      draw();
    } catch (e) { toast(e.message); }
  }

  function noteDialog() {
    let dlg = $('twnNote');
    if (!dlg) { dlg = document.createElement('dialog'); dlg.id = 'twnNote'; dlg.className = 'twn-dlg'; document.body.appendChild(dlg); }
    const d = T.d;
    const draft = `${hhmm(d.as_of)}, ${dayLabel(d.as_of)}. Digital Twin flagged: ${d.headline.toLowerCase()}. ${d.low.text} ${d.checks[0] || ''}. Reviewed by ${(typeof currentUser !== 'undefined' && currentUser && currentUser.name) || 'doctor'}. Plan: `;
    dlg.innerHTML = '<form method="dialog"><b style="font-size:17px">Clinical note</b><p class="twn-sub" style="margin:6px 0 10px">Draft from what the twin saw. Edit, then save.</p><textarea id="twnNoteText"></textarea><div class="twn-acts"><button class="primary" value="save">Save note</button><button class="ghost" value="cancel">Cancel</button></div></form>';
    $('twnNoteText').value = draft;
    dlg.onclose = async () => {
      if (dlg.returnValue !== 'save') return;
      const note = $('twnNoteText').value.trim(); if (!note) return;
      try { await apiFetch('POST', `/patients/${T.sel}/notes`, { note }); toast('Note saved.'); await loadDetail(T.sel, T.d.as_of); } catch (e) { toast(e.message); }
    };
    dlg.showModal();
  }

  function howDialog() {
    let dlg = $('twnHow');
    if (!dlg) { dlg = document.createElement('dialog'); dlg.id = 'twnHow'; dlg.className = 'twn-dlg'; document.body.appendChild(dlg); }
    dlg.innerHTML = '<b style="font-size:17px">How the twin works</b>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It looks at</b> the sugar so far, how fast it is changing, meals, insulin, the time of day and facts about the patient. <b>It predicts</b> sugar for the next 2 hours with a likely range, and the chance of a low or a high.</p>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It was checked</b> on 21 patients it had never seen: forecasts were off by about 5 mg/dL at 15 minutes and 17 at 60 minutes, better than assuming sugar stays the same (6.7 and 21). The low alert warned about 24 of 25 low events at 1.5 alerts per patient a day. Most alerts are precautionary: about 1 in 14 is followed by a real low within the hour.</p>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It cannot</b> diagnose or change treatment. It learned from 100 patients in Shanghai, China, and a sensor that reads wrong can cause false alarms. How it reacts to missed doses is untested.</p>' +
      `<p class="twn-quiet" style="margin-top:8px">${esc(T.d ? T.d.model_version : '')}. ${esc(T.d ? T.d.training_data : '')}.</p>` +
      '<form method="dialog" style="margin-top:12px"><button class="primary">Close</button></form>';
    dlg.showModal();
  }

  // ---- guided tour: a pointer glides to each control and says what it is for ------------------------------------
  const TOUR = [
    { sel: '[data-twn=play]', text: 'Press Play to watch the patient\'s day unfold.', click: () => { if (!T.playing) { setPlaying(true); draw(); } }, ms: 4200 },
    { sel: '#twnScrub', text: 'Or drag to any moment.', before: () => { if (T.playing) { setPlaying(false); draw(); } }, ms: 3200 },
    { sel: '.twn-mini', text: 'Black line: sugar so far. Green band: where it is expected to go.', ms: 4200 },
    { sel: '.twn-bar8', text: 'The next 2 hours, 15 minutes per block. Green calm, amber watch, red alert.', ms: 4200 },
    { sel: '[data-twn=why]', text: 'Tap Why? to see what drove it.', ms: 3200 },
    { sel: '#twnActs', text: 'Then write a note or mark it reviewed.', optional: true, ms: 3400 },
  ];

  function tourEnd(done) {
    if (T.tour) { clearTimeout(T.tour.timer); T.tour = null; }
    ['twnPtr', 'twnTip'].forEach((id) => { const el = $(id); if (el) el.remove(); });
    if (T.playing) { setPlaying(false); draw(); }
    if (done) { try { localStorage.setItem('twn_tour_done', '1'); } catch (e) { /* storage blocked */ } }
  }

  function tourStart() {
    if (!T.d || !$('view-twin') || $('view-twin').style.display === 'none') return;
    tourEnd(false);
    T.tour = { i: -1, timer: null, waiting: false };
    const ptr = document.createElement('div'); ptr.id = 'twnPtr'; ptr.setAttribute('aria-hidden', 'true');
    ptr.innerHTML = '<svg viewBox="0 0 24 24" width="30" height="30"><path d="M4 2l15 9-6.5 1.7L9 19z" fill="#16241F" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg><span class="twn-ripple"></span>';
    const tip = document.createElement('div'); tip.id = 'twnTip'; tip.setAttribute('role', 'dialog'); tip.setAttribute('aria-label', 'Quick tour');
    document.body.appendChild(ptr); document.body.appendChild(tip);
    tourNext(1);
  }

  function tourNext(dir) {
    const t = T.tour; if (!t) return;
    let i = t.i + dir;
    while (i >= 0 && i < TOUR.length && TOUR[i].optional && !document.querySelector(TOUR[i].sel)) i += dir;
    if (i >= TOUR.length) { tourEnd(true); return; }
    if (i < 0) i = 0;
    t.i = i; tourShow();
  }

  function tourShow() {
    const t = T.tour; if (!t) return;
    const s = TOUR[t.i];
    if (s.before) s.before();
    const el = document.querySelector(s.sel);
    if (!el) { t.waiting = true; return; }
    clearTimeout(t.timer);
    el.scrollIntoView({ block: 'center', behavior: calmMotion() ? 'auto' : 'smooth' });
    setTimeout(() => {
      if (!T.tour) return;
      const el2 = document.querySelector(s.sel); if (!el2) return;
      const r = el2.getBoundingClientRect(), big = r.width > 260;
      const x = big ? r.left + Math.min(90, r.width / 4) : r.left + r.width / 2, y = big ? r.top + Math.min(50, r.height / 2) : r.top + r.height / 2;
      const ptr = $('twnPtr'), tip = $('twnTip'); if (!ptr || !tip) return;
      ptr.style.transition = calmMotion() ? 'none' : 'transform .9s cubic-bezier(.45,.05,.2,1)';
      ptr.style.transform = `translate(${x}px, ${y}px)`;
      const below = y < window.innerHeight - 190;
      tip.className = below ? 'below' : 'above';
      tip.innerHTML = `<div class="twn-tip-text">${esc(s.text)}</div><div class="twn-tip-row"><span class="twn-dots">${TOUR.map((_, k) => `<i class="${k === t.i ? 'on' : ''}"></i>`).join('')}</span><span><button class="twn-link" data-tour="back"${t.i === 0 ? ' style="visibility:hidden"' : ''}>Back</button><button class="primary small" data-tour="next">${t.i === TOUR.length - 1 ? 'Done' : 'Next'}</button><button class="twn-link" data-tour="skip">Skip</button></span></div>`;
      const tw = Math.min(320, window.innerWidth - 24);
      tip.style.width = tw + 'px';
      tip.style.left = Math.max(12, Math.min(window.innerWidth - tw - 12, x - 40)) + 'px';
      tip.style.top = below ? (y + 30) + 'px' : (y - 30 - 118) + 'px';
      const arrive = calmMotion() ? 0 : 950;
      setTimeout(() => { if (!T.tour) return; ptr.classList.remove('click'); void ptr.offsetWidth; ptr.classList.add('click'); if (s.click) s.click(); }, arrive);
      t.timer = setTimeout(() => tourNext(1), (s.ms || 3600) + arrive);
    }, calmMotion() ? 0 : 350);
  }

  document.addEventListener('click', (e) => {
    const tb = e.target.closest('[data-tour]');
    if (tb) { const a = tb.dataset.tour; if (a === 'next') tourNext(1); else if (a === 'back') tourNext(-1); else tourEnd(true); return; }
    const view = $('view-twin'); if (!view || !view.contains(e.target)) return;
    const s = e.target.closest('[data-twn-sel]');
    if (s) { T.sel = +s.dataset.twnSel; T.d = null; T.why = T.curve = false; setPlaying(false); draw(); loadDetail(T.sel, (T.list.find((p) => p.patient_id === T.sel) || {}).as_of || null); return; }
    const b = e.target.closest('[data-twn]'); if (!b) return;
    const a = b.dataset.twn;
    if (a === 'why') { T.why = !T.why; draw(); } else if (a === 'curve') { T.curve = !T.curve; draw(); } else if (a === 'how') howDialog();
    else if (a === 'note') noteDialog(); else if (a === 'play') { setPlaying(!T.playing); draw(); } else if (a === 'tour') tourStart();
    else if (a === 'reopen') act('reopened'); else if (a === 'reviewed' || a === 'snoozed') act(a);
  });
  document.addEventListener('input', (e) => {
    if (e.target.id !== 'twnScrub' || !T.d) return;
    setPlaying(false);
    const at = toIso(new Date(new Date(T.d.range.start).getTime() + (+e.target.value) * STEP_MS));
    clearTimeout(T.wait); T.wait = setTimeout(() => loadDetail(T.sel, at), 150);
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && T.tour) tourEnd(true); });

  window.twinRender = function () { if (!T.enabled) return; if (!T.loaded) { $('view-twin').innerHTML = '<h2>Digital Twin</h2><div class="card"><p class="twn-quiet">Loading...</p></div>'; load(); } else draw(); };
  init();
})();
