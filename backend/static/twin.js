// SmartPoli: Digital Twin tab for the doctor dashboard (demo copy).
//
// The nav item stays hidden unless GET /twin/info answers (it is a 404 on the live app, where the twin is off).
// The twin only informs: every output is a risk flag for the doctor, never a diagnosis or a treatment change.
// Doctor actions go to the server (AuditLog / clinical notes), so they show on the patient timeline and report.

(function () {
  const T = { enabled: false, loaded: false, list: [], others: [], sel: null, d: null, at: null, why: false, curve: false, playing: false, timer: null, seq: 0, wait: null };
  const COL = { alert: '#D3402A', watch: '#B4791F', calm: '#12876F', none: '#B4BDB8' };
  const STEP_MS = 15 * 60000;
  const DATA_CREDIT = { ShanghaiT2DM: 'Sugar readings, meals and insulin: ShanghaiT2DM dataset (Zhao et al., Scientific Data 2023, CC BY 4.0), anonymised and replayed.', CGMacros: 'Sugar readings, meals and heart rate: CGMacros dataset (PhysioNet), anonymised and replayed.' };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const hhmm = (iso) => iso.slice(11, 16);
  const pad = (n) => String(n).padStart(2, '0');
  const toIso = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00`;
  const dayLabel = (iso, year = true) => new Date(iso).toLocaleDateString([], year ? { day: 'numeric', month: 'long', year: 'numeric' } : { day: 'numeric', month: 'long' });

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
    return list.map((r) => `<div class="twn-reason"><span>${esc(r.text)}</span><span class="twn-bar" title="${r.up ? 'raises' : 'lowers'} the risk"><i style="width:${Math.max(8, Math.min(100, r.weight))}%;background:${r.up ? '#D3402A' : '#12876F'}"></i></span></div>`).join('') +
      '<p class="twn-quiet" style="margin:6px 0 0">Red raises the risk, green lowers it. A longer bar is a bigger push.</p>';
  }

  function hrSvg(d, W, t0, now, X) {
    if (!d.heart_rate || d.heart_rate.length < 3) return '';
    const v = d.heart_rate.map((p) => p[1]), lo = Math.min(...v, 50), hi = Math.max(...v, 100), H = 70;
    const Y = (b) => 8 + (hi - b) / (hi - lo) * (H - 16);
    const pts = d.heart_rate.map((p) => `${X(new Date(p[0]))},${Y(p[1])}`).join(' ');
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Heart rate over the same hours" style="margin-top:6px"><text x="4" y="12" font-size="12" fill="#5C6E68">Heart rate</text><polyline fill="none" stroke="#D3402A" stroke-width="2" points="${pts}"/><text x="${W - 10}" y="12" text-anchor="end" font-size="12" fill="#5C6E68">${Math.round(lo)} to ${Math.round(hi)} beats per minute</text></svg>`;
  }

  function chartSvg(d, width) {
    const W = Math.max(330, Math.min(900, width)), H = 250, L = 36, R = 10, top = 10, ch = 190;
    const hist = d.history.map((p) => [new Date(p[0]).getTime(), p[1]]);
    const t0 = hist.length ? hist[0][0] : new Date(d.as_of).getTime() - 6 * 3600000, now = new Date(d.as_of).getTime(), t1 = now + 125 * 60000;
    const X = (t) => L + (t - t0) / (t1 - t0) * (W - L - R);
    const Y = (v) => top + ch - (Math.max(40, Math.min(240, v)) - 40) / 200 * ch;
    let s = `<rect x="${L}" y="${Y(70)}" width="${W - L - R}" height="${Y(40) - Y(70)}" fill="#FBE2DD" opacity=".6"/><rect x="${L}" y="${Y(240)}" width="${W - L - R}" height="${Y(180) - Y(240)}" fill="#FBEBD3" opacity=".6"/>`;
    [70, 120, 180].forEach((g) => { s += `<line x1="${L}" x2="${W - R}" y1="${Y(g)}" y2="${Y(g)}" stroke="#8B9A94" stroke-dasharray="4 4"/><text x="${L - 6}" y="${Y(g) + 4}" text-anchor="end" font-size="12" fill="#5C6E68">${g}</text>`; });
    for (let t = Math.ceil(t0 / 3600000) * 3600000; t <= t1; t += (W < 600 ? 2 : 1) * 3600000) s += `<text x="${X(t)}" y="${top + ch + 18}" text-anchor="middle" font-size="12" fill="#5C6E68">${pad(new Date(t).getHours())}:00</text>`;
    const f = d.forecast, up = f.map((p) => `${X(new Date(p.time))},${Y(p.high)}`), dn = f.slice().reverse().map((p) => `${X(new Date(p.time))},${Y(p.low)}`);
    s += `<polygon points="${X(now)},${Y(d.current)} ${up.join(' ')} ${dn.join(' ')}" fill="#DFF1EA"/><polyline fill="none" stroke="#12876F" stroke-width="2.5" stroke-dasharray="6 5" points="${X(now)},${Y(d.current)} ${f.map((p) => `${X(new Date(p.time))},${Y(p.mg_dl)}`).join(' ')}"/>`;
    if (d.what_really_happened.length) s += `<polyline fill="none" stroke="#16241F" stroke-width="2" stroke-dasharray="2 4" opacity=".6" points="${X(now)},${Y(d.current)} ${d.what_really_happened.map((p) => `${X(new Date(p[0]))},${Y(p[1])}`).join(' ')}"/>`;
    s += `<polyline fill="none" stroke="#16241F" stroke-width="3" stroke-linejoin="round" points="${hist.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}"/><line x1="${X(now)}" x2="${X(now)}" y1="${top}" y2="${top + ch}" stroke="#12876F"/><circle cx="${X(now)}" cy="${Y(d.current)}" r="5" fill="#16241F"/>`;
    const lane = top + ch + 40;
    d.events.forEach((e) => { const x = X(new Date(e.time)); if (x < L || x > W - R) return; const o = new Date(e.time).getTime() <= now ? 1 : .35; s += e.kind === 'insulin' ? `<path d="M${x} ${lane - 9} l8 14 h-16z" fill="#2E6FE0" opacity="${o}"><title>${esc(e.text)}</title></path>` : `<circle cx="${x}" cy="${lane + 14}" r="6" fill="#B4791F" opacity="${o}"><title>${esc(e.text)}</title></circle>`; });
    const hr = hrSvg(d, W, t0, now, X);
    return `<svg viewBox="0 0 ${W} ${H + 20}" role="img" aria-label="Sugar so far, forecast and likely range">${s}</svg>` +
      hr + '<p class="twn-quiet" style="margin:6px 0 0">Black: sugar so far. Green dashed: forecast with its likely range (real values land inside about 8 times in 10). Dotted: what really happened next. Blue triangle: insulin. Amber dot: meal.</p>';
  }

  function detailHtml() {
    const d = T.d;
    if (!d) return '<div class="card"><p class="twn-quiet">Loading...</p></div>';
    const rv = d.review.state !== 'open', tier = d.low.tier, col = rv ? 'var(--ink)' : COL[tier];
    let h = `<div class="card"><div class="twn-head"><div><b style="font-size:18px">${esc(d.label)}</b><div class="twn-quiet">${esc(d.summary || '')}</div></div>`;
    if (d.review.state === 'reviewed') h += `<span class="twn-chip reviewed">Reviewed by ${esc(d.review.info.by)}</span>`;
    if (d.review.state === 'snoozed') h += '<span class="twn-chip">Snoozed for 1 hour</span>';
    h += `</div><div class="twn-big" style="color:${col}">${esc(d.headline)}</div><p class="twn-sub">${esc(d.low.text)}</p>`;
    h += `<div class="twn-strip">${d.forecast.map((p) => `<div class="twn-cell ${tierOf(p, d)}"><b>${Math.round(p.mg_dl)}</b><span>${hhmm(p.time)}</span></div>`).join('')}</div><div class="twn-quiet">Expected sugar (mg/dL) for the next 2 hours</div>`;
    const hw = d.high.already_high ? 'Already high' : { calm: 'Low chance', watch: 'Moderate chance', alert: 'High chance' }[d.high.tier];
    h += `<p class="twn-sub" style="margin-top:10px"><b style="color:${d.high.already_high ? '#1b4aa0' : COL[d.high.tier]}">High sugar: ${hw}.</b> ${esc(d.high.text)}</p>`;
    if (d.confidence !== 'ok') h += `<div class="twn-heart">Lower confidence: ${esc(d.confidence_reason || 'this patient is outside what the model learned from.')}</div>`;
    if (d.heart_note && !rv) h += '<div class="twn-heart">Heart note: this patient has heart disease. Low sugar is linked to heart strain, so this alert deserves a quick look.</div>';
    if (d.review.state === 'reviewed') h += '<p class="twn-sub" style="margin-top:14px">The twin re-checks every 15 minutes and flags this patient again if the forecast changes. <button class="twn-link" data-twn="reopen">Reopen</button></p>';
    else if (d.review.state === 'snoozed') h += '<p class="twn-sub" style="margin-top:14px"><button class="twn-link" data-twn="reopen">Reopen</button></p>';
    else if (tier === 'calm') h += '<p class="twn-sub" style="margin-top:14px">Nothing to do right now. The twin re-checks every 15 minutes.</p>';
    else h += '<div class="twn-acts"><button class="primary" data-twn="note">Write a note</button><button class="ghost" data-twn="reviewed">Mark as reviewed</button><button class="ghost" data-twn="snoozed">Remind me in 1 hour</button></div>';
    h += `<div class="twn-more"><button class="twn-link" data-twn="why">${T.why ? 'Hide reasons' : 'Why?'}</button><button class="twn-link" data-twn="curve">${T.curve ? 'Hide the curve' : 'Show the curve'}</button><button class="twn-link" data-twn="how">How does this work?</button></div>`;
    if (T.why) h += `<div class="twn-panel"><div style="font-weight:600;margin-bottom:2px">Why the low-sugar forecast looks like this</div>${reasonsHtml(d.low.reasons)}<div style="font-weight:600;margin:12px 0 2px">Worth checking</div><ul class="twn-checks">${d.checks.map((c) => `<li>${esc(c)}</li>`).join('')}</ul><div style="font-weight:600;margin:12px 0 2px">Behind the high-sugar chance</div>${reasonsHtml(d.high.reasons)}</div>`;
    if (T.curve) h += `<div class="twn-panel" id="twnCurve">${chartSvg(d, ($('twnDetail') ? $('twnDetail').clientWidth : 700) - 40)}</div>`;
    h += '</div>';
    h += `<div class="card"><details${d.activity.length ? ' open' : ''}><summary style="cursor:pointer;font-weight:600;min-height:32px;display:flex;align-items:center">Activity${d.activity.length ? ' (' + d.activity.length + ')' : ''}</summary>` +
      (d.activity.length ? d.activity.map((a) => `<div class="twn-log"><b>${a.at ? new Date(a.at + 'Z').toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</b><span>${esc(a.text)} <span class="twn-quiet">(${esc(a.by)})</span></span></div>`).join('') + '<p class="twn-quiet" style="margin:8px 0 0">Recorded on the patient timeline and in the report.</p>' : '<p class="twn-quiet">Nothing yet. Notes and reviews are recorded here and on the patient timeline.</p>') + '</details></div>';
    h += `<p class="twn-quiet">${esc(d.model_version)}. Trained on ${esc(d.training_data)}. ${esc(d.disclaimer)}</p>`;
    h += `<p class="twn-quiet">${esc(DATA_CREDIT[d.data_source] || '')}</p>`;
    return h;
  }

  function listHtml() {
    const rowTier = (p) => (p.review && p.review !== 'open' ? 'reviewed' : p.tier);
    const lab = { alert: 'Alert', watch: 'Watch', calm: 'Calm', reviewed: 'Reviewed' };
    const col = { alert: COL.alert, watch: COL.watch, calm: COL.calm, reviewed: COL.calm };
    let h = T.list.map((p) => { const t = rowTier(p); return `<button class="twn-item${T.sel === p.patient_id ? ' sel' : ''}" data-twn-sel="${p.patient_id}"><span class="twn-dot" style="background:${col[t]}"></span><span><b>${esc(p.label)}</b><small style="color:${col[t]}">${esc(t === 'reviewed' ? 'Reviewed' : p.headline)}</small></span><span class="twn-chip ${t}">${lab[t]}</span></button>`; }).join('');
    if (T.others.length) h += '<div class="twn-sep">Without a sugar sensor</div>' + T.others.map((n) => `<div class="twn-item twn-off"><span class="twn-dot" style="background:${COL.none}"></span><span><b>${esc(n)}</b></span><span></span></div>`).join('');
    return h;
  }

  function draw() {
    const v = $('view-twin'); if (!v) return;
    const d = T.d, n = { alert: 0, watch: 0, calm: 0, reviewed: 0 };
    T.list.forEach((p) => { n[p.review && p.review !== 'open' ? 'reviewed' : p.tier]++; });
    let ctl = '';
    if (d) {
      const total = Math.round((new Date(d.range.end) - new Date(d.range.start)) / STEP_MS), idx = Math.round((new Date(d.as_of) - new Date(d.range.start)) / STEP_MS);
      ctl = `<div class="twn-controls"><span class="twn-quiet">Replay</span><button class="ghost small" data-twn="play">${T.playing ? 'Pause' : 'Play'}</button><input type="range" id="twnScrub" min="0" max="${total}" step="1" value="${idx}" aria-label="Replay time"><b style="font-size:14px;min-width:92px">${dayLabel(d.as_of, false)}, ${hhmm(d.as_of)}</b></div>`;
    }
    v.innerHTML = `<div class="twn-top"><h2 style="margin:0">Digital Twin</h2>${ctl}</div>` +
      `<div class="twn-row" style="margin-bottom:12px">${n.alert ? `<span class="twn-chip alert">${n.alert} alert</span>` : ''}${n.watch ? `<span class="twn-chip watch">${n.watch} watch</span>` : ''}${n.calm ? `<span class="twn-chip calm">${n.calm} calm</span>` : ''}${n.reviewed ? `<span class="twn-chip reviewed">${n.reviewed} reviewed</span>` : ''}<span class="twn-quiet">Replay of anonymised dataset patients</span></div>` +
      `<div class="twn-layout"><div class="card twn-list">${listHtml()}</div><div id="twnDetail">${detailHtml()}</div></div>`;
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
      toast({ reviewed: 'Marked as reviewed. The twin re-checks in 15 minutes.', snoozed: 'Snoozed for 1 hour.', reopened: 'Back to open.' }[action]);
      draw();
    } catch (e) { toast(e.message); }
  }

  function noteDialog() {
    let dlg = $('twnNote');
    if (!dlg) { dlg = document.createElement('dialog'); dlg.id = 'twnNote'; dlg.className = 'twn-dlg'; document.body.appendChild(dlg); }
    const d = T.d;
    const draft = `${hhmm(d.as_of)}, ${dayLabel(d.as_of)}. Digital Twin flagged: ${d.headline.toLowerCase()}. ${d.low.text} ${d.checks[0] || ''}. Reviewed by ${(typeof currentUser !== 'undefined' && currentUser && currentUser.name) || 'doctor'}. Plan: `;
    dlg.innerHTML = '<form method="dialog"><b style="font-size:17px">Clinical note</b><p class="twn-sub" style="margin:6px 0 10px">Filled in from what the twin saw. Edit it, then save.</p><textarea id="twnNoteText"></textarea><div class="twn-acts"><button class="primary" value="save">Save note</button><button class="ghost" value="cancel">Cancel</button></div></form>';
    $('twnNoteText').value = draft;
    dlg.onclose = async () => {
      if (dlg.returnValue !== 'save') return;
      const note = $('twnNoteText').value.trim(); if (!note) return;
      try { await apiFetch('POST', `/patients/${T.sel}/notes`, { note }); toast('Note saved to the clinical notes.'); await loadDetail(T.sel, T.d.as_of); } catch (e) { toast(e.message); }
    };
    dlg.showModal();
  }

  function howDialog() {
    let dlg = $('twnHow');
    if (!dlg) { dlg = document.createElement('dialog'); dlg.id = 'twnHow'; dlg.className = 'twn-dlg'; document.body.appendChild(dlg); }
    dlg.innerHTML = '<b style="font-size:17px">How the twin works</b>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It looks at</b> the sugar so far, how fast it is changing, meals, insulin, the time of day and facts about the patient. <b>It predicts</b> sugar for the next 2 hours with a likely range, and how likely a low or a high is.</p>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It was checked</b> on 21 patients it had never seen: forecasts were off by about 5 mg/dL at 15 minutes and 17 at 60 minutes, better than assuming sugar stays the same (6.7 and 21). The low alert warned about 24 of 25 low events at 1.5 alerts per patient a day. Most alerts are precautionary: about 1 in 14 is followed by a real low within the hour.</p>' +
      '<p class="twn-sub" style="margin-top:8px"><b>It cannot</b> diagnose or change treatment. It learned from 100 patients in Shanghai, China, and a sensor that reads wrong can cause false alarms. How it reacts to missed doses is untested.</p>' +
      '<form method="dialog" style="margin-top:12px"><button class="primary">Close</button></form>';
    dlg.showModal();
  }

  document.addEventListener('click', (e) => {
    const view = $('view-twin'); if (!view || !view.contains(e.target)) return;
    const s = e.target.closest('[data-twn-sel]');
    if (s) { T.sel = +s.dataset.twnSel; T.d = null; T.why = T.curve = false; setPlaying(false); draw(); loadDetail(T.sel, (T.list.find((p) => p.patient_id === T.sel) || {}).as_of || null); return; }
    const b = e.target.closest('[data-twn]'); if (!b) return;
    const a = b.dataset.twn;
    if (a === 'why') { T.why = !T.why; draw(); } else if (a === 'curve') { T.curve = !T.curve; draw(); } else if (a === 'how') howDialog();
    else if (a === 'note') noteDialog(); else if (a === 'play') { setPlaying(!T.playing); draw(); }
    else if (a === 'reopen') act('reopened'); else if (a === 'reviewed' || a === 'snoozed') act(a);
  });
  document.addEventListener('input', (e) => {
    if (e.target.id !== 'twnScrub' || !T.d) return;
    setPlaying(false);
    const at = toIso(new Date(new Date(T.d.range.start).getTime() + (+e.target.value) * STEP_MS));
    clearTimeout(T.wait); T.wait = setTimeout(() => loadDetail(T.sel, at), 150);
  });

  window.twinRender = function () { if (!T.enabled) return; if (!T.loaded) { $('view-twin').innerHTML = '<h2>Digital Twin</h2><div class="card"><p class="twn-quiet">Loading...</p></div>'; load(); } else draw(); };
  init();
})();
