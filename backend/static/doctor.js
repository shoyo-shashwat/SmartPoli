// SmartPoli — doctor dashboard (Parts 1 and 7 of the brief).
//
// Same pattern as caregiver.js: every request carries the doctor's real
// bearer token, and doctor_router.py is what actually enforces that this
// account only ever reaches a patient it holds an ACTIVE DoctorLink for.
// A correction here is never silent — see submitCorrection() below, which
// always shows the server's before/after + who + why back to the doctor
// immediately after saving it.
//
// The per-patient detail is split across tabs (Overview / Prescriptions /
// Adherence / Triage history / Notes) reached through the hamburger drawer,
// instead of one long page — the full report is fetched once per patient
// selection (state.currentReport) and each tab just renders its own slice
// of it, so switching tabs never re-fetches.

const currentUser = requireRole('doctor');
const api = apiFetch;

const state = { patients: [], patientId: null, templates: [], activeTab: 'patients', currentReport: null };

const CORRECTABLE_FIELDS = ['name', 'dose_amount', 'dose_unit', 'schedule_code', 'food', 'duration_days'];

function badgeClass(status) { return (status || '').toLowerCase(); }

const PRIORITY_LABELS = { emergency: 'Emergency', high: 'High', medium: 'Medium', routine: 'Routine' };

function renderPriorityBadge(priority) {
  if (!priority) return '';
  return `<span class="badge ${priority.level}">${PRIORITY_LABELS[priority.level] || priority.level}</span>`;
}

function renderPriorityReasons(priority) {
  if (!priority || !priority.reasons.length) return '';
  return `<ul class="priority-reasons">${priority.reasons.map(r => `<li>${r}</li>`).join('')}</ul>`;
}

function priorityFor(patientId) {
  const p = state.patients.find(p => p.id === patientId);
  return p ? p.priority : null;
}

function renderSessionChip() {
  const chip = document.getElementById('sessionChip');
  if (!chip || !currentUser) return;
  chip.innerHTML = `
    <span class="who">${currentUser.name}</span>
    <span class="role-tag">Doctor</span>
    <button class="ghost small" id="logoutBtn">Log out</button>
  `;
  document.getElementById('logoutBtn').addEventListener('click', logout);
}

function applyA11yMode() {
  const on = localStorage.getItem('smartpoli_a11y') === '1';
  document.documentElement.dataset.a11y = on ? 'large' : '';
  document.getElementById('a11yToggleBtn')?.setAttribute('aria-pressed', String(on));
}
document.getElementById('a11yToggleBtn').addEventListener('click', () => {
  const on = localStorage.getItem('smartpoli_a11y') === '1';
  localStorage.setItem('smartpoli_a11y', on ? '0' : '1');
  applyA11yMode();
});

/** Injects each nav button's icon from icons.js, same helper as app.js —
 * one injection point instead of hand-writing SVG markup per page. */
function injectNavIcons() {
  document.querySelectorAll('nav.pill-nav button[data-icon]').forEach((btn) => {
    if (btn.querySelector('.nav-icon')) return;
    const icon = ICONS[btn.dataset.icon];
    if (!icon) return;
    btn.insertAdjacentHTML('afterbegin', `<span class="nav-icon">${icon}</span>`);
  });
}

const hamburgerBtn = document.getElementById('hamburgerBtn');
const navDrawer = document.getElementById('navDrawer');
const navOverlay = document.getElementById('navOverlay');
const closeDrawerBtn = document.getElementById('closeDrawerBtn');
function openDrawer() {
  navDrawer.classList.add('open');
  navOverlay.classList.add('open');
  hamburgerBtn.setAttribute('aria-expanded', 'true');
}
function closeDrawer() {
  navDrawer.classList.remove('open');
  navOverlay.classList.remove('open');
  hamburgerBtn.setAttribute('aria-expanded', 'false');
}
hamburgerBtn.addEventListener('click', openDrawer);
closeDrawerBtn.addEventListener('click', closeDrawer);
navOverlay.addEventListener('click', closeDrawer);

/** Switches the visible section + nav highlight only — never renders stale
 * data itself. Used both for a plain tab click (caller re-renders from the
 * already-fetched report right after) and from selectPatient() (which lands
 * on Overview then lets loadPatientDetail()'s own fetch/render take over, so
 * the previous patient's data never flashes first). */
function setActiveNavAndSection(tabName) {
  document.querySelectorAll('nav.pill-nav button[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
  state.activeTab = tabName;
  document.querySelectorAll('main.content > section').forEach(s => s.style.display = 'none');
  document.getElementById(`view-${tabName}`).style.display = ''; // let doctor.css pick block/grid per breakpoint
  renderCurrentPatientBar();
}

/** The picker (search + link-a-patient) lives only on the Patients tab now;
 * every other patient-specific tab shows this compact bar instead, so the
 * doctor always knows whose record they're viewing and can jump back. */
function renderCurrentPatientBar() {
  const bar = document.getElementById('currentPatientBar');
  if (!bar) return;
  if (state.activeTab === 'patients' || state.activeTab === 'settings' || state.activeTab === 'twin' || !state.patientId) {
    bar.style.display = 'none';
    return;
  }
  const p = state.patients.find(p => p.id === state.patientId);
  bar.style.display = 'flex';
  bar.innerHTML = `
    <span><strong>${p ? p.name : 'Patient'}</strong>${p && p.age ? ' · ' + p.age + ' yrs' : ''}${p && p.sex ? ', ' + p.sex : ''}</span>
    <button type="button" class="ghost small" id="switchPatientBtn">Switch patient</button>
  `;
  document.getElementById('switchPatientBtn').addEventListener('click', () => setActiveNavAndSection('patients'));
}

document.querySelectorAll('nav.pill-nav button[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => {
    setActiveNavAndSection(btn.dataset.tab);
    renderActiveDoctorTab();
    closeDrawer();
  });
});

document.getElementById('linkPatientBtn').addEventListener('click', () => {
  document.getElementById('linkForm').style.display = 'block';
});
document.getElementById('cancelLinkBtn').addEventListener('click', () => {
  document.getElementById('linkForm').style.display = 'none';
});
document.getElementById('redeemLinkBtn').addEventListener('click', async () => {
  const code = document.getElementById('linkCodeInput').value.trim();
  const status = document.getElementById('linkStatus');
  if (!code) return;
  try {
    const result = await api('POST', '/doctor/link/redeem', { code });
    status.style.color = 'var(--teal-dark)';
    status.textContent = `Linked to ${result.patient.name}.`;
    document.getElementById('linkCodeInput').value = '';
    await loadPatients();
    selectPatient(result.patient.id, { switchToOverview: true });
  } catch (e) {
    status.style.color = 'var(--alarm)';
    status.textContent = e.message;
  }
});

async function loadPatients() {
  state.patients = await api('GET', '/doctor/patients');
  if (!state.patients.length) {
    document.getElementById('patientPicker').innerHTML = '<div class="empty">No linked patients yet — use "+ Link a patient" above.</div>';
    return;
  }
  renderPatientPicker(state.patients);
  if (!state.patientId) selectPatient(state.patients[0].id);
}

function renderPatientPicker(patients) {
  const picker = document.getElementById('patientPicker');
  picker.innerHTML = patients.map(p => `
    <button type="button" class="patient-picker-card ${p.id === state.patientId ? 'active' : ''}" data-pid="${p.id}">
      <div class="name">${p.name}</div>
      <div class="meta">${p.age ? p.age + ' yrs' : ''}${p.sex ? ', ' + p.sex : ''}</div>
      <div class="priority-row">
        ${renderPriorityBadge(p.priority)}
        ${renderPriorityReasons(p.priority)}
      </div>
    </button>
  `).join('') || '<div class="empty">No patients match that search.</div>';
  picker.querySelectorAll('[data-pid]').forEach(btn => {
    // On a laptop the Patients tab already previews the selected patient
    // below the picker, so a click just swaps that preview in place; on
    // smaller screens there's no room for it, so jump to Overview as before.
    btn.addEventListener('click', () => selectPatient(Number(btn.dataset.pid), { switchToOverview: !isLaptopWidth() }));
  });
}

document.getElementById('patientSearchInput').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  const filtered = q ? state.patients.filter(p => p.name.toLowerCase().includes(q)) : state.patients;
  renderPatientPicker(filtered);
});

function selectPatient(id, { switchToOverview = false } = {}) {
  state.patientId = id;
  document.querySelectorAll('#patientPicker [data-pid]').forEach(btn => {
    btn.classList.toggle('active', Number(btn.dataset.pid) === id);
  });
  if (switchToOverview) setActiveNavAndSection('overview');
  else renderCurrentPatientBar();
  loadPatientDetail();
}

function isLaptopWidth() { return window.matchMedia('(min-width: 1024px)').matches; }

const PATIENT_DRIVEN_TABS = ['overview', 'prescriptions', 'adherence', 'triage', 'notes'];

async function loadPatientDetail() {
  if (state.activeTab === 'settings') return; // settings is patient-independent, nothing to fetch
  if (PATIENT_DRIVEN_TABS.includes(state.activeTab)) {
    const view = document.getElementById(`view-${state.activeTab}`);
    if (view) view.innerHTML = '<div class="empty">Loading...</div>';
  }
  state.currentReport = await api('GET', `/doctor/patients/${state.patientId}`);
  renderActiveDoctorTab();
}

function renderActiveDoctorTab() {
  if (state.activeTab === 'settings') return; // static content, already in the page
  if (state.activeTab === 'twin') { if (window.twinRender) window.twinRender(); return; } // Digital Twin (demo copy): needs no selected patient
  if (!state.patientId || !state.currentReport) return;
  if (state.activeTab === 'patients') renderPatientPreview();
  else if (state.activeTab === 'overview') renderOverviewTab();
  else if (state.activeTab === 'prescriptions') renderPrescriptionsTab();
  else if (state.activeTab === 'adherence') renderAdherenceTab();
  else if (state.activeTab === 'triage') renderTriageTab();
  else if (state.activeTab === 'notes') renderNotesTab();
}

function renderPatientPreview() {
  const preview = document.getElementById('patientPreview');
  renderOverviewTab(preview);
  preview.querySelector('h2').insertAdjacentHTML('beforeend',
    '<button type="button" class="ghost small" id="openFullRecordBtn">Open full record</button>');
  document.getElementById('openFullRecordBtn').addEventListener('click', () => {
    setActiveNavAndSection('overview');
    renderActiveDoctorTab();
  });
}

function renderOverviewTab(view = document.getElementById('view-overview')) {
  const r = state.currentReport;
  const alertsHtml = r.alerts.map(a => `<div class="alert-item">${a}</div>`).join('') || '<div class="empty">No alerts.</div>';

  const b = r.brief;
  const latestCheck = b.latest_symptom_check;
  const briefHtml = `
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
      ${renderPriorityBadge(priorityFor(state.patientId))}
    </div>
    ${renderPriorityReasons(priorityFor(state.patientId))}
    <div class="change-row"><span class="label">Current medicines</span><span class="value">${b.current_medicine_count}${b.current_medicine_names.length ? ' — ' + b.current_medicine_names.join(', ') : ''}</span></div>
    <div class="change-row"><span class="label">Allergies</span><span class="value">${b.allergies}</span></div>
    <div class="change-row"><span class="label">Adherence</span><span class="value">${b.adherence_percent === null ? '—' : b.adherence_percent + '%'}</span></div>
    <div class="change-row"><span class="label">Pending verification</span><span class="value">${b.pending_verification_count}</span></div>
    <div class="change-row">
      <span class="label">Latest symptom check</span>
      <span class="value">${latestCheck
        ? `<span class="badge ${badgeClass(latestCheck.severity)}">${latestCheck.severity}</span> ${new Date(latestCheck.created_at).toLocaleDateString()}`
        : 'None recorded'}</span>
    </div>
  `;

  const since = r.since_last_visit;
  const sinceHtml = since ? `
    <div class="card">
      <div class="card-head">${iconBadge('teal', 'history')}<h3>Since last visit — ${since.days_since} day${since.days_since === 1 ? '' : 's'} ago</h3></div>
      ${since.changes.map(c => `<div class="change-row"><span class="label">${c.label}</span><span class="value">${c.value}</span></div>`).join('') || '<div class="empty">No new activity since your last review.</div>'}
    </div>
  ` : `
    <div class="card">
      <div class="card-head">${iconBadge('teal', 'history')}<h3>Since last visit</h3></div>
      <div class="empty">First time reviewing this patient — nothing to compare yet. This will fill in after your first note or correction.</div>
    </div>
  `;

  view.innerHTML = `
    <h2 style="margin-top:26px;">${r.patient.name}</h2>
    <div style="color:var(--ink-soft);font-size:13px;margin-bottom:14px;">
      ${r.patient.age ? r.patient.age + ' yrs' : ''}${r.patient.sex ? ', ' + r.patient.sex : ''}
    </div>

    <div class="card">
      <div class="card-head">${iconBadge('teal', 'clipboard')}<h3>Patient brief — before consultation</h3></div>
      ${briefHtml}
    </div>

    ${sinceHtml}

    <div class="card">
      <div class="card-head">${iconBadge('amber', 'alertCircle')}<h3>Alerts</h3></div>
      ${alertsHtml}
    </div>
  `;
}

function renderPrescriptionsTab() {
  const r = state.currentReport;
  const view = document.getElementById('view-prescriptions');

  const medsHtml = r.prescriptions.flatMap(p => p.medicines).map(m => `
    <div class="medicine-line ${m.status === 'needs_confirmation' ? 'needs_confirmation' : ''}" data-med-id="${m.id}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
        <div>
          <strong>${m.name || '(name not read)'}</strong> ${m.dose_amount ? m.dose_amount + (m.dose_unit || '') : ''}
          ${m.schedule_code ? ' — ' + m.schedule_code : ''}
          <div class="raw">"${m.raw_text}"</div>
        </div>
        <span class="badge ${badgeClass(m.status)}">${m.status.replace('_', ' ')}</span>
      </div>
      <div style="margin-top:6px;">
        ${Object.entries(m.field_confidence).filter(([k]) => k !== 'source').map(([k, v]) => `
          <div class="confidence-row">
            <span class="field-name">${k}</span>
            <span class="confidence-track"><span class="confidence-fill ${badgeClass(m.status)}" style="width:${Math.round(v * 100)}%"></span></span>
            <span class="confidence-pct">${Math.round(v * 100)}%</span>
          </div>
        `).join('')}
      </div>
      <button class="ghost small" data-correct-id="${m.id}" style="margin-top:8px;">Add a correction</button>
      <div class="correction-form" id="correction-${m.id}" style="display:none;">
        <select id="correctField-${m.id}">
          ${CORRECTABLE_FIELDS.map(f => `<option value="${f}">${f.replace('_', ' ')}</option>`).join('')}
        </select>
        <input type="text" id="correctValue-${m.id}" placeholder="Corrected value">
        <input type="text" id="correctReason-${m.id}" placeholder="Reason (required — kept in the audit trail)">
        <button class="primary small" data-submit-correction="${m.id}">Save correction</button>
        <div id="correctStatus-${m.id}" style="font-size:12px;margin-top:6px;"></div>
      </div>
      <div id="correctionHistory-${m.id}" style="font-size:12px;color:var(--ink-soft);margin-top:6px;"></div>
    </div>
  `).join('') || '<div class="empty">No prescriptions on file.</div>';

  view.innerHTML = `
    <h2 style="margin-top:26px;">Prescriptions</h2>

    <div class="card">
      <div class="card-head">${iconBadge('teal', 'fileText')}<h3>Prescription &amp; extraction review</h3></div>
      ${medsHtml}
    </div>

    <div class="card">
      <div class="card-head">${iconBadge('teal', 'fileText')}<h3>Write a prescription</h3></div>
      <p style="color:var(--ink-soft);font-size:13px;margin-top:0;">
        Same shorthand as the patient's own entry — one medicine per line. Goes through the
        same parser and confidence gate; the patient still confirms it before anything is scheduled.
      </p>
      <div class="field-inline" style="margin-bottom:10px;">
        <label for="templateSelect">Start from a saved template (optional)</label>
        <select id="templateSelect">
          <option value="">— none —</option>
          ${state.templates.map(t => `<option value="${t.id}">${t.label}</option>`).join('')}
        </select>
      </div>
      <textarea id="newPrescriptionLines" rows="4" placeholder="Tab Dolo 650mg 1-0-1 PC x5d"></textarea>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
        <button class="primary small" id="sendPrescriptionBtn">Send to patient</button>
        <button class="ghost small" id="saveTemplateBtn">Save as template</button>
      </div>
      <div id="prescriptionFormStatus" style="font-size:12px;margin-top:6px;"></div>
    </div>
  `;

  view.querySelectorAll('[data-correct-id]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.correctId;
      const form = document.getElementById(`correction-${id}`);
      form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });
  });

  view.querySelectorAll('[data-submit-correction]').forEach(btn => {
    btn.addEventListener('click', () => submitCorrection(btn.dataset.submitCorrection));
  });

  document.getElementById('templateSelect').addEventListener('change', (e) => {
    const template = state.templates.find(t => String(t.id) === e.target.value);
    document.getElementById('newPrescriptionLines').value = template ? template.lines.join('\n') : '';
  });

  document.getElementById('sendPrescriptionBtn').addEventListener('click', async () => {
    const status = document.getElementById('prescriptionFormStatus');
    const lines = document.getElementById('newPrescriptionLines').value.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) {
      status.style.color = 'var(--alarm)';
      status.textContent = 'Enter at least one medicine line.';
      return;
    }
    try {
      await api('POST', `/doctor/patients/${state.patientId}/prescriptions`, { lines });
      status.style.color = 'var(--teal-dark)';
      status.textContent = 'Sent — the patient will see this as a draft to confirm.';
      await loadPatientDetail();
    } catch (e) {
      status.style.color = 'var(--alarm)';
      status.textContent = e.message;
    }
  });

  document.getElementById('saveTemplateBtn').addEventListener('click', async () => {
    const status = document.getElementById('prescriptionFormStatus');
    const lines = document.getElementById('newPrescriptionLines').value.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) {
      status.style.color = 'var(--alarm)';
      status.textContent = 'Nothing to save — type at least one line first.';
      return;
    }
    const label = prompt('Template name?');
    if (!label) return;
    await api('POST', '/doctor/templates', { label, lines });
    await loadTemplates();
    status.style.color = 'var(--teal-dark)';
    status.textContent = `Saved as "${label}".`;
    const select = document.getElementById('templateSelect');
    select.innerHTML = `<option value="">— none —</option>` +
      state.templates.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
  });

  // Correction history is fetched lazily per medicine, only if any exist,
  // so a fresh prescription with no corrections doesn't cost extra calls.
  r.prescriptions.flatMap(p => p.medicines).forEach(async (m) => {
    try {
      const corrections = await api('GET', `/doctor/medicines/${m.id}/corrections`);
      if (!corrections.length) return;
      const el = document.getElementById(`correctionHistory-${m.id}`);
      if (el) {
        el.innerHTML = 'Corrections: ' + corrections.map(c =>
          `${c.field} "${c.original_value}" → "${c.corrected_value}"`
        ).join('; ');
      }
    } catch (e) { /* non-critical history fetch */ }
  });
}

function renderAdherenceTab() {
  const r = state.currentReport;
  const view = document.getElementById('view-adherence');
  const missedHtml = r.missed_doses.map(d => `
    <tr><td data-label="When">${new Date(d.scheduled_at).toLocaleString()}</td><td data-label="Medicine">${d.medicine_name}</td></tr>
  `).join('') || '<tr><td colspan="2" class="empty">None</td></tr>';

  const srcNote = r.data_source ? `<p style="color:var(--ink-soft);font-size:14px;line-height:1.5;margin:0 4px 14px;">From a research record (${String(r.data_source).replace(/[<>&"']/g, '')}): these are the insulin doses written in the record, each shown as taken. The record does not list missed doses, so 100% here is not a measured adherence.</p>` : '';
  view.innerHTML = `
    <h2 style="margin-top:26px;">Adherence</h2>
    ${srcNote}

    <div class="stat-row" style="margin-bottom:16px;">
      <div class="stat">${iconBadge('teal', 'chartBar')}<div><div class="num">${r.adherence.adherence_percent ?? '—'}${r.adherence.adherence_percent !== null ? '%' : ''}</div><div class="label">Adherence</div></div></div>
      <div class="stat">${iconBadge('blue', 'pill')}<div><div class="num">${r.adherence.taken}</div><div class="label">Taken</div></div></div>
      <div class="stat">${iconBadge('alarm', 'alertCircle')}<div><div class="num">${r.adherence.missed}</div><div class="label">Missed</div></div></div>
    </div>

    <div class="card">
      <div class="card-head">${iconBadge('alarm', 'clock')}<h3>Missed doses</h3></div>
      <div class="table-scroll"><table class="responsive-table"><thead><tr><th>When</th><th>Medicine</th></tr></thead><tbody>${missedHtml}</tbody></table></div>
    </div>
  `;
}

function renderTriageTab() {
  const r = state.currentReport;
  const view = document.getElementById('view-triage');
  const triageHtml = r.symptom_history.map(c => `
    <tr>
      <td data-label="Date">${new Date(c.created_at).toLocaleDateString()}</td>
      <td data-label="Symptoms">${c.symptoms.join(', ')}</td>
      <td data-label="Severity"><span class="badge ${badgeClass(c.severity)}">${c.severity}</span></td>
      <td data-label="Action">${c.action}</td>
    </tr>
  `).join('') || '<tr><td colspan="4" class="empty">None</td></tr>';
  const srcNote = r.data_source && !r.symptom_history.length
    ? `<p style="color:var(--ink-soft);font-size:14px;line-height:1.5;margin:12px 4px 0;">No symptom checks: this demo patient comes from a research dataset (${String(r.data_source).replace(/[<>&"']/g, '')}), which records sugar, meals and doses but not symptoms.</p>` : '';

  view.innerHTML = `
    <h2 style="margin-top:26px;">Triage history</h2>
    <div class="card">
      <div class="card-head">${iconBadge('teal', 'stethoscope')}<h3>Symptoms &amp; triage history</h3></div>
      <div class="table-scroll"><table class="responsive-table"><thead><tr><th>Date</th><th>Symptoms</th><th>Severity</th><th>Action</th></tr></thead><tbody>${triageHtml}</tbody></table></div>${srcNote}
    </div>
  `;
}

function renderNotesTab() {
  const r = state.currentReport;
  const view = document.getElementById('view-notes');
  const notesHtml = r.doctor_caregiver_notes.map(n => `
    <div style="padding:6px 0;border-bottom:1px solid var(--line);font-size:13px;">
      <span style="color:var(--ink-soft);">${new Date(n.at).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'})}</span>
      <div>${n.note}</div>
    </div>
  `).join('') || '<div class="empty">No notes yet.</div>';

  view.innerHTML = `
    <h2 style="margin-top:26px;">Notes</h2>
    <div class="card">
      <div class="card-head">${iconBadge('teal', 'clipboard')}<h3>Clinical notes</h3></div>
      ${notesHtml}
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
        <input type="text" id="noteText" placeholder="Add a consultation / follow-up note..." style="flex:1 1 200px;min-width:0;">
        <button class="primary small" id="addNoteBtn">Add</button>
      </div>
    </div>
  `;

  document.getElementById('addNoteBtn').addEventListener('click', async () => {
    const note = document.getElementById('noteText').value.trim();
    if (!note) return;
    await api('POST', `/patients/${state.patientId}/notes`, { note });
    await loadPatientDetail();
  });
}

async function submitCorrection(medicineId) {
  const field = document.getElementById(`correctField-${medicineId}`).value;
  const corrected_value = document.getElementById(`correctValue-${medicineId}`).value.trim();
  const reason = document.getElementById(`correctReason-${medicineId}`).value.trim();
  const status = document.getElementById(`correctStatus-${medicineId}`);
  if (!corrected_value || !reason) {
    status.style.color = 'var(--alarm)';
    status.textContent = 'Both a corrected value and a reason are required — corrections are never silent.';
    return;
  }
  try {
    const result = await api('POST', `/doctor/medicines/${medicineId}/correction`, { field, corrected_value, reason });
    status.style.color = 'var(--teal-dark)';
    status.textContent = `Saved. ${result.warning || ''}`;
    setTimeout(loadPatientDetail, 800);
  } catch (e) {
    status.style.color = 'var(--alarm)';
    status.textContent = e.message;
  }
}

async function loadTemplates() {
  state.templates = await api('GET', '/doctor/templates');
}

(async function boot() {
  if (!currentUser) return;
  renderSessionChip();
  applyA11yMode();
  injectNavIcons();
  await loadTemplates();
  await loadPatients();
})();
