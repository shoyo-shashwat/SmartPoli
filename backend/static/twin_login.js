// SmartPoli: "Doctor demo" card on the login page of the DEMO copy only.
// GET /twin/info answers 404 on the live app, so nothing is added there. When SMARTPOLI_DEMO_LOGIN=1 it also returns
// the demo doctor's credentials, shown in plain text with Copy buttons plus a one-tap sign-in.
(async function () {
  let info;
  try {
    const r = await fetch('/twin/info');
    if (!r.ok) return;
    info = await r.json();
  } catch (e) { return; }
  const demo = info.demo_login;
  if (!demo) return;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const card = document.createElement('div');
  card.className = 'card twn-demo';
  card.innerHTML = '<h2>Doctor demo</h2><p>See the Digital Twin on two anonymised patients. No sign-up needed.</p>' +
    '<button type="button" class="primary" id="demoEnter" style="width:100%">Open the doctor demo</button>' +
    `<div class="twn-cred"><div><span>Email</span><code>${esc(demo.email)}</code></div><button type="button" class="ghost small" data-copy="${esc(demo.email)}">Copy</button></div>` +
    `<div class="twn-cred"><div><span>Password</span><code>${esc(demo.password)}</code></div><button type="button" class="ghost small" data-copy="${esc(demo.password)}">Copy</button></div>` +
    '<div id="demoError" class="auth-error" role="alert"></div>';
  const target = document.querySelector('.auth-card');
  target.parentNode.insertBefore(card, target);
  const note = document.querySelector('.auth-demo-note');
  if (note) note.style.display = 'none';
  card.addEventListener('click', async (e) => {
    const c = e.target.closest('[data-copy]');
    if (c) { try { await navigator.clipboard.writeText(c.dataset.copy); } catch (err) { /* clipboard blocked */ } const o = c.textContent; c.textContent = 'Copied'; setTimeout(() => { c.textContent = o; }, 1200); return; }
    if (!e.target.closest('#demoEnter')) return;
    const btn = document.getElementById('demoEnter'); btn.disabled = true;
    try {
      const res = await fetch('/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: demo.email, password: demo.password }) });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || 'Could not sign in to the demo.');
      setAuth(body.token, body.user);
      window.location.href = landingAfterLogin(body.user.role, window.location.search);
    } catch (err) {
      document.getElementById('demoError').textContent = err.message; btn.disabled = false;
    }
  });
})();
