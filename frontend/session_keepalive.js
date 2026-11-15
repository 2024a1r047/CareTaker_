/**
 * Keeps the login alive on pages that stay open for hours (room camera, voice assistant, dashboard).
 *
 * A login token is valid for 60 minutes. Without this, a monitoring page left open silently
 * stops saving falls and medicine sightings and the History tab shows "Could not load".
 * - Every 20 minutes (and when the page becomes visible again) the token is swapped for a fresh one.
 * - An already expired login is renewed too (the server checks the signature and the account),
 *   so the caretaker is never asked to log in again on these pages; only Logout ends the session.
 * Fires a "caretaker-token" event with the new token so page scripts can pick it up.
 */
(function () {
    const API = 'http://127.0.0.1:8000/api';
    const EVERY_MS = 20 * 60 * 1000;
    let lastTry = 0, busy = false;

    // The LATEST login is in localStorage. sessionStorage can still hold an older login of this tab
    // (e.g. a caretaker who used the AI assistant before a doctor logged in): renewing that one
    // used to overwrite the new login. Use localStorage first, and drop a different stale copy.
    function current() {
        const local = localStorage.getItem('token');
        const sess = sessionStorage.getItem('token');
        if (local && sess && sess !== local) {
            try { sessionStorage.setItem('token', local); } catch (e) { /* ignore */ }
        }
        return local || sess;
    }

    function store(tok) {
        localStorage.setItem('token', tok);
        if (sessionStorage.getItem('token')) sessionStorage.setItem('token', tok);
        try { window.dispatchEvent(new CustomEvent('caretaker-token', { detail: tok })); } catch (e) { /* old browser */ }
    }

    // No "login expired" bar any more: the session renews itself. Kept as a quiet console note.
    function showExpired() {
        console.warn('[Session] login could not be renewed (account removed or logged out)');
        return;
        /* eslint-disable no-unreachable */
        if (document.getElementById('loginExpiredBar')) return;
        const bar = document.createElement('div');
        bar.id = 'loginExpiredBar';
        bar.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#c0392b;color:#fff;' +
            'padding:10px 16px;font:600 15px "Segoe UI",sans-serif;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.25)';
        bar.innerHTML = 'Your login has expired — falls and medicine sightings are not being saved. ' +
            '<a href="login.html" style="color:#fff;text-decoration:underline;margin-left:6px;">Log in again</a>';
        document.body.appendChild(bar);
    }

    let inflight = null;
    // A renewal already running is shared: callers (e.g. a page retrying after a 401) wait for it
    // instead of getting back at once with the old token.
    function refresh(force) {
        if (inflight) return inflight;
        inflight = doRefresh(force).finally(() => { inflight = null; });
        return inflight;
    }

    async function doRefresh(force) {
        const tok = current();
        if (!tok || busy) return;
        const now = Date.now();
        if (!force && now - lastTry < EVERY_MS - 5000) return;
        busy = true; lastTry = now;
        try {
            const r = await fetch(`${API}/refresh-token`, { method: 'POST', headers: { 'Authorization': `Bearer ${tok}` } });
            if (r.status === 401) { showExpired(); return; }
            if (!r.ok) return;                                   // server busy: try again next time
            const d = await r.json();
            if (d && d.access_token) store(d.access_token);
            const bar = document.getElementById('loginExpiredBar'); if (bar) bar.remove();
        } catch (e) { /* backend offline: try again later */ }
        finally { busy = false; }
    }

    // once on load (an old page may already hold a nearly expired token), then every 20 minutes
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => refresh(true));
    else refresh(true);
    setInterval(() => refresh(false), 60 * 1000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(false); });
    window.addEventListener('focus', () => refresh(false));

    // laptop woke up / tab came back after a long time: renew right away, not at the next 20-minute mark
    let lastTick = Date.now();
    setInterval(() => { const n = Date.now(); if (n - lastTick > 3 * 60 * 1000) refresh(true); lastTick = n; }, 30 * 1000);

    window.CaretakerSession = { refresh: () => refresh(true), current };
})();
