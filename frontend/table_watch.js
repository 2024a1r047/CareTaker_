/**
 * table_watch.js — "Table Watch": the camera watches the MEDICINES and the WATER GLASS
 * on the table, finds them by itself, and keeps reminding the elder at the dose time.
 *
 * AUTOMATIC DETECTION (no marking needed)
 *   - When the camera starts it learns the EMPTY table (background model, 160x120).
 *   - Anything new that is put on the table and stays still for a few seconds, with no hand
 *     or person over it, becomes an object automatically:
 *        cup / wine glass / bowl (COCO-SSD)  -> WATER glass
 *        anything else (strips, pill bottle, box, pills on a plate) -> MEDICINE
 *   - A cup/glass already on the table at start is also picked up by COCO-SSD.
 *
 * TAKEN = GONE
 *   - Each medicine spot is compared with the empty table under it. When the spot looks
 *     empty again (and no hand is over it) for a few seconds -> "medicine taken".
 *   - Water: glass gone -> picked up; back and changed -> "drank".
 *
 * REMINDER AT THE DOSE TIME
 *   - From the due time, while medicines are still on the table: beep + spoken reminder
 *     ("Dadi ji, dawai lene ka samay ho gaya hai ... le lijiye") repeated every N seconds
 *     until they are gone, or the caretaker presses "Taken" / "Snooze".
 *   - After the grace period the caretaker is alerted (dashboard alert, saved in history).
 *   - Water untouched too long -> spoken "paani pi lijiye" reminder, repeated every 15 min.
 *
 * Manual marking (drag a box) is still available as a fallback.
 */
'use strict';

const TableWatch = (() => {
    const API = (typeof API_BASE !== 'undefined') ? API_BASE : 'http://127.0.0.1:8000/api';
    const GW = 160, GH = 120, GN = GW * GH;   // analysis grid
    const PIX_THR = 0.11;          // per-pixel difference that counts as "something here"
    const PRESENT_FRAC = 0.22;     // share of a spot that must differ from the empty table = object present
    const STABLE_NEW_MS = 5000;    // new object must stay still this long to be registered
    const GONE_MS = 4000;          // spot must look empty this long = taken
    const DETECT_EVERY_MS = 2500;
    const GLASS = ['cup', 'wine glass', 'bowl'];
    // Water-glass tracking is switched off (medicines only). A glass is still recognised so it is
    // never mistaken for a medicine, but it is not tracked, reminded about or logged.
    const WATER_TRACKING = false;
    const MAX_MED_SPOTS = 5;

    const st = {
        video: null, canvas: null, ctx: null, stream: null, running: false, raf: null, tick: null,
        model: null, detections: [], lastDetect: 0,
        work: document.createElement('canvas'),
        bg: null, bgAt: 0,
        candidates: [],
        spots: [],                 // {id, kind, rect(video px), g(grid box), auto, emptyRef(bool), ref(Float32 patch), present, goneSince, state}
        nextId: 1,
        markMode: null, dragStart: null, dragRect: null,
        water: { lastDrinkAt: null, filledAt: null, alertedAt: 0 },
        med: { doneSlots: new Set(), snoozeUntil: 0, lastRemindAt: 0, escalatedSlot: null, takenAt: null, nothingSpokenSlot: null },
        schedule: [], patientName: '', log: [],
    };

    const $ = id => document.getElementById(id);
    // patient: from the URL, else the one picked on the dashboard, else the caretaker's own patient
    const recipientId = () => new URLSearchParams(location.search).get('recipient_id')
        || sessionStorage.getItem('selectedRecipientId') || localStorage.getItem('selectedRecipientId')
        || (window._twDefaultRecipient || '');
    async function ensureRecipient() {
        if (recipientId()) return recipientId();
        try {
            const r = await fetch(`${API}/video-monitoring/default-recipient`, { headers: auth() });
            if (r.ok) window._twDefaultRecipient = String((await r.json()).recipient_id || '');
        } catch (e) { /* offline */ }
        return recipientId();
    }
    const timeStr = d => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const fmtAgo = ms => { const m = Math.floor(ms / 60000); return m < 1 ? 'just now' : m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`; };

    function settings() {
        return {
            waterMin: Math.max(5, Number(($('twWaterMin') || {}).value) || 60),
            medDue: (($('twMedDue') || {}).value) || '',
            medGrace: Math.max(5, Number(($('twMedGrace') || {}).value) || 30),
            repeatSec: Math.max(20, Number(($('twRepeat') || {}).value) || 60),
            voice: !$('twVoice') || $('twVoice').checked,
        };
    }

    // ── frame grid ─────────────────────────────────────────────────────────
    function grab() {
        const c = st.work; c.width = GW; c.height = GH;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(st.video, 0, 0, GW, GH);
        const d = g.getImageData(0, 0, GW, GH).data;
        const f = { gray: new Float32Array(GN), r: new Float32Array(GN), g: new Float32Array(GN), b: new Float32Array(GN), mean: 0 };
        let s = 0;
        for (let i = 0; i < GN; i++) {
            const r = d[i * 4] / 255, gg = d[i * 4 + 1] / 255, b = d[i * 4 + 2] / 255;
            f.r[i] = r; f.g[i] = gg; f.b[i] = b; f.gray[i] = 0.299 * r + 0.587 * gg + 0.114 * b; s += f.gray[i];
        }
        f.mean = s / GN;
        return f;
    }

    function copyFrame(f) {
        return { gray: f.gray.slice(), r: f.r.slice(), g: f.g.slice(), b: f.b.slice(), mean: f.mean };
    }

    /** per-pixel difference to the background, corrected for overall light changes */
    function diffMap(f) {
        const bg = st.bg, k = bg.mean > 0.01 ? f.mean / bg.mean : 1;
        const out = new Float32Array(GN);
        for (let i = 0; i < GN; i++) {
            const dg = Math.abs(f.gray[i] - bg.gray[i] * k);
            const dc = (Math.abs(f.r[i] - bg.r[i] * k) + Math.abs(f.g[i] - bg.g[i] * k) + Math.abs(f.b[i] - bg.b[i] * k)) / 3;
            out[i] = Math.max(dg, dc);
        }
        return out;
    }

    const toGrid = r => {
        const sx = GW / st.video.videoWidth, sy = GH / st.video.videoHeight;
        return { x: Math.max(0, Math.floor(r.x * sx)), y: Math.max(0, Math.floor(r.y * sy)),
                 x2: Math.min(GW - 1, Math.ceil((r.x + r.w) * sx)), y2: Math.min(GH - 1, Math.ceil((r.y + r.h) * sy)) };
    };
    const toVideo = g => {
        const sx = st.video.videoWidth / GW, sy = st.video.videoHeight / GH;
        return { x: g.x * sx, y: g.y * sy, w: (g.x2 - g.x + 1) * sx, h: (g.y2 - g.y + 1) * sy };
    };
    const bboxToRect = b => ({ x: b[0], y: b[1], w: b[2], h: b[3] });

    function overlapFrac(a, b) {      // share of a covered by b (video px rects)
        const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
        return (x2 <= x1 || y2 <= y1) ? 0 : ((x2 - x1) * (y2 - y1)) / (a.w * a.h);
    }
    function iou(a, b) {
        const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
        if (x2 < x1 || y2 < y1) return 0;
        const inter = (x2 - x1 + 1) * (y2 - y1 + 1);
        const area = r => (r.x2 - r.x + 1) * (r.y2 - r.y + 1);
        return inter / (area(a) + area(b) - inter);
    }

    function personGridMask() {
        const m = new Uint8Array(GN);
        for (const p of st.detections.filter(d => d.class === 'person')) {
            const r = bboxToRect(p.bbox);
            const g = toGrid({ x: r.x - r.w * 0.1, y: r.y - r.h * 0.1, w: r.w * 1.2, h: r.h * 1.2 });
            for (let y = g.y; y <= g.y2; y++) for (let x = g.x; x <= g.x2; x++) m[y * GW + x] = 1;
        }
        return m;
    }

    const personOver = rect => st.detections.some(d => d.class === 'person' && overlapFrac(rect, bboxToRect(d.bbox)) > 0.1);

    // ── logging / backend / sound ──────────────────────────────────────────
    function addLog(kind, text, cls = 'secondary') {
        st.log.unshift({ at: new Date(), kind, text, cls });
        st.log = st.log.slice(0, 40);
        const el = $('twLog');
        if (el) el.innerHTML = st.log.map(l => `
            <div class="alert alert-${l.cls} py-2 px-3 mb-2 small">
                <i class="fas ${l.kind === 'water' ? 'fa-glass-water' : 'fa-pills'} me-1"></i>
                <strong>${timeStr(l.at)}</strong> — ${l.text}
            </div>`).join('');
    }

    async function report(item, event, detail, extra = {}) {
        const token = localStorage.getItem('token');
        if (!token) return;
        try {
            await fetch(`${API}/video-monitoring/table-watch/event`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ item, event, detail, recipient_id: recipientId() ? Number(recipientId()) : null, ...extra }),
            });
        } catch (e) { console.warn('[TableWatch] report failed', e); }
    }

    function beep(urgent) {
        try {
            if (typeof ensureAlarmAudio === 'function') ensureAlarmAudio();
            if (urgent && typeof playSiren === 'function') return playSiren();
            if (typeof playAlertSound === 'function') return playAlertSound();
        } catch (e) { /* ignore */ }
    }

    function speak(text) {
        if (!settings().voice || !window.speechSynthesis) return;
        try {
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            const voices = window.speechSynthesis.getVoices();
            const hi = voices.find(v => /hi[-_]IN/i.test(v.lang)) || voices.find(v => /en[-_]IN/i.test(v.lang));
            if (hi) u.voice = hi;
            u.lang = hi ? hi.lang : 'hi-IN';
            u.rate = 0.9; u.volume = 1;
            window.speechSynthesis.speak(u);
        } catch (e) { /* ignore */ }
    }

    function remind(text, urgent) {
        beep(urgent);
        setTimeout(() => speak(text), urgent ? 2300 : 1200);   // speak after the beep
    }

    // ── spots (auto or manual) ─────────────────────────────────────────────
    function spotPatchFrac(spot, f, dmap) {
        // share of the spot that differs from the EMPTY table (= object still there)
        let n = 0, on = 0;
        for (let y = spot.g.y; y <= spot.g.y2; y++) for (let x = spot.g.x; x <= spot.g.x2; x++) {
            const i = y * GW + x; n++; if (dmap[i] > PIX_THR) on++;
        }
        return n ? on / n : 0;
    }

    function patchOf(f, g) {
        const out = [];
        for (let y = g.y; y <= g.y2; y++) for (let x = g.x; x <= g.x2; x++) out.push(f.gray[y * GW + x]);
        return Float32Array.from(out);
    }

    function patchChange(a, b) {      // share of pixels that changed vs the saved look (manual spots)
        if (!a || !b || a.length !== b.length) return 0;
        let ma = 0, mb = 0; for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.length; mb /= b.length;
        let c = 0; for (let i = 0; i < a.length; i++) if (Math.abs((a[i] - ma) - (b[i] - mb)) > PIX_THR) c++;
        return c / a.length;
    }

    function addSpot(kind, rect, auto, f, emptyRef) {
        if (kind === 'water' && !WATER_TRACKING) return null;
        const g = toGrid(rect);
        const spot = { id: st.nextId++, kind, rect: toVideo(g), g, auto, emptyRef, ref: f ? patchOf(f, g) : null,
                       present: true, goneSince: null, changedSince: null, addedAt: Date.now(), state: 'on table' };
        st.spots.push(spot);
        if (kind === 'water') {
            st.water.filledAt = Date.now(); st.water.alertedAt = 0;
            addLog('water', auto ? 'Water glass found on the table (treated as full)' : 'Water glass marked (treated as full)', 'info');
            report('water', 'baseline', 'Glass on the table, treated as full');
        } else {
            const n = st.spots.filter(s => s.kind === 'medicine').length;
            addLog('medicine', `${auto ? 'Medicine found on the table' : 'Medicine area marked'} (${n} item${n > 1 ? 's' : ''})`, 'info');
            if (n === 1) report('medicine', 'baseline', 'Medicines placed on the table');
        }
        return spot;
    }

    // ── main loop (every second) ───────────────────────────────────────────
    async function analyse() {
        if (!st.running || !st.video.videoWidth) return;
        const now = Date.now();
        if (st.model && now - st.lastDetect > DETECT_EVERY_MS && !fallBusy()) {   // fall detection first
            st.lastDetect = now;
            try { st.detections = await st.model.detect(st.video, 15, 0.35); } catch (e) { st.detections = []; }
        }
        const f = grab();
        if (!st.bg) { st.bg = copyFrame(f); st.bgAt = now; setInfo(); return; }

        const dmap = diffMap(f);
        const pmask = personGridMask();
        const inSpot = new Uint8Array(GN);
        for (const s of st.spots) if (!s.learned) for (let y = s.g.y; y <= s.g.y2; y++) for (let x = s.g.x; x <= s.g.x2; x++) inSpot[y * GW + x] = 1;

        // Camera moved / light changed: a big part of the picture differs -> relearn the table
        // instead of reporting phantom objects (curtains, walls, auto-exposure).
        let changedPx = 0, freePx = 0;
        for (let i = 0; i < GN; i++) { if (pmask[i] || inSpot[i]) continue; freePx++; if (dmap[i] > PIX_THR) changedPx++; }
        if (freePx && changedPx / freePx > 0.08) {
            st.bg = copyFrame(f); st.candidates = [];
            const before = st.spots.length;
            st.spots = st.spots.filter(s => s.learned || !s.emptyRef || s.kind === 'water');   // drop change-based spots (no longer valid)
            if (now - (st.lastRelearn || 0) > 15000) {
                st.lastRelearn = now;
                addLog('medicine', 'Camera view or light changed — relearned the table' + (before !== st.spots.length ? ' (removed unsure items)' : ''), 'secondary');
            }
            await learnedMedicineTick(now, f);
            medicineReminder(now); waterReminder(now); renderStatus(); setInfo();
            return;
        }

        // 1) slowly keep the empty-table model up to date (not under people or objects)
        const k = st.bg.mean > 0.01 ? f.mean / st.bg.mean : 1;
        for (let i = 0; i < GN; i++) {
            if (pmask[i] || inSpot[i]) continue;
            const a = dmap[i] > PIX_THR ? 0.004 : 0.05;     // changed pixels adapt very slowly (new objects stay visible)
            st.bg.gray[i] += a * (f.gray[i] / k - st.bg.gray[i]) * 1;
            st.bg.r[i] += a * (f.r[i] / k - st.bg.r[i]);
            st.bg.g[i] += a * (f.g[i] / k - st.bg.g[i]);
            st.bg.b[i] += a * (f.b[i] / k - st.bg.b[i]);
        }
        st.bg.mean = st.bg.gray.reduce((s, v) => s + v, 0) / GN;

        // 2) taught medicines (recognised by look, anywhere in the picture, even in the hand)
        await learnedMedicineTick(now, f);

        // 2b) optional: any new still object on the table counts as a medicine
        if ($('twAnyObject') && $('twAnyObject').checked) findNewObjects(dmap, pmask, inSpot, f, now);
        else st.candidates = [];

        // 3) glass already on the table (COCO-SSD)
        if (st.model && !st.spots.some(s => s.kind === 'water')) {
            const cup = st.detections.find(d => GLASS.includes(d.class) && d.score > 0.5);
            if (cup && !personOver(bboxToRect(cup.bbox))) addSpot('water', bboxToRect(cup.bbox), true, f, false);
        }

        // 4) check each spot: still there or gone?
        for (const s of [...st.spots]) {
            if (s.learned) continue;                                  // handled by learnedMedicineTick
            if (personOver(s.rect)) { s.touched = now; s.goneSince = null; s.changedSince = null; continue; }
            if (s.touched && now - s.touched < 1500) continue;      // let the hand move away
            let present;
            if (s.emptyRef) {
                present = spotPatchFrac(s, f, dmap) > PRESENT_FRAC * 0.6;
            } else if (s.kind === 'water' && st.model) {
                present = st.detections.some(d => GLASS.includes(d.class) && overlapFrac(s.rect, bboxToRect(d.bbox)) > 0.2);
            } else {
                // manual spot without an empty-table reference: a lasting change = taken
                present = patchChange(patchOf(f, s.g), s.ref) < 0.3;
            }
            if (!present) { s.goneSince = s.goneSince || now; } else { s.goneSince = null; }

            if (s.kind === 'medicine' && s.goneSince && now - s.goneSince >= GONE_MS) {
                st.spots = st.spots.filter(x => x !== s);
                const left = st.spots.filter(x => x.kind === 'medicine').length;
                if (left === 0) medicinesTaken('camera');
                else addLog('medicine', `One medicine picked up — ${left} still on the table`, 'warning');
            }
            if (s.kind === 'water') {
                if (s.goneSince && now - s.goneSince >= GONE_MS && s.state !== 'away') {
                    s.state = 'away'; s.awayAt = now;
                    addLog('water', 'Glass picked up from the table', 'secondary');
                    report('water', 'missing', 'Glass picked up from the table');
                } else if (!s.goneSince && s.state === 'away') {
                    s.state = 'on table'; s.ref = patchOf(f, s.g);
                    st.water.lastDrinkAt = now; st.water.alertedAt = 0;
                    hideReminder('water');
                    addLog('water', 'Glass put back — likely drank water ✓', 'success');
                    report('water', 'taken', 'Glass picked up and put back', { confidence: 65 });
                } else if (s.state === 'away' && now - s.awayAt > 10 * 60000) {
                    st.spots = st.spots.filter(x => x !== s);     // glass taken away for good
                }
            }
        }

        medicineReminder(now);
        waterReminder(now);
        renderStatus();
        setInfo();
    }

    function findNewObjects(dmap, pmask, inSpot, f, now) {
        const mask = new Uint8Array(GN);
        for (let i = 0; i < GN; i++) mask[i] = (dmap[i] > PIX_THR && !pmask[i] && !inSpot[i]) ? 1 : 0;
        // connected components (4-neighbour)
        const seen = new Uint8Array(GN), comps = [];
        for (let i = 0; i < GN; i++) {
            if (!mask[i] || seen[i]) continue;
            let x1 = GW, y1 = GH, x2 = 0, y2 = 0, area = 0;
            const stack = [i]; seen[i] = 1;
            while (stack.length) {
                const p = stack.pop(), x = p % GW, y = (p / GW) | 0;
                area++; if (x < x1) x1 = x; if (x > x2) x2 = x; if (y < y1) y1 = y; if (y > y2) y2 = y;
                for (const q of [p - 1, p + 1, p - GW, p + GW]) {
                    if (q < 0 || q >= GN || seen[q] || !mask[q]) continue;
                    if ((p % GW === 0 && q === p - 1) || (p % GW === GW - 1 && q === p + 1)) continue;
                    seen[q] = 1; stack.push(q);
                }
            }
            const w = x2 - x1 + 1, h = y2 - y1 + 1;
            if (area >= 10 && area <= GN * 0.12 && w >= 3 && h >= 3 && area / (w * h) >= 0.2) {
                comps.push({ x: Math.max(0, x1 - 1), y: Math.max(0, y1 - 1), x2: Math.min(GW - 1, x2 + 1), y2: Math.min(GH - 1, y2 + 1), area });
            }
        }
        // track candidates over time; register the ones that stay still
        const next = [];
        for (const c of comps) {
            const prev = st.candidates.find(p => iou(p.g, c) > 0.3);
            next.push({ g: c, first: prev ? prev.first : now });
        }
        st.candidates = next;
        for (const c of st.candidates) {
            if (now - c.first < STABLE_NEW_MS) continue;
            const rect = toVideo(c.g);
            if (personOver(rect)) continue;
            if (st.spots.some(s => iou(s.g, c.g) > 0.2)) continue;
            const obj = st.detections.find(d => d.class !== 'person' && overlapFrac(rect, bboxToRect(d.bbox)) > 0.3);
            const isGlass = obj && GLASS.includes(obj.class);
            if (isGlass && !st.spots.some(s => s.kind === 'water')) {
                addSpot('water', rect, true, f, true);
            } else if (!isGlass && st.spots.filter(s => s.kind === 'medicine').length < MAX_MED_SPOTS) {
                if (obj && ['chair', 'couch', 'bed', 'dining table', 'tv', 'laptop', 'potted plant', 'cat', 'dog'].includes(obj.class)) continue;
                addSpot('medicine', rect, true, f, true);
                st.med.takenAt = null;
            }
            c.first = Infinity;      // don't register twice
        }
    }

    // ── taught medicines (MobileNet look-alike search) ─────────────────────
    // The caretaker shows each medicine pack to the camera once ("Teach medicine").
    // MobileNet turns the picture into a fingerprint; every 1.5 s the whole frame is scanned
    // in windows of several sizes and compared with the taught fingerprints.
    const MED = {
        net: null, loading: false, taught: [], negs: [],
        hits: [], lastSeen: 0, firstSeen: 0, present: false, everSeen: false,
        box: null, bestSim: 0, bestName: '', lastScan: 0, teach: null,
    };
    const LEARN_KEY = 'tableWatchMedicinesV1';
    const LEARNED_GONE_MS = 8000;
    const SCAN_EVERY_MS = 2000;

    function simThreshold() {
        // sensitivity slider 1..5 (3 = default); the built-in fingerprint needs its own levels
        const s = Number(($('twSens') || {}).value) || 3;
        if (!MED.net) return { pos: 0.62 - (s - 3) * 0.05, margin: 0.08 - (s - 3) * 0.02 };
        // Measured with the real model: a photo-taught strip scores ~0.65-0.75 in the room, plain wall ~0.4.
        // So: a moderate absolute level + a clear lead over the room's own "not medicine" samples.
        return { pos: 0.64 - (s - 3) * 0.04, margin: 0.10 - (s - 3) * 0.02 };
    }

    // ── medicines list (saved on the server, cached in this browser) ───────
    // Each entry: { id, name, mode, protos[], negs[], image, source, rt: {hits, present, lastSeen, box, sim, loc, locHi, reportedAt, reportedLoc} }
    const newRt = () => ({ hits: [], present: false, lastSeen: 0, box: null, sim: 0, loc: '', locHi: '', reportedAt: 0, reportedLoc: '' });
    const auth = () => ({ 'Authorization': `Bearer ${localStorage.getItem('token')}`, 'Content-Type': 'application/json' });
    const toF32 = arr => (arr || []).map(p => Float32Array.from(p));
    const toArr = arr => (arr || []).map(p => Array.from(p, v => +v.toFixed(4)));

    function loadTaughtLocal() {
        try {
            const raw = JSON.parse(localStorage.getItem(LEARN_KEY) || '[]');
            return raw.map(m => ({ id: m.id || null, name: m.name, mode: m.mode || 'mobilenet', image: m.image || null,
                                   source: m.source || 'camera', protos: toF32(m.protos), negs: toF32(m.negs), rt: newRt() }));
        } catch (e) { return []; }
    }

    function saveTaught() {
        try {
            localStorage.setItem(LEARN_KEY, JSON.stringify(MED.taught.map(m => ({
                id: m.id, name: m.name, mode: m.mode, image: m.image, source: m.source,
                protos: toArr(m.protos), negs: toArr(m.negs),
            }))));
        } catch (e) { console.warn('[TableWatch] could not cache taught medicines', e); }
    }

    async function serverSave(m) {
        try {
            const res = await fetch(`${API}/video-monitoring/medicine-refs`, {
                method: 'POST', headers: auth(),
                body: JSON.stringify({ name: m.name, recipient_id: recipientId() ? Number(recipientId()) : null,
                                       image: m.image, source: m.source, mode: m.mode,
                                       embeddings: toArr(m.protos), negatives: toArr(m.negs) }),
            });
            if (res.ok) { const d = await res.json(); m.id = d.medicine && d.medicine.id; }
        } catch (e) { console.warn('[TableWatch] could not save medicine on the server', e); }
    }

    /** Load the list from the server (shared with other devices and the AI assistant). */
    const PENDING_DEL_KEY = 'tableWatchPendingDeleteV1';
    function pendingDeletes() { try { return JSON.parse(localStorage.getItem(PENDING_DEL_KEY) || '[]'); } catch (e) { return []; } }
    function addPendingDelete(id) {
        const l = pendingDeletes(); if (!l.includes(id)) l.push(id);
        try { localStorage.setItem(PENDING_DEL_KEY, JSON.stringify(l)); } catch (e) { /* ignore */ }
    }
    async function flushPendingDeletes() {
        const left = [];
        for (const id of pendingDeletes()) {
            try {
                const r = await fetch(`${API}/video-monitoring/medicine-refs/${id}`, { method: 'DELETE', headers: auth() });
                if (!(r.ok || r.status === 404)) left.push(id);                  // 404 = already gone
            } catch (e) { left.push(id); }
        }
        try { localStorage.setItem(PENDING_DEL_KEY, JSON.stringify(left)); } catch (e) { /* ignore */ }
        return left;
    }

    async function loadTaught() {
        await ensureRecipient();
        await flushPendingDeletes();
        const local = loadTaughtLocal();
        MED.taught = local;
        try {
            const res = await fetch(`${API}/video-monitoring/medicine-refs?recipient_id=${recipientId() || ''}`, { headers: auth() });
            if (res.ok) {
                const d = await res.json();
                const stillDeleting = pendingDeletes();
                const server = (d.medicines || []).filter(m => !stillDeleting.includes(m.id)).map(m => ({
                    id: m.id, name: m.name, mode: m.mode || 'mobilenet', image: m.image, source: m.source,
                    protos: toF32(m.embeddings), negs: toF32(m.negatives), rt: newRt(),
                    lastLocation: m.last_location, lastSeenAt: m.last_seen_at,
                }));
                // medicines taught on this device before the server list existed -> upload them once
                for (const m of local.filter(l => !l.id)) { await serverSave(m); if (m.id) server.push(m); }
                MED.taught = server;
                saveTaught();
            }
        } catch (e) { /* offline: keep the cached list */ }
        syncNegs();
        renderTaught();
    }

    /** taught medicines usable with the current recognition method */
    const usableTaught = () => MED.taught.filter(m => m.mode === embedMode());
    function syncNegs() { MED.negs = [...(MED.roomNegs || [])]; usableTaught().forEach(m => MED.negs.push(...(m.negs || []))); }

    const esc = s => String(s == null ? '' : s).replace(/[<>&"]/g, '');
    function seenText(m) {
        const rt = m.rt || {};
        if (rt.present) return `<span class="text-success fw-bold">In view${rt.conf ? ' (' + rt.conf + '% sure)' : ''}</span> — ${esc(rt.loc || 'in the room')}`;
        const ago = ms => { const t = fmtAgo(ms); return t === 'just now' ? 'Last seen just now' : `Last seen ${t} ago`; };
        if (rt.lastSeen) return `${ago(Date.now() - rt.lastSeen)} — ${esc(rt.loc || 'in the room')}`;
        if (m.lastSeenAt) return `${ago(Date.now() - new Date(m.lastSeenAt).getTime())} — ${esc(m.lastLocation || 'in the room')}`;
        return '<span class="text-muted">Not seen yet</span>';
    }

    /** Camera's best guess (not sure yet): one tap confirms it and teaches that view. */
    function guessHtml(m, i) {
        return '';   // no confirmation needed: the camera records sightings by itself ("Wrong place" fixes a mistake)
        /* eslint-disable no-unreachable */
        const rt = m.rt || {}, c = rt.cand;
        if (rt.present || !c || !rt.candEmb || Date.now() - c.at > 8000 || c.pos < c.need - GUESS_GAP) return '';
        return `<div class="mt-1" style="font-size:0.75rem;">
                    <span class="text-warning fw-bold">Camera guess: ${confidencePct(c.pos, simThreshold())}% sure</span> — yellow box on the video
                    <button class="btn btn-sm btn-warning py-0 px-2 ms-1" style="font-size:0.72rem;" onclick="TableWatch.confirmGuess(${i})">
                        <i class="fas fa-check me-1"></i>Yes, this is it</button>
                    <button class="btn btn-sm btn-outline-secondary py-0 px-2 ms-1" style="font-size:0.72rem;" onclick="TableWatch.notHere(${i})">
                        <i class="fas fa-xmark me-1"></i>No</button>
                </div>`;
    }

    function renderTaught() {
        const item = (m, i) => `
            <div class="d-flex align-items-center gap-2 py-1 border-bottom">
                ${m.image ? `<img src="${m.image}" alt="" style="width:38px;height:38px;object-fit:cover;border-radius:8px;">`
                          : '<div style="width:38px;height:38px;border-radius:8px;background:#f3e8eb;display:grid;place-items:center;"><i class="fas fa-pills" style="color:#b0566b;"></i></div>'}
                <div class="flex-grow-1 small" style="min-width:0;">
                    <div class="fw-bold text-truncate">${esc(m.name)}${m.mode !== embedMode() ? ' <span class="text-warning">(teach again)</span>' : ''}</div>
                    <div class="text-muted" style="white-space:normal; line-height:1.3;">${seenText(m)}</div>
                    ${m.rt && m.rt.present ? `<button class="btn btn-sm btn-outline-secondary py-0 px-2 mt-1" style="font-size:0.72rem;"
                        onclick="TableWatch.notHere(${i})" title="The box is not on this medicine">
                        <i class="fas fa-xmark me-1"></i>Wrong place</button>` : ''}
                    ${guessHtml(m, i)}
                </div>
                <a href="#" class="text-muted" onclick="TableWatch.forget(${i});return false;" title="Forget this medicine">×</a>
            </div>`;
        const html = MED.taught.length ? MED.taught.map(item).join('')
            : '<span class="text-muted small">No medicine added yet — upload a photo of the medicine.</span>';
        ['twTaught', 'liveMedList'].forEach(id => { const el = $(id); if (el) el.innerHTML = html; });
    }

    async function loadNet() {
        if (MED.net || MED.loading || MED.netFailed || typeof mobilenet === 'undefined') return;
        MED.loading = true;
        // The model is stored with the app (frontend/models/...): TF Hub now redirects to Kaggle
        // and the browser could not download it there.
        const urls = ['models/mobilenet_v2_050/model.json', '/models/mobilenet_v2_050/model.json'];
        for (const modelUrl of urls) {
            try { MED.net = await mobilenet.load({ version: 2, alpha: 0.5, modelUrl }); break; }
            catch (e) { console.warn('[TableWatch] MobileNet load failed from', modelUrl, e); }
        }
        if (!MED.net) {
            MED.netFailed = true;
            addLog('medicine', 'Recognition model not available — using the simpler built-in recognition', 'warning');
        }
        MED.loading = false;
    }

    const embedMode = () => (MED.net ? 'mobilenet' : 'simple');
    const srcSize = src => ({ W: src.videoWidth || src.naturalWidth || src.width, H: src.videoHeight || src.naturalHeight || src.height });

    /**
     * Built-in fingerprint (no model needed): colour histogram + edge-direction histogram +
     * small brightness layout. Less precise than MobileNet but always works offline.
     */
    function embedSimple(boxes, src = st.video) {
        const { W, H } = srcSize(src), S = 32;
        const c = document.createElement('canvas'); c.width = S; c.height = S;
        const g = c.getContext('2d', { willReadFrequently: true });
        return boxes.map(([y1, x1, y2, x2]) => {
            g.drawImage(src, x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H, 0, 0, S, S);
            const d = g.getImageData(0, 0, S, S).data;
            const col = new Float32Array(64), edge = new Float32Array(128), lay = new Float32Array(64), gray = new Float32Array(S * S);
            for (let i = 0; i < S * S; i++) {
                const r = d[i * 4], gg = d[i * 4 + 1], b = d[i * 4 + 2];
                col[(r >> 6) * 16 + (gg >> 6) * 4 + (b >> 6)] += 1;
                gray[i] = (0.299 * r + 0.587 * gg + 0.114 * b) / 255;
            }
            for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
                const gx = gray[y * S + x + 1] - gray[y * S + x - 1], gy = gray[(y + 1) * S + x] - gray[(y - 1) * S + x];
                const mag = Math.hypot(gx, gy);
                if (mag < 0.03) continue;
                const bin = Math.floor(((Math.atan2(gy, gx) + Math.PI) / (2 * Math.PI)) * 8) % 8;
                edge[(Math.floor(y / 8) * 4 + Math.floor(x / 8)) * 8 + bin] += mag;
            }
            for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) lay[Math.floor(y / 4) * 8 + Math.floor(x / 4)] += gray[y * S + x] / 16;
            const norm = a => { let m = 0; for (const v of a) m += v; m /= a.length; let n = 0; const o = a.map(v => v - m); for (const v of o) n += v * v; n = Math.sqrt(n) || 1; return o.map(v => v / n); };
            const v = [...norm(col), ...norm(edge), ...norm(lay)].map(x => x / Math.sqrt(3));
            return Float32Array.from(v);
        });
    }

    /** L2-normalised fingerprints for normalised boxes [y1,x1,y2,x2] of a video / image / canvas */
    async function embed(boxes, src = st.video) {
        if (!MED.net) return embedSimple(boxes, src);
        const t = tf.tidy(() => {
            // functional ops only: the page loads tfjs-core, which has no chained tensor methods
            const img = tf.expandDims(tf.cast(tf.browser.fromPixels(src), 'float32'), 0);
            const crops = tf.image.cropAndResize(img, tf.tensor2d(boxes), new Array(boxes.length).fill(0), [224, 224]);
            const e = MED.net.infer(crops, true);
            return tf.div(e, tf.norm(e, 2, 1, true));
        });
        const arr = await t.array();
        t.dispose();
        return arr.map(a => Float32Array.from(a));
    }

    const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

    function scanWindows() {
        const W = st.video.videoWidth, H = st.video.videoHeight, m = Math.min(W, H);
        const out = [];
        // small windows too: in a whole room a medicine strip is small
        for (const frac of [0.14, 0.2, 0.3, 0.44]) {
            const bw = (m * frac) / W, bh = (m * frac) / H, step = 0.55;
            for (let y = 0; y + bh <= 1.0001; y += bh * step) for (let x = 0; x + bw <= 1.0001; x += bw * step) {
                out.push([y, x, Math.min(1, y + bh), Math.min(1, x + bw)]);
            }
        }
        return out;
    }

    /** small JPEG thumbnail (data URL) of an image/video region */
    function thumbnail(src, box = [0, 0, 1, 1], max = 220) {
        const { W, H } = srcSize(src);
        const [y1, x1, y2, x2] = box, sw = (x2 - x1) * W, sh = (y2 - y1) * H, k = Math.min(1, max / Math.max(sw, sh));
        const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(sw * k)); c.height = Math.max(1, Math.round(sh * k));
        c.getContext('2d').drawImage(src, x1 * W, y1 * H, sw, sh, 0, 0, c.width, c.height);
        return c.toDataURL('image/jpeg', 0.8);
    }

    /** "not this medicine" samples from the room picture, skipping anything that looks like the medicine */
    async function roomNegatives(protos, count = 24) {
        if (!st.running || !st.video || !st.video.videoWidth) return [];
        const boxes = scanWindows().sort(() => Math.random() - 0.5).slice(0, 60);
        const embs = await embed(boxes);
        return embs.filter(e => protos.every(p => dot(e, p) < 0.6)).slice(0, count);
    }

    // ── teach: from the camera, or from an uploaded photo ─────────────────
    async function teach(nameArg) {
        if (!st.running) { alert('Start the camera first (press "Start camera").'); return; }
        if (MED.teach) return;                                    // already teaching
        try { await teachInner(nameArg); }
        catch (e) {
            console.error('[TableWatch] teach failed', e);
            MED.teach = null;
            addLog('medicine', 'Could not learn the medicine: ' + (e && e.message ? e.message : e), 'danger');
        }
    }

    async function teachInner(nameArg) {
        if (!MED.net && !MED.netFailed) {
            addLog('medicine', 'Loading the recognition model…', 'secondary');
            await loadNet();
        }
        const name = (nameArg || (($('twMedName') || {}).value) || '').trim() || `Medicine ${MED.taught.length + 1}`;
        MED.teach = { name, until: Date.now() + 3000 };
        addLog('medicine', `Hold "${name}" inside the white box in the middle of the picture…`, 'info');
        await new Promise(r => { const iv = setInterval(() => { if (Date.now() >= MED.teach.until) { clearInterval(iv); r(); } }, 100); });
        const protos = [];
        const ar = st.video.videoHeight / st.video.videoWidth;
        for (let k = 0; k < 8; k++) {
            const j = () => (Math.random() - 0.5) * 0.06, s = 0.16 + Math.random() * 0.08;
            const cy = 0.5 + j(), cx = 0.5 + j();
            protos.push(...(await embed([[cy - s, cx - s * ar, cy + s, cx + s * ar]])));
            await new Promise(r => setTimeout(r, 200));
        }
        const image = thumbnail(st.video, [0.3, 0.5 - 0.2 * ar, 0.7, 0.5 + 0.2 * ar]);
        const negBoxes = scanWindows().filter(b => !(b[0] < 0.62 && b[2] > 0.38 && b[1] < 0.62 && b[3] > 0.38));
        const negs = await embed(negBoxes.sort(() => Math.random() - 0.5).slice(0, 24));
        MED.teach = null;
        await addMedicine({ name, mode: embedMode(), protos, negs, image, source: 'camera' });
        if ($('twMedName')) $('twMedName').value = '';
        speak(`Theek hai, maine ${name} pehchan li.`);
    }

    function loadImageFile(file) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(file);
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('This file is not a picture the browser can open'));
            img.src = url;
        });
    }

    /** rotated / flipped copies of the photo on a canvas (so the camera finds it at other angles) */
    function variant(img, angleDeg, flip) {
        const W = img.naturalWidth, H = img.naturalHeight;
        const c = document.createElement('canvas'); c.width = W; c.height = H;
        const g = c.getContext('2d');
        g.fillStyle = '#808080'; g.fillRect(0, 0, W, H);
        g.translate(W / 2, H / 2); g.rotate(angleDeg * Math.PI / 180); if (flip) g.scale(-1, 1);
        g.drawImage(img, -W / 2, -H / 2);
        return c;
    }

    /** "Upload photo": learn a medicine from a picture of it. */
    async function teachFromPhoto(fileArg, nameArg) {
        const file = fileArg || (($('twMedPhoto') || {}).files || [])[0] || (($('liveMedPhoto') || {}).files || [])[0];
        if (!file) { alert('Choose a photo of the medicine first.'); return; }
        const name = (nameArg || (($('twMedName') || {}).value) || (($('liveMedName') || {}).value) || '').trim()
            || file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').slice(0, 40) || `Medicine ${MED.taught.length + 1}`;
        try {
            addLog('medicine', `Learning "${name}" from the photo…`, 'secondary');
            if (!MED.net && !MED.netFailed) await loadNet();
            const img = await loadImageFile(file);
            const protos = [];
            // whole photo + centre crops, at a few angles and mirrored -> robust to how it lies in the room
            for (const [ang, flip] of [[0, false], [0, true], [12, false], [-12, false], [90, false]]) {
                const v = variant(img, ang, flip);
                protos.push(...(await embed([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]], v)));
            }
            const negs = await roomNegatives(protos);
            await addMedicine({ name, mode: embedMode(), protos, negs, image: thumbnail(img), source: 'upload' });
            if (st.running) buildRoomViews(MED.taught[MED.taught.length - 1]).catch(() => {});
            ['twMedPhoto', 'liveMedPhoto', 'twMedName', 'liveMedName'].forEach(id => { if ($(id)) $(id).value = ''; });
            if (!st.running) addLog('medicine', 'Start the room camera — the medicine will then be found automatically.', 'info');
        } catch (e) {
            console.error('[TableWatch] photo teach failed', e);
            addLog('medicine', 'Could not learn from the photo: ' + (e && e.message ? e.message : e), 'danger');
        }
    }

    async function addMedicine(m) {
        m.rt = newRt();
        await serverSave(m);
        MED.taught.push(m);
        syncNegs(); saveTaught(); renderTaught();
        addLog('medicine', `Learned "${m.name}" ✓ — the room camera will now look for it`, 'success');
    }

    async function forget(i) {
        const m = MED.taught[i];
        if (!m || !confirm(`Stop looking for "${m.name}"?`)) return;
        MED.taught.splice(i, 1);
        if (m.id) {
            // remembered until the server confirms: a failed delete (offline, old login) is retried later,
            // otherwise the "deleted" medicine comes back and the voice assistant keeps talking about it
            addPendingDelete(m.id);
            await flushPendingDeletes();
        }
        syncNegs(); saveTaught(); renderTaught();
        st.spots = st.spots.filter(s => !(s.learned && s.medKey === m));
    }

    // ── where is it? (room description) ───────────────────────────────────
    const PLACES = {
        'dining table': ['on the dining table', 'dining table par', true], 'bed': ['on the bed', 'bed par', true],
        'couch': ['on the sofa', 'sofa par', true], 'chair': ['on / near the chair', 'kursi ke paas', true],
        'bench': ['on the bench', 'bench par', true], 'tv': ['near the TV', 'TV ke paas', false],
        'refrigerator': ['near the fridge', 'fridge ke paas', false], 'sink': ['near the sink', 'sink ke paas', false],
        'oven': ['near the oven', 'oven ke paas', false], 'microwave': ['near the microwave', 'microwave ke paas', false],
        'toilet': ['near the toilet', 'toilet ke paas', false], 'potted plant': ['near the plant', 'gamle ke paas', false],
        'laptop': ['next to the laptop', 'laptop ke paas', false], 'book': ['next to the books', 'kitabon ke paas', false],
        'clock': ['near the clock', 'ghadi ke paas', false], 'vase': ['next to the vase', 'phooldaan ke paas', false],
        'bowl': ['next to a bowl', 'katori ke paas', false], 'cup': ['next to a cup', 'cup ke paas', false],
        'bottle': ['next to a bottle', 'bottle ke paas', false], 'remote': ['next to the TV remote', 'remote ke paas', false],
        'cell phone': ['next to the phone', 'phone ke paas', false], 'handbag': ['near the handbag', 'handbag ke paas', false],
        'backpack': ['near the bag', 'bag ke paas', false], 'suitcase': ['near the suitcase', 'suitcase ke paas', false],
    };

    function describeLocation(box) {
        const W = st.video.videoWidth, H = st.video.videoHeight;
        const cx = (box.x + box.w / 2) / W, cy = (box.y + box.h / 2) / H;
        const dispX = 1 - cx;                                  // the room picture is shown mirrored
        const side = dispX < 0.34 ? ['left side of the room', 'kamre ki left side mein']
            : dispX > 0.66 ? ['right side of the room', 'kamre ki right side mein'] : ['middle of the room', 'kamre ke beech mein'];
        const height = cy > 0.75 ? [', low down (near the floor)', ', neeche'] : cy < 0.3 ? [', high up', ', upar'] : ['', ''];
        // in someone's hand?
        const inHand = st.detections.some(d => d.class === 'person' && overlapFrac(box, bboxToRect(d.bbox)) > 0.6);
        if (inHand) return { en: `in someone's hand, ${side[0]}`, hi: `kisi ke haath mein, ${side[1]}` };
        // Detailed place: what it lies on (bed, table...), which part of it, and the object right
        // beside it -> "on the bed, right side, right beside the phone".
        const detail = detailedPlace(box, cx, cy, W, H, side);
        if (detail) return detail;
        // furniture / object it is on or next to (COCO-SSD)
        let best = null;
        for (const d of st.detections) {
            const p = PLACES[d.class];
            if (!p) continue;
            const r = bboxToRect(d.bbox);
            const inside = cx * W >= r.x && cx * W <= r.x + r.w && cy * H >= r.y - box.h && cy * H <= r.y + r.h;
            const dist = Math.hypot((r.x + r.w / 2) / W - cx, (r.y + r.h / 2) / H - cy);
            const score = (inside ? 0 : 1) + dist;
            if ((inside || dist < 0.3) && (!best || score < best.score)) best = { p, inside, score };
        }
        if (best) {
            const [en, hi, surface] = best.p;
            const enTxt = best.inside || !surface ? en : en.replace(/^on (the )?/, 'near the ');
            return { en: `${enTxt}, ${side[0]}`, hi: `${hi}, ${side[1]}` };
        }
        return { en: `${side[0]}${height[0]}`, hi: `${side[1]}${height[1]}` };
    }

    function detailedPlace(box, cx, cy, W, H, side) {
        const dets = (st.detections || []).filter(d => PLACES[d.class]);
        // what it lies on: a surface whose box contains the medicine's centre
        let surface = null;
        for (const d of dets) {
            const p = PLACES[d.class];
            if (!p[2]) continue;
            const r = bboxToRect(d.bbox);
            if (cx * W >= r.x && cx * W <= r.x + r.w && cy * H >= r.y - box.h && cy * H <= r.y + r.h) {
                if (!surface || r.w * r.h < surface.r.w * surface.r.h) surface = { d, p, r };   // the smallest = most specific
            }
        }
        // the object right beside it (phone, cup, remote, bottle...): close to the medicine, not the medicine itself
        let near = null;
        const mx = box.x + box.w / 2, my = box.y + box.h / 2, msz = Math.max(box.w, box.h);
        for (const d of dets) {
            const p = PLACES[d.class];
            if (p[2]) continue;
            const r = bboxToRect(d.bbox);
            if (overlapFrac(box, r) > 0.5) continue;                       // the detector may call the strip a "phone"
            const gap = Math.max(0, Math.max(r.x - (box.x + box.w), box.x - (r.x + r.w)))
                      + Math.max(0, Math.max(r.y - (box.y + box.h), box.y - (r.y + r.h)));
            const dist = Math.hypot(r.x + r.w / 2 - mx, r.y + r.h / 2 - my);
            if (gap <= 1.5 * msz && (!near || dist < near.dist)) near = { p, r, dist, gap };
        }
        if (!surface && !near) return null;
        const en = [], hi = [];
        if (surface) {
            const [sEn, sHi] = surface.p;
            // part of the surface, as seen on the (mirrored) screen
            const rel = 1 - ((cx * W - surface.r.x) / Math.max(1, surface.r.w));
            const part = rel < 0.34 ? ['left side', 'left side mein'] : rel > 0.66 ? ['right side', 'right side mein'] : ['middle', 'beech mein'];
            en.push(`${sEn}, ${part[0]}`); hi.push(`${sHi} ${part[1]}`);
        }
        if (near) {
            const [nEn, nHi] = near.p;
            const close = near.gap <= 0.5 * msz;
            en.push(close ? nEn.replace(/^(next to|near) /, 'right beside ') : nEn);
            hi.push(close ? nHi.replace(/ ke paas$/, ' ke bilkul paas') : nHi);
        }
        if (!surface) { en.push(side[0]); hi.push(side[1]); }
        return { en: en.join(', '), hi: hi.join(', ') };
    }

    async function reportSeen(m) {
        const rt = m.rt;
        if (!m.id || !rt.box) return;
        const now = Date.now();
        if (rt.loc === rt.reportedLoc && now - rt.reportedAt < 60000) return;   // same place: at most once a minute
        rt.reportedAt = now; rt.reportedLoc = rt.loc;
        const W = st.video.videoWidth, H = st.video.videoHeight;
        try {
            const res = await fetch(`${API}/video-monitoring/medicine-refs/${m.id}/seen`, {
                method: 'POST', headers: auth(),
                body: JSON.stringify({ location: rt.loc, location_hi: rt.locHi, confidence: rt.conf || confidencePct(rt.sim, simThreshold(), { accepted: true }),
                                       box: { x: rt.box.x / W, y: rt.box.y / H, w: rt.box.w / W, h: rt.box.h / H } }),
            });
            if (!res.ok) { console.warn('[TableWatch] could not save sighting', res.status); rt.reportedAt = 0; }
        } catch (e) { console.warn('[TableWatch] could not save sighting', e); rt.reportedAt = 0; }
    }

    /**
     * Strong but not yet confirmed guess: save it as a PROBABLE place, so the voice assistant can
     * already say "shayad takiye par…" instead of "camera ne nahi dekha". A confirmed sighting
     * always wins over a probable one (the server keeps the confirmed one for 10 minutes).
     */
    async function reportProbable(m, box, pos) {
        const rt = m.rt;
        if (!m.id || !box || rt.present) return;
        const now = Date.now();
        const loc = describeLocation(box);
        const en = withPlace(rt, loc.en, 'en'), hi = withPlace(rt, loc.hi, 'hi');
        if (en === rt.probReportedLoc && now - (rt.probReportedAt || 0) < 60000) return;
        rt.probReportedAt = now; rt.probReportedLoc = en;
        const W = st.video.videoWidth, H = st.video.videoHeight;
        try {
            await fetch(`${API}/video-monitoring/medicine-refs/${m.id}/seen`, {
                method: 'POST', headers: auth(),
                body: JSON.stringify({ location: en, location_hi: hi, confidence: confidencePct(pos, simThreshold()), probable: true,
                                       box: { x: box.x / W, y: box.y / H, w: box.w / W, h: box.h / H } }),
            });
        } catch (e) { rt.probReportedAt = 0; }
    }

    // ── colour model: where in the room could the medicine be? ─────────────
    // 28 bins: 4 grey levels (low saturation, e.g. silver foil) + 12 hues x 2 saturations.
    const NBIN = 28;
    function colorBin(r, g, b) {              // r,g,b in 0..1
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b), v = mx, s = mx ? (mx - mn) / mx : 0;
        if (s < 0.2 || v < 0.15) return Math.min(3, Math.floor(v * 4));
        let h;
        if (mx === r) h = ((g - b) / (mx - mn) + 6) % 6; else if (mx === g) h = (b - r) / (mx - mn) + 2; else h = (r - g) / (mx - mn) + 4;
        return 4 + Math.min(11, Math.floor(h * 2)) * 2 + (s > 0.55 ? 1 : 0);
    }

    /** colour histogram of a region of an image/video/canvas (box = [y1,x1,y2,x2] fractions) */
    function regionHist(src, box) {
        const { W, H } = srcSize(src), S = 48;
        const c = document.createElement('canvas'); c.width = S; c.height = S;
        const g = c.getContext('2d', { willReadFrequently: true });
        const [y1, x1, y2, x2] = box;
        g.drawImage(src, x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H, 0, 0, S, S);
        const d = g.getImageData(0, 0, S, S).data, h = new Float32Array(NBIN);
        for (let i = 0; i < S * S; i++) h[colorBin(d[i * 4] / 255, d[i * 4 + 1] / 255, d[i * 4 + 2] / 255)]++;
        return h;
    }

    /** the medicine's own colours = centre of the picture minus its surroundings (table, hand, background) */
    function colorModel(src, inner = [0.2, 0.2, 0.8, 0.8]) {
        try {
            const c = regionHist(src, inner), all = regionHist(src, [0, 0, 1, 1]);
            const ring = all.map((v, i) => Math.max(0, v - c[i] * 0.36));      // approx. the border area
            const cs = c.reduce((a, b) => a + b, 0) || 1, rs = ring.reduce((a, b) => a + b, 0) || 1;
            const m = c.map((v, i) => Math.max(0, v / cs - 0.6 * ring[i] / rs));
            const ms = m.reduce((a, b) => a + b, 0);
            return ms > 0.05 ? Array.from(m, v => v / ms) : Array.from(c, v => v / cs);
        } catch (e) { return null; }
    }

    async function ensureColor(m) {
        if (m.color || m._colorLoading || !m.image) return;
        m._colorLoading = true;
        try {
            const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = m.image; });
            m.color = colorModel(img);
        } catch (e) { /* no colour hint for this one */ }
        m._colorLoading = false;
    }

    /**
     * Candidate places for a medicine: pixels whose colour is common in the medicine but rare in the
     * rest of the room (ratio back-projection), grouped into blobs -> tight boxes [y1,x1,y2,x2].
     */
    function colorProposals(m, f, max = 4) {
        if (!m.color || !f) return [];
        const frameHist = new Float32Array(NBIN), bins = new Uint8Array(GN);
        for (let i = 0; i < GN; i++) { const b = colorBin(f.r[i], f.g[i], f.b[i]); bins[i] = b; frameHist[b]++; }
        for (let i = 0; i < NBIN; i++) frameHist[i] /= GN;
        const w = m.color.map((v, i) => Math.min(1, v / (frameHist[i] * 6 + 0.002)));
        const prob = new Float32Array(GN);
        for (let i = 0; i < GN; i++) prob[i] = w[bins[i]];
        // 3x3 smoothing
        const sm = new Float32Array(GN);
        for (let y = 1; y < GH - 1; y++) for (let x = 1; x < GW - 1; x++) {
            let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += prob[(y + dy) * GW + x + dx];
            sm[y * GW + x] = s / 9;
        }
        let mx = 0; for (let i = 0; i < GN; i++) if (sm[i] > mx) mx = sm[i];
        if (mx < 0.25) return [];
        const thr = Math.max(0.25, mx * 0.55), seen = new Uint8Array(GN), blobs = [];
        for (let i = 0; i < GN; i++) {
            if (seen[i] || sm[i] < thr) continue;
            let x1 = GW, y1 = GH, x2 = 0, y2 = 0, mass = 0; const stack = [i]; seen[i] = 1;
            while (stack.length) {
                const p = stack.pop(), x = p % GW, y = (p / GW) | 0;
                mass += sm[p]; if (x < x1) x1 = x; if (x > x2) x2 = x; if (y < y1) y1 = y; if (y > y2) y2 = y;
                for (const q of [p - 1, p + 1, p - GW, p + GW]) {
                    if (q < 0 || q >= GN || seen[q] || sm[q] < thr) continue;
                    if ((p % GW === 0 && q === p - 1) || (p % GW === GW - 1 && q === p + 1)) continue;
                    seen[q] = 1; stack.push(q);
                }
            }
            const bw = x2 - x1 + 1, bh = y2 - y1 + 1;
            if (mass < 4 || bw * bh > GN * 0.25) continue;
            // square-ish box around the blob with some margin (MobileNet sees a 224x224 square)
            const side = Math.max(bw, bh) * 1.5 + 4, cx = (x1 + x2 + 1) / 2, cy = (y1 + y2 + 1) / 2;
            blobs.push({ mass, box: [Math.max(0, (cy - side / 2) / GH), Math.max(0, (cx - side / 2) / GW),
                                     Math.min(1, (cy + side / 2) / GH), Math.min(1, (cx + side / 2) / GW)] });
        }
        return blobs.sort((a, b) => b.mass - a.mass).slice(0, max).map(b => b.box);
    }

    function boxIou(a, b) {
        const y1 = Math.max(a[0], b[0]), x1 = Math.max(a[1], b[1]), y2 = Math.min(a[2], b[2]), x2 = Math.min(a[3], b[3]);
        if (y2 <= y1 || x2 <= x1) return 0;
        const i = (y2 - y1) * (x2 - x1), ar = r => (r[2] - r[0]) * (r[3] - r[1]);
        return i / (ar(a) + ar(b) - i);
    }

    /** the windows to check this round: around last sightings + colour hints + a rotating slice of the grid */
    /**
     * Fall detection always comes first. Medicine recognition runs on the same graphics chip,
     * so it pauses while a fall is being confirmed / the person is on the floor, and whenever
     * the fall camera's frame rate drops below 70 % of its normal rate (or under 8 FPS).
     */
    function fallBusy() {
        const s = (typeof _tfjsSession !== 'undefined') ? _tfjsSession : null;
        if (!s || !s.isRunning) return false;
        if (s.isFallen || (s.detector && s.detector.state !== 'normal')) return true;
        const fps = s.fps || 0;
        if (!fps) return true;
        // remember the normal frame rate (measured when the medicine scanner is resting)
        if (!MED.scanning && Date.now() - (MED.lastScan || 0) > 1500) st.fpsBase = Math.max((st.fpsBase || 0) * 0.995, fps);
        return fps < 8 || (st.fpsBase && fps < 0.7 * st.fpsBase);
    }

    function candidateBoxes(meds, f, capOverride = 0) {
        const W = st.video.videoWidth, H = st.video.videoHeight, out = [];
        const add = b => { if (b && b[2] - b[0] > 0.04 && b[3] - b[1] > 0.04 && !out.some(o => boxIou(o, b) > 0.7)) out.push(b); };
        for (const m of meds) {
            const rt = m.rt;
            // re-check where it was seen — also right after a FIRST hit, so the 2nd hit can confirm it
            if (rt && rt.box && (Date.now() - rt.lastSeen < 60000 || Date.now() - (rt.hitAt || 0) < 15000)) {
                const cy = (rt.box.y + rt.box.h / 2) / H, cx = (rt.box.x + rt.box.w / 2) / W, hh = rt.box.h / H / 2, hw = rt.box.w / W / 2;
                for (const k of [1, 0.75, 1.35]) add([Math.max(0, cy - hh * k), Math.max(0, cx - hw * k), Math.min(1, cy + hh * k), Math.min(1, cx + hw * k)]);
            }
            // zoom in on a good guess: tighter and shifted crops around it fit the medicine better
            const c = rt && rt.cand;
            if (c && Date.now() - c.at < 6000 && c.pos >= c.need - GUESS_GAP) {
                const cy = (c.box.y + c.box.h / 2) / H, cx = (c.box.x + c.box.w / 2) / W, hh = c.box.h / H / 2, hw = c.box.w / W / 2;
                for (const k of [1, 0.45, 0.65, 0.85, 1.25]) add([Math.max(0, cy - hh * k), Math.max(0, cx - hw * k), Math.min(1, cy + hh * k), Math.min(1, cx + hw * k)]);
                for (const [dx, dy] of [[-0.3, 0], [0.3, 0], [0, -0.3], [0, 0.3]]) {
                    const ny = cy + dy * hh * 2, nx = cx + dx * hw * 2, k = 0.6;
                    add([Math.max(0, ny - hh * k), Math.max(0, nx - hw * k), Math.min(1, ny + hh * k), Math.min(1, nx + hw * k)]);
                }
            }
            colorProposals(m, f).forEach(add);
        }
        const hot = USE_HOT_ZOOM ? (MED.hot || []).filter(h => Date.now() - h.at < 5000).slice(0, 2) : [];
        for (const h of hot) {
            const [y1, x1, y2, x2] = h.box, bh = y2 - y1, bw = x2 - x1;
            for (const k of [0.5, 0.35]) {                              // half and a third of the window
                const sh = bh * k, sw = bw * k;
                for (const fy of [0, 0.5, 1]) for (const fx of [0, 0.5, 1]) {
                    if (k === 0.35 && (fy !== 0.5 || fx !== 0.5) && (fy === fx || fy + fx === 1)) continue;   // fewer tiny ones
                    const ty = y1 + (bh - sh) * fy, tx = x1 + (bw - sw) * fx;
                    add([ty, tx, Math.min(1, ty + sh), Math.min(1, tx + sw)]);
                }
            }
        }
        if (!MED.grid || MED.grid.w !== W) {
            // shuffled sweep order: every part of the room (table, bed, floor) gets looked at early,
            // instead of going row by row from the top of the picture
            const g = scanWindows();
            for (let i = g.length - 1; i > 0; i--) { const j = (i * 7919 + 13) % (i + 1); [g[i], g[j]] = [g[j], g[i]]; }
            MED.grid = g; MED.grid.w = W; MED.gridPos = 0;
        }
        // medicine is often in the patient's hand: look around each person's hands / lap first
        for (const p of st.detections.filter(d => d.class === 'person').slice(0, 2)) {
            const [px, py, pw, ph] = p.bbox, s = Math.max(0.08, Math.min(0.3, (ph / H) * 0.28));
            for (const k of [1, 0.55]) {                                   // a pack in the hand is small
                const spots = k === 1 ? [[0.15, 0.55], [0.85, 0.55], [0.5, 0.65], [0.5, 0.45]]
                    // small windows: a 3x3 net over hands / waist / lap (edges of the body included)
                    : [[0.2, 0.4], [0.5, 0.4], [0.8, 0.4], [0.2, 0.55], [0.5, 0.55], [0.8, 0.55], [0.2, 0.7], [0.5, 0.7], [0.8, 0.7]];
                for (const [fx, fy] of spots) {
                    const cx = (px + pw * fx) / W, cy = (py + ph * fy) / H, hw = s * k * H / W / 2, hh = s * k / 2;
                    add([Math.max(0, cy - hh), Math.max(0, cx - hw), Math.min(1, cy + hh), Math.min(1, cx + hw)]);
                }
            }
        }
        // sweep the rest of the room: faster when the fall camera has spare capacity
        const s = (typeof _tfjsSession !== 'undefined') ? _tfjsSession : null;
        const roomy = !s || !s.isRunning || !st.fpsBase || (s.fps || 0) >= 0.85 * st.fpsBase;
        // worker (CPU, parallel to fall detection): fill up to capOverride windows per pass
        const cap = capOverride || (roomy ? 32 : 16);
        const gridN = capOverride ? Math.max(0, capOverride - out.length) : (roomy ? 20 : 8);
        for (let k = 0; k < gridN && out.length < cap; k++) { add(MED.grid[MED.gridPos % MED.grid.length]); MED.gridPos++; }
        return out.slice(0, cap);
    }

    // ── background worker: medicine recognition on the CPU, in parallel with fall detection ──
    // Fall detection (MoveNet) uses the graphics chip in the page; the worker uses WebAssembly
    // on the CPU, so the two never wait for each other and medicine search runs all the time.
    const WK = { worker: null, ready: false, failed: false, seq: 0, pending: new Map(), canvas: null };

    function startWorker() {
        if (WK.worker || WK.failed || typeof Worker === 'undefined') return;
        try {
            WK.worker = new Worker('med_worker.js?v=3');
            WK.worker.onmessage = ev => {
                const m = ev.data || {};
                if (m.type === 'tick') {
                    // hidden tab: page timers are throttled, so the worker's heartbeat drives the search
                    if (st.running && typeof document !== 'undefined' && document.hidden) analyse();
                    return;
                }
                if (m.type === 'ready') {
                    WK.ready = true;
                    WK.worker.postMessage({ type: 'heartbeat', ms: 1000 });
                    addLog('medicine', `Medicine search running in the background (${m.backend}) — fall detection keeps full speed`, 'secondary');
                } else if (m.type === 'embeds' || m.type === 'classes' || (m.type === 'error' && m.id)) {
                    const p = WK.pending.get(m.id);
                    if (!p) return;
                    WK.pending.delete(m.id);
                    if (m.type === 'error') return p.reject(new Error(m.message));
                    if (m.type === 'classes') return p.resolve(m.classes || []);
                    const flat = new Float32Array(m.data), out = [];
                    for (let i = 0; i < flat.length; i += m.dim) out.push(flat.subarray(i, i + m.dim));
                    p.resolve(out);
                } else if (m.type === 'error') {
                    WK.failed = true; WK.ready = false;
                    console.warn('[TableWatch] worker unavailable, using the page instead:', m.message);
                }
            };
            WK.worker.onerror = e => { WK.failed = true; WK.ready = false; console.warn('[TableWatch] worker error', e.message); };
            WK.worker.postMessage({
                type: 'init',
                modelUrl: new URL('models/mobilenet_v2_050/model.json', location.href).href,
                wasmBase: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.17.0/dist/',
            });
        } catch (e) { WK.failed = true; console.warn('[TableWatch] could not start worker', e); }
    }

    /** Fingerprints of boxes in the current camera frame, computed in the worker. */
    function embedInWorker(boxes) {
        const v = st.video, W = v.videoWidth, H = v.videoHeight;
        if (!WK.canvas) WK.canvas = document.createElement('canvas');
        const c = WK.canvas; if (c.width !== W) { c.width = W; c.height = H; }
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(v, 0, 0, W, H);
        const img = g.getImageData(0, 0, W, H);
        const id = ++WK.seq;
        return new Promise((resolve, reject) => {
            WK.pending.set(id, { resolve, reject });
            setTimeout(() => { if (WK.pending.has(id)) { WK.pending.delete(id); reject(new Error('worker timeout')); } }, 8000);
            WK.worker.postMessage({ type: 'embed', id, w: W, h: H, data: img.data.buffer, boxes }, [img.data.buffer]);
        });
    }

    // ── learning from the camera: confident sightings become extra references ──
    // An uploaded photo looks different from the room camera. Once the medicine is found with
    // high confidence, that real camera view is kept as an extra reference, so later matches
    // score much higher (typically 80-95 %) and are found faster.
    const CAM_KEY = 'tableWatchCamRefsV1';
    const MAX_CAM_REFS = 12;
    function camRefs() { try { return JSON.parse(localStorage.getItem(CAM_KEY) || '{}'); } catch (e) { return {}; } }
    function loadCamRefs(m) {
        if (m.camProtos) return m.camProtos;
        const all = camRefs(), key = String(m.id || m.name);
        m.camProtos = (all[key] || []).map(a => Float32Array.from(a));
        return m.camProtos;
    }
    function saveCamRefs(m) {
        try {
            const all = camRefs();
            all[String(m.id || m.name)] = m.camProtos.map(p => Array.from(p, v => +v.toFixed(4)));
            localStorage.setItem(CAM_KEY, JSON.stringify(all));
        } catch (e) { /* storage full / unavailable: keep in memory */ }
    }
    function maybeLearnView(m, e, pos, margin, thr) {
        const cams = loadCamRefs(m);
        if (pos < thr.pos + 0.05 || margin < thr.margin + 0.06) return;          // only very sure sightings
        if (cams.some(p => dot(p, e) > 0.93)) return;                              // already have this view
        cams.push(Float32Array.from(e));
        if (cams.length > MAX_CAM_REFS) cams.shift();
        saveCamRefs(m);
    }
    const refsOf = m => m.protos.concat(loadCamRefs(m), m.roomViews || []);

    /**
     * An UPLOADED photo is sharp, close-up and on another background; in the room camera the same strip
     * is small, soft and lies on the bed / table, so photo and camera view match only ~45-55 %.
     * Teaching with the camera works because it captures exactly that look. This builds the same kind
     * of references from the photo: the photo is shrunk to the size it would have in the camera, softened
     * like a webcam picture, and placed on real patches of the current room picture (a few sizes and
     * angles). Built when the camera starts (kept in memory for this session).
     */
    async function buildRoomViews(m) {
        if (!m || m.source === 'camera' || !m.image || (m.roomViews && m.roomViews.length)) return;
        if (!st.running || !st.video || !st.video.videoWidth) return;
        if (!MED.net && !MED.netFailed) await loadNet();
        if (!MED.net) return;
        let img;
        try {
            img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = m.image; });
        } catch (e) { return; }
        const W = st.video.videoWidth, H = st.video.videoHeight;
        const frame = document.createElement('canvas'); frame.width = W; frame.height = H;
        frame.getContext('2d').drawImage(st.video, 0, 0, W, H);
        const T = 224, cols = 4, rows = 4, n = cols * rows;
        const mos = document.createElement('canvas'); mos.width = T * cols; mos.height = T * rows;
        const g = mos.getContext('2d');
        const wins = scanWindows().filter(b => (b[2] - b[0]) <= 0.45);
        const ar = (img.naturalWidth || 1) / (img.naturalHeight || 1);
        const boxes = [];
        for (let k = 0; k < n; k++) {
            const [y1, x1, y2, x2] = wins[(k * 7919 + 13) % wins.length];
            const sx = x1 * W, sy = y1 * H, sw = (x2 - x1) * W, sh = (y2 - y1) * H;
            const ox = (k % cols) * T, oy = Math.floor(k / cols) * T;
            g.drawImage(frame, sx, sy, sw, sh, ox, oy, T, T);                         // real room background
            const share = [0.5, 0.62, 0.75, 0.88][k % 4];                              // how much of the window it fills
            const ang = [0, 15, -15, 90, 180, 30, -30, 0][k % 8] * Math.PI / 180;
            let pw = T * share, ph = pw / ar;
            if (ph > T * share) { ph = T * share; pw = ph * ar; }
            // render at the pixel size it really has in the camera, then scale up = soft like a webcam
            const realW = Math.max(12, Math.round(pw * sw / T)), realH = Math.max(8, Math.round(realW / ar));
            const small = document.createElement('canvas'); small.width = realW; small.height = realH;
            const sg = small.getContext('2d');
            sg.filter = ['saturate(0.8) contrast(0.88) brightness(0.95)', 'saturate(0.7) contrast(0.8) brightness(1.05)',
                         'saturate(0.9) contrast(0.95) brightness(0.85)'][k % 3];
            sg.drawImage(img, 0, 0, realW, realH);
            g.save();
            g.beginPath(); g.rect(ox, oy, T, T); g.clip();
            g.translate(ox + T / 2 + ((k % 2) ? 10 : -10), oy + T / 2 + ((k % 3) - 1) * 10);
            g.rotate(ang);
            g.imageSmoothingEnabled = true;
            g.drawImage(small, -pw / 2, -ph / 2, pw, ph);
            g.restore();
            boxes.push([oy / mos.height, ox / mos.width, (oy + T) / mos.height, (ox + T) / mos.width]);
        }
        try {
            m.roomViews = await embed(boxes, mos);
            addLog('medicine', `"${m.name}": prepared ${m.roomViews.length} room views from the photo — it can be recognised like a camera-taught medicine now.`, 'secondary');
        } catch (e) { console.warn('[TableWatch] room views failed', e); }
    }
    async function buildAllRoomViews() {
        for (const m of usableTaught()) { await buildRoomViews(m); if (typeof tf !== 'undefined' && tf.nextFrame) await tf.nextFrame(); }
    }

    /** Remove learned camera views that look like the empty room (wrongly learned background). */
    // Places the caretaker marked as "Wrong place" for a medicine (pillow, phone...): never that medicine
    const WRONG_KEY = 'tableWatchWrongV1';
    function loadWrong(m) {
        if (m.wrong) return m.wrong;
        try {
            const all = JSON.parse(localStorage.getItem(WRONG_KEY) || '{}');
            m.wrong = (all[String(m.id || m.name)] || []).map(a => Float32Array.from(a));
        } catch (e) { m.wrong = []; }
        return m.wrong;
    }
    function saveWrong(m) {
        try {
            const all = JSON.parse(localStorage.getItem(WRONG_KEY) || '{}');
            all[String(m.id || m.name)] = m.wrong.slice(-20).map(p => Array.from(p, v => +v.toFixed(4)));
            localStorage.setItem(WRONG_KEY, JSON.stringify(all));
        } catch (e) { /* storage unavailable: keep in memory */ }
    }
    const photoSim = (m, e) => { let b = -1; for (const p of m.protos) { const v = dot(e, p); if (v > b) b = v; } return b; };

    /** "Wrong place": the box is not on the medicine. Unlearn that look and search again. */
    function notHere(i) {
        const m = MED.taught[i];
        if (!m || !m.rt) return;
        const rt = m.rt, e = rt.boxEmb || rt.candEmb;
        if (e) {
            loadWrong(m).push(Float32Array.from(e)); saveWrong(m);
            const cams = loadCamRefs(m);
            const keep = cams.filter(p => dot(p, e) < 0.85);           // forget learned views that look like that spot
            if (keep.length < cams.length) { m.camProtos = keep; saveCamRefs(m); }
        }
        rt.present = false; rt.conf = 0; rt.box = null; rt.hits = []; rt.steady = 0; rt.cand = null; rt.candEmb = null;
        rt.absentChecks = 0; rt.boxEmb = null;
        st.spots = st.spots.filter(sp => !(sp.learned && sp.medKey === m));
        addLog('medicine', `Okay — that was not "${m.name}". The camera will not pick that spot again and keeps looking.`, 'info');
        renderTaught();
    }

    function pruneBackgroundViews() {
        if (!MED.roomNegs || !MED.roomNegs.length) return;
        for (const m of MED.taught) {
            const cams = loadCamRefs(m);
            // drop learned views that look like the empty room, or that look nothing like the medicine photo
            const keep = cams.filter(p => !MED.roomNegs.some(n => dot(p, n) > 0.9) && (!USE_PHOTO_CHECK || !m.protos.length || photoSim(m, p) >= 0.30));
            if (keep.length < cams.length) {
                m.camProtos = keep; saveCamRefs(m);
                console.log(`[TableWatch] "${m.name}": removed ${cams.length - keep.length} learned view(s) that matched the room background`);
            }
        }
    }

    /**
     * A real guess: the photo match is decent (>= ~42 %) AND clearly ahead of the room's own
     * "not medicine" samples. A photo of a strip often scores only 45-55 % against the camera
     * view, so the lead over the background matters more than the raw number. Repeated on the
     * same spot it is saved as "probably here" (2 scans), accepted (3) and learned (5) — after
     * learning, that camera view scores 85-100 %.
     */
    // Kept tight: a static background patch (bed sheet, curtain) at ~50 % used to win "steady" and
    // be accepted as the medicine. A guess must now be within 10 points of the threshold.
    const GUESS_GAP = 0.22;          // as in the version that recognised well
    const USE_PHOTO_CHECK = false;   // later experiment (match must resemble the photo): off, it blocked real sightings
    const USE_HOT_ZOOM = false;      // later experiment (extra zoom windows): off

    /**
     * Confidence shown to people (0-99 %), calibrated so that 80 % = "found":
     * the raw fingerprint similarity of a medicine strip in a room picture is only ~0.6-0.75 even when
     * it is certainly there, so the raw number (e.g. "50 % match") looks wrong to a caretaker.
     *  - accepted on a strong match:  80 % at the threshold -> 99 % at threshold + 0.2
     *  - accepted after 4+ steady scans on the same spot: 80-88 % (more scans = surer)
     *  - a guess that is not accepted yet: 40-79 %
     */
    function confidencePct(pos, thr, opts = {}) {
        if (pos == null || isNaN(pos)) return 0;
        if (pos >= thr.pos) return Math.round(80 + 19 * Math.min(1, (pos - thr.pos) / 0.2));
        if (opts.accepted) return Math.round(80 + 2 * Math.min(4, Math.max(0, (opts.steady || 4) - 4)));
        const lo = thr.pos - GUESS_GAP;
        return Math.max(0, Math.min(79, Math.round(40 + 39 * (pos - lo) / GUESS_GAP)));
    }
    function isLikely(best, thr) {
        return best.pos >= thr.pos - GUESS_GAP && best.pos - Math.max(0, best.neg) >= thr.margin + 0.04;
    }

    /** Keep a camera view of the medicine as an extra reference (no confidence check). */
    function learnView(m, e) {
        const cams = loadCamRefs(m);
        if (!e || cams.some(p => dot(p, e) > 0.93)) return false;
        cams.push(Float32Array.from(e));
        if (cams.length > MAX_CAM_REFS) cams.shift();
        saveCamRefs(m);
        return true;
    }

    /** "Yes, this is it": the caretaker confirms the camera's guess for medicine i. */
    function confirmGuess(i) {
        const m = MED.taught[i];
        if (!m || !m.rt || !m.rt.candEmb || !m.rt.cand) return;
        learnView(m, m.rt.candEmb);
        m.rt.box = m.rt.cand.box; m.rt.hitAt = Date.now(); m.rt.hits = [true, true]; m.rt.boxEmb = m.rt.candEmb;
        const loc = describeLocation(m.rt.box); m.rt.loc = withPlace(m.rt, loc.en, 'en'); m.rt.locHi = withPlace(m.rt, loc.hi, 'hi');
        // it is here now: show it as found and tell the voice assistant right away
        const now = Date.now();
        m.rt.present = true; m.rt.lastSeen = now; m.rt.absentChecks = 0; m.rt.sim = Math.max(m.rt.sim || 0, 0.9);
        m.rt.conf = 95;                                          // confirmed by a person
        let spot = st.spots.find(s => s.learned && s.medKey === m);
        if (!spot) { spot = { id: st.nextId++, kind: 'medicine', learned: true, medKey: m, auto: true, state: 'on table', addedAt: now }; st.spots.push(spot); }
        spot.rect = m.rt.box; spot.g = toGrid(m.rt.box); spot.name = m.name; spot.goneSince = null;
        m.rt.reportedAt = 0; reportSeen(m);
        refinePlace(m, now);
        addLog('medicine', `Thanks — "${m.name}" confirmed ${m.rt.loc}. The camera now knows how it looks here, and the voice assistant knows where it is.`, 'success');
        renderTaught();
    }

    // ── what is the medicine lying on? (pillow, blanket, table, shelf…) ────
    // The recognition model knows 1000 everyday things; the area around the medicine tells the surface.
    const SURFACES = [
        [/^pillow/, 'on the pillow', 'takiye par'],
        [/^quilt|comforter/, 'on the blanket', 'razai / kambal par'],
        [/four-poster|studio couch|day bed|^crib|cradle/, 'on the bed', 'bed par'],
        [/^dining table/, 'on the table', 'table par'],
        [/^desk/, 'on the desk', 'desk par'],
        [/^tray/, 'in a tray', 'tray mein'],
        [/^plate/, 'on a plate', 'plate mein'],
        [/^bookcase|^shelf/, 'on the shelf', 'shelf par'],
        [/chiffonier|commode/, 'on the dresser', 'dresser par'],
        [/wardrobe|closet/, 'near the cupboard', 'almari ke paas'],
        [/medicine chest|medicine cabinet/, 'in the medicine cabinet', 'dawai ki almari mein'],
        [/rocking chair|folding chair|barber chair|throne/, 'on the chair', 'kursi par'],
        [/doormat|prayer rug/, 'on the floor mat', 'zameen par daree par'],
        [/table lamp/, 'next to the lamp', 'lamp ke paas'],
        [/refrigerator|icebox/, 'near the fridge', 'fridge ke paas'],
        [/microwave/, 'near the microwave', 'microwave ke paas'],
        [/window shade|window screen/, 'near the window', 'khidki ke paas'],
        [/^sleeping bag|bath towel/, 'on the cloth / towel', 'kapde par'],
    ];

    function withPlace(rt, base, lang) {
        const p = rt.place && Date.now() - rt.place.at < 120000 ? rt.place : null;
        if (!p) return base;
        const txt = lang === 'hi' ? p.hi : p.en;
        // "on the pillow" + "on the bed, left side" -> "on the pillow, on the bed, left side"
        return base.toLowerCase().includes(txt.toLowerCase()) ? base : `${txt}, ${base}`;
    }

    async function refinePlace(m, now) {
        const rt = m.rt;
        if (!WK.ready || !rt.box || WK.classifying || now - (rt.placeAt || 0) < 10000) return;
        rt.placeAt = now; WK.classifying = true;
        try {
            const v = st.video, W = v.videoWidth, H = v.videoHeight;
            const cx = (rt.box.x + rt.box.w / 2) / W, cy = (rt.box.y + rt.box.h / 2) / H;
            const s = Math.min(0.9, Math.max(rt.box.w / W, rt.box.h / H) * 3) / 2;    // 3x around the medicine
            const box = [Math.max(0, cy - s), Math.max(0, cx - s * H / W), Math.min(1, cy + s), Math.min(1, cx + s * H / W)];
            if (!WK.canvas) WK.canvas = document.createElement('canvas');
            const c = WK.canvas; if (c.width !== W) { c.width = W; c.height = H; }
            const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(v, 0, 0, W, H);
            const img = g.getImageData(0, 0, W, H), id = ++WK.seq;
            const classes = await new Promise((resolve, reject) => {
                WK.pending.set(id, { resolve, reject, classify: true });
                setTimeout(() => { if (WK.pending.has(id)) { WK.pending.delete(id); reject(new Error('timeout')); } }, 8000);
                WK.worker.postMessage({ type: 'classify', id, w: W, h: H, data: img.data.buffer, box }, [img.data.buffer]);
            });
            for (const c2 of classes) {
                if (c2.probability < 0.06) continue;
                const hitS = SURFACES.find(([re]) => re.test(c2.className));
                if (hitS) {
                    const changed = !rt.place || rt.place.en !== hitS[1];
                    rt.place = { en: hitS[1], hi: hitS[2], at: now };
                    if (changed) {
                        const loc = describeLocation(rt.box); rt.loc = withPlace(rt, loc.en, 'en'); rt.locHi = withPlace(rt, loc.hi, 'hi');
                        reportSeen(m);
                    }
                    break;
                }
            }
        } catch (e) { /* place is optional */ }
        WK.classifying = false;
    }

    // ── scan the room for every medicine ───────────────────────────────────
    async function learnedMedicineTick(now, f) {
        if (!MED.net && !MED.netFailed) { await loadNet(); syncNegs(); renderTaught(); }
        const meds = usableTaught();
        // photo-taught medicine without its room views yet (list just loaded / reloaded): build them
        if (Date.now() - (MED.roomViewsAt || 0) > 15000 && meds.some(m => m.source !== 'camera' && m.image && !(m.roomViews && m.roomViews.length))) {
            MED.roomViewsAt = Date.now();
            buildAllRoomViews().catch(e => console.warn('[TableWatch] room views', e));
        }
        if (!meds.length) return;
        if (!WK.worker && !WK.failed && embedMode() === 'mobilenet') startWorker();
        const useWorker = WK.ready && embedMode() === 'mobilenet';
        // worker: scan continuously (about once a second); page fallback: every 2 s and only when the fall camera has room
        const gap = useWorker ? 700 : SCAN_EVERY_MS;
        if (MED.teach || now - MED.lastScan < gap || MED.scanning) return;
        // the background search keeps running while you use another tab (e.g. the voice assistant);
        // only the in-page fallback (graphics chip) rests in a hidden tab
        if (!useWorker && typeof document !== 'undefined' && document.hidden) return;
        if (!useWorker && fallBusy()) {
            meds.forEach(m => { if (m.rt && m.rt.present) m.rt.lastSeen = Math.max(m.rt.lastSeen, now - 1000); });
            MED.paused = true;
            return;
        }
        MED.paused = false;
        MED.lastScan = now;
        meds.forEach(m => { m.rt = m.rt || newRt(); ensureColor(m); });
        const boxes = candidateBoxes(meds, f, useWorker ? 40 : 0);
        if (!boxes.length) return;
        MED.scanning = true;
        let embs = [];
        try {
            if (useWorker) {
                embs = await embedInWorker(boxes);                 // one batch, off the main thread
            } else {
                for (let i = 0; i < boxes.length; i += 6) {        // page fallback: small batches
                    embs.push(...(await embed(boxes.slice(i, i + 6))));
                    if (i + 6 < boxes.length) {
                        if (typeof tf !== 'undefined' && tf.nextFrame) await tf.nextFrame();
                        if (fallBusy()) break;
                    }
                }
            }
        } catch (e) { console.warn('[TableWatch] scan failed', e); MED.scanning = false; return; }
        MED.scanning = false;
        if (!embs.length || !st.running) return;
        boxes.length = embs.length;
        if (!MED.roomNegs && embs.length) {
            MED.roomNegs = embs.filter(e => meds.every(m => m.protos.every(p => dot(e, p) < 0.55))).slice(0, 20).map(e => Float32Array.from(e));
            syncNegs();
            pruneBackgroundViews();
        }
        const thr = simThreshold();
        const W = st.video.videoWidth, H = st.video.videoHeight;
        const toRect = b => ({ x: b[1] * W, y: b[0] * H, w: (b[3] - b[1]) * W, h: (b[2] - b[0]) * H });
        let bestOverall = 0;
        for (const m of meds) {
            const rt = m.rt || (m.rt = newRt());
            const refs = refsOf(m);
            const wrong = loadWrong(m);
            let best = null, localPos = -1;
            embs.forEach((e, i) => {
                let pos = -1; for (const p of refs) { const s = dot(e, p); if (s > pos) pos = s; }
                let neg = -1; for (const n of MED.negs) { const s = dot(e, n); if (s > neg) neg = s; }
                for (const n of wrong) { const s = dot(e, n); if (s > neg) neg = s; }   // spots marked "Wrong place"
                const score = pos - Math.max(0, neg);
                if (!best || score > best.score) best = { i, pos, neg, score, photo: photoSim(m, e) };
                // how the LAST KNOWN place looks now (for "is it still there?")
                if (rt.box && overlapFrac(rt.box, toRect(boxes[i])) > 0.5 && pos > localPos) localPos = pos;
            });
            // promising windows (resemble the photo, ahead of the room): zoom into them next scan
            const ranked = [];
            embs.forEach((e, i) => {
                const ph = photoSim(m, e);
                if (ph >= thr.pos - 0.25) ranked.push({ i, ph });
            });
            ranked.sort((a, b) => b.ph - a.ph);
            MED.hot = ranked.slice(0, 2).map(r => ({ box: boxes[r.i], at: now, ph: r.ph }))
                .concat((MED.hot || []).filter(h => now - h.at < 5000)).sort((a, b) => b.ph - a.ph).slice(0, 3);
            rt.sim = best ? best.pos : 0;
            bestOverall = Math.max(bestOverall, rt.sim);
            // A match that only comes from a learned camera view must still loosely resemble the medicine
            // photo: a wrongly learned view (pillow, sheet) can no longer win on its own.
            const photoOk = !USE_PHOTO_CHECK || !best || !m.protos.length || best.photo >= thr.pos - 0.22;
            const strictHit = best && photoOk && best.pos >= thr.pos && best.pos - best.neg >= thr.margin;
            // Steady guess: the same spot is the best match scan after scan, clearly ahead of the
            // room's "not medicine" samples -> accept it even if the photo match is a bit lower.
            let steadyHit = false;
            if (best) {
                const cb = toRect(boxes[best.i]);
                const likely = photoOk && isLikely(best, thr);
                const prev = rt.cand && now - rt.cand.at < 6000 ? rt.cand.box : null;
                const sameSpot = prev && Math.hypot((cb.x + cb.w / 2) - (prev.x + prev.w / 2), (cb.y + cb.h / 2) - (prev.y + prev.h / 2))
                                         < 0.6 * Math.max(prev.w, prev.h);
                rt.steady = likely ? (sameSpot ? (rt.steady || 0) + 1 : 1) : 0;
                steadyHit = rt.steady >= 3;                 // 'likely' already requires photo resemblance + lead over the room
                rt.cand = { box: cb, pos: best.pos, need: thr.pos, at: now, steady: rt.steady };
                rt.candEmb = embs[best.i];
            }
            const hit = strictHit || steadyHit;
            rt.hits.push(!!hit); if (rt.hits.length > 3) rt.hits.shift();
            if (hit) {
                rt.hitAt = now;
                rt.box = toRect(boxes[best.i]);
                rt.boxEmb = embs[best.i];
                const loc = describeLocation(rt.box); rt.loc = withPlace(rt, loc.en, 'en'); rt.locHi = withPlace(rt, loc.hi, 'hi');
            }
            const seen = rt.hits.filter(Boolean).length >= 2 || (strictHit && best.pos >= thr.pos + 0.12);   // very sure: no 2nd look needed
            let spot = st.spots.find(s => s.learned && s.medKey === m);
            // not confirmed yet, but the same strong guess twice: tell the voice assistant "probably here"
            // (same level as the yellow "maybe" box on the video, so what you see is what the assistant knows)
            if (!seen && !rt.present && best && rt.steady >= 2 && isLikely(best, thr)) {
                reportProbable(m, rt.cand.box, best.pos);
            }
            if (seen) {
                rt.lastSeen = now; rt.absentChecks = 0;
                const conf = confidencePct(best.pos, thr, { accepted: true, steady: rt.steady });
                rt.conf = rt.present ? Math.max(rt.conf || 0, conf) : conf;
                if (strictHit) maybeLearnView(m, embs[best.i], best.pos, best.pos - best.neg, thr);
                // Same spot 5 scans in a row AND it clearly resembles the medicine photo: learn this view,
                // so from now on it scores high from this distance / angle (no button press needed).
                // The photo check is what keeps a bed sheet / pillow from being learned.
                else if (rt.steady >= 5 && isLikely(best, thr)) learnView(m, embs[best.i]);
                refinePlace(m, now);
                if (!rt.present) {
                    rt.present = true; st.med.takenAt = null;
                    addLog('medicine', `"${m.name}" found — ${rt.loc} (${Math.round(best.pos * 100)}% match)`, 'info');
                    report('medicine', 'baseline', `${m.name} seen ${rt.loc}`);
                }
                if (!spot) {
                    spot = { id: st.nextId++, kind: 'medicine', learned: true, medKey: m, auto: true, state: 'on table', addedAt: now };
                    st.spots.push(spot);
                }
                if (rt.box) { spot.rect = rt.box; spot.g = toGrid(rt.box); }
                spot.name = m.name; spot.goneSince = null;
                reportSeen(m);
            } else if (rt.present) {
                // Keep knowing where it is: only "gone" when its place is clearly visible and empty
                // several times in a row (a person / hand in front of it does not count).
                const blocked = rt.box && personOver(rt.box);
                // never count it missing during a fall (someone may be lying in front of it) or when the
                // "is a person in front of it?" check is out of date
                const s = (typeof _tfjsSession !== 'undefined') ? _tfjsSession : null;
                const fallNow = s && s.isRunning && (s.isFallen || (s.detector && s.detector.state !== 'normal'));
                const peopleFresh = !st.model || now - (st.lastDetect || 0) < 4000;
                if (!blocked && !fallNow && peopleFresh && localPos >= 0 && localPos < thr.pos - 0.08) {
                    rt.absentChecks = (rt.absentChecks || 0) + 1;
                }
                if (spot) spot.goneSince = spot.goneSince || now;
                if ((rt.absentChecks || 0) >= 5 && now - rt.lastSeen >= LEARNED_GONE_MS) {
                    rt.present = false; rt.conf = 0;
                    st.spots = st.spots.filter(s => s !== spot);
                    addLog('medicine', `"${m.name}" no longer at its place (last ${rt.loc})`, 'secondary');
                    if (!st.spots.some(s => s.kind === 'medicine')) medicinesTaken('camera');
                }
            }
        }
        MED.bestSim = bestOverall;
        if (now - (MED.lastListRender || 0) > 3000) { MED.lastListRender = now; renderTaught(); }
    }

    // ── medicine reminder logic ────────────────────────────────────────────
    function currentSlot() {
        const due = settings().medDue;
        if (!due) return null;
        const [h, m] = due.split(':').map(Number);
        const d = new Date(); d.setHours(h, m, 0, 0);
        const meds = (st.schedule.find(s => s.time === due) || {}).meds || [];
        return { key: `${d.toDateString()} ${due}`, due: d, time: due, meds };
    }

    function medicineReminder(now) {
        const slot = currentSlot();
        const onTable = st.spots.filter(s => s.kind === 'medicine').length;
        if (!slot || now < slot.due.getTime() || st.med.doneSlots.has(slot.key)) { hideReminder('medicine'); return; }
        const lateMin = (now - slot.due.getTime()) / 60000;
        if (lateMin > 6 * 60) { hideReminder('medicine'); return; }     // a dose more than 6 h ago is not chased any more

        if (!onTable) {
            // nothing seen on the table at the due time: one gentle spoken reminder only
            if (st.med.nothingSpokenSlot !== slot.key) {
                st.med.nothingSpokenSlot = slot.key;
                remind(`${who()}, ${slot.time} baje ki dawai ka samay ho gaya hai.${medList(slot)}`, false);
                addLog('medicine', `Dose time ${slot.time} — no medicines seen on the table`, 'warning');
            }
            hideReminder('medicine');
            return;
        }

        showReminder('medicine', 'Medicine time — please take it now!',
            `${slot.time} dose${slot.meds.length ? ': ' + slot.meds.join(', ') : ''} · still on the table · ${Math.round(lateMin)} min since due`);

        if (now >= st.med.snoozeUntil && now - st.med.lastRemindAt >= settings().repeatSec * 1000) {
            st.med.lastRemindAt = now;
            const urgent = lateMin >= settings().medGrace;
            remind(`${who()}, dawai lene ka samay ho gaya hai.${medList(slot)} Kripya abhi dawai le lijiye.`, urgent);
            addLog('medicine', `Reminder: medicines still on the table (${Math.round(lateMin)} min after ${slot.time})`, urgent ? 'danger' : 'warning');
        }
        if (lateMin >= settings().medGrace && st.med.escalatedSlot !== slot.key) {
            st.med.escalatedSlot = slot.key;
            const txt = `Medicines still on the table ${Math.round(lateMin)} min after the ${slot.time} dose${slot.meds.length ? ' (' + slot.meds.join(', ') + ')' : ''}.`;
            report('medicine', 'untouched_alert', txt, { idle_minutes: Math.round(lateMin) });
            addLog('medicine', 'Caretaker alerted: ' + txt, 'danger');
        }
    }

    function medicinesTaken(how) {
        const slot = currentSlot();
        const now = Date.now();
        st.med.takenAt = now;
        if (slot && now >= slot.due.getTime() - 2 * 3600000) st.med.doneSlots.add(slot.key);
        hideReminder('medicine');
        try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
        const txt = how === 'camera' ? 'Medicines are no longer on the table — taken ✓' : 'Marked as taken by the caretaker ✓';
        addLog('medicine', txt, 'success');
        report('medicine', 'taken', how === 'camera' ? 'Medicines picked up from the table (camera)' : 'Marked taken by caretaker', { confidence: how === 'camera' ? 80 : 100 });
        if (how !== 'camera') st.spots = st.spots.filter(s => s.kind !== 'medicine');
        advanceSlot();
    }

    function advanceSlot() {
        // move "Dose due at" to the next time in today's schedule
        const times = st.schedule.map(s => s.time).sort();
        const nowHM = new Date().toTimeString().slice(0, 5);
        const next = times.find(t => t > nowHM);
        if (next && $('twMedDue')) $('twMedDue').value = next;
    }

    function waterReminder(now) {
        if (!WATER_TRACKING) { hideReminder('water'); return; }
        const w = st.spots.find(s => s.kind === 'water');
        if (!w || w.state === 'away') { hideReminder('water'); return; }
        const since = Math.max(st.water.filledAt || now, st.water.lastDrinkAt || 0);
        const idleMin = (now - since) / 60000;
        if (idleMin < settings().waterMin) { hideReminder('water'); return; }
        if (!document.getElementById('twReminder') || $('twReminder').dataset.kind !== 'medicine') {
            showReminder('water', 'Time to drink water', `Glass not touched for ${Math.round(idleMin)} min — still full`);
        }
        if (now - st.water.alertedAt >= 15 * 60000) {
            const first = !st.water.alertedAt;
            st.water.alertedAt = now;
            remind(`${who()}, paani pi lijiye. Aapne kaafi der se paani nahi piya hai.`, false);
            addLog('water', `Water not touched for ${Math.round(idleMin)} min — reminder given`, 'danger');
            if (first) report('water', 'untouched_alert', `Water glass not touched for ${Math.round(idleMin)} min — still full.`, { idle_minutes: Math.round(idleMin) });
        }
    }

    const who = () => st.patientName || 'Ji';
    const medList = slot => slot && slot.meds.length ? ` ${slot.meds.join(', ')}.` : '';

    function showReminder(kind, title, text) {
        const b = $('twReminder');
        if (!b) return;
        if (kind === 'water' && b.style.display !== 'none' && b.dataset.kind === 'medicine') return;  // medicine wins
        b.dataset.kind = kind;
        b.className = 'tw-reminder' + (kind === 'water' ? ' water' : '');
        $('twReminderTitle').textContent = title;
        $('twReminderText').textContent = text;
        b.style.display = 'block';
        // same reminder on the Live Monitoring screen (one room camera)
        const l = $('liveMedReminder');
        if (l) {
            l.dataset.kind = kind;
            l.className = 'tw-reminder' + (kind === 'water' ? ' water' : '');
            $('liveMedReminderTitle').textContent = title;
            $('liveMedReminderText').textContent = text;
            l.style.display = 'block';
        }
    }
    function hideReminder(kind) {
        ['twReminder', 'liveMedReminder'].forEach(id => {
            const b = $(id);
            if (b && b.dataset.kind === kind) b.style.display = 'none';
        });
    }

    /**
     * Draw the medicine / water boxes on the Live Monitoring picture (called every frame by
     * the fall-detection session). That picture is mirrored, so x is flipped.
     */
    function paintOverlay(ctx, W, H, mirrored) {
        if (!st.running || !st.video || !st.video.videoWidth) return;
        const sx = W / st.video.videoWidth, sy = H / st.video.videoHeight;
        // Boxes are drawn smaller (70 % around the centre) so they sit on the medicine itself
        const map = r => {
            const k = 0.7, w0 = r.w * k, h0 = r.h * k;
            const x = (r.x + (r.w - w0) / 2) * sx, w = w0 * sx;
            return { x: mirrored ? W - x - w : x, y: (r.y + (r.h - h0) / 2) * sy, w, h: h0 * sy };
        };
        const now = Date.now();
        ctx.save();
        for (const s of st.spots) {
            if (!s.rect) continue;
            // Show the box only for a few seconds when the medicine is found, moves or goes missing;
            // after that the picture stays clean (tracking continues in the background).
            const cx = s.rect.x + s.rect.w / 2, cy = s.rect.y + s.rect.h / 2, gone = !!s.goneSince;
            const f = s._flash;
            if (!f || f.gone !== gone || Math.hypot(cx - f.cx, cy - f.cy) > 0.5 * Math.max(s.rect.w, s.rect.h)) {
                s._flash = { cx, cy, gone, until: now + 5000 };
            }
            if (now > s._flash.until) continue;
            const r = map(s.rect);
            const col = s.kind === 'water' ? '#2f72b8' : '#b0566b';
            const txt = s.kind === 'water' ? (s.state === 'away' ? 'Water: picked up' : 'Water glass')
                                           : (s.learned ? (s.name || 'Medicine') : 'Medicine');
            ctx.setLineDash(s.goneSince ? [6, 4] : []); ctx.lineWidth = 2; ctx.strokeStyle = col;
            ctx.strokeRect(r.x, r.y, r.w, r.h); ctx.setLineDash([]);
            ctx.font = 'bold 13px "Segoe UI", sans-serif';
            const tw = ctx.measureText(txt).width + 10;
            ctx.fillStyle = col; ctx.fillRect(r.x, Math.max(0, r.y - 18), tw, 18);
            ctx.fillStyle = '#fff'; ctx.fillText(txt, r.x + 5, Math.max(13, r.y - 5));
        }
        // "Maybe" boxes: the best guess for each medicine not found yet, with its match score
        for (const m of MED.taught) {
            const c = m.rt && m.rt.cand;
            if (!c || m.rt.present || Date.now() - c.at > 4000 || c.pos < c.need - GUESS_GAP || (c.steady || 0) < 2) continue;
            const r = map(c.box);
            const txt = `${m.name}? ${confidencePct(c.pos, simThreshold())}% (found at 80%)`;
            ctx.setLineDash([5, 5]); ctx.lineWidth = 2; ctx.strokeStyle = '#f5c518';
            ctx.strokeRect(r.x, r.y, r.w, r.h); ctx.setLineDash([]);
            ctx.font = '12px "Segoe UI", sans-serif';
            const tw = ctx.measureText(txt).width + 8;
            ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(r.x, r.y + r.h, tw, 16);
            ctx.fillStyle = '#f5c518'; ctx.fillText(txt, r.x + 4, r.y + r.h + 12);
        }
        if (MED.teach) {
            const s = 0.2 * Math.min(W, H);
            ctx.lineWidth = 4; ctx.strokeStyle = '#fff'; ctx.strokeRect(W / 2 - s, H / 2 - s, 2 * s, 2 * s);
            const left = Math.max(0, Math.ceil((MED.teach.until - Date.now()) / 1000));
            ctx.fillStyle = '#fff'; ctx.font = 'bold 18px "Segoe UI", sans-serif';
            ctx.fillText(left > 0 ? `Hold "${MED.teach.name}" in the box… ${left}` : 'Learning… keep still', W / 2 - s, H / 2 - s - 10);
        }
        ctx.restore();
    }

    // ── drawing ────────────────────────────────────────────────────────────
    function draw() {
        if (!st.running) return;
        const { ctx, canvas, video } = st;
        // This tab's picture is only needed while it is on screen (Live Monitoring draws its own):
        // drawing a hidden 640x480 canvas every frame made the room camera lag.
        if (!canvas.offsetParent || (typeof document !== 'undefined' && document.hidden)) {
            st.raf = null; setTimeout(() => { if (st.running) st.raf = requestAnimationFrame(draw); }, 500);
            return;
        }
        if (video.videoWidth && canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const label = (r, color, text, dashed) => {
            ctx.setLineDash(dashed ? [6, 4] : []); ctx.lineWidth = 3; ctx.strokeStyle = color;
            ctx.strokeRect(r.x, r.y, r.w, r.h); ctx.setLineDash([]);
            ctx.font = 'bold 14px "Segoe UI", sans-serif';
            const w = ctx.measureText(text).width + 10;
            ctx.fillStyle = color; ctx.fillRect(r.x, Math.max(0, r.y - 20), w, 20);
            ctx.fillStyle = '#fff'; ctx.fillText(text, r.x + 5, Math.max(15, r.y - 5));
        };
        for (const c of st.candidates) {
            if (c.first === Infinity) continue;
            label(toVideo(c.g), '#d4a017', 'checking…', true);
        }
        for (const s of st.spots) {
            const touched = s.touched && Date.now() - s.touched < 1500;
            const col = touched ? '#f59e0b' : (s.kind === 'water' ? '#2f72b8' : '#b0566b');
            const txt = s.kind === 'water' ? (s.state === 'away' ? 'Water: picked up' : 'Water glass')
                                           : (s.goneSince ? 'Medicine: gone?' : (s.learned ? (s.name || 'Medicine') : 'Medicine (new object)'));
            label(s.rect, col, touched ? 'hand nearby' : txt, s.goneSince);
        }
        if (st.dragRect) { ctx.setLineDash([6, 4]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.strokeRect(st.dragRect.x, st.dragRect.y, st.dragRect.w, st.dragRect.h); ctx.setLineDash([]); }
        if (MED.teach) {
            // guide box for "Teach medicine"
            const W = canvas.width, H = canvas.height, s = 0.2 * Math.min(W, H);
            ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, W, H);
            ctx.clearRect(W / 2 - s, H / 2 - s, 2 * s, 2 * s);
            ctx.drawImage(video, W / 2 - s, H / 2 - s, 2 * s, 2 * s, W / 2 - s, H / 2 - s, 2 * s, 2 * s);
            ctx.lineWidth = 4; ctx.strokeStyle = '#fff'; ctx.strokeRect(W / 2 - s, H / 2 - s, 2 * s, 2 * s);
            const left = Math.max(0, Math.ceil((MED.teach.until - Date.now()) / 1000));
            ctx.fillStyle = '#fff'; ctx.font = 'bold 22px "Segoe UI", sans-serif';
            ctx.fillText(left > 0 ? `Hold "${MED.teach.name}" in the box… ${left}` : 'Learning… keep still', 16, 34);
        }
        if (st.markMode) {
            ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(0, 0, canvas.width, 34);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 15px "Segoe UI", sans-serif';
            ctx.fillText(`Drag a box around the ${st.markMode === 'water' ? 'WATER GLASS' : 'MEDICINES'}`, 12, 23);
        }
        st.raf = requestAnimationFrame(draw);
    }

    function canvasPoint(ev) {
        const r = st.canvas.getBoundingClientRect(), p = ev.touches ? ev.touches[0] : ev;
        return { x: (p.clientX - r.left) * (st.canvas.width / r.width), y: (p.clientY - r.top) * (st.canvas.height / r.height) };
    }
    let _bound = false;
    function bindDrawing() {
        if (_bound) return; _bound = true;
        const c = st.canvas;
        const down = ev => { if (!st.markMode) return; ev.preventDefault(); st.dragStart = canvasPoint(ev); st.dragRect = null; };
        const move = ev => {
            if (!st.markMode || !st.dragStart) return;
            const p = canvasPoint(ev);
            st.dragRect = { x: Math.min(p.x, st.dragStart.x), y: Math.min(p.y, st.dragStart.y), w: Math.abs(p.x - st.dragStart.x), h: Math.abs(p.y - st.dragStart.y) };
        };
        const up = () => {
            if (!st.markMode || !st.dragRect) { st.dragStart = null; return; }
            const kind = st.markMode, r = st.dragRect;
            st.markMode = null; st.dragStart = null; st.dragRect = null;
            if (r.w < 20 || r.h < 20) { addLog(kind, 'Box too small — drag a bigger box', 'warning'); return; }
            if (kind === 'water') st.spots = st.spots.filter(s => s.kind !== 'water');
            addSpot(kind, r, false, grab(), false);
        };
        c.addEventListener('mousedown', down); c.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
        c.addEventListener('touchstart', down, { passive: false }); c.addEventListener('touchmove', move); c.addEventListener('touchend', up);
    }

    // ── status text ────────────────────────────────────────────────────────
    function renderStatus() {
        const now = Date.now();
        const w = st.spots.find(s => s.kind === 'water');
        const meds = st.spots.filter(s => s.kind === 'medicine');
        if ($('twWaterStatus')) {
            let t;
            if (!w) t = '<span class="text-muted">No glass on the table yet</span>';
            else if (w.state === 'away') t = `<span class="text-warning fw-bold">Glass picked up</span> (${fmtAgo(now - w.awayAt)} ago)`;
            else {
                const since = Math.max(st.water.filledAt || now, st.water.lastDrinkAt || 0), idle = now - since, lim = settings().waterMin * 60000;
                t = (st.water.lastDrinkAt ? `<span class="text-success fw-bold">Drank ${fmtAgo(now - st.water.lastDrinkAt)} ago</span>` : '<span class="fw-bold">Still full — not touched</span>')
                    + ` · ${idle >= lim ? '<span class="text-danger fw-bold">reminder due</span>' : 'reminder in ' + fmtAgo(lim - idle)}`;
            }
            $('twWaterStatus').innerHTML = t;
            if ($('liveWaterStatus')) $('liveWaterStatus').innerHTML = t;
        }
        if ($('twMedStatus')) {
            const slot = currentSlot();
            let t;
            if (meds.length) {
                t = `<span class="fw-bold">${meds.length} medicine item${meds.length > 1 ? 's' : ''} on the table</span>`;
                if (slot) {
                    const late = (now - slot.due.getTime()) / 60000;
                    t += late < 0 ? ` · due at ${slot.time}` : ` · <span class="text-danger fw-bold">${Math.round(late)} min overdue</span>`;
                }
            } else if (st.med.takenAt) {
                t = `<span class="text-success fw-bold">Taken ✓</span> at ${timeStr(new Date(st.med.takenAt))}`;
            } else {
                t = '<span class="text-muted">No medicines seen on the table</span>' + (slot ? ` · next dose ${slot.time}` : '');
            }
            $('twMedStatus').innerHTML = t;
            if ($('liveMedStatus')) $('liveMedStatus').innerHTML = t;
        }
    }

    function setInfo() {
        const el = $('twAutoInfo');
        if (!el) return;
        if (!st.running) { el.textContent = 'Auto-detect: camera off'; return; }
        const c = st.candidates.filter(x => x.first !== Infinity).length;
        el.innerHTML = `<i class="fas fa-wand-magic-sparkles me-1"></i>Auto-detect on · ${st.spots.length} item(s) tracked`
            + (c ? ` · ${c} new item(s) being checked` : '') + (st.model ? '' : ' · (object model loading)')
            + (usableTaught().length ? ` · medicine match ${confidencePct(MED.bestSim, simThreshold())}% (found at 80%)`
                                 : ' · no medicine taught yet')
            + (MED.paused ? ' · <span class="text-warning">medicine search paused — fall detection first</span>' : '')
            + (WK.ready ? ' · <span class="text-success">background search on</span>' : '');
    }

    // ── schedule & patient name ────────────────────────────────────────────
    async function loadSchedule() {
        const rid = recipientId(), token = localStorage.getItem('token');
        const box = $('twSchedule');
        if (!rid || !token) return;
        try {
            const res = await fetch(`${API}/care-recipients/${rid}/profile`, { headers: { 'Authorization': `Bearer ${token}` } });
            if (!res.ok) return;
            const data = await res.json();
            const gi = data.general_info || {};
            st.patientName = (gi.preferred_name || (gi.name || '').split(' ')[0] || '').trim();
            const slots = {};
            (data.medications || []).forEach(md => {
                (String(md.schedule_time || '').match(/\b\d{1,2}:\d{2}\b/g) || []).forEach(t => {
                    const k = t.padStart(5, '0');
                    (slots[k] = slots[k] || []).push(`${md.medicine_name}${md.dosage ? ' ' + md.dosage : ''}`);
                });
            });
            st.schedule = Object.keys(slots).sort().map(t => ({ time: t, meds: slots[t] }));
            if (!box) return;
            if (!st.schedule.length) { box.textContent = 'No timed medicines in the profile.'; return; }
            const nowHM = new Date().toTimeString().slice(0, 5);
            // the dose still being chased (due within the last 2 h) or the next one
            const recent = st.schedule.filter(s => s.time <= nowHM && minutesBetween(s.time, nowHM) <= 120).pop();
            const next = (recent || st.schedule.find(s => s.time > nowHM) || st.schedule[0]).time;
            if ($('twMedDue') && !$('twMedDue').value) $('twMedDue').value = next;
            box.innerHTML = st.schedule.map(s => `<div class="${s.time === next ? 'fw-bold' : ''}">${s.time} — ${s.meds.join(', ')}${s.time === next ? ' (current)' : ''}</div>`).join('');
        } catch (e) { /* ignore */ }
    }
    function minutesBetween(a, b) { const [h1, m1] = a.split(':').map(Number), [h2, m2] = b.split(':').map(Number); return (h2 * 60 + m2) - (h1 * 60 + m1); }

    /** progress text: hidden loader + the visible status line in the Live Monitoring panel */
    function loadingMsg(t) {
        const a = $('twLoadingMsg'); if (a) a.textContent = t;
        const b = $('twAutoInfo'); if (b) b.textContent = t;
    }

    // ── public API ─────────────────────────────────────────────────────────
    // ONE ROOM CAMERA: the Live Monitoring session (fall detection) owns the camera.
    // Medicine & water tracking attaches to the same <video> stream instead of opening
    // a second camera. "Start camera" here simply starts the room camera.
    async function start() {
        if (st.running) return;
        if (typeof _tfjsSession !== 'undefined' && _tfjsSession && _tfjsSession.isRunning) {
            return attach(document.getElementById('tfjs-video'));
        }
        if (typeof startLiveMonitoring === 'function') {
            $('twLoading').style.display = 'flex'; loadingMsg('Starting the room camera (falls + medicines)…');
            await startLiveMonitoring();          // attaches Table Watch itself once the camera is on
            if (!st.running) $('twLoading').style.display = 'none';
            return;
        }
        alert('Room camera is not available on this page.');
    }

    /** Use the room camera's video for medicine & water tracking (called by Live Monitoring). */
    async function attach(videoEl) {
        if (st.running || !videoEl) return;
        st.attached = true;
        st.video = videoEl; st.canvas = $('twCanvas'); st.ctx = st.canvas.getContext('2d');
        $('twStartBtn').classList.add('d-none'); $('twStopBtn').classList.remove('d-none');
        $('twLoading').style.display = 'flex'; loadingMsg('Starting medicine tracking…');
        if (typeof ensureAlarmAudio === 'function') ensureAlarmAudio();
        try { window.speechSynthesis && window.speechSynthesis.getVoices(); } catch (e) { /* warm up voices */ }
        try {
            if (!videoEl.videoWidth) await new Promise(r => { videoEl.addEventListener('loadedmetadata', r, { once: true }); setTimeout(r, 3000); });
            st.running = true; st.bg = null; st.spots = []; st.candidates = [];
            MED.roomNegs = null;                       // re-learn what this room looks like
            draw(); bindDrawing();
            loadingMsg('Loading object detector…');
            try { if (typeof cocoSsd !== 'undefined') st.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' }); }
            catch (e) { console.warn('[TableWatch] COCO-SSD unavailable, using change detection only', e); }
            $('twLoading').style.display = 'none';
            if (!st.schedule.length) loadSchedule();
            loadTaught();
            $('twLoading').style.display = 'flex'; loadingMsg('Loading medicine recognition…');
            await loadNet();
            syncNegs(); renderTaught();
            setTimeout(() => { buildAllRoomViews().catch(e => console.warn('[TableWatch] room views', e)); }, 3000);
            $('twLoading').style.display = 'none';
            if (MED.net) addLog('medicine', 'Medicine recognition ready', 'secondary');
            if (!MED.taught.length) addLog('medicine', 'Tip: press "Teach medicine" and hold each medicine pack in the box once — then it is found automatically.', 'info');
            addLog('medicine', 'Room camera on — watching for falls and for medicines in the room.', 'info');
            st.tick = setInterval(analyse, 1000);
            // medicine boxes on their own transparent layer over the live picture (4x per second)
            if (st.overlayTimer) clearInterval(st.overlayTimer);
            st.overlayTimer = setInterval(drawMedOverlay, 250);
        } catch (e) {
            console.error('[TableWatch] attach failed', e);
            $('twLoading').style.display = 'none';
            detach();
        }
    }

    /** Room camera stopped: stop tracking (the camera itself is owned by Live Monitoring). */
    function drawMedOverlay() {
        const c = $('medOverlay');
        if (!c || !st.video || !st.video.videoWidth) return;
        if (c.width !== st.video.videoWidth) { c.width = st.video.videoWidth; c.height = st.video.videoHeight; }
        const g = c.getContext('2d');
        g.clearRect(0, 0, c.width, c.height);
        if (st.running) paintOverlay(g, c.width, c.height, true);     // the live picture is mirrored
    }

    function detach() {
        st.running = false; st.attached = false;
        if (st.raf) cancelAnimationFrame(st.raf);
        if (st.tick) clearInterval(st.tick);
        if (st.overlayTimer) { clearInterval(st.overlayTimer); st.overlayTimer = null; }
        const oc = $('medOverlay'); if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height);
        try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
        ['twReminder', 'liveMedReminder'].forEach(id => { const b = $(id); if (b) b.style.display = 'none'; });
        const sb = $('twStartBtn'), xb = $('twStopBtn');
        if (sb) sb.classList.remove('d-none');
        if (xb) xb.classList.add('d-none');
        if ($('twLoading')) $('twLoading').style.display = 'none';
        setInfo();
    }

    /** "Stop" in the medicine tab stops the whole room camera (falls + medicines). */
    function stop() {
        if (st.attached && typeof stopLiveMonitoring === 'function') { stopLiveMonitoring(); return; }
        st.running = false;
        if (st.raf) cancelAnimationFrame(st.raf);
        if (st.tick) clearInterval(st.tick);
        if (st.stream) st.stream.getTracks().forEach(t => t.stop());
        st.stream = null;
        try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
        const b = $('twReminder'); if (b) b.style.display = 'none';
        const sb = $('twStartBtn'), xb = $('twStopBtn');
        if (sb) sb.classList.remove('d-none');
        if (xb) xb.classList.add('d-none');
        setInfo();
    }

    function rescan() {
        if (!st.running) { alert('Start the camera first.'); return; }
        st.bg = null; st.spots = []; st.candidates = [];
        addLog('medicine', 'Rescanning — the table is treated as empty now', 'info');
    }

    function mark(kind) { if (kind === 'water' && !WATER_TRACKING) return; if (!st.running) { alert('Start the camera first.'); return; } st.markMode = kind; }

    function confirmBaseline(kind) {
        if (kind === 'water' && !WATER_TRACKING) return;
        if (!st.running) { alert('Start the camera first.'); return; }
        if (kind === 'water') {
            st.water.filledAt = Date.now(); st.water.lastDrinkAt = null; st.water.alertedAt = 0;
            hideReminder('water');
            addLog('water', 'Glass refilled', 'info');
            report('water', 'baseline', 'Glass refilled');
        } else {
            addLog('medicine', 'Place the medicines on the table — they are found automatically', 'info');
        }
    }

    function markTaken() { medicinesTaken('manual'); }

    function snooze(min) {
        st.med.snoozeUntil = Date.now() + min * 60000;
        try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
        addLog('medicine', `Reminder snoozed for ${min} min`, 'secondary');
    }

    document.addEventListener('DOMContentLoaded', () => {
        loadSchedule();
        loadTaught();
        setInterval(() => { if (st.running) renderStatus(); }, 15000);
        // the reminder must also run when the medicine tab is not visible
    });

    /** Beds and sofas the object detector currently sees (video pixel coords, not mirrored). */
    function furniture() {
        if (!st.running || !st.detections) return Object.assign([], { checkedAt: 0 });
        const out = st.detections.filter(d => (d.class === 'bed' || d.class === 'couch') && d.score >= 0.5)
            .map(d => ({ kind: d.class === 'couch' ? 'sofa' : 'bed', x: d.bbox[0], y: d.bbox[1], w: d.bbox[2], h: d.bbox[3], score: d.score }));
        out.checkedAt = st.lastDetect || 0;       // when the object detector last looked at the room
        return out;
    }

    return { start, stop, mark, confirmBaseline, loadSchedule, rescan, markTaken, snooze, furniture, notHere,
             teach, teachFromPhoto, forget, confirmGuess, attach, detach, paintOverlay,
             _test: { st, analyse, MED, colorBin, colorProposals } };   // used by the offline simulation tests only
})();
