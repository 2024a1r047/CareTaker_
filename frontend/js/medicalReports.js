/**
 * Medical Report Management — Consolidated JS
 * Handles: sidebar, tabs, report upload/listing, clinical intelligence,
 * doctor's summary, and all Chart.js visualizations.
 */
(function () {
    'use strict';

    const API = 'http://127.0.0.1:8000/api';
    const token = localStorage.getItem('token');
    if (!token) { window.location.href = 'index.html'; return; }

    // ── State ──
    let recipientsCache = [];
    let selectedRecipientId = null;
    let selectedRecipientName = '';
    let currentReports = [];
    let trendChart = null;
    let timelineChart = null;
    let severityChart = null;

    // Status badge colors/icons
    const STATUS_COLORS = {
        active: '#C96B55', worsening: '#A94F3D', improving: '#91AA8A',
        controlled: '#B8A9C9', resolved: '#6b7280', chronic_stable: '#8b5cf6',
    };
    const STATUS_ICONS = {
        active: '\u26a0\ufe0f', worsening: '\ud83d\udd34', improving: '\u2705',
        controlled: '\ud83d\udfe2', resolved: '\u26aa', chronic_stable: '\ud83d\udd35',
    };

    // ── DOM refs ──
    const $ = id => document.getElementById(id);

    // ── Toast ──
    function showToast(msg, type = 'success') {
        const el = $('toast');
        if (!el) return;
        el.className = type;
        $('toastIcon').textContent = type === 'success' ? '\u2713' : '\u2717';
        $('toastMsg').textContent = msg;
        el.classList.add('show');
        setTimeout(() => el.classList.remove('show'), 3500);
    }

    // ═══════════════════════════════════════
    //  TABS
    // ═══════════════════════════════════════
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const target = $(btn.dataset.tab);
            if (target) target.classList.add('active');
        });
    });

    // ═══════════════════════════════════════
    //  SIDEBAR — Recipients
    // ═══════════════════════════════════════
    async function fetchProfile() {
        try {
            const res = await fetch(API + '/profile', { headers: { 'Authorization': 'Bearer ' + token } });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            recipientsCache = data.care_recipients || [];
            renderRecipients(recipientsCache);
        } catch (e) {
            console.error('Profile fetch error:', e);
            $('recipientsList').innerHTML = '<p class="muted">Unable to load.</p>';
        }
    }

    function renderRecipients(list) {
        const el = $('recipientsList');
        el.innerHTML = '';
        if (!list.length) { el.innerHTML = '<p class="muted">No recipients.</p>'; return; }
        list.forEach(r => {
            const card = document.createElement('div');
            card.className = 'recipient-card';
            card.dataset.id = r.id;
            card.innerHTML = `<h5>${r.full_name || 'Unnamed'}</h5><p>Age: ${r.age || '-'} &middot; ${r.gender || '-'}</p>`;
            card.addEventListener('click', () => selectRecipient(r.id, r.full_name));
            el.appendChild(card);
        });
        highlightActive();
    }

    function highlightActive() {
        document.querySelectorAll('.recipient-card').forEach(c => {
            c.classList.toggle('active', String(c.dataset.id) === String(selectedRecipientId));
        });
    }

    function selectRecipient(id, name) {
        selectedRecipientId = id;
        selectedRecipientName = name || '';
        if (id) {
            localStorage.setItem('selectedRecipientId', id);
        } else {
            localStorage.removeItem('selectedRecipientId');
        }
        $('headerRecipientName').textContent = name || 'All';
        $('btnRunAnalysis').disabled = !id;
        $('btnRefresh').disabled = !id;
        const uploadPanel = $('uploadPanel');
        if (uploadPanel) uploadPanel.style.display = id ? 'block' : 'none';
        highlightActive();
        loadReports();
        loadClinicalData();
    }

    $('recipientSearch')?.addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        renderRecipients(recipientsCache.filter(r => (r.full_name || '').toLowerCase().includes(q)));
    });

    // ═══════════════════════════════════════
    //  TAB 1 — Reports & Upload
    // ═══════════════════════════════════════

    // ── Upload ──
    const fileInput = $('reportFileInput');
    const uploadBtn = $('uploadReportBtn');
    const fileNameEl = $('uploadFileName');
    const progressWrap = $('uploadProgress');
    const progressBar = $('uploadProgressBar');
    const dropZone = $('uploadDropZone');

    fileInput?.addEventListener('change', () => {
        const f = fileInput.files[0];
        if (f && selectedRecipientId) { fileNameEl.textContent = f.name; fileNameEl.style.display = 'block'; uploadBtn.disabled = false; }
        else { fileNameEl.style.display = 'none'; uploadBtn.disabled = true; }
    });

    ['dragover', 'dragenter'].forEach(evt => dropZone?.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('dragover'); }));
    ['dragleave', 'dragend', 'drop'].forEach(evt => dropZone?.addEventListener(evt, () => dropZone?.classList.remove('dragover')));
    dropZone?.addEventListener('drop', e => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (f && selectedRecipientId) {
            const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files;
            fileNameEl.textContent = f.name; fileNameEl.style.display = 'block'; uploadBtn.disabled = false;
        } else if (!selectedRecipientId) {
            showToast('Please select a recipient first', 'error');
        }
    });

    uploadBtn?.addEventListener('click', uploadReport);

    async function uploadReport() {
        if (!selectedRecipientId) return;
        const file = fileInput.files[0];
        if (!file) return;

        uploadBtn.disabled = true; uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
        progressWrap.style.display = 'block'; progressBar.style.width = '30%';

        const animTimer = setInterval(() => {
            const cur = parseFloat(progressBar.style.width) || 30;
            if (cur < 85) progressBar.style.width = (cur + 5) + '%';
        }, 300);

        const fd = new FormData(); fd.append('file', file);
        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/reports`, {
                method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: fd
            });
            clearInterval(animTimer); progressBar.style.width = '100%';
            if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || 'Upload failed'); }

            showToast('Report uploaded! Pipeline processing...', 'success');
            resetUploadUI();

            // Auto-refresh after short delay to let pipeline start
            await loadReports();

            // Poll for pipeline completion
            pollPipelineStatus();
        } catch (e) {
            clearInterval(animTimer);
            progressBar.style.width = '100%';
            progressBar.style.background = 'linear-gradient(90deg, var(--danger), #ff6b6b)';
            showToast(e.message || 'Upload failed', 'error');
            setTimeout(resetUploadUI, 2000);
        }
    }

    function resetUploadUI() {
        fileInput.value = '';
        fileNameEl.style.display = 'none'; fileNameEl.textContent = '';
        uploadBtn.disabled = true; uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload Report';
        progressBar.style.width = '0%'; progressBar.style.background = '';
        progressWrap.style.display = 'none';
    }

    // Poll for pipeline completion and auto-refresh
    async function pollPipelineStatus() {
        let attempts = 0;
        const maxAttempts = 20; // ~40s max
        const poll = setInterval(async () => {
            attempts++;
            if (attempts >= maxAttempts || !selectedRecipientId) { clearInterval(poll); return; }
            try {
                const res = await fetch(`${API}/recipients/${selectedRecipientId}/reports`, {
                    headers: { 'Authorization': 'Bearer ' + token }
                });
                if (!res.ok) return;
                const data = await res.json();
                const reports = data?.result?.reports || [];
                const anyProcessing = reports.some(r => r.processing_status === 'processing' || r.processing_status === 'pending');
                currentReports = reports;
                renderReportsList(reports);
                updateReportStats(reports);
                if (!anyProcessing) {
                    clearInterval(poll);
                    showToast('Analysis complete! Refreshing data...', 'success');
                    loadClinicalData();
                }
            } catch (e) { /* continue polling */ }
        }, 2000);
    }

    // ── Report List ──
    async function loadReports() {
        const list = $('reportsList');
        if (!list) return;

        if (!selectedRecipientId) {
            list.innerHTML = '<p class="muted" style="text-align:center; padding:40px;">Select a care recipient to view reports.</p>';
            return;
        }
        list.innerHTML = '<p class="muted" style="text-align:center; padding:20px;"><i class="fas fa-spinner fa-spin"></i> Loading...</p>';
        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/reports`, { headers: { 'Authorization': 'Bearer ' + token } });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            currentReports = data?.result?.reports || [];
            sortAndRenderReports();
            updateReportStats(currentReports);
        } catch (e) {
            console.error('Load reports error:', e);
            list.innerHTML = '<p class="muted">Unable to load reports.</p>';
        }
    }

    function updateReportStats(reports) {
        if ($('statTotalReports')) $('statTotalReports').textContent = reports.length;
        if ($('statProcessed')) $('statProcessed').textContent = reports.filter(r => r.processing_status === 'completed').length;
        const pending = reports.filter(r => r.processing_status === 'pending' || r.processing_status === 'processing').length;
        const failed = reports.filter(r => r.processing_status === 'failed').length;
        if ($('statPending')) $('statPending').textContent = `${pending} / ${failed}`;
    }

    function sortAndRenderReports() {
        const sort = $('reportSortOrder').value;
        const sorted = [...currentReports].sort((a, b) => {
            if (sort === 'date_desc') return new Date(b.uploaded_at) - new Date(a.uploaded_at);
            if (sort === 'date_asc') return new Date(a.uploaded_at) - new Date(b.uploaded_at);
            if (sort === 'name_asc') return a.filename.localeCompare(b.filename);
            return 0;
        });
        renderReportsList(sorted);
    }

    $('reportSortOrder')?.addEventListener('change', sortAndRenderReports);

    function renderReportsList(reports) {
        const el = $('reportsList');
        if (!reports.length) {
            el.innerHTML = '<p class="muted" style="text-align:center; padding:40px;">No reports uploaded yet.</p>';
            return;
        }
        el.innerHTML = '';
        reports.forEach(r => {
            const ext = (r.filename.split('.').pop() || '').toLowerCase();
            const icon = ext === 'pdf' ? 'fa-file-pdf' : ['png', 'jpg', 'jpeg'].includes(ext) ? 'fa-file-image' : 'fa-file-alt';
            const statusClass = r.processing_status || 'unknown';
            const statusLabel = statusClass === 'completed' ? 'Analyzed' : statusClass;

            const card = document.createElement('div');
            card.className = 'report-card';
            card.innerHTML = `
                <h4 style="display:flex; align-items:center; gap:8px;">
                    <i class="fas ${icon}" style="color:var(--accent); font-size:1.1rem;"></i>
                    <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${r.filename}">${r.filename}</span>
                    <span class="status-badge ${statusClass}">${statusLabel}</span>
                </h4>
                <div class="report-meta">
                    <span><i class="fas fa-calendar-alt"></i> ${new Date(r.uploaded_at).toLocaleDateString()}</span>
                    ${r.report_date ? `<span><i class="fas fa-notes-medical"></i> Report: ${r.report_date}</span>` : ''}
                </div>
                ${r.analysis_summary ? `<p style="margin:8px 0 0; font-size:0.82rem; color:var(--muted); line-height:1.5; border-top:1px dashed rgba(0,0,0,0.06); padding-top:8px;">${r.analysis_summary.substring(0, 200)}${r.analysis_summary.length > 200 ? '...' : ''}</p>` : ''}
                <div class="report-actions">
                    <button class="ghost-btn" onclick="window._downloadReport(${r.id}, '${r.filename}')"><i class="fas fa-download"></i> Download</button>
                    <button class="danger-btn" onclick="window._deleteReport(${r.id})"><i class="fas fa-trash"></i> Delete</button>
                </div>
            `;
            el.appendChild(card);
        });
    }

    // Global handlers for inline onclick
    window._downloadReport = async function (reportId, filename) {
        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/reports/${reportId}/download`, {
                headers: { 'Authorization': 'Bearer ' + token }
            });
            if (!res.ok) throw new Error('Download failed');
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 4000);
        } catch (e) { showToast('Download failed', 'error'); }
    };

    window._deleteReport = async function (reportId) {
        if (!confirm('Delete this report? This will recalculate all analysis.')) return;
        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/reports/${reportId}`, {
                method: 'DELETE', headers: { 'Authorization': 'Bearer ' + token }
            });
            if (!res.ok) throw new Error('Delete failed');
            showToast('Report deleted. Recalculating analysis...', 'success');
            await loadReports();
            // Auto-refresh clinical data after deletion
            setTimeout(() => loadClinicalData(), 1000);
        } catch (e) { showToast('Delete failed', 'error'); }
    };

    // ═══════════════════════════════════════
    //  TAB 2 — Clinical Intelligence
    // ═══════════════════════════════════════
    async function loadClinicalData() {
        if (!selectedRecipientId) return;
        try {
            const [stateRes, trendsRes, alertsRes, recsRes] = await Promise.all([
                fetch(`${API}/recipients/${selectedRecipientId}/medical-state`, { headers: { 'Authorization': 'Bearer ' + token } }),
                fetch(`${API}/recipients/${selectedRecipientId}/trends`, { headers: { 'Authorization': 'Bearer ' + token } }),
                fetch(`${API}/recipients/${selectedRecipientId}/alerts`, { headers: { 'Authorization': 'Bearer ' + token } }),
                fetch(`${API}/recipients/${selectedRecipientId}/recommendations`, { headers: { 'Authorization': 'Bearer ' + token } }),
            ]);

            const state = await stateRes.json();
            const trends = await trendsRes.json();
            const alerts = await alertsRes.json();
            const recommendationsData = await recsRes.json();
            const recommendations = recommendationsData.result || [];

            const hasData = state.active_conditions?.length || state.past_conditions?.length || state.latest_labs?.length || state.risk_score;

            if (hasData) {
                $('clinicalEmpty').style.display = 'none';
                $('clinicalDashboard').style.display = 'block';
                renderRiskGauge(state.risk_score);
                renderConditions(state.active_conditions, state.past_conditions);
                renderAlerts(alerts);
                renderTrendChart(trends);
                renderTimeline(state.active_conditions, state.past_conditions);
                setupSeverityTracker(state.active_conditions, state.past_conditions);
                
                // Render the new actionable clinical recommendations
                if (typeof renderRecommendations === 'function') {
                    renderRecommendations(recommendations);
                }
            } else {
                $('clinicalDashboard').style.display = 'none';
                $('clinicalEmpty').style.display = 'block';
            }

            // Update doctor's summary tab
            renderDoctorSummary(state, alerts);
        } catch (e) {
            console.error('Clinical data load error:', e);
        }
    }

    // ── Risk Score Gauge ──
    function renderRiskGauge(risk) {
        if (!risk) {
            $('riskCategory').textContent = '\u2014';
            $('riskTrajectory').textContent = '';
            $('riskFactors').innerHTML = '<p class="muted" style="font-size:0.82rem;">No risk data. Run analysis.</p>';
            return;
        }

        const canvas = $('riskGauge');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2, cy = canvas.height - 10, r = 80;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background arc
        ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0, false);
        ctx.lineWidth = 18; ctx.strokeStyle = '#e5e7eb'; ctx.stroke();

        // Score arc
        const pct = Math.min(risk.risk_score / 100, 1);
        const end = Math.PI + pct * Math.PI;
        const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
        grad.addColorStop(0, '#91AA8A'); grad.addColorStop(0.5, '#E8B89A'); grad.addColorStop(1, '#C96B55');
        ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, end, false);
        ctx.lineWidth = 18; ctx.strokeStyle = grad; ctx.lineCap = 'round'; ctx.stroke();

        // Score text
        ctx.fillStyle = '#1A202C'; ctx.font = 'bold 28px Inter, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(Math.round(risk.risk_score), cx, cy - 15);
        ctx.font = '12px Inter, sans-serif'; ctx.fillStyle = '#6b7280';
        ctx.fillText('/ 100', cx, cy);

        // Category & trajectory
        const catColors = { Low: '#91AA8A', Moderate: '#E8B89A', High: '#E8B89A', Critical: '#C96B55' };
        $('riskCategory').innerHTML = `<span style="color:${catColors[risk.risk_category] || '#6b7280'}">${risk.risk_category}</span>`;
        const trajMap = { increasing: '\u2191 Increasing', stable: '\u2192 Stable', improving: '\u2193 Improving' };
        $('riskTrajectory').textContent = trajMap[risk.risk_trajectory] || '';

        // Factors
        const fEl = $('riskFactors');
        if (risk.factors?.length) {
            fEl.innerHTML = `<p style="font-size:0.75rem; font-weight:700; color:var(--muted); margin-bottom:6px;">Contributing Factors</p>` +
                risk.factors.map(f => `<div style="display:flex; justify-content:space-between; font-size:0.8rem; padding:3px 0; border-bottom:1px solid rgba(0,0,0,0.03);"><span>${f.factor}</span><span style="font-weight:700; color:var(--danger);">+${f.contribution}</span></div>`).join('');
        } else {
            fEl.innerHTML = '<p class="muted" style="font-size:0.82rem;">No significant risk factors.</p>';
        }
    }

    // ── Condition Cards ──
    function renderConditions(active, past) {
        $('activeConditions').innerHTML = (active || []).map(c => conditionCard(c)).join('') || '<p class="muted">No active conditions detected.</p>';
        if (past?.length) {
            $('pastConditionsToggle').style.display = 'block';
            $('pastConditions').innerHTML = past.map(c => conditionCard(c, true)).join('');
        } else {
            $('pastConditionsToggle').style.display = 'none';
        }
    }

    function conditionCard(c, isPast = false) {
        const col = STATUS_COLORS[c.status] || '#6b7280';
        const icon = STATUS_ICONS[c.status] || '\u25cf';
        const src = c.source_type === 'explicit_diagnosis' ? 'Doctor Diagnosed' : 'Lab Inferred';
        const conf = Math.round((c.confidence_score || 0) * 100);
        return `
        <article class="info-card" style="border-left:4px solid ${col}; padding:16px; ${isPast ? 'opacity:0.65;' : ''}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                <strong>${c.disease_name}</strong>
                <span style="font-size:0.72rem; background:${col}12; color:${col}; padding:2px 10px; border-radius:12px; font-weight:700; text-transform:capitalize;">${icon} ${(c.status || '').replace('_', ' ')}</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:3px 14px; font-size:0.8rem; color:var(--muted);">
                <span>Severity: <strong>${c.severity || '\u2014'}</strong></span>
                <span>Confidence: <strong>${conf}%</strong> (${src})</span>
                <span>Since: <strong>${c.first_detected || '\u2014'}</strong></span>
                <span>Updated: <strong>${c.last_updated || '\u2014'}</strong></span>
                ${c.baseline_value ? `<span style="grid-column:span 2;">Baseline: ${c.baseline_value} (${c.baseline_date || '\u2014'})</span>` : ''}
            </div>
        </article>`;
    }

    // ── Alerts ──
    function renderAlerts(alertsData) {
        const badge = $('alertBadge');
        const list = $('alertsList');
        const count = alertsData.unread_count || 0;
        const alerts = alertsData.alerts || [];

        badge.style.display = count > 0 ? 'inline' : 'none';
        badge.textContent = `${count} alert${count > 1 ? 's' : ''}`;

        if (alerts.length) {
            list.style.display = 'block';
            const sevColors = { low: '#91AA8A', medium: '#E8B89A', high: '#E8B89A', critical: '#C96B55' };
            list.innerHTML = `<p style="font-size:0.75rem; font-weight:700; color:var(--muted); margin-bottom:8px;">Recent Alerts</p>` +
                alerts.slice(0, 5).map(a => `<div style="padding:8px 12px; margin-bottom:6px; border-radius:10px; background:${sevColors[a.severity] || '#e5e7eb'}08; border:1px solid ${sevColors[a.severity] || '#e5e7eb'}30; font-size:0.82rem;">
                    ${a.message}
                    <span style="display:block; font-size:0.7rem; color:var(--muted); margin-top:2px;">${a.created_at ? new Date(a.created_at).toLocaleDateString() : ''}</span>
                </div>`).join('');
        } else {
            list.style.display = 'none';
        }
    }

    // ── Lab Trend Chart ──
    function renderTrendChart(trends) {
        const select = $('metricSelect');
        const metrics = trends.available_metrics || [];
        select.innerHTML = metrics.map(m => `<option value="${m}">${m}</option>`).join('');
        if (metrics.length) {
            select.onchange = () => drawTrend(trends.metrics.find(m => m.metric_name === select.value));
            drawTrend(trends.metrics[0]);
        }
    }

    function drawTrend(metric) {
        if (!metric) return;
        const badges = $('trendBadges');
        const trendIcons = { increasing: '↑', decreasing: '↓', stable: '→', fluctuating: '↕' };
        const trendColors = { increasing: '#C96B55', decreasing: '#91AA8A', stable: '#6b7280', fluctuating: '#E8B89A' };
        const last = metric.data_points?.[metric.data_points.length - 1];
        const pPrev = last?.pct_change_from_previous;

        // ── Confidence threshold filter (default: show all ≥ 0.0) ──────────
        const confThreshold = parseFloat($('confThreshold')?.value || '0');
        const filteredPoints = metric.data_points.filter(d =>
            d.confidence_score == null || d.confidence_score >= confThreshold
        );
        const lowConfCount = metric.data_points.filter(d =>
            d.confidence_score != null && d.confidence_score < 0.8
        ).length;

        // ── Confidence-based point colour ────────────────────────────────────
        // regex=0.95 → strong blue  |  fuzzy=0.82 → teal  |  llm=0.65 → orange
        const CONF_COLORS = { regex: '#4A90E2', template: '#2563eb', fuzzy: '#06b6d4', llm: '#E8B89A' };
        const ptColors = filteredPoints.map(d => {
            if (d.is_abnormal) return '#C96B55';
            return CONF_COLORS[d.extraction_source] || '#4A90E2';
        });
        const ptSizes = filteredPoints.map(d =>
            d.confidence_score != null && d.confidence_score < 0.8 ? 7 : 5
        );

        badges.innerHTML = `
            <span style="background:${trendColors[metric.trend_direction]}15; color:${trendColors[metric.trend_direction]}; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700;">${trendIcons[metric.trend_direction] || ''} ${metric.trend_direction}</span>
            <span style="background:rgba(0,0,0,0.04); padding:3px 12px; border-radius:20px; font-size:0.75rem;">Volatility: ${metric.volatility_label || 'Low'}</span>
            ${pPrev != null ? `<span style="background:rgba(0,0,0,0.04); padding:3px 12px; border-radius:20px; font-size:0.75rem;">${pPrev > 0 ? '↑' : '↓'} ${Math.abs(pPrev).toFixed(1)}% from last</span>` : ''}
            ${lowConfCount > 0 ? `<span style="background:rgba(249,115,22,0.12); color:#E8B89A; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700;" title="${lowConfCount} point(s) extracted by AI (lower confidence)">⚠ ${lowConfCount} AI-extracted</span>` : '<span style="background:rgba(34,197,94,0.12); color:#91AA8A; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700;">✓ Rule-verified</span>'}
        `;

        if (trendChart) trendChart.destroy();
        const labels = filteredPoints.map(d => d.date);
        const values = filteredPoints.map(d => d.value);

        const datasets = [{
            label: metric.metric_name, data: values, borderColor: '#4A90E2', backgroundColor: 'rgba(74,144,226,0.08)',
            fill: true, tension: 0.3, pointBackgroundColor: ptColors, pointRadius: ptSizes, pointHoverRadius: 8,
            pointBorderColor: filteredPoints.map(d =>
                d.confidence_score != null && d.confidence_score < 0.8 ? '#E8B89A' : 'transparent'
            ),
            pointBorderWidth: 2,
        }];

        if (metric.reference_range_low != null && metric.reference_range_high != null) {
            datasets.push({
                label: 'Normal Range', data: new Array(labels.length).fill(metric.reference_range_high),
                borderColor: 'transparent', backgroundColor: 'rgba(34,197,94,0.08)',
                fill: { target: { value: metric.reference_range_low }, above: 'rgba(34,197,94,0.08)' }, pointRadius: 0,
            });
        }
        if (metric.baseline_value) {
            datasets.push({
                label: 'Baseline', data: new Array(labels.length).fill(metric.baseline_value),
                borderColor: '#E8B89A', borderDash: [6, 4], borderWidth: 2, pointRadius: 0, fill: false,
            });
        }

        trendChart = new Chart($('trendChart'), {
            type: 'line', data: { labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 10 } } },
                    tooltip: {
                        callbacks: {
                            afterLabel: ctx => {
                                const pt = filteredPoints[ctx.dataIndex];
                                if (!pt) return '';
                                const lines = [];
                                if (pt.confidence_score != null)
                                    lines.push(`Confidence: ${Math.round(pt.confidence_score * 100)}% (${pt.extraction_source || 'rule'})`);
                                if (pt.source_text)
                                    lines.push(`Source: "${pt.source_text.substring(0, 60)}${pt.source_text.length > 60 ? '...' : ''}"`); 
                                return lines;
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 10 } } },
                }
            }
        });
    }

    // ── Disease Timeline ──
    function renderTimeline(active, past) {
        if (timelineChart) timelineChart.destroy();
        const all = [...(active || []), ...(past || [])];
        if (!all.length) return;

        const now = new Date();
        const durations = all.map(c => {
            const s = new Date(c.first_detected);
            const e = c.resolved_date ? new Date(c.resolved_date) : now;
            return Math.max(1, Math.round((e - s) / (1000 * 60 * 60 * 24 * 30)));
        });
        const colors = all.map(c => STATUS_COLORS[c.status] || '#6b7280');

        timelineChart = new Chart($('timelineChart'), {
            type: 'bar',
            data: {
                labels: all.map(c => c.disease_name),
                datasets: [{ label: 'Duration (months)', data: durations, backgroundColor: colors.map(c => c + '40'), borderColor: colors, borderWidth: 2, borderRadius: 8 }]
            },
            options: {
                indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { afterLabel: ctx => { const c = all[ctx.dataIndex]; return `Status: ${c.status}\nSince: ${c.first_detected}`; } } }
                },
                scales: {
                    x: { title: { display: true, text: 'Months', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)' } },
                    y: { grid: { display: false }, ticks: { font: { size: 11 } } },
                }
            }
        });
    }

    // ── Condition Severity Tracker ──
    function setupSeverityTracker(active, past) {
        const select = $('conditionSelect');
        if (!select) return;

        const all = [...(active || []), ...(past || [])];
        select.innerHTML = all.map(c => `<option value="${c.id}">${c.disease_name}</option>`).join('');

        if (all.length) {
            select.onchange = () => loadAndDrawSeverity(select.value);
            loadAndDrawSeverity(select.value);
        }
    }

    async function loadAndDrawSeverity(conditionId) {
        if (!conditionId || !selectedRecipientId) return;

        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/conditions/${conditionId}/timeline`, {
                headers: { 'Authorization': 'Bearer ' + token }
            });
            if (!res.ok) throw new Error('Failed to load condition timeline');
            const data = await res.json();
            drawSeverityChart(data.condition, data.timeline);
        } catch (e) {
            console.error('Severity fetch error:', e);
        }
    }

    function drawSeverityChart(condition, timeline) {
        if (severityChart) severityChart.destroy();

        const el = $('severityChart');
        if (!el || !timeline || !timeline.length) return;

        // Severity mapping
        const sevMap = { 'mild': 1, 'moderate': 2, 'severe': 3, 'critical': 4 };
        const sevLabels = ['', 'Mild', 'Moderate', 'Severe', 'Critical'];

        // Sort timeline
        const sorted = [...timeline].sort((a, b) => new Date(a.recorded_at) - new Date(b.recorded_at));

        const labels = sorted.map(t => new Date(t.recorded_at).toLocaleDateString());
        const data = sorted.map(t => sevMap[t.new_severity] || 0);

        // Status tracking for point colors
        const pointColors = sorted.map(t => STATUS_COLORS[t.new_status] || '#6b7280');

        severityChart = new Chart(el, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Severity Level',
                    data,
                    borderColor: '#E8B89A',
                    backgroundColor: 'rgba(249,115,22,0.1)',
                    fill: true,
                    stepped: true,
                    pointBackgroundColor: pointColors,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const t = sorted[ctx.dataIndex];
                                return [
                                    `Severity: ${t.new_severity}`,
                                    `Status: ${t.new_status}`,
                                    t.clinical_interpretation ? `Note: ${t.clinical_interpretation}` : ''
                                ].filter(Boolean);
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: {
                        min: 0, max: 4,
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: {
                            stepSize: 1,
                            callback: val => sevLabels[val] || '',
                            font: { size: 10 }
                        }
                    },
                }
            }
        });
    }

    // ═══════════════════════════════════════
    //  TAB 3 — Doctor's Summary
    // ═══════════════════════════════════════
    function renderDoctorSummary(state, alertsData) {
        const hasData = state.active_conditions?.length || state.past_conditions?.length || state.latest_labs?.length || state.risk_score;
        $('summaryEmpty').style.display = hasData ? 'none' : 'block';
        $('summaryDashboard').style.display = hasData ? 'block' : 'none';
        if (!hasData) return;

        // Quick stats
        const catColors = { Low: 'var(--success)', Moderate: 'var(--warning)', High: '#E8B89A', Critical: 'var(--danger)' };
        $('sumRiskLevel').innerHTML = state.risk_score ? `<span style="color:${catColors[state.risk_score.risk_category] || 'inherit'}">${state.risk_score.risk_category} (${Math.round(state.risk_score.risk_score)})</span>` : '\u2014';
        $('sumConditionCount').textContent = (state.active_conditions?.length || 0);
        $('sumAlertCount').textContent = alertsData?.unread_count || 0;

        // Summary blocks
        const blocks = $('summaryBlocks');
        blocks.innerHTML = '';

        // Current Problems
        if (state.active_conditions?.length) {
            blocks.innerHTML += `<div class="summary-block">
                <h4><i class="fas fa-exclamation-triangle" style="margin-right:6px;"></i>Current Problems</h4>
                <ul style="margin:0; padding-left:16px; font-size:0.88rem; line-height:1.8; color:var(--text);">
                    ${state.active_conditions.map(c => `<li><strong>${c.disease_name}</strong> &mdash; ${(c.status || '').replace('_', ' ')} (${c.severity || 'unknown'} severity, since ${c.first_detected || 'unknown'})</li>`).join('')}
                </ul>
            </div>`;
        }

        // Key Lab Values
        if (state.latest_labs?.length) {
            blocks.innerHTML += `<div class="summary-block">
                <h4><i class="fas fa-flask" style="margin-right:6px;"></i>Key Lab Values</h4>
                <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:8px;">
                    ${state.latest_labs.map(l => {
                const abnormal = l.is_abnormal;
                return `<div style="background:${abnormal ? 'rgba(239,68,68,0.06)' : 'rgba(74,144,226,0.04)'}; padding:10px 12px; border-radius:12px; border:1px solid ${abnormal ? 'rgba(239,68,68,0.15)' : 'rgba(74,144,226,0.1)'};">
                            <div style="font-size:0.72rem; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:4px;">${l.metric_name}</div>
                            <div style="font-size:1.1rem; font-weight:800; color:${abnormal ? '#C96B55' : 'var(--text)'};">${l.value} <span style="font-size:0.7rem; font-weight:400;">${l.unit || ''}</span></div>
                        </div>`;
            }).join('')}
                </div>
            </div>`;
        }

        // Recent Alerts
        const alerts = alertsData?.alerts || [];
        if (alerts.length) {
            blocks.innerHTML += `<div class="summary-block">
                <h4><i class="fas fa-bell" style="margin-right:6px;"></i>Recent Alerts</h4>
                <div style="font-size:0.88rem; line-height:1.7;">${alerts.slice(0, 5).map(a => `<div style="padding:6px 0; border-bottom:1px solid rgba(0,0,0,0.04);">${a.message}</div>`).join('')}</div>
            </div>`;
        }

        // Past Conditions
        if (state.past_conditions?.length) {
            blocks.innerHTML += `<div class="summary-block">
                <h4><i class="fas fa-history" style="margin-right:6px;"></i>Past Conditions</h4>
                <ul style="margin:0; padding-left:16px; font-size:0.88rem; line-height:1.7; color:var(--muted);">
                    ${state.past_conditions.map(c => `<li>${c.disease_name} (resolved ${c.resolved_date || 'date unknown'})</li>`).join('')}
                </ul>
            </div>`;
        }

        // Risk Factors
        if (state.risk_score?.factors?.length) {
            blocks.innerHTML += `<div class="summary-block">
                <h4><i class="fas fa-chart-line" style="margin-right:6px;"></i>Risk Factors</h4>
                <div style="font-size:0.88rem;">${state.risk_score.factors.map(f =>
                `<div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(0,0,0,0.03);"><span>${f.factor}</span><span style="font-weight:700; color:var(--danger);">+${f.contribution}</span></div>`
            ).join('')}</div>
            </div>`;
        }
    }

    // ═══════════════════════════════════════
    //  Action Buttons
    // ═══════════════════════════════════════
    $('btnRunAnalysis')?.addEventListener('click', async () => {
        if (!selectedRecipientId) return;
        const btn = $('btnRunAnalysis');
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...'; btn.disabled = true;
        try {
            const res = await fetch(`${API}/recipients/${selectedRecipientId}/analyze`, {
                method: 'POST', headers: { 'Authorization': 'Bearer ' + token }
            });
            if (res.ok) {
                const result = await res.json();
                // Show AI interpretation
                const el = $('healthStatus');
                if (el && result.overall_health_status) {
                    el.innerHTML = `<strong>${result.overall_health_status}</strong>`;
                    if (result.explanation) el.innerHTML += `<p style="font-size:0.82rem; color:var(--muted); margin-top:6px;">${result.explanation}</p>`;
                    if (result.recommendations?.length) {
                        el.innerHTML += `<ul style="font-size:0.82rem; color:var(--muted); margin-top:6px; padding-left:18px;">${result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>`;
                    }
                    if (result.monitoring_frequency) {
                        el.innerHTML += `<p style="font-size:0.78rem; color:var(--accent); margin-top:6px;"><i class="fas fa-calendar-check"></i> ${result.monitoring_frequency}</p>`;
                    }
                }
                showToast('Analysis complete!', 'success');
                await loadClinicalData();
                // Switch to clinical tab to show results
                document.querySelector('[data-tab="tab-clinical"]')?.click();
            }
        } catch (e) { console.error('Analysis failed:', e); showToast('Analysis failed', 'error'); }
        btn.innerHTML = '<i class="fas fa-brain"></i> Run Full Analysis'; btn.disabled = false;
    });

    // ── Clinical Recommendations ──
    window.renderRecommendations = function(recommendations) {
        let container = $('recommendationsContainer');
        if (!container) {
            // Create container below alerts if it doesn't exist
            const insightsCol = document.querySelector('.insights-column');
            if (insightsCol) {
                container = document.createElement('div');
                container.id = 'recommendationsContainer';
                container.className = 'panel';
                container.style.marginTop = '24px';
                
                // Find where to insert (after alerts list)
                const alertsPanel = $('alertsList')?.closest('.panel');
                if (alertsPanel) {
                    alertsPanel.after(container);
                } else {
                    insightsCol.appendChild(container); // Fallback
                }
            }
        }
        
        if (!container) return;

        if (!recommendations || recommendations.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        
        // Group by severity
        const groups = { critical: [], high: [], medium: [], low: [], suggestion: [] };
        recommendations.forEach(r => {
            if (groups[r.severity]) groups[r.severity].push(r);
            else groups.suggestion.push(r); // default to suggestion
        });

        let html = `<div class="panel-header" style="margin-bottom: 16px;"><h2 class="panel-title"><i class="fas fa-user-md" style="color:var(--accent);"></i> Actionable Recommendations</h2></div><div style="display:flex; flex-direction:column; gap:12px;">`;

        const colors = { critical: '#fef2f2', high: '#fffbeb', medium: '#f0fdf4', suggestion: '#FFF9F2', low: '#FFF9F2' };
        const borders = { critical: '#C96B55', high: '#E8B89A', medium: '#91AA8A', suggestion: '#B8A9C9', low: '#756D70' };
        const iconColors = { critical: '#C96B55', high: '#E8B89A', medium: '#91AA8A', suggestion: '#B8A9C9', low: '#756D70' };
        const icons = { critical: 'fa-exclamation-triangle', high: 'fa-bolt', medium: 'fa-info-circle', suggestion: 'fa-lightbulb', low: 'fa-info-circle' };
        
        // Render groups in priority order
        ['critical', 'high', 'medium', 'suggestion', 'low'].forEach(sev => {
            if (groups[sev].length > 0) {
                groups[sev].forEach(rec => {
                    const actionsHtml = (rec.actions || []).map(action => {
                        let aIcon = 'fa-check';
                        if (action.type === 'doctor_visit') aIcon = 'fa-stethoscope';
                        else if (action.type === 'diet') aIcon = 'fa-apple-alt';
                        else if (action.type === 'lifestyle') aIcon = 'fa-running';
                        else if (action.type === 'test') aIcon = 'fa-vial';
                        else if (action.type === 'home_remedy') aIcon = 'fa-leaf';
                        return `<li style="margin-top:4px;"><i class="fas ${aIcon}" style="color:${iconColors[sev]}; width:16px;"></i> ${action.text}</li>`;
                    }).join('');
                    
                    const triggerHtml = rec.trigger_value ? `<span style="float:right; font-size:0.75rem; background:#fff; padding:2px 8px; border-radius:12px; border:1px solid ${borders[sev]}50; color:${borders[sev]};">Trigger: ${rec.trigger_value} ${rec.reference_range ? `(Ref: ${rec.reference_range})` : ''}</span>` : '';

                    html += `
                    <div style="background:${colors[sev]}; border-left:4px solid ${borders[sev]}; padding:12px; border-radius:6px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                        <div style="font-weight:600; font-size:0.9rem; margin-bottom:4px; display:flex; align-items:center;">
                            <i class="fas ${icons[sev]}" style="color:${iconColors[sev]}; margin-right:8px;"></i>
                            ${rec.metric}
                            <span style="flex-grow:1;"></span>
                            ${triggerHtml}
                        </div>
                        <p style="font-size:0.85rem; color:var(--text); margin-bottom:8px;">${rec.message}</p>
                        ${actionsHtml ? `<ul style="font-size:0.8rem; color:var(--muted); list-style:none; padding-left:4px;">${actionsHtml}</ul>` : ''}
                    </div>`;
                });
            }
        });

        html += `</div>`;
        container.innerHTML = html;
    };


    $('btnRefresh')?.addEventListener('click', () => {
        if (selectedRecipientId) { loadReports(); loadClinicalData(); }
    });

    // ═══════════════════════════════════════
    //  Init
    // ═══════════════════════════════════════
    document.addEventListener('DOMContentLoaded', () => {
        fetchProfile();
        // Auto-select previously selected recipient
        const savedId = localStorage.getItem('selectedRecipientId');
        if (savedId) {
            setTimeout(() => {
                const r = recipientsCache.find(r => String(r.id) === savedId);
                if (r) selectRecipient(r.id, r.full_name);
            }, 1000);
        }
    });
})();
