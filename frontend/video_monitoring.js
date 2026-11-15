// Video Monitoring JavaScript
const API_BASE = 'http://127.0.0.1:8000/api';
let currentProcessId = null;
let currentSessionId = null;
let checkStatusInterval = null;
let checkAlertsInterval = null;
let selectedCameraIndex = 0;    // Legacy — kept for upload-related code
let selectedCameraDeviceId = null; // TF.js live: exact deviceId string

/** TF.js live session singleton. */
let _tfjsSession = null;

/** Fall alert auto-dismiss timer. */
let _fallAlertTimer = null;
let _tfjsOpenFall = null;   // latest live fall waiting for the person to get up

// Authentication check
let token = localStorage.getItem('token');
if (!token) {
    window.location.href = 'index.html';
}
// session_keepalive.js swaps the login token every 20 minutes: use the fresh one from then on
window.addEventListener('caretaker-token', e => { if (e.detail) token = e.detail; });

// Initialize on page load
// Initial listener removed to avoid duplication with the one at the bottom

// Load user information
async function loadUserInfo() {
    try {
        const response = await fetch(`${API_BASE}/profile`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('userBadge').textContent = data.caretaker.full_name || data.caretaker.username;
        }
    } catch (error) {
        console.error('Error loading user info:', error);
    }
}

// Setup drag and drop upload zone
function setupUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    const videoInput = document.getElementById('videoInput');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', handleDrop, false);

    // Handle file input change
    videoInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    // Click to upload
    uploadZone.addEventListener('click', function (e) {
        if (e.target === uploadZone || e.target.closest('.upload-zone')) {
            videoInput.click();
        }
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }
}

async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file', 'danger');
        return;
    }

    // Validate file size (100MB max)
    if (file.size > 100 * 1024 * 1024) {
        showAlert('File size exceeds 100MB limit', 'danger');
        return;
    }

    // Show upload status
    document.getElementById('uploadStatus').classList.remove('d-none');
    document.getElementById('resultsSection').classList.add('d-none');
    updateProgress(0, 'Uploading video...');

    const formData = new FormData();
    formData.append('file', file);

    // Check for recipient_id in URL
    const urlParams = new URLSearchParams(window.location.search);
    const recipientId = urlParams.get('recipient_id');
    if (recipientId) {
        formData.append('recipient_id', recipientId);
        console.log('Uploading for recipient:', recipientId);
    }

    try {
        // Do NOT set Content-Type header manually when sending FormData
        // browser sets it automatically with boundary
        const response = await fetch(`${API_BASE}/video-monitoring/upload-video`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        currentProcessId = data.process_id;

        updateProgress(30, 'Processing video...');

        // Start checking status
        checkProcessingStatus();

    } catch (error) {
        console.error('Upload error:', error);
        showAlert(`Upload failed: ${error.message}`, 'danger');
        document.getElementById('uploadStatus').classList.add('d-none');
    }
}

async function checkProcessingStatus() {
    if (!currentProcessId) return;

    try {
        const response = await fetch(`${API_BASE}/video-monitoring/status/${currentProcessId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to get status');
        }

        const status = await response.json();

        if (status.status === 'completed') {
            updateProgress(100, 'Analysis complete!');
            setTimeout(() => {
                document.getElementById('uploadStatus').classList.add('d-none');
                showResults(status);
            }, 1000);
        } else if (status.status === 'error') {
            showAlert(`Processing error: ${status.error}`, 'danger');
            document.getElementById('uploadStatus').classList.add('d-none');
        } else {
            // Still processing
            const progress = status.progress || 50;
            updateProgress(progress, 'Analyzing video...');

            // Check again in 2 seconds
            setTimeout(checkProcessingStatus, 2000);
        }

    } catch (error) {
        console.error('Status check error:', error);
        showAlert('Failed to check processing status', 'danger');
        document.getElementById('uploadStatus').classList.add('d-none');
    }
}

function updateProgress(percent, message) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const statusText = document.getElementById('statusText');

    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
    statusText.textContent = message;
}



function escapeFallText(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function showResults(results) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsAlert = document.getElementById('resultsAlert');
    const fallDetails = document.getElementById('fallDetails');

    resultsSection.classList.remove('d-none');

    const hasFalls = results.has_falls || (results.falls_detected && results.falls_detected.length > 0);
    const outputFilename = results.output_filename || results.output_file;

    console.log('[Video URL Debug] output_filename:', outputFilename);
    console.log('[Video URL Debug] processed_url from backend:', results.processed_url);

    // FORCE MANUAL CONSTRUCTION: Ignore backend processed_url because it might be cached/wrong
    // Videos are served from /processed_videos/ on port 8000
    // Ensure outputFilename is clean (no path)
    const cleanFilename = outputFilename ? outputFilename.split(/[/\\]/).pop() : null;

    // Use the NEW path: /processed_videos/ (matches directory name)
    const processedUrl = cleanFilename ? `http://127.0.0.1:8000/processed_videos/${cleanFilename}` : null;

    console.log('[Video URL Debug] Final processedUrl:', processedUrl);

    // Construct Video Player HTML
    let videoHtml = '';
    if (processedUrl) {
        console.log('[Video Display] Video URL:', processedUrl);

        videoHtml = `
            <div class="card mb-4">
                <div class="card-header bg-dark text-white">
                    <h5 class="mb-0"><i class="fas fa-play-circle me-2"></i>Analysis Playback</h5>
                </div>
                <div class="card-body p-0">
                    <div class="ratio ratio-16x9">
                        <video controls autoplay class="w-100" preload="metadata">
                            <source src="${processedUrl}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
            </div>
        `;
    } else {
        console.warn('[Video Display] No output filename/url in results:', results);
    }

    if (hasFalls) {
        resultsAlert.className = 'alert-custom alert-danger-custom';
        resultsAlert.innerHTML = `
            <h5><i class="fas fa-exclamation-triangle me-2"></i>Fall Detected!</h5>
            <p class="mb-0">
                ${results.fall_summary ? escapeFallText(results.fall_summary) + '<br>' : 'A fall was detected in the video analysis. '}
                Please review the playback and details below. The fall summary is saved in the database.
            </p>
        `;

        // Show fall details
        let detailsHTML = videoHtml; // Add video at the top
        detailsHTML += '<div class="monitoring-card mt-3"><h5>Fall Details</h5><div class="table-responsive"><table class="table table-striped">';
        detailsHTML += '<thead><tr><th>Time in video</th><th>Confidence</th><th>Got up?</th><th>Summary</th></tr></thead><tbody>';

        results.falls_detected.forEach(fall => {
            const t = Number(fall.timestamp) || 0;
            const timeTxt = `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`;
            const confTxt = fall.confidence ? `${Math.round(fall.confidence)}%` : '—';
            const upTxt = fall.got_up
                ? `<span class="badge bg-success">Yes, after ${Number(fall.time_to_get_up_sec || 0).toFixed(1)} s</span>`
                : (fall.got_up === false
                    ? `<span class="badge bg-danger">No${fall.time_on_ground_sec != null ? ` (still down ${Number(fall.time_on_ground_sec).toFixed(1)} s)` : ''}</span>`
                    : '—');
            detailsHTML += `
                <tr>
                    <td>${timeTxt}</td>
                    <td>${confTxt}</td>
                    <td>${upTxt}</td>
                    <td>${escapeFallText(fall.summary || fall.message || 'Fall detected')}</td>
                </tr>
            `;
        });

        detailsHTML += '</tbody></table></div></div>';
        fallDetails.innerHTML = detailsHTML;

        // Play alert sound
        playAlertSound();

    } else {
        resultsAlert.className = 'alert-custom alert alert-success';
        resultsAlert.innerHTML = `
            <h5><i class="fas fa-check-circle me-2"></i>No Falls Detected</h5>
            <p class="mb-0">The video has been analyzed and no falls were detected.</p>
        `;

        let detailsHTML = videoHtml; // Add video even if no falls
        detailsHTML += '<div class="text-center mt-3 text-muted"><p>No abnormal events found during analysis.</p></div>';
        fallDetails.innerHTML = detailsHTML;
    }



    // Update stats and history
    loadStats();
    loadHistory();

    // Update charts and insights
    updateCharts(results);
}

// ── Camera Selector Logic ──────────────────────────────────────────────────

/**
 * Enumerate connected video devices using the browser MediaDevices API.
 * Prompts for camera permission if labels are empty (first run).
 */
async function refreshCameraList() {
    const select   = document.getElementById('cameraSelect');
    const statusEl = document.getElementById('cameraStatus');

    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        statusEl.innerHTML = '<i class="fas fa-exclamation-triangle me-1 text-warning"></i>Camera enumeration not supported in this browser.';
        return;
    }

    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Detecting cameras...';

    try {
        // Brief permission request so labels are populated
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (permErr) {
            console.warn('[Camera] Permission prompt error:', permErr.message);
        }

        const devices     = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        if (stream) stream.getTracks().forEach(t => t.stop());

        if (videoDevices.length === 0) {
            statusEl.innerHTML = '<i class="fas fa-times-circle me-1 text-danger"></i>No cameras detected.';
            select.innerHTML   = '<option value="">No cameras found</option>';
            return;
        }

        // Store previous selection
        const prevId = select.value;
        select.innerHTML = '';

        // Default option
        const defOpt = document.createElement('option');
        defOpt.value       = '';
        defOpt.textContent = 'Default Camera';
        select.appendChild(defOpt);

        videoDevices.forEach((device, idx) => {
            const label = device.label || `Camera ${idx}`;
            const opt   = document.createElement('option');
            opt.value       = device.deviceId;
            opt.textContent = label;
            if (device.deviceId === prevId) opt.selected = true;
            select.appendChild(opt);
        });

        selectedCameraDeviceId = select.value || null;
        statusEl.innerHTML = `<i class="fas fa-check-circle me-1 text-success"></i>${videoDevices.length} camera(s) detected.`;

    } catch (err) {
        console.error('[Camera] Enumeration error:', err);
        statusEl.innerHTML = '<i class="fas fa-exclamation-triangle me-1 text-danger"></i>Failed to enumerate cameras: ' + err.message;
    }
}

// Camera dropdown change handler — called from DOMContentLoaded below
function setupCameraSelector() {
    const select = document.getElementById('cameraSelect');
    if (select) {
        select.addEventListener('change', function () {
            selectedCameraDeviceId = this.value || null;
            console.log('[Camera] Selected deviceId:', selectedCameraDeviceId);
        });
    }

    // Auto-refresh camera list when user switches to Live tab
    const liveTabBtn = document.getElementById('live-tab');
    if (liveTabBtn) {
        liveTabBtn.addEventListener('shown.bs.tab', function () {
            refreshCameraList();
        });
    }
    // Live Monitoring is the first (default) tab now: fill the camera list on page load too
    const livePaneOnLoad = document.getElementById('live');
    if (livePaneOnLoad && livePaneOnLoad.classList.contains('active')) refreshCameraList();

    // Auto-refresh history when user switches to History tab
    const historyTabBtn = document.getElementById('history-tab');
    if (historyTabBtn) {
        historyTabBtn.addEventListener('shown.bs.tab', function () {
            loadHistory();
            loadHistoryDashboard();
        });
    }

    // Hot-plug detection
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', async () => {
            console.log('[Camera] Device change detected — refreshing list...');
            const livePane = document.getElementById('live');
            if (livePane && livePane.classList.contains('active')) {
                await refreshCameraList();
                const statusEl = document.getElementById('cameraStatus');
                if (statusEl) {
                    const prev = statusEl.innerHTML;
                    statusEl.innerHTML = '<i class="fas fa-plug me-1 text-primary"></i><strong>Camera list updated</strong> — a device was connected or removed.';
                    setTimeout(() => { statusEl.innerHTML = prev; }, 3000);
                }
            }
        });
        console.log('[Camera] Hot-plug detection active.');
    }
}

// ── IP Camera Test ──────────────────────────────────────────────────────────
async function testIpCamera() {
    const urlInput = document.getElementById('ipCameraUrl');
    const statusEl = document.getElementById('cameraStatus');
    const url = urlInput?.value?.trim();

    if (!url) {
        statusEl.innerHTML = '<i class="fas fa-exclamation-triangle me-1 text-warning"></i>Please enter a stream URL first.';
        return;
    }

    // Validate format
    if (!url.startsWith('rtsp://') && !url.startsWith('http://') && !url.startsWith('https://')) {
        statusEl.innerHTML = '<i class="fas fa-times-circle me-1 text-danger"></i>URL must start with <code>rtsp://</code>, <code>http://</code>, or <code>https://</code>';
        return;
    }

    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Testing connection via backend...';

    try {
        // Ask the backend to validate the URL (it tries cv2.VideoCapture briefly)
        const response = await fetch(`${API_BASE}/video-monitoring/test-ip-camera`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ camera_url: url })
        });

        const data = await response.json();
        if (response.ok && data.reachable) {
            statusEl.innerHTML = '<i class="fas fa-check-circle me-1 text-success"></i><strong>Connected!</strong> Camera is reachable. Click Start Monitoring.';
        } else {
            statusEl.innerHTML = `<i class="fas fa-times-circle me-1 text-danger"></i><strong>Unreachable:</strong> ${data.detail || 'Could not connect to the camera URL.'}`;
        }
    } catch (err) {
        // Backend endpoint may not exist yet — just do a format check
        statusEl.innerHTML = '<i class="fas fa-check-circle me-1 text-success"></i>URL format looks valid. Click Start Monitoring to connect.';
    }
}

// ── TF.js Live Monitoring ────────────────────────────────────────────────────

async function startLiveMonitoring() {
    if (_tfjsSession && _tfjsSession.isRunning) {
        showAlert('Monitoring is already active.', 'warning');
        return;
    }

    const sensitivity = document.getElementById('sensitivitySelect')?.value || 'medium';
    const deviceId    = selectedCameraDeviceId || null;

    // Show the video container & loading overlay immediately
    const liveVideoContainer = document.getElementById('liveVideoContainer');
    liveVideoContainer.classList.remove('d-none');
    document.getElementById('liveInfoBanner').classList.add('d-none');

    const loadingOverlay = document.getElementById('tfjsLoadingOverlay');
    const loadingMsg     = document.getElementById('tfjsLoadingMsg');
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
    if (loadingMsg)     loadingMsg.textContent = 'Loading MoveNet model…';

    // Update header UI
    document.getElementById('startLiveBtn').classList.add('d-none');
    document.getElementById('stopLiveBtn').classList.remove('d-none');
    const liveIndicator = document.getElementById('liveIndicator');
    if (liveIndicator) liveIndicator.innerHTML = '<span class="live-indicator"></span>';

    try {
        const videoEl  = document.getElementById('tfjs-video');
        const canvasEl = document.getElementById('tfjs-canvas');

        _tfjsSession = new TFJSLiveSession(videoEl, canvasEl, {
            sensitivity,

            // ── Fall confirmed (fires ONCE per fall; no repeats while the person is down) ──
            onFall(event) {
                const openFall = { at: Date.now(), eventId: null, longLie: false };
                _tfjsOpenFall = openFall;

                showFallBanner('fall', `Detected at ${new Date(event.timestamp).toLocaleTimeString()} — waiting for the person to get up`);
                _appendLiveAlert({ type: 'fall', timestamp: event.timestamp, confidence: event.confidence,
                    message: 'Beeping until the person gets up or you press "I am attending"' });
                startFallAlarm();

                // Save the fall in the database (fall_events); keep its id to record "got up" later
                _reportFallToBackend(event)
                    .then(id => { openFall.eventId = id; })
                    .catch(e => console.warn('[Fall] Backend report failed:', e));
            },

            // ── Every frame while on the floor: live timer towards the 2-minute limit ──
            onFloorTick({ secondsOnFloor, personVisible }) {
                updateFallTimer(secondsOnFloor, personVisible);
            },

            // ── 2 minutes on the floor: escalate (siren + caretaker alert) ──
            onLongLie({ timestamp, secondsOnFloor }) {
                const f = _tfjsOpenFall;
                if (f) f.longLie = true;
                showFallBanner('longlie', 'Not up after 2 minutes — please check on them NOW');
                _appendLiveAlert({ type: 'long_lie', timestamp,
                    message: `Still on the floor after ${fmtDuration(secondsOnFloor)} — caretaker alerted` });
                escalateFallAlarm();
                _reportLongLieToBackend(f, secondsOnFloor).catch(e => console.warn('[Fall] long-lie report failed:', e));
            },

            // ── Person stood up again ──
            onRecover({ timestamp, secondsOnFloor, withinLimit }) {
                const f = _tfjsOpenFall;
                _tfjsOpenFall = null;
                stopFallAlarm();
                playRecoveredChime();
                const verdict = withinLimit
                    ? `Got up after ${fmtDuration(secondsOnFloor)} — within 2 minutes ✓`
                    : `Got up after ${fmtDuration(secondsOnFloor)} — took longer than 2 minutes ⚠`;
                showFallBanner(withinLimit ? 'ok' : 'late', verdict);
                clearTimeout(_fallAlertTimer);
                _fallAlertTimer = setTimeout(hideFallBanner, 8000);
                _appendLiveAlert({ type: 'got_up', timestamp, message: verdict });
                if (f) _reportGotUpToBackend(f, secondsOnFloor).catch(e => console.warn('[Fall] got-up report failed:', e));
            },

            onFrame(features, fps) {
                const fpEl  = document.getElementById('metricFps');
                const angEl = document.getElementById('metricAngle');
                const aspEl = document.getElementById('metricAspect');

                if (fpEl)  fpEl.textContent  = fps.toFixed(1);
                if (angEl) angEl.textContent = features.torsoAngle  != null ? features.torsoAngle.toFixed(1)  + '°' : '--';
                if (aspEl) aspEl.textContent = features.aspectRatio != null ? features.aspectRatio.toFixed(2) : '--';
            },
        });

        // A click on "Start" is a user gesture: unlock audio now so the alarm can sound later
        ensureAlarmAudio();

        if (loadingMsg) loadingMsg.textContent = 'Opening camera…';
        await _tfjsSession.start(deviceId);

        // Hide loading overlay once running
        if (loadingOverlay) loadingOverlay.style.display = 'none';

        // Same camera also tracks the medicines in the room
        if (typeof TableWatch !== 'undefined') {
            TableWatch.attach(document.getElementById('tfjs-video'))
                .catch(e => console.warn('[Room camera] medicine tracking could not start:', e));
        }

        showAlert('✅ Room camera started — watching for falls and medicines', 'success');
        console.log('[startLiveMonitoring] TF.js session running at', sensitivity, 'sensitivity.');

    } catch (err) {
        console.error('[startLiveMonitoring] Error:', err);
        if (loadingOverlay) loadingOverlay.style.display = 'none';

        // Roll back UI
        document.getElementById('startLiveBtn').classList.remove('d-none');
        document.getElementById('stopLiveBtn').classList.add('d-none');
        liveVideoContainer.classList.add('d-none');
        document.getElementById('liveInfoBanner').classList.remove('d-none');
        if (liveIndicator) liveIndicator.innerHTML = '';

        let msg = 'Failed to start live monitoring.';
        if (err.name === 'NotAllowedError')  msg = 'Camera permission denied. Please allow camera access and try again.';
        if (err.name === 'NotFoundError')    msg = 'No camera found. Please connect a camera and refresh.';
        showAlert(msg, 'danger');
    }
}

function stopLiveMonitoring() {
    if (typeof TableWatch !== 'undefined') TableWatch.detach();   // medicine tracking uses the same camera
    if (_tfjsSession) {
        // Stopped while the person is still on the floor: save how long they were down
        if (_tfjsSession.isFallen && _tfjsOpenFall) {
            _reportStoppedDownToBackend(_tfjsOpenFall, _tfjsSession.secondsOnFloor)
                .catch(e => console.warn('[Fall] stopped-down report failed:', e));
        }
        _tfjsSession.stop();
        _tfjsSession = null;
    }
    _tfjsOpenFall = null;
    stopFallAlarm();

    // UI reset
    document.getElementById('startLiveBtn').classList.remove('d-none');
    document.getElementById('stopLiveBtn').classList.add('d-none');
    document.getElementById('liveVideoContainer').classList.add('d-none');
    document.getElementById('liveInfoBanner').classList.remove('d-none');

    const liveIndicator = document.getElementById('liveIndicator');
    if (liveIndicator) liveIndicator.innerHTML = '';

    const banner = document.getElementById('fallAlertBanner');
    if (banner) banner.classList.remove('show');

    // Reset metrics
    ['metricFps', 'metricAngle', 'metricAspect'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '--';
    });

    showAlert('Live monitoring stopped.', 'info');
    console.log('[stopLiveMonitoring] TF.js session stopped.');
}

/** Append a new entry to the live alerts list panel. */
function _appendLiveAlert(alert) {
    const alertsList = document.getElementById('liveAlertsList');
    if (!alertsList) return;
    const card = document.getElementById('liveAlertsCard');
    if (card) card.style.display = '';          // first event: show the card

    // Remove 'no alerts' placeholder
    const placeholder = alertsList.querySelector('p.text-muted');
    if (placeholder) placeholder.remove();

    const alertClass = (alert.type === 'fall' || alert.type === 'long_lie') ? 'danger'
        : (alert.type === 'got_up' ? 'success' : (alert.type === 'info' ? 'secondary' : 'warning'));
    const icon       = alert.type === 'fall' ? 'exclamation-triangle'
        : (alert.type === 'long_lie' ? 'stopwatch' : (alert.type === 'got_up' ? 'person-walking' : 'exclamation-circle'));
    // Browser detector sends 0-1, the camera engine sends 0-100: show both as a normal percentage
    const pct        = alert.confidence == null ? null : (alert.confidence <= 1 ? alert.confidence * 100 : alert.confidence);
    const conf       = pct != null ? ` &mdash; ${pct.toFixed(0)}% conf.` : '';
    const title      = { got_up: 'GOT UP', long_lie: 'LONG LIE (2+ MIN)', info: 'NOTE' }[alert.type] || alert.type.toUpperCase();
    const msg        = alert.message ? `${escapeFallText(alert.message)}<br>` : '';

    const div = document.createElement('div');
    div.className = `alert alert-${alertClass} mb-2`;
    div.innerHTML = `
        <small>
            <i class="fas fa-${icon} me-1"></i>
            <strong>${title}</strong>${conf}<br>
            ${msg}${new Date(alert.timestamp).toLocaleString()}
        </small>`;

    alertsList.insertBefore(div, alertsList.firstChild);

    // Keep only the last 10
    while (alertsList.children.length > 10) alertsList.removeChild(alertsList.lastChild);
}

/** Fire-and-forget: log fall event to backend for DB storage. */
async function _reportFallToBackend(event) {
    if (!token) return;
    const urlParams   = new URLSearchParams(window.location.search);
    const recipientId = urlParams.get('recipient_id');

    const body = {
        type:        'fall',
        timestamp:   event.timestamp,
        confidence:  event.confidence,
        source:      'tfjs_live',
    };
    if (recipientId) body.recipient_id = recipientId;

    const res = await fetch(`${API_BASE}/video-monitoring/log-event`, {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return data.event_id || null;      // used to record "got up" on the same fall
}

/** Person got up after a live fall: update the saved fall (waits briefly for its id). */
async function _reportGotUpToBackend(fall, secs) {
    if (!token) return;
    for (let i = 0; i < 50 && !fall.eventId; i++) await new Promise(r => setTimeout(r, 100));
    if (!fall.eventId) return;
    await fetch(`${API_BASE}/video-monitoring/log-event`, {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify({ type: 'got_up', event_id: fall.eventId, time_to_get_up_sec: Math.round(secs * 10) / 10 }),
    });
}

// --- Settings Functions ---
async function loadInactivitySettings() {
    try {
        const response = await fetch(`${API_BASE}/video-monitoring/get-inactivity-threshold`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
            const data = await response.json();
            document.getElementById('inactivityThreshold').value = data.threshold_seconds;
        }
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

async function saveInactivitySettings() {
    const threshold = parseInt(document.getElementById('inactivityThreshold').value);
    if (!threshold || threshold < 5 || threshold > 300) {
        showAlert('Please enter a value between 5 and 300 seconds', 'warning');
        return;
    }

    try {
        // 1. Update default setting
        const response = await fetch(`${API_BASE}/video-monitoring/set-inactivity-threshold`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ threshold_seconds: threshold })
        });

        if (!response.ok) {
            throw new Error('Failed to save default settings');
        }

        // 2. If live session is active, update it dynamically
        if (currentSessionId) {
            const liveResponse = await fetch(`${API_BASE}/video-monitoring/update-threshold/${currentSessionId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ threshold_seconds: threshold })
            });

            if (liveResponse.ok) {
                showAlert('Settings saved and live session updated!', 'success');
            } else {
                showAlert('Settings saved, but failed to update live session.', 'warning');
            }
        } else {
            showAlert('Settings saved successfully', 'success');
        }

    } catch (error) {
        console.error('Error saving settings:', error);
        showAlert('Failed to save settings', 'danger');
    }
}

// --- Alert Checking ---
async function checkLiveAlerts() {
    if (!currentSessionId) return;

    try {
        // Use the dedicated alerts endpoint
        const response = await fetch(`${API_BASE}/video-monitoring/session/${currentSessionId}/alerts`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 404) {
            console.warn('[Alerts] Session ended or not found. Stopping poll.');
            if (checkAlertsInterval) {
                clearInterval(checkAlertsInterval);
                checkAlertsInterval = null;
            }
            return;
        }

        if (!response.ok) return;

        const data = await response.json();
        // data.alerts is an array of alert objects
        if (data.alerts && data.alerts.length > 0) {
            displayLiveAlerts(data.alerts);
        }
    } catch (error) {
        console.error('Error checking alerts:', error);
    }
}

function displayLiveAlerts(alerts) {
    const alertsList = document.getElementById('liveAlertsList');
    const card = document.getElementById('liveAlertsCard');

    if (alerts.length === 0) {
        alertsList.innerHTML = '<p class="text-muted text-center">No active alerts</p>';
        if (card) card.style.display = 'none';  // nothing to show: keep the card hidden
        return;
    }
    if (card) card.style.display = '';

    let html = '';
    alerts.slice(-5).reverse().forEach(alert => {
        const alertClass = alert.type === 'fall' ? 'danger' : 'warning';
        const icon = alert.type === 'fall' ? 'exclamation-triangle' : 'exclamation-circle';

        html += `
            <div class="alert alert-${alertClass} mb-2">
                <small>
                    <i class="fas fa-${icon} me-1"></i>
                    <strong>${alert.type.toUpperCase()}</strong><br>
                    ${new Date(alert.timestamp).toLocaleString()}
                </small>
            </div>
        `;
    });

    alertsList.innerHTML = html;

    // Play sound for new fall alerts
    const latestAlert = alerts[alerts.length - 1];
    if (latestAlert.type === 'fall') {
        playAlertSound();
    }
}

// Utility Functions
function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at top of current tab
    const activeTab = document.querySelector('.tab-pane.active');
    activeTab.insertBefore(alertDiv, activeTab.firstChild);

    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// High-decibel Web Audio API siren alert for fall detection
let _alertAudioCtx = null;

function playAlertSound() {
    try {
        if (!_alertAudioCtx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                _alertAudioCtx = new AudioCtx();
            }
        }
        if (_alertAudioCtx && _alertAudioCtx.state === 'suspended') {
            _alertAudioCtx.resume();
        }

        if (_alertAudioCtx) {
            // Urgent, piercing emergency alarm pattern
            // 4 high-frequency square/sawtooth pulses for maximum audibility
            const now = _alertAudioCtx.currentTime;
            const beeps = [
                { time: 0.00, freq: 1600, dur: 0.20 },
                { time: 0.25, freq: 2200, dur: 0.20 },
                { time: 0.50, freq: 1600, dur: 0.20 },
                { time: 0.75, freq: 2600, dur: 0.40 }
            ];

            beeps.forEach(b => {
                const osc = _alertAudioCtx.createOscillator();
                const gain = _alertAudioCtx.createGain();

                osc.type = 'sawtooth'; // Piercing harmonics for maximum loudness
                osc.frequency.setValueAtTime(b.freq, now + b.time);

                // Peak volume envelope
                gain.gain.setValueAtTime(0.001, now + b.time);
                gain.gain.linearRampToValueAtTime(1.0, now + b.time + 0.02);
                gain.gain.setValueAtTime(1.0, now + b.time + b.dur - 0.03);
                gain.gain.linearRampToValueAtTime(0.001, now + b.time + b.dur);

                osc.connect(gain);
                gain.connect(_alertAudioCtx.destination);

                osc.start(now + b.time);
                osc.stop(now + b.time + b.dur);
            });
            console.log('[Alert] Loud emergency fall alarm played via Web Audio API');
            return;
        }
    } catch (err) {
        console.warn('Web Audio alarm error:', err);
    }

    // Fallback: standard audio element
    try {
        const audio = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3');
        audio.volume = 1.0;
        audio.play().catch(e => console.log('Could not play fallback alert sound'));
    } catch (e) {
        console.log('Alert sound not available');
    }
}

async function loadStats() {
    try {
        const urlParams = new URLSearchParams(window.location.search);
        const recipientId = urlParams.get('recipient_id');
        let statsUrl = `${API_BASE}/video-monitoring/stats`;
        if (recipientId) {
            statsUrl += `?care_recipient_id=${recipientId}`;
        }

        const response = await fetch(statsUrl, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('totalUploads').textContent = data.total_uploads;
            document.getElementById('totalAlerts').textContent = data.total_alerts;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/video-monitoring/history`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        const historyList = document.getElementById('historyList');

        if (!response.ok) {
            historyList.innerHTML = '<p class="text-danger">Failed to load history</p>';
            return;
        }

        const history = await response.json();

        if (history.length === 0) {
            historyList.innerHTML = '<p class="text-muted text-center">No monitoring history found</p>';
            return;
        }

        let html = '<div class="table-responsive"><table class="table table-hover">';
        html += '<thead><tr><th>Date</th><th>File</th><th>Size</th><th>Action</th></tr></thead><tbody>';

        history.forEach(item => {
            const date = new Date(item.timestamp).toLocaleString();
            const size = (item.size / (1024 * 1024)).toFixed(2) + ' MB';

            // CLEAN URL CONSTRUCTION
            // Ignore whatever backend says about processed_url
            // Use proven manual construction: http://127.0.0.1:8000/processed_videos/{filename}
            const cleanFilename = item.filename.split(/[/\\]/).pop();
            const videoUrl = `http://127.0.0.1:8000/processed_videos/${cleanFilename}`;

            html += `
                <tr>
                    <td>${date}</td>
                    <td>${item.filename}</td>
                    <td>${size}</td>
                    <td>
                        <a href="${videoUrl}" class="btn btn-sm btn-outline-primary" target="_blank" download>
                            <i class="fas fa-download"></i>
                        </a>
                        <button class="btn btn-sm btn-outline-info ms-1" onclick="previewVideo('${videoUrl}')">
                            <i class="fas fa-play"></i>
                        </button>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
        historyList.innerHTML = html;

    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('historyList').innerHTML = '<p class="text-danger">Error loading history</p>';
    }
}

// --- Chart Functions ---
let activityChart = null;
let postureChart = null;

function updateCharts(results) {
    const analysisCard = document.getElementById('analysisCard');
    analysisCard.classList.remove('d-none');

    // 1. Prepare Data
    const hasFalls = results.has_falls;
    // Mock data based on results (since backend doesn't return full timeline yet)
    // In a real scenario, backend would return time-series data
    const totalDuration = 100; // Mock percentage
    const fallDuration = hasFalls ? 5 : 0;
    const inactiveDuration = results.inactivity_duration || (hasFalls ? 10 : 30);
    const activeDuration = totalDuration - fallDuration - inactiveDuration;

    // 2. Activity Chart (Pie)
    const ctxActivity = document.getElementById('activityChart').getContext('2d');

    if (activityChart) activityChart.destroy();

    activityChart = new Chart(ctxActivity, {
        type: 'doughnut',
        data: {
            labels: ['Active', 'Inactive', 'Fall Event'],
            datasets: [{
                data: [activeDuration, inactiveDuration, fallDuration],
                backgroundColor: ['#34d399', '#f6c23e', '#fb7185'], // Updated theme colors
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
                title: { display: true, text: 'Activity Distribution' }
            }
        }
    });

    // 3. Posture/Health Chart (Bar)
    const ctxPosture = document.getElementById('postureChart').getContext('2d');
    if (postureChart) postureChart.destroy();

    postureChart = new Chart(ctxPosture, {
        type: 'bar',
        data: {
            labels: ['Mobility', 'Stability', 'Response Time'],
            datasets: [{
                label: 'Health Scores (0-10)',
                data: [
                    hasFalls ? 4 : 8, // Lower mobility if fall detected
                    hasFalls ? 2 : 9, // Lower stability if fall detected
                    hasFalls ? 7 : 0  // Response urgency
                ],
                backgroundColor: ['#4A90E2', '#6BB5FF', '#fb7185'] // Updated theme colors
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true, max: 10 } },
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Health Risk Assessment' }
            }
        }
    });

    // 4. Update Conclusion Text
    const conclusionDiv = document.querySelector('#analysisConclusion p');
    let conclusion = '';

    if (hasFalls) {
        conclusion = `<strong>CRITICAL ATTENTION NEEDED:</strong> A fall event was detected. 
        The stability score is low (2/10), indicating a high risk of recurrence. 
        Immediate physical assessment recommended.`;
        document.getElementById('analysisConclusion').classList.replace('border-primary', 'border-danger');
        document.querySelector('#analysisConclusion h6').className = 'text-danger';
    } else {
        conclusion = `<strong>Good Health Status:</strong> The subject shows good mobility and stability levels. 
        Activity balance is within normal range (${activeDuration}% active). 
        No immediate risks detected.`;
        document.getElementById('analysisConclusion').classList.replace('border-danger', 'border-primary');
        document.querySelector('#analysisConclusion h6').className = 'text-primary';
    }
    conclusionDiv.innerHTML = conclusion;
}


// ── Initialisation ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    loadUserInfo();
    setupUploadZone();
    loadStats();
    loadHistory();
    loadInactivitySettings();
    setupCameraSelector();  // Wire up camera dropdown + live-tab auto-detect

    // Refresh stats/history periodically
    setInterval(loadStats, 30000);
});

function handleLogout() {
    localStorage.removeItem('token');
    window.location.href = 'index.html';
}

// --- Video Preview Function ---
window.previewVideo = function (videoUrl) {
    console.log('[Preview] Opening video:', videoUrl);

    // Ensure modal element exists
    const modalEl = document.getElementById('videoPreviewModal');
    if (!modalEl) {
        console.error('Preview modal not found in DOM');
        alert('Video preview modal not found. Please refresh the page.');
        return;
    }

    const modal = new bootstrap.Modal(modalEl);
    const videoPlayer = document.getElementById('previewVideoPlayer');
    const videoSource = videoPlayer.querySelector('source');

    videoSource.src = videoUrl;
    videoPlayer.load();
    modal.show();

    // Stop video when modal is closed
    modalEl.addEventListener('hidden.bs.modal', function () {
        videoPlayer.pause();
        videoSource.src = '';
    }, { once: true });
};


// ============================================================================
//  FALL ALARM, 2-MINUTE TIMER AND BANNER
//  - Beeps every 3 s from the moment a fall is confirmed until the person gets up
//    or someone presses "I am attending".
//  - After 2 minutes on the floor: a louder siren + caretaker alert (backend).
//  - When the person stands up: alarm stops, a soft chime plays, the time is recorded.
// ============================================================================
const _fallAlarm = { timer: null, mode: null, acknowledged: false };
let _lastTimerPaint = 0;

function ensureAlarmAudio() {
    try {
        if (!_alertAudioCtx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) _alertAudioCtx = new AudioCtx();
        }
        if (_alertAudioCtx && _alertAudioCtx.state === 'suspended') _alertAudioCtx.resume();
    } catch (e) { /* audio not available */ }
}

function _tone(freqStart, freqEnd, start, dur, type = 'square', vol = 0.8) {
    if (!_alertAudioCtx) return;
    const osc = _alertAudioCtx.createOscillator();
    const gain = _alertAudioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freqStart, start);
    if (freqEnd !== freqStart) osc.frequency.linearRampToValueAtTime(freqEnd, start + dur);
    gain.gain.setValueAtTime(0.001, start);
    gain.gain.linearRampToValueAtTime(vol, start + 0.02);
    gain.gain.setValueAtTime(vol, start + dur - 0.03);
    gain.gain.linearRampToValueAtTime(0.001, start + dur);
    osc.connect(gain);
    gain.connect(_alertAudioCtx.destination);
    osc.start(start);
    osc.stop(start + dur);
}

/**
 * Long-lie alarm (2 minutes on the floor): a two-tone "nee-naw" emergency tone, clearly
 * different from the fall beeps, plus a spoken warning every few rounds.
 */
let _sirenRound = 0;
function playSiren() {
    ensureAlarmAudio();
    if (!_alertAudioCtx) return playAlertSound();
    try { if (window.speechSynthesis && window.speechSynthesis.speaking) return; } catch (e) { /* ignore */ }   // let the warning be heard
    const now = _alertAudioCtx.currentTime;
    for (let i = 0; i < 4; i++) {
        _tone(960, 960, now + i * 0.6, 0.28, 'triangle', 0.9);
        _tone(640, 640, now + i * 0.6 + 0.3, 0.28, 'triangle', 0.9);
    }
    if (_sirenRound++ % 4 === 0) _speakLongLieWarning();
}

function _speakLongLieWarning() {
    try {
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance('Attention. The patient has been on the floor for more than two minutes. Please check on them now.');
        u.rate = 0.95; u.volume = 1;
        setTimeout(() => window.speechSynthesis.speak(u), 2500);   // after the tone
    } catch (e) { /* speech not available */ }
}

/** Soft two-note chime when the person is back on their feet. */
function playRecoveredChime() {
    ensureAlarmAudio();
    if (!_alertAudioCtx) return;
    const now = _alertAudioCtx.currentTime;
    _tone(660, 660, now, 0.18, 'sine', 0.4);
    _tone(990, 990, now + 0.2, 0.3, 'sine', 0.4);
}

function startFallAlarm() {
    stopFallAlarm();
    _fallAlarm.mode = 'fall';
    _fallAlarm.acknowledged = false;
    playAlertSound();
    _fallAlarm.timer = setInterval(() => { if (!_fallAlarm.acknowledged) playAlertSound(); }, 3000);
    _setAckButton(true);
}

function escalateFallAlarm() {
    if (_fallAlarm.timer) clearInterval(_fallAlarm.timer);
    _fallAlarm.mode = 'longlie';
    // A long lie always sounds again, even if the first alarm was acknowledged
    _fallAlarm.acknowledged = false;
    playSiren();
    _sirenRound = 0;
    _fallAlarm.timer = setInterval(() => { if (!_fallAlarm.acknowledged) playSiren(); }, 3200);
    _setAckButton(true);
}

function stopFallAlarm() {
    if (_fallAlarm.timer) clearInterval(_fallAlarm.timer);
    _fallAlarm.timer = null;
    if (_fallAlarm.mode === 'longlie') { try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ } }
    _fallAlarm.mode = null;
    _setAckButton(false);
}

/** "I am attending": silence the alarm; the on-floor timer and recording continue. */
function acknowledgeFallAlarm() {
    _fallAlarm.acknowledged = true;
    try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    _setAckButton(false);
    const st = document.getElementById('fallAlertStatus');
    if (st) st.textContent = 'Alarm silenced — someone is attending. Timer still running.';
    _appendLiveAlert({ type: 'info', timestamp: new Date().toISOString(), message: 'Alarm acknowledged by caretaker' });
}

function _setAckButton(show) {
    const b = document.getElementById('fallAckBtn');
    if (b) b.style.display = show ? 'inline-block' : 'none';
}

function showFallBanner(kind, statusText) {
    const banner = document.getElementById('fallAlertBanner');
    if (!banner) return;
    const title = document.getElementById('fallAlertTitle');
    const status = document.getElementById('fallAlertStatus');
    const colors = {
        fall:    'linear-gradient(135deg, #ff3b3b, #c0392b)',
        longlie: 'linear-gradient(135deg, #8b0000, #4a0000)',
        ok:      'linear-gradient(135deg, #2e9d5b, #1e7a44)',
        late:    'linear-gradient(135deg, #d97706, #b45309)',
    };
    const titles = {
        fall: 'FALL DETECTED!', longlie: 'STILL ON THE FLOOR — 2 MINUTES PASSED',
        ok: 'PERSON GOT UP', late: 'PERSON GOT UP (LATE)',
    };
    banner.style.background = colors[kind] || colors.fall;
    banner.style.animation = (kind === 'ok' || kind === 'late') ? 'none' : '';
    if (title) title.textContent = titles[kind] || titles.fall;
    if (status) status.textContent = statusText || '';
    if (kind === 'ok' || kind === 'late') {
        const timer = document.getElementById('fallFloorTimer');
        if (timer) timer.textContent = '';
    }
    banner.classList.add('show');
    const timeEl = document.getElementById('fallAlertTime');
    if (timeEl && kind === 'fall') timeEl.textContent = '';
}

function hideFallBanner() {
    const banner = document.getElementById('fallAlertBanner');
    if (banner) banner.classList.remove('show');
}

function updateFallTimer(secondsOnFloor, personVisible) {
    const now = Date.now();
    if (now - _lastTimerPaint < 400) return;          // repaint ~2x per second
    _lastTimerPaint = now;
    const el = document.getElementById('fallFloorTimer');
    if (!el) return;
    const left = Math.max(0, 120 - secondsOnFloor);
    el.textContent = secondsOnFloor < 120
        ? `On the floor: ${fmtDuration(secondsOnFloor)}  ·  ${fmtDuration(left)} left to get up`
        : `On the floor: ${fmtDuration(secondsOnFloor)}  ·  over the 2-minute limit`;
    if (!personVisible) el.textContent += '  ·  (not visible to camera)';
}

/** 2 minutes on the floor: mark the fall as a long lie and alert the caretaker. */
async function _reportLongLieToBackend(fall, secs) {
    if (!token || !fall) return;
    for (let i = 0; i < 50 && !fall.eventId; i++) await new Promise(r => setTimeout(r, 100));
    if (!fall.eventId) return;
    await fetch(`${API_BASE}/video-monitoring/log-event`, {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify({ type: 'long_lie', event_id: fall.eventId, time_on_ground_sec: Math.round(secs) }),
    });
}

/** Monitoring stopped while the person was still down: record how long they were on the floor. */
async function _reportStoppedDownToBackend(fall, secs) {
    if (!token || !fall || !fall.eventId) return;
    await fetch(`${API_BASE}/video-monitoring/log-event`, {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify({ type: 'stopped_down', event_id: fall.eventId, time_on_ground_sec: Math.round(secs) }),
        keepalive: true,
    });
}

// ============================================================================
//  HISTORY DASHBOARD: falls (2-minute rule), medicine checks
// ============================================================================
let _histDays = 30;

function _fmtMinSec(sec) {
    if (sec == null || isNaN(sec)) return '--';
    sec = Math.round(sec);
    const m = Math.floor(sec / 60), s = sec % 60;
    return m ? `${m} min ${s} s` : `${s} s`;
}

function _histErr(e, what) {
    if (String(e && e.message) === '401') {
        // login token too old: renew it quietly and load again (no "log in again" message)
        if (window.CaretakerSession && !_histErr._renewing && Date.now() - (_histErr._lastRenew || 0) > 60000) {
            _histErr._renewing = true; _histErr._lastRenew = Date.now();
            window.CaretakerSession.refresh().finally(() => {
                setTimeout(() => { _histErr._renewing = false; loadHistoryDashboard(); }, 500);
            });
        }
        return '<p class="text-muted mb-0">Loading ' + what + '…</p>';
    }
    return '<p class="text-danger mb-0">Could not load ' + what + '. Is the server running?</p>';
}

async function loadHistoryDashboard() {
    const auth = { headers: { 'Authorization': `Bearer ${token}` } };
    // patient from the URL, or (page opened directly) the caretaker's own patient
    let rid = new URLSearchParams(window.location.search).get('recipient_id');
    if (!rid) {
        try {
            const r = await fetch(`${API_BASE}/video-monitoring/default-recipient`, auth);
            if (r.ok) rid = (await r.json()).recipient_id;
            else if (r.status === 401) {
                const msg = _histErr(new Error('401'), 'the history');
                ['histFalls', 'histTable', 'histSightings'].forEach(id => { const el = document.getElementById(id); if (el) el.innerHTML = msg; });
                return;
            }
        } catch (e) { /* stays empty */ }
    }
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.innerHTML = v; };

    // ── Falls ──
    if (!rid) {
        set('histFalls', '<p class="text-muted">Open this page from a care recipient\'s profile to see their fall history.</p>');
    } else {
        try {
            const r = await fetch(`${API_BASE}/video-monitoring/recipient/${rid}/fall-analysis?days=${_histDays}`, auth);
            if (!r.ok) throw new Error(r.status);
            const d = await r.json();
            set('hkFalls', d.total_falls);
            set('hkFallsSub', `previous ${d.days} days: ${d.previous_period_falls}`);
            set('hkOk', d.got_up_within_limit);
            set('hkOkSub', d.avg_time_to_get_up_sec != null ? `avg ${_fmtMinSec(d.avg_time_to_get_up_sec)} to get up` : '&nbsp;');
            set('hkLong', d.long_lies);
            const risk = d.risk_level;
            set('histRisk', risk === 'high' ? '<span class="hist-badge bad"><i class="fas fa-triangle-exclamation"></i> Fall risk: High</span>'
                : risk === 'moderate' ? '<span class="hist-badge warn"><i class="fas fa-circle-exclamation"></i> Fall risk: Moderate</span>'
                : '<span class="hist-badge ok"><i class="fas fa-check"></i> Fall risk: Low</span>');
            set('histInsights', (d.insights || []).map(t => `<div class="hist-insight"><i class="fas fa-lightbulb me-1 text-warning"></i>${escapeFallText(t)}</div>`).join(''));
            const src = s => ({ tfjs_live: 'Live monitor', live_stream: 'Home camera', upload: 'Uploaded video' }[s] || s || '--');
            const rows = (d.incidents || []).map(i => {
                const when = new Date(i.occurred_at).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
                const up = i.got_up && i.time_to_get_up_sec != null ? _fmtMinSec(i.time_to_get_up_sec)
                    : i.got_up ? 'Got up' : (i.time_on_ground_sec ? `still down after ${_fmtMinSec(i.time_on_ground_sec)}` : 'Not recorded');
                const res = i.outcome === 'ok' ? '<span class="hist-badge ok"><i class="fas fa-check"></i> Within 2 min</span>'
                    : i.outcome === 'long_lie' ? '<span class="hist-badge bad"><i class="fas fa-stopwatch"></i> Long lie</span>'
                    : '<span class="hist-badge na"><i class="fas fa-question"></i> Unknown</span>';
                return `<tr><td>${when}</td><td>${escapeFallText(src(i.source))}${i.video ? '<br><small class="text-muted">' + escapeFallText(i.video) + '</small>' : ''}</td>
                        <td>${i.confidence ? i.confidence + '%' : '--'}</td><td>${up}</td><td>${res}</td></tr>`;
            });
            set('histFalls', rows.length
                ? `<div class="table-responsive"><table class="table table-sm table-hover align-middle mb-0">
                     <thead><tr><th>When</th><th>Detected by</th><th>Conf.</th><th>Got up after</th><th>Result</th></tr></thead>
                     <tbody>${rows.join('')}</tbody></table></div>`
                : `<p class="text-muted mb-0">No falls in the last ${d.days} days. 👍</p>`);
        } catch (e) {
            console.warn('[History] fall analysis failed', e);
            set('histFalls', _histErr(e, 'the fall log'));
        }
    }

    // ── Medicines ──
    try {
        const r = await fetch(`${API_BASE}/video-monitoring/table-watch/events?days=${Math.min(_histDays, 90)}${rid ? '&recipient_id=' + rid : ''}`, auth);
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        set('hkMed', d.summary.medicine_taken);
        set('hkMedSub', d.summary.medicine_alerts ? `${d.summary.medicine_alerts} missed-dose alert(s)` : 'no missed doses');
        const label = { baseline: 'Seen / placed', taken: 'Taken', missing: 'Moved', returned: 'Back', untouched_alert: 'Alert' };
        const cls = { taken: 'ok', untouched_alert: 'bad', missing: 'warn' };
        const items = (d.events || []).filter(e => e.item !== 'water').slice(0, 40).map(e => `
            <div class="d-flex gap-2 align-items-start py-1 border-bottom small">
                <i class="fas ${e.item === 'water' ? 'fa-glass-water' : 'fa-pills'} mt-1" style="color:${e.item === 'water' ? '#2f72b8' : '#b0566b'}"></i>
                <div class="flex-grow-1">${escapeFallText(e.detail || e.event)}<br>
                    <span class="text-muted">${new Date(e.occurred_at).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</span></div>
                <span class="hist-badge ${cls[e.event] || 'na'}">${label[e.event] || e.event}</span>
            </div>`);
        set('histTable', items.length ? items.join('')
            : '<p class="text-muted mb-0">No medicine events yet. Start the room camera in <strong>Live Monitoring</strong>.</p>');
    } catch (e) {
        console.warn('[History] table watch failed', e);
        set('histTable', _histErr(e, 'the medicine log'));
    }

    // ── Where the camera found each medicine ──
    try {
        const r = await fetch(`${API_BASE}/video-monitoring/medicine-sightings?days=${_histDays}${rid ? '&recipient_id=' + rid : ''}`, auth);
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        const fmt = iso => new Date(iso).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
        const dur = mins => mins < 1 ? 'a moment' : mins < 60 ? `${Math.round(mins)} min` : `${Math.floor(mins / 60)} h ${Math.round(mins % 60)} min`;
        set('histSightSummary', d.medicines.length
            ? d.medicines.map(m => `<strong>${escapeFallText(m.name)}</strong>: ${m.visits} sighting${m.visits === 1 ? '' : 's'}`).join(' · ')
            : '');
        const rows = (d.sightings || []).map(s => `
            <tr>
                <td><i class="fas fa-pills me-1" style="color:#b0566b;"></i>${escapeFallText(s.medicine)}</td>
                <td>${escapeFallText((s.location || '').replace(/^probably /, ''))}</td>
                <td>${fmt(s.first_seen_at)}${s.minutes >= 1 ? ` <span class="text-muted">· for ${dur(s.minutes)}</span>` : ''}</td>
                <td>${s.confidence != null ? Math.round(s.confidence) + '%' : '--'}</td>
                <td>${s.probable ? '<span class="hist-badge warn"><i class="fas fa-question"></i> Probably</span>'
                                 : '<span class="hist-badge ok"><i class="fas fa-check"></i> Seen</span>'}</td>
            </tr>`);
        set('histSightings', rows.length
            ? `<div class="table-responsive" style="max-height:360px; overflow-y:auto;"><table class="table table-sm table-hover align-middle mb-0">
                 <thead><tr><th>Medicine</th><th>Where</th><th>When</th><th>Match</th><th>Status</th></tr></thead>
                 <tbody>${rows.join('')}</tbody></table></div>`
            : `<p class="text-muted mb-0">No medicine sightings in the last ${d.days} days. Start the room camera in <strong>Live Monitoring</strong> with a medicine in view.</p>`);
    } catch (e) {
        console.warn('[History] medicine sightings failed', e);
        set('histSightings', _histErr(e, 'the medicine sightings'));
    }
}

document.addEventListener('click', e => {
    const b = e.target.closest('#histRange button');
    if (!b) return;
    _histDays = Number(b.dataset.days) || 30;
    document.querySelectorAll('#histRange button').forEach(x => x.classList.toggle('active', x === b));
    loadHistoryDashboard();
});

// Keep the History tab up to date while it is open (new falls / medicine sightings appear by themselves)
setInterval(() => {
    const pane = document.getElementById('history');
    if (pane && pane.classList.contains('active') && !document.hidden) loadHistoryDashboard();
}, 30000);
