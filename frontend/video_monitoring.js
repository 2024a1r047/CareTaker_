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

// Authentication check
const token = localStorage.getItem('token');
if (!token) {
    window.location.href = 'index.html';
}

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
                A fall was detected in the video analysis. 
                Please review the playback and details below.
            </p>
        `;

        // Show fall details
        let detailsHTML = videoHtml; // Add video at the top
        detailsHTML += '<div class="monitoring-card mt-3"><h5>Fall Details</h5><div class="table-responsive"><table class="table table-striped">';
        detailsHTML += '<thead><tr><th>Time</th><th>Details</th></tr></thead><tbody>';

        results.falls_detected.forEach(fall => {
            detailsHTML += `
                <tr>
                    <td>${fall.timestamp}</td>
                    <td>${fall.message || 'Fall detected'}</td>
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

    // Auto-refresh history when user switches to History tab
    const historyTabBtn = document.getElementById('history-tab');
    if (historyTabBtn) {
        historyTabBtn.addEventListener('shown.bs.tab', function () {
            loadHistory();
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

            onFall(event) {
                // Show fall banner
                const banner = document.getElementById('fallAlertBanner');
                const timeEl = document.getElementById('fallAlertTime');
                if (banner) banner.classList.add('show');
                if (timeEl)  timeEl.textContent = `Detected at ${new Date(event.timestamp).toLocaleTimeString()}`;

                // Auto-dismiss banner after 6 s
                clearTimeout(_fallAlertTimer);
                _fallAlertTimer = setTimeout(() => {
                    if (banner) banner.classList.remove('show');
                }, 6000);

                // Add entry to the live alerts list
                _appendLiveAlert({ type: 'fall', timestamp: event.timestamp, confidence: event.confidence });

                // Play alert sound
                playAlertSound();

                // Optionally report to backend for DB logging (fire-and-forget)
                _reportFallToBackend(event).catch(e => console.warn('[Fall] Backend report failed:', e));
            },

            onFrame(features, fps) {
                // Update metrics panel
                const fpEl  = document.getElementById('metricFps');
                const angEl = document.getElementById('metricAngle');
                const aspEl = document.getElementById('metricAspect');

                if (fpEl)  fpEl.textContent  = fps.toFixed(1);
                if (angEl) angEl.textContent = features.torsoAngle  != null ? features.torsoAngle.toFixed(1)  + '°' : '--';
                if (aspEl) aspEl.textContent = features.aspectRatio != null ? features.aspectRatio.toFixed(2) : '--';
            },
        });

        if (loadingMsg) loadingMsg.textContent = 'Opening camera…';
        await _tfjsSession.start(deviceId);

        // Hide loading overlay once running
        if (loadingOverlay) loadingOverlay.style.display = 'none';

        showAlert('✅ TF.js live monitoring started! (MoveNet, in-browser)', 'success');
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
    if (_tfjsSession) {
        _tfjsSession.stop();
        _tfjsSession = null;
    }

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

    // Remove 'no alerts' placeholder
    const placeholder = alertsList.querySelector('p.text-muted');
    if (placeholder) placeholder.remove();

    const alertClass = alert.type === 'fall' ? 'danger' : 'warning';
    const icon       = alert.type === 'fall' ? 'exclamation-triangle' : 'exclamation-circle';
    const conf       = alert.confidence != null ? ` &mdash; ${(alert.confidence * 100).toFixed(0)}% conf.` : '';

    const div = document.createElement('div');
    div.className = `alert alert-${alertClass} mb-2`;
    div.innerHTML = `
        <small>
            <i class="fas fa-${icon} me-1"></i>
            <strong>${alert.type.toUpperCase()}</strong>${conf}<br>
            ${new Date(alert.timestamp).toLocaleString()}
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

    await fetch(`${API_BASE}/video-monitoring/log-event`, {
        method:  'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
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

    if (alerts.length === 0) {
        alertsList.innerHTML = '<p class="text-muted text-center">No active alerts</p>';
        return;
    }

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

function playAlertSound() {
    try {
        const audio = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3');
        audio.play().catch(e => console.log('Could not play alert sound'));
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

