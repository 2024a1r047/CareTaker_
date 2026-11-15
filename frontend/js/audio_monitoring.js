
// Audio Monitoring JavaScript
const API_BASE = 'http://127.0.0.1:8000'; // Socket.IO connects to root usually, or base URL

// ── Authentication Check ───────────────────────────────────────────
const token = localStorage.getItem('token');
if (!token) {
    window.location.href = 'index.html';
}

// ── Socket.IO ──────────────────────────────────────────────────────
const socket = io(API_BASE, {
    transports: ['websocket'],
    upgrade: false
});

// Get recipient ID from URL parameters
const urlParams = new URLSearchParams(window.location.search);
const recipientId = urlParams.get('recipient_id');

socket.on('connect', () => {
    console.log("Connected to WebSocket");
    if (token) {
        socket.emit('authenticate', {
            token: token,
            care_recipient_id: recipientId ? parseInt(recipientId) : null
        });
    }
});

socket.on('auth_ack', (data) => {
    console.log("Authentication successful:", data);
});

socket.on('auth_error', (data) => {
    console.error("Authentication failed:", data);
    alert("Authentication failed. Please login again.");
    window.location.href = 'index.html';
});

// ── DOM Refs ───────────────────────────────────────────────────────
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const waveCanvas = document.getElementById("waveform-canvas");
const spectroCanvas = document.getElementById("spectrogram-canvas");
const alertList = document.getElementById("alert-list");
const btnClearLogs = document.getElementById("btn-clear-logs");
const emptyAlerts = document.getElementById("empty-alerts");

// Class confidence bars & values
const coughBar = document.getElementById("bar-cough");
const sneezeBar = document.getElementById("bar-sneeze");
const talkingBar = document.getElementById("bar-talking");
const noiseBar = document.getElementById("bar-noise");
const coughVal = document.getElementById("val-cough");
const sneezeVal = document.getElementById("val-sneeze");
const talkingVal = document.getElementById("val-talking");
const noiseVal = document.getElementById("val-noise");
const predLabel = document.getElementById("predicted-label");
const predConf = document.getElementById("predicted-conf");
const predIcon = document.getElementById("predicted-icon");

// Stats counters
const statCough = document.getElementById("stat-cough");
const statSneeze = document.getElementById("stat-sneeze");
const statTotal = document.getElementById("stat-total");

// ── State ──────────────────────────────────────────────────────────
let audioCtx = null;
let workletNode = null;
let sourceNode = null;
let analyserNode = null;
let stream = null;
let isMonitoring = false;
let totalAlerts = 0;
let coughCount = 0;
let sneezeCount = 0;
let animFrameId = null;
let spectroFrameId = null;
let lastWaveTime = 0;
let lastSpectroTime = 0;
const WAVE_FPS = 30;
const SPECTRO_FPS = 15;

const TARGET_SR = 16000;
const CHUNK_MS = 500;
const CHUNK_SAMPLES = TARGET_SR * CHUNK_MS / 1000;
const OVERLAP = 0.5;
let nativeSR = 44100;
let pcmBuffer = [];

const SPECTRO_COLS = 200;
let spectroHistory = [];

// ── Helpers ────────────────────────────────────────────────────────
function downsample(buffer, fromRate, toRate) {
    if (fromRate === toRate) return buffer;
    const ratio = fromRate / toRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
        result[i] = buffer[Math.min(Math.round(i * ratio), buffer.length - 1)];
    }
    return result;
}

function timestamp() {
    return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

function classIcon(cls) {
    switch (cls) {
        case "Cough": return "🫁";
        case "Sneeze": return "🤧";
        case "Talking": return "🗣️";
        case "Noise": return "🔇";
        default: return "🎙️";
    }
}

function handleLogout() {
    localStorage.removeItem('token'); sessionStorage.removeItem('token');
    window.location.href = 'index.html';
}

function drawWaveform(ts) {
    if (!analyserNode || !isMonitoring) return;
    animFrameId = requestAnimationFrame(drawWaveform);
    if (ts - lastWaveTime < 1000 / WAVE_FPS) return;
    lastWaveTime = ts;

    const ctx = waveCanvas.getContext("2d");
    const W = waveCanvas.width;
    const H = waveCanvas.height;
    const bufLen = analyserNode.fftSize;
    const data = new Float32Array(bufLen);
    analyserNode.getFloatTimeDomainData(data);

    ctx.clearRect(0, 0, W, H);

    // Centre line
    ctx.strokeStyle = "rgba(0,0,0,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    // Waveform gradient
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "#00f0ff");
    grad.addColorStop(0.5, "#7b61ff");
    grad.addColorStop(1, "#ff4d6d");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.beginPath();

    const sliceW = W / bufLen;
    for (let i = 0; i < bufLen; i++) {
        const y = (data[i] + 1) / 2 * H;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * sliceW, y);
    }
    ctx.stroke();
}

// ── Spectrogram Drawing ────────────────────────────────────────────
function drawSpectrogram(ts) {
    if (!analyserNode || !isMonitoring) return;
    spectroFrameId = requestAnimationFrame(drawSpectrogram);
    if (ts - lastSpectroTime < 1000 / SPECTRO_FPS) return;
    lastSpectroTime = ts;

    const ctx = spectroCanvas.getContext("2d");
    const W = spectroCanvas.width;
    const H = spectroCanvas.height;
    const bins = analyserNode.frequencyBinCount;
    const data = new Uint8Array(bins);
    analyserNode.getByteFrequencyData(data);

    spectroHistory.push(data.slice());
    if (spectroHistory.length > SPECTRO_COLS) spectroHistory.shift();

    ctx.clearRect(0, 0, W, H);
    const colW = W / SPECTRO_COLS;

    for (let col = 0; col < spectroHistory.length; col++) {
        const colData = spectroHistory[col];
        const binH = H / colData.length;
        for (let b = 0; b < colData.length; b++) {
            const val = colData[b] / 255;
            const hue = 240 - val * 240;
            ctx.fillStyle = `hsl(${hue}, 100%, ${val * 60 + 10}%)`;
            ctx.fillRect(col * colW, H - (b + 1) * binH, colW + 1, binH + 1);
        }
    }
}

// ── Start / Stop / Reset Monitoring ────────────────────────────────
async function startMonitoring() {
    if (isMonitoring) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
    } catch (err) {
        console.error("Microphone access denied:", err);
        alert("Microphone access denied. Please allow microphone using browser settings.");
        return;
    }

    audioCtx = new AudioContext();
    nativeSR = audioCtx.sampleRate;

    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048;
    analyserNode.smoothingTimeConstant = 0.7;

    sourceNode = audioCtx.createMediaStreamSource(stream);
    sourceNode.connect(analyserNode);

    // Load Audio Worklet
    try {
        await audioCtx.audioWorklet.addModule("js/audio-processor.js");
        workletNode = new AudioWorkletNode(audioCtx, "audio-chunk-processor");
        sourceNode.connect(workletNode);
    } catch (e) {
        console.error("Failed to load audio worklet:", e);
        alert("Failed to load audio processor. Make sure you are serving this file via a server (not file://).");
        return;
    }

    pcmBuffer = [];
    workletNode.port.onmessage = (e) => {
        const raw = e.data;
        const down = downsample(raw, nativeSR, TARGET_SR);
        pcmBuffer.push(...down);

        if (pcmBuffer.length > CHUNK_SAMPLES * 10) {
            pcmBuffer = [];
            return;
        }

        const slideBy = Math.floor(CHUNK_SAMPLES * (1 - OVERLAP));
        let chunksSent = 0;

        while (pcmBuffer.length >= CHUNK_SAMPLES && chunksSent < 2) {
            const chunk = new Float32Array(pcmBuffer.slice(0, CHUNK_SAMPLES));
            socket.emit("audio_chunk", chunk.buffer);
            pcmBuffer.splice(0, slideBy);
            chunksSent++;
        }

        if (pcmBuffer.length > CHUNK_SAMPLES * 4) {
            pcmBuffer.splice(0, pcmBuffer.length - CHUNK_SAMPLES);
        }
    };

    isMonitoring = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    statusDot.classList.add("live");
    statusText.textContent = "LIVE";

    drawWaveform();
    drawSpectrogram();
}

function stopMonitoring() {
    if (!isMonitoring) return;
    isMonitoring = false;

    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (spectroFrameId) { cancelAnimationFrame(spectroFrameId); spectroFrameId = null; }

    pcmBuffer = [];
    spectroHistory = [];

    btnStart.disabled = false;
    btnStop.disabled = true;
    statusDot.classList.remove("live");
    statusText.textContent = "OFFLINE";

    const waveCtx = waveCanvas.getContext("2d");
    waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);

    const specCtx = spectroCanvas.getContext("2d");
    specCtx.clearRect(0, 0, spectroCanvas.width, spectroCanvas.height);
}

function resetAll() {
    stopMonitoring();
    clearLogs();
    coughCount = 0;
    sneezeCount = 0;
    updateStats();

    predLabel.textContent = "Waiting…";
    predLabel.className = "pred-label";
    predConf.textContent = "—";
    predIcon.textContent = "🎙️";
    predIcon.className = "pred-icon";

    updateBar(coughBar, coughVal, 0);
    updateBar(sneezeBar, sneezeVal, 0);
    updateBar(talkingBar, talkingVal, 0);
    updateBar(noiseBar, noiseVal, 0);

    // Tell server to reset its buffer too
    socket.emit("reset");
}

// ── Socket Events ──────────────────────────────────────────────────
socket.on("classification", (data) => {
    updateBar(coughBar, coughVal, data.Cough);
    updateBar(sneezeBar, sneezeVal, data.Sneeze);
    updateBar(talkingBar, talkingVal, data.Talking);
    updateBar(noiseBar, noiseVal, data.Noise);

    predLabel.textContent = data.predicted;
    predConf.textContent = data.confidence + "%";
    predLabel.className = "pred-label " + data.predicted.toLowerCase();
    predIcon.className = "pred-icon " + data.predicted.toLowerCase();
    predIcon.textContent = classIcon(data.predicted);
});

socket.on("alert", (data) => {
    if (emptyAlerts) emptyAlerts.style.display = "none";

    totalAlerts++;
    if (data.type === "Cough") coughCount++;
    if (data.type === "Sneeze") sneezeCount++;
    updateStats();

    const entry = document.createElement("div");
    entry.className = `alert-entry ${data.type.toLowerCase()}`;
    entry.innerHTML = `
        <span class="alert-time">[${data.timestamp}]</span>
        <span class="alert-tag tag-${data.type.toLowerCase()}">${data.type}</span>
        <span class="alert-conf">${data.confidence}% confidence</span>
    `;

    alertList.prepend(entry);
    requestAnimationFrame(() => entry.classList.add("visible"));
});

// ── Utilities ──────────────────────────────────────────────────────
function updateBar(barEl, valEl, pct) {
    barEl.style.width = pct + "%";
    valEl.textContent = pct + "%";
    if (pct > 70) barEl.classList.add("high");
    else barEl.classList.remove("high");
}

function updateStats() {
    if (statCough) statCough.textContent = coughCount;
    if (statSneeze) statSneeze.textContent = sneezeCount;
    if (statTotal) statTotal.textContent = totalAlerts;
}

function clearLogs() {
    const entries = alertList.querySelectorAll(".alert-entry");
    entries.forEach(e => e.remove());
    totalAlerts = 0;
    coughCount = 0;
    sneezeCount = 0;
    updateStats();
    if (emptyAlerts) emptyAlerts.style.display = "flex"; // Re-show empty state
}

// ── Event Listeners ────────────────────────────────────────────────
btnStart.addEventListener("click", startMonitoring);
btnStop.addEventListener("click", stopMonitoring);
btnReset.addEventListener("click", resetAll);
btnClearLogs.addEventListener("click", clearLogs);

// Resize canvases
function resizeCanvases() {
    const waveWrap = waveCanvas.parentElement;
    const spectroWrap = spectroCanvas.parentElement;

    // Use device pixel ratio for sharper rendering if desired, but standard is fine
    waveCanvas.width = waveWrap.clientWidth;
    waveCanvas.height = waveWrap.clientHeight;

    spectroCanvas.width = spectroWrap.clientWidth;
    spectroCanvas.height = spectroWrap.clientHeight;
}
window.addEventListener("resize", resizeCanvases);
window.addEventListener("load", resizeCanvases);
