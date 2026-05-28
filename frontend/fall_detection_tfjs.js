/**
 * fall_detection_tfjs.js — In-browser fall detection using TensorFlow.js + MoveNet
 *
 * Mirrors the Python FallDetector logic:
 *   - Torso angle (shoulder → hip vs vertical axis)
 *   - Body aspect ratio (width / height of bounding box)
 *   - Vertical hip speed (px / second)
 *   - Temporal smoothing (5-frame rolling mean)
 *   - 2-of-3 criteria required to trigger a fall
 *   - Cooldown period to suppress duplicate alerts
 *
 * COCO keypoint indices (MoveNet uses the same):
 *   0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
 *   5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
 *   9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
 *   13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
 */

'use strict';

// ── Skeleton connections for drawing ─────────────────────────────────────────
const SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],          // head
    [5, 6],                                     // shoulders
    [5, 7], [7, 9],                             // left arm
    [6, 8], [8, 10],                            // right arm
    [5, 11], [6, 12], [11, 12],                 // torso
    [11, 13], [13, 15],                         // left leg
    [12, 14], [14, 16],                         // right leg
];

// Keypoint indices
const L_SHO = 5, R_SHO = 6;
const L_HIP = 11, R_HIP = 12;
const L_KNEE = 13, R_KNEE = 14;

// ── Maths helpers ─────────────────────────────────────────────────────────────

/**
 * Angle of vector p1→p2 relative to the vertical axis (degrees).
 * Returns 90° when perfectly horizontal, 0° when perfectly vertical.
 */
function angleDeg(p1, p2) {
    const vx = p2.x - p1.x;
    const vy = p2.y - p1.y;
    const norm = Math.max(1e-6, Math.sqrt(vx * vx + vy * vy));
    const angH = Math.abs((Math.atan2(vy, vx) * 180) / Math.PI);
    return 90.0 - Math.min(angH, 180 - angH);
}

/**
 * Body bounding box aspect ratio: width / height.
 * High ratio → person is more horizontal (lying down).
 */
function calcAspectRatio(keypoints) {
    const valid = keypoints.filter(kp => kp.score > 0.2 && kp.x > 0 && kp.y > 0);
    if (valid.length < 3) return 0;

    const xs = valid.map(k => k.x);
    const ys = valid.map(k => k.y);
    const width  = Math.max(...xs) - Math.min(...xs);
    const height = Math.max(...ys) - Math.min(...ys);
    return height < 1e-4 ? 0 : width / height;
}

/** Simple rolling average over a fixed-length array. */
function rollingMean(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((s, v) => s + v, 0) / arr.length;
}

// ── Main detector class ───────────────────────────────────────────────────────

class FallDetectorTFJS {
    /**
     * @param {object} opts
     * @param {string} opts.sensitivity  'low' | 'medium' | 'high'
     * @param {number} opts.cooldownMs   Milliseconds between consecutive alerts (default 3000)
     * @param {number} opts.temporalWindow  Frames to smooth over (default 5)
     * @param {number} opts.minKeypointScore  Minimum MoveNet confidence to use a keypoint (default 0.2)
     */
    constructor(opts = {}) {
        const {
            sensitivity       = 'medium',
            cooldownMs        = 3000,
            temporalWindow    = 5,
            minKeypointScore  = 0.2,
        } = opts;

        // Sensitivity presets (mirrors Python)
        const presets = {
            low:    { angle: 25.0, speed: 50.0, aspect: 1.5 },
            medium: { angle: 35.0, speed: 30.0, aspect: 1.3 },
            high:   { angle: 45.0, speed: 20.0, aspect: 1.1 },
        };
        const p = presets[sensitivity] || presets.medium;

        this.angleThr        = p.angle;
        this.speedThr        = p.speed;
        this.aspectThr       = p.aspect;
        this.cooldownMs      = cooldownMs;
        this.temporalWindow  = temporalWindow;
        this.minScore        = minKeypointScore;

        // Rolling history for temporal smoothing
        this._angleHistory  = [];
        this._aspectHistory = [];

        // State
        this._prevHipY       = null;
        this._prevTimeMs     = null;
        this._cooldownUntil  = 0;
        this._lastStandingMs = Date.now();
    }

    /**
     * Process one frame's pose keypoints and decide if a fall occurred.
     *
     * @param {Array} keypoints  MoveNet keypoint array [{x,y,score,name}, ...]
     * @returns {{ fallDetected:boolean, confidence:number, features:object }}
     */
    detect(keypoints) {
        const now = Date.now();
        const result = { fallDetected: false, confidence: 0, features: {} };

        if (!keypoints || keypoints.length < 17) return result;

        // ── Extract positions ──────────────────────────────────────────────
        const get = idx => {
            const kp = keypoints[idx];
            return (kp && kp.score >= this.minScore) ? kp : null;
        };

        const lSho = get(L_SHO), rSho = get(R_SHO);
        const lHip = get(L_HIP), rHip = get(R_HIP);
        const lKne = get(L_KNEE), rKne = get(R_KNEE);

        if (!lSho || !rSho || !lHip || !rHip) return result; // can't compute core features

        const shoulder = { x: (lSho.x + rSho.x) / 2, y: (lSho.y + rSho.y) / 2 };
        const hip      = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };

        // ── Feature extraction ────────────────────────────────────────────
        const torsoAngle  = angleDeg(hip, shoulder);
        const aspectRatio = calcAspectRatio(keypoints);
        const hipY        = hip.y;

        // Hip-knee vertical relationship
        let hipKneeDiff = 0;
        if (lKne && rKne) {
            const kneeY = (lKne.y + rKne.y) / 2;
            hipKneeDiff = kneeY - hipY; // positive → hips above knees (standing)
        }

        // ── Temporal smoothing ────────────────────────────────────────────
        this._angleHistory.push(torsoAngle);
        this._aspectHistory.push(aspectRatio);
        if (this._angleHistory.length  > this.temporalWindow) this._angleHistory.shift();
        if (this._aspectHistory.length > this.temporalWindow) this._aspectHistory.shift();

        const smoothAngle  = rollingMean(this._angleHistory);
        const smoothAspect = rollingMean(this._aspectHistory);

        // ── Vertical speed ────────────────────────────────────────────────
        let speed = 0;
        if (this._prevTimeMs !== null && this._prevHipY !== null) {
            const dtSec = Math.max(1e-3, (now - this._prevTimeMs) / 1000);
            speed = (hipY - this._prevHipY) / dtSec;   // px/s downward positive
        }
        this._prevHipY   = hipY;
        this._prevTimeMs = now;

        // ── Standing history ──────────────────────────────────────────────
        const timeSinceStanding = now - this._lastStandingMs;
        const wasStanding = smoothAngle > 50.0 || timeSinceStanding < 2000;
        if (smoothAngle > 50.0) this._lastStandingMs = now;

        // ── Multi-criteria fall detection ─────────────────────────────────
        let criteriaM = 0;
        let confidence = 0;

        if (smoothAngle < this.angleThr)   { criteriaM++; confidence += 0.35; }
        if (smoothAspect > this.aspectThr) { criteriaM++; confidence += 0.35; }
        if (speed > this.speedThr)         { criteriaM++; confidence += 0.30; }

        let fallDetected = false;
        if (now >= this._cooldownUntil && wasStanding && criteriaM >= 2) {
            fallDetected = true;
            this._cooldownUntil = now + this.cooldownMs;
        }

        result.fallDetected = fallDetected;
        result.confidence   = Math.min(1.0, confidence);
        result.features     = {
            torsoAngle:  smoothAngle,
            aspectRatio: smoothAspect,
            speed,
            criteriaM,
            hipKneeDiff,
        };
        return result;
    }

    reset() {
        this._angleHistory  = [];
        this._aspectHistory = [];
        this._prevHipY      = null;
        this._prevTimeMs    = null;
        this._cooldownUntil = 0;
        this._lastStandingMs = Date.now();
    }
}

// ── Canvas drawing helpers ────────────────────────────────────────────────────

/**
 * Draw MoveNet keypoints and skeleton connections on a canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} keypoints  MoveNet keypoints
 * @param {boolean} fallDetected
 * @param {number} minScore  Minimum confidence to draw
 */
function drawPose(ctx, keypoints, fallDetected, minScore = 0.2) {
    const boneColor = fallDetected ? '#ff3b3b' : '#00e5ff';
    const dotColor  = fallDetected ? '#ff8c00' : '#39ff14';
    const lineW     = fallDetected ? 3 : 2;

    // Skeleton lines
    ctx.lineWidth   = lineW;
    ctx.strokeStyle = boneColor;
    for (const [a, b] of SKELETON_CONNECTIONS) {
        const kpA = keypoints[a], kpB = keypoints[b];
        if (!kpA || !kpB) continue;
        if (kpA.score < minScore || kpB.score < minScore) continue;
        ctx.beginPath();
        ctx.moveTo(kpA.x, kpA.y);
        ctx.lineTo(kpB.x, kpB.y);
        ctx.stroke();
    }

    // Keypoint dots
    for (const kp of keypoints) {
        if (!kp || kp.score < minScore) continue;
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = dotColor;
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

/**
 * Draw the status overlay (status banner, metrics) on the canvas.
 */
function drawOverlay(ctx, W, H, fallDetected, confidence, features, fps) {
    if (fallDetected) {
        // Red pulsing banner
        const alpha = 0.55 + 0.2 * Math.sin(Date.now() / 200);
        ctx.fillStyle = `rgba(220, 30, 30, ${alpha})`;
        ctx.fillRect(0, 0, W, 90);

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 30px "Segoe UI", sans-serif';
        ctx.fillText('⚠ FALL DETECTED!', 20, 42);

        ctx.font = '16px "Segoe UI", sans-serif';
        ctx.fillText(`Confidence: ${(confidence * 100).toFixed(0)}%`, 20, 70);
    } else {
        // Normal status
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillRect(0, 0, W, 36);
        ctx.fillStyle = '#39ff14';
        ctx.font = 'bold 16px "Segoe UI", sans-serif';
        ctx.fillText('● Monitoring Active', 14, 23);
    }

    // Metrics strip at bottom
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, H - 80, W, 80);

    ctx.fillStyle = '#e0e0e0';
    ctx.font = '13px "Segoe UI Mono", monospace';

    const angle   = features.torsoAngle  != null ? features.torsoAngle.toFixed(1)  : '--';
    const aspect  = features.aspectRatio != null ? features.aspectRatio.toFixed(2) : '--';
    const speed   = features.speed       != null ? features.speed.toFixed(1)       : '--';
    const crit    = features.criteriaM   != null ? features.criteriaM : '--';

    ctx.fillText(`Torso angle: ${angle}°`, 12, H - 58);
    ctx.fillText(`Aspect ratio: ${aspect}`, 12, H - 40);
    ctx.fillText(`Vert. speed: ${speed} px/s`, 12, H - 22);

    ctx.fillStyle = '#aaaaaa';
    ctx.font = '13px "Segoe UI Mono", monospace';
    ctx.fillText(`Criteria: ${crit}/3`, W / 2 - 30, H - 40);
    ctx.fillText(`FPS: ${fps.toFixed(1)}`, W / 2 - 30, H - 22);
}

// ── Live session controller ───────────────────────────────────────────────────

class TFJSLiveSession {
    /**
     * @param {HTMLVideoElement}  videoEl   Webcam video element
     * @param {HTMLCanvasElement} canvasEl  Overlay canvas
     * @param {object} opts  Passed to FallDetectorTFJS + callbacks
     * @param {function} opts.onFall  Called with (event) when fall detected
     * @param {function} opts.onFrame Called with (features, fps) every frame
     * @param {string}  opts.sensitivity
     */
    constructor(videoEl, canvasEl, opts = {}) {
        this.videoEl  = videoEl;
        this.canvasEl = canvasEl;
        this.ctx      = canvasEl.getContext('2d');
        this.onFall   = opts.onFall  || (() => {});
        this.onFrame  = opts.onFrame || (() => {});

        this.detector = new FallDetectorTFJS({ sensitivity: opts.sensitivity || 'medium' });

        this._poseDetector = null;   // MoveNet instance
        this._running      = false;
        this._rafId        = null;
        this._stream       = null;

        // FPS tracking
        this._frameTimes = [];
        this._fps = 0;
    }

    /** Load MoveNet and start the detection loop. */
    async start(deviceId = null) {
        if (this._running) return;

        // 1. Load MoveNet (Lightning = fastest, suitable for real-time)
        console.log('[TFJSLiveSession] Loading MoveNet Lightning...');
        const model = poseDetection.SupportedModels.MoveNet;
        this._poseDetector = await poseDetection.createDetector(model, {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        });
        console.log('[TFJSLiveSession] MoveNet loaded.');

        // 2. Open webcam
        const constraints = {
            video: deviceId
                ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
                : { width: 640, height: 480, facingMode: 'user' },
            audio: false,
        };
        this._stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.videoEl.srcObject = this._stream;
        await new Promise(res => { this.videoEl.onloadedmetadata = res; });
        await this.videoEl.play();

        // Match canvas to video dimensions
        this.canvasEl.width  = this.videoEl.videoWidth  || 640;
        this.canvasEl.height = this.videoEl.videoHeight || 480;

        this._running = true;
        this._loop();
        console.log('[TFJSLiveSession] Detection loop started.');
    }

    /** Main per-frame loop. */
    async _loop() {
        if (!this._running) return;

        const W = this.canvasEl.width;
        const H = this.canvasEl.height;

        // Estimate pose
        let poses = [];
        try {
            poses = await this._poseDetector.estimatePoses(this.videoEl);
        } catch (err) {
            console.error('[TFJSLiveSession] Pose estimation error:', err);
        }

        // Clear canvas & mirror-draw the video frame
        this.ctx.save();
        this.ctx.scale(-1, 1);
        this.ctx.translate(-W, 0);
        this.ctx.drawImage(this.videoEl, 0, 0, W, H);
        this.ctx.restore();

        let fallResult = { fallDetected: false, confidence: 0, features: {} };

        if (poses && poses.length > 0) {
            const kps = poses[0].keypoints.map(kp => ({
                ...kp,
                // Mirror X for the flipped canvas
                x: W - kp.x,
            }));

            fallResult = this.detector.detect(kps);

            drawPose(this.ctx, kps, fallResult.fallDetected, 0.2);

            if (fallResult.fallDetected) {
                this.onFall({
                    timestamp:  new Date().toISOString(),
                    confidence: fallResult.confidence,
                    features:   fallResult.features,
                });
            }
        }

        // FPS
        const now = performance.now();
        this._frameTimes.push(now);
        if (this._frameTimes.length > 30) this._frameTimes.shift();
        if (this._frameTimes.length >= 2) {
            const elapsed = this._frameTimes[this._frameTimes.length - 1] - this._frameTimes[0];
            this._fps = ((this._frameTimes.length - 1) / elapsed) * 1000;
        }

        drawOverlay(this.ctx, W, H, fallResult.fallDetected, fallResult.confidence, fallResult.features, this._fps);

        this.onFrame(fallResult.features, this._fps);

        this._rafId = requestAnimationFrame(() => this._loop());
    }

    /** Stop detection and release the camera. */
    stop() {
        this._running = false;
        if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
        this.videoEl.srcObject = null;
        this.detector.reset();

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

        console.log('[TFJSLiveSession] Stopped.');
    }

    get isRunning() { return this._running; }
    get fps()       { return this._fps; }
}
