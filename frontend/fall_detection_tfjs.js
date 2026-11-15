/**
 * fall_detection_tfjs.js — In-browser fall detection using TensorFlow.js + MoveNet
 *
 * Elder-care fall logic (state machine):
 *
 *   NORMAL ──(upright → rapid drop → lying posture)──► CANDIDATE
 *   CANDIDATE ──(still lying after confirmMs)──► FALLEN      → onFall()  (one alarm per fall)
 *   CANDIDATE ──(upright again / sat down)──► NORMAL          (bending, sitting: no alarm)
 *   FALLEN ──(upright for standHoldMs)──► NORMAL               → onRecover({secondsOnFloor})
 *   FALLEN: no new fall alarms; time on the floor keeps counting even if the
 *           person slips out of view (missing keypoints never count as "got up").
 *
 * Why these rules:
 *   - A fall = the person WAS upright, the hips DROPPED quickly (normalised by body size,
 *     so it works near or far from the camera), and the body is now LYING (torso near
 *     horizontal or a wide, flat bounding box).
 *   - Lying down slowly on a bed has no rapid drop → not a fall (sleep / rest is ignored).
 *   - Sitting down keeps the torso upright → not a fall.
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

// The 2-minute rule: getting up within this time is fine; longer is a "long lie"
const LONG_LIE_SEC = 120;

// ── Maths helpers ─────────────────────────────────────────────────────────────

/**
 * Angle of vector p1→p2 relative to the HORIZONTAL axis, folded to 0..90 degrees.
 * 90° = torso perfectly vertical (standing / sitting), 0° = horizontal (lying).
 */
function angleDeg(p1, p2) {
    // Kept for backwards compatibility; same as angleFromHorizontal (90 = upright).
    return angleFromHorizontal(p1, p2);
}

/**
 * Body bounding box aspect ratio: width / height.
 * High ratio → person is more horizontal (lying down).
 */
function calcAspectRatio(keypoints, minScore = 0.2) {
    const valid = keypoints.filter(kp => kp && kp.score > minScore && kp.x > 0 && kp.y > 0);
    if (valid.length < 5) return 0;

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

function fmtDuration(sec) {
    sec = Math.max(0, Math.round(sec));
    const m = Math.floor(sec / 60), s = sec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
}

// ── Main detector class ───────────────────────────────────────────────────────

class FallDetectorTFJS {
    /**
     * @param {object} opts
     * @param {string} opts.sensitivity  'low' | 'medium' | 'high'
     * @param {number} opts.confirmMs    How long the lying posture must last to confirm a fall (default 1000)
     * @param {number} opts.standHoldMs  How long the person must stay upright to count as "got up" (default 2000)
     * @param {number} opts.temporalWindow  Frames to smooth over (default 5)
     * @param {number} opts.minKeypointScore  Minimum MoveNet confidence to use a keypoint (default 0.25)
     */
    constructor(opts = {}) {
        const {
            sensitivity      = 'medium',
            confirmMs        = 1000,
            standHoldMs      = 2000,
            temporalWindow   = 5,
            minKeypointScore = 0.25,
        } = opts;

        // lyingAngle: torso angle (deg from horizontal) below which the body counts as lying
        // dropSpeed:  hip descent speed in torso-lengths per second
        // dropDist:   total hip drop (torso lengths) within the look-back window
        const presets = {
            low:    { lyingAngle: 30, aspect: 1.5, dropSpeed: 1.6, dropDist: 0.8 },
            medium: { lyingAngle: 40, aspect: 1.3, dropSpeed: 1.2, dropDist: 0.6 },
            high:   { lyingAngle: 50, aspect: 1.15, dropSpeed: 0.9, dropDist: 0.45 },
        };
        const p = presets[sensitivity] || presets.medium;

        this.lyingAngle     = p.lyingAngle;
        this.aspectThr      = p.aspect;
        this.dropSpeedThr   = p.dropSpeed;
        this.dropDistThr    = p.dropDist;
        this.uprightAngle   = 60;          // torso this close to vertical = upright
        this.confirmMs      = confirmMs;
        this.standHoldMs    = standHoldMs;
        this.lookBackMs     = 1500;        // the drop must happen within this window
        this.uprightRecentMs = 2500;       // ...and the person must have been upright this recently
        this.temporalWindow = temporalWindow;
        this.minScore       = minKeypointScore;

        this.reset();
    }

    reset() {
        this._angleHistory  = [];
        this._aspectHistory = [];
        this._hipTrack      = [];     // [{t, y, torso}] for the drop look-back
        this._lastUprightMs = 0;
        this.state          = 'normal';   // normal | candidate | fallen
        this._candidateAt   = 0;
        this._candidate     = null;
        this.fallStartMs    = null;       // when the current fall happened
        this._standSince    = null;
        this._lastSeenMs    = Date.now();
        this.longLieNotified = false;
        this.restZones      = this.restZones || [];   // beds / sofas [{kind,x,y,w,h}] in keypoint coords
        this._lastRestMs    = 0;          // last time the person was lying on a bed / sofa
        this._restZone      = null;
        // legacy fields (kept for callers that read them)
        this._cooldownUntil  = 0;
        this._lastStandingMs = Date.now();
    }

    /** Seconds on the floor for the current fall (0 when not fallen). */
    get secondsOnFloor() {
        return this.state === 'fallen' && this.fallStartMs ? (Date.now() - this.fallStartMs) / 1000 : 0;
    }

    /**
     * Process one frame's pose keypoints.
     *
     * @param {Array} keypoints  MoveNet keypoint array [{x,y,score,name}, ...]
     * @returns {{ fallDetected:boolean, recovered:boolean, secondsOnFloor:number,
     *             confidence:number, state:string, features:object }}
     */
    detect(keypoints) {
        const now = Date.now();
        const result = {
            fallDetected: false, recovered: false, secondsOnFloor: 0,
            confidence: 0, state: this.state, features: {}, personVisible: false,
        };

        const get = idx => {
            const kp = keypoints && keypoints[idx];
            return (kp && kp.score >= this.minScore) ? kp : null;
        };
        const lSho = get(L_SHO), rSho = get(R_SHO);
        const lHip = get(L_HIP), rHip = get(R_HIP);
        const lKne = get(L_KNEE), rKne = get(R_KNEE);

        // Need at least one shoulder and one hip on the same body
        const sho = lSho && rSho ? { x: (lSho.x + rSho.x) / 2, y: (lSho.y + rSho.y) / 2 } : (lSho || rSho);
        const hip = lHip && rHip ? { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 } : (lHip || rHip);

        // Head position (nose / eyes / ears) - used only to tell whether a fallen person got up
        const headPts = [0, 1, 2, 3, 4].map(get).filter(Boolean);
        const headY = headPts.length ? headPts.reduce((a, k) => a + k.y, 0) / headPts.length : null;
        const frameH = this.frameH || 480;
        const headRaised = (headY !== null && this._fallHeadY != null && headY < this._fallHeadY - 0.3 * frameH);

        if (!sho || !hip) {
            // Body not fully visible. After a fall, a head held well above where it lay on the floor
            // (person got up close to the camera, only head/shoulders in view) counts as "got up".
            if (this.state === 'candidate' && now - this._candidateAt > 3000) this.state = 'normal';
            if (this.state === 'fallen' && headRaised) {
                if (!this._standSince) this._standSince = now;
                if (now - this._standSince >= this.standHoldMs) {
                    result.recovered = true;
                    result.secondsOnFloor = Math.max(0, (this._standSince - this.fallStartMs) / 1000);
                    this.state = 'normal';
                    this.fallStartMs = null;
                    this._standSince = null;
                    this._hipTrack = [];
                }
                result.personVisible = true;
                result.state = this.state;
                if (this.state === 'fallen') result.secondsOnFloor = this.secondsOnFloor;
                return result;
            }
            this._standSince = null;
            result.state = this.state;
            result.secondsOnFloor = this.secondsOnFloor;
            return result;
        }
        result.personVisible = true;
        this._lastSeenMs = now;

        // ── Features ──────────────────────────────────────────────────────
        const torsoLen = Math.max(20, Math.hypot(sho.x - hip.x, sho.y - hip.y));
        const rawAngle = angleFromHorizontal(hip, sho);          // 90 = vertical
        const aspect   = calcAspectRatio(keypoints, this.minScore);

        this._angleHistory.push(rawAngle);
        this._aspectHistory.push(aspect);
        if (this._angleHistory.length  > this.temporalWindow) this._angleHistory.shift();
        if (this._aspectHistory.length > this.temporalWindow) this._aspectHistory.shift();
        const angle    = rollingMean(this._angleHistory);
        const aspectSm = rollingMean(this._aspectHistory);

        // Hip track for the "rapid drop" test (normalised by torso length)
        this._hipTrack.push({ t: now, y: hip.y, torso: torsoLen });
        while (this._hipTrack.length && now - this._hipTrack[0].t > this.lookBackMs) this._hipTrack.shift();
        let dropDist = 0, dropSpeed = 0;
        if (this._hipTrack.length >= 2) {
            const minY = Math.min(...this._hipTrack.map(h => h.y));            // highest hip position
            const ref  = rollingMean(this._hipTrack.map(h => h.torso));
            dropDist = (hip.y - minY) / ref;                                     // + = moved down
            const a = this._hipTrack[Math.max(0, this._hipTrack.length - 4)];
            const dt = Math.max(0.05, (now - a.t) / 1000);
            dropSpeed = ((hip.y - a.y) / ref) / dt;
        }

        // Hips above knees = standing; hips level with knees + vertical torso = sitting
        let hipAboveKnee = null;
        if (lKne || rKne) {
            const kneeY = lKne && rKne ? (lKne.y + rKne.y) / 2 : (lKne || rKne).y;
            hipAboveKnee = (kneeY - hip.y) / torsoLen;    // > 0.3: hips clearly above knees
        }

        // Lying on a bed / sofa (resting or sleeping) is not a fall. The body must be over the
        // furniture and the hips above its lower edge (the floor in front of a bed is below that).
        const bodyX = (sho.x + hip.x) / 2;
        const restZone = (this.restZones || []).find(z =>
            bodyX >= z.x - 0.1 * z.w && bodyX <= z.x + 1.1 * z.w &&
            hip.y >= z.y - 0.15 * z.h && hip.y <= z.y + 0.8 * z.h) || null;

        const upright = angle >= this.uprightAngle;
        const lying   = angle <= this.lyingAngle || (aspectSm >= this.aspectThr && angle < this.uprightAngle - 10);
        if (upright) {
            this._lastUprightMs = now;
            this._lastStandingMs = now;
        }

        result.features = {
            torsoAngle: angle, aspectRatio: aspectSm, speed: dropSpeed, dropDist,
            criteriaM: (lying ? 1 : 0) + (dropSpeed >= this.dropSpeedThr ? 1 : 0) + (dropDist >= this.dropDistThr ? 1 : 0),
            hipKneeDiff: hipAboveKnee,
        };

        // ── State machine ─────────────────────────────────────────────────
        if (this.state === 'normal') {
            const wasUpright = now - this._lastUprightMs <= this.uprightRecentMs;
            const rapidDrop  = dropSpeed >= this.dropSpeedThr || dropDist >= this.dropDistThr;
            // Fell OFF the bed / sofa: was lying on it moments ago, now lying below / beside it
            const rz = this._restZone;
            const offFurniture = !restZone && rz && now - this._lastRestMs <= 4000 &&
                (dropDist >= 0.5 || hip.y > rz.y + 0.8 * rz.h);
            if (lying && restZone) {
                result.resting = restZone.kind;
                this._lastRestMs = now; this._restZone = restZone;
            } else if (lying && ((wasUpright && rapidDrop) || offFurniture)) {
                this.state = 'candidate';
                this._candidateAt = now;
                this._candidate = { dropSpeed, dropDist, angle, aspect: aspectSm, fromFurniture: !!offFurniture };
            }
        } else if (this.state === 'candidate') {
            if (lying && restZone) {
                this.state = 'normal';                 // lay down on the bed / sofa: resting, not a fall
                result.resting = restZone.kind;
                this._lastRestMs = now; this._restZone = restZone;
            } else if (upright) {
                this.state = 'normal';                 // bent down / caught themselves: no alarm
            } else if (lying && now - this._candidateAt >= this.confirmMs) {
                this.state = 'fallen';
                this.fallStartMs = this._candidateAt;
                this._standSince = null;
                this.longLieNotified = false;
                this._fallHipY = hip.y; this._fallTorso = torsoLen;
                this._fallHeadY = headY !== null ? headY : Math.min(sho.y, hip.y);
                const c = this._candidate || {};
                // Confidence: posture + how fast/far the drop was
                let conf = 0.55;
                conf += Math.min(0.2, (c.dropSpeed || 0) / this.dropSpeedThr * 0.08);
                conf += Math.min(0.15, (c.dropDist || 0) / this.dropDistThr * 0.06);
                if ((c.aspect || 0) >= this.aspectThr) conf += 0.05;
                if ((c.angle || 90) <= this.lyingAngle) conf += 0.05;
                if (c.fromFurniture) { conf = Math.max(conf, 0.7); result.features.fromFurniture = true; }
                result.fallDetected = true;
                result.confidence = Math.min(0.99, conf);
            } else if (now - this._candidateAt > this.confirmMs + 2000) {
                this.state = 'normal';
            }
        } else if (this.state === 'fallen') {
            // Got up = upright torso (and hips above knees when knees are visible) held for standHoldMs
            const standing = upright && (hipAboveKnee === null || hipAboveKnee > 0.3);
            // Also got up: sitting on a chair/bed (upright torso, hips clearly higher than on the floor),
            // or head held well above where it lay. Sitting up ON the floor keeps hips at floor level: no.
            const hipRaised = this._fallHipY != null && hip.y < this._fallHipY - 0.6 * (this._fallTorso || torsoLen);
            const gotUp = standing || (upright && (hipRaised || headRaised));
            if (lying && !this._standSince) {
                // still on the floor: keep the floor position current (person may slide/roll)
                this._fallHipY = hip.y; this._fallTorso = torsoLen;
                if (headY !== null) this._fallHeadY = headY;
            }
            if (gotUp) {
                if (!this._standSince) this._standSince = now;
                if (now - this._standSince >= this.standHoldMs) {
                    result.recovered = true;
                    result.secondsOnFloor = Math.max(0, (this._standSince - this.fallStartMs) / 1000);
                    this.state = 'normal';
                    this.fallStartMs = null;
                    this._standSince = null;
                    this._hipTrack = [];
                }
            } else {
                this._standSince = null;
            }
        }

        result.state = this.state;
        if (this.state === 'fallen') result.secondsOnFloor = this.secondsOnFloor;
        return result;
    }
}

/** 90° = torso vertical, 0° = horizontal. */
function angleFromHorizontal(p1, p2) {
    const vx = p2.x - p1.x;
    const vy = p2.y - p1.y;
    const a = Math.abs((Math.atan2(vy, vx) * 180) / Math.PI);   // 0..180
    return Math.min(a, 180 - a);                                  // 0..90 from horizontal
}

// ── Canvas drawing helpers ────────────────────────────────────────────────────

/**
 * Draw MoveNet keypoints and skeleton connections on a canvas.
 */
function drawPose(ctx, keypoints, fallDetected, minScore = 0.2) {
    const boneColor = fallDetected ? '#ff3b3b' : '#00e5ff';
    const dotColor  = fallDetected ? '#ff8c00' : '#39ff14';
    const lineW     = fallDetected ? 3 : 2;

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
 * Draw the status overlay (status banner, on-floor timer, metrics) on the canvas.
 * `fallen` = currently in the FALLEN state; `secondsOnFloor` drives the 2-minute timer.
 */
function drawOverlay(ctx, W, H, fallen, confidence, features, fps, secondsOnFloor = 0, personVisible = true,
                     zones = [], resting = null) {
    // Beds / sofas: lying there is resting, not a fall
    for (const z of zones) {
        ctx.save();
        ctx.setLineDash([8, 6]); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(120, 200, 255, 0.85)';
        ctx.strokeRect(z.x, z.y, z.w, z.h);
        ctx.setLineDash([]);
        ctx.font = 'bold 13px "Segoe UI", sans-serif';
        const label = z.kind === 'sofa' ? 'Sofa (resting area)' : 'Bed (resting area)';
        ctx.fillStyle = 'rgba(20, 60, 90, 0.75)';
        ctx.fillRect(z.x, Math.max(0, z.y - 20), ctx.measureText(label).width + 12, 20);
        ctx.fillStyle = '#d6f0ff';
        ctx.fillText(label, z.x + 6, Math.max(14, z.y - 6));
        ctx.restore();
    }
    if (fallen) {
        const longLie = secondsOnFloor >= LONG_LIE_SEC;
        const alpha = 0.6 + 0.2 * Math.sin(Date.now() / 200);
        ctx.fillStyle = longLie ? `rgba(150, 0, 0, ${alpha})` : `rgba(220, 30, 30, ${alpha})`;
        ctx.fillRect(0, 0, W, 104);

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 28px "Segoe UI", sans-serif';
        ctx.fillText(longLie ? '⚠ STILL ON THE FLOOR — CHECK NOW' : '⚠ FALL DETECTED', 20, 38);
        ctx.font = 'bold 18px "Segoe UI", sans-serif';
        ctx.fillText(`On the floor: ${fmtDuration(secondsOnFloor)}  (should get up within 2:00)`, 20, 66);
        if (!personVisible) {
            ctx.font = '14px "Segoe UI", sans-serif';
            ctx.fillText('Person not visible to the camera — timer keeps running', 20, 90);
        }

        // Progress bar towards 2 minutes
        const pct = Math.min(1, secondsOnFloor / LONG_LIE_SEC);
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillRect(20, 96, W - 40, 5);
        ctx.fillStyle = longLie ? '#ffeb3b' : '#ffffff';
        ctx.fillRect(20, 96, (W - 40) * pct, 5);
    } else {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillRect(0, 0, W, 36);
        ctx.fillStyle = personVisible ? '#39ff14' : '#ffc107';
        ctx.font = 'bold 16px "Segoe UI", sans-serif';
        ctx.fillText(resting ? ('● Resting on the ' + resting + ' — not a fall')
                   : personVisible ? '● Monitoring Active' : '● Monitoring — no person in view', 14, 23);
    }

    // Metrics strip at bottom
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, H - 80, W, 80);

    ctx.fillStyle = '#e0e0e0';
    ctx.font = '13px "Segoe UI Mono", monospace';
    const angle  = features.torsoAngle  != null ? features.torsoAngle.toFixed(1)  : '--';
    const aspect = features.aspectRatio != null ? features.aspectRatio.toFixed(2) : '--';
    const speed  = features.speed       != null ? features.speed.toFixed(2)       : '--';
    ctx.fillText(`Torso angle: ${angle}° (90 = upright)`, 12, H - 58);
    ctx.fillText(`Aspect ratio: ${aspect}`, 12, H - 40);
    ctx.fillText(`Drop speed: ${speed} torso/s`, 12, H - 22);

    ctx.fillStyle = '#aaaaaa';
    ctx.fillText(`Criteria: ${features.criteriaM != null ? features.criteriaM : '--'}/3`, W / 2 + 20, H - 40);
    ctx.fillText(`FPS: ${fps.toFixed(1)}`, W / 2 + 20, H - 22);
}

function _iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / Math.max(1, a.w * a.h + b.w * b.h - inter);
}

// ── Live session controller ───────────────────────────────────────────────────

class TFJSLiveSession {
    /**
     * @param {HTMLVideoElement}  videoEl   Webcam video element
     * @param {HTMLCanvasElement} canvasEl  Overlay canvas
     * @param {object} opts
     * @param {function} opts.onFall      ({timestamp, confidence, features}) once per fall
     * @param {function} opts.onRecover   ({timestamp, secondsOnFloor, withinLimit}) when the person got up
     * @param {function} opts.onLongLie   ({timestamp, secondsOnFloor}) once, when 2 minutes pass on the floor
     * @param {function} opts.onFloorTick ({secondsOnFloor, personVisible}) every frame while fallen
     * @param {function} opts.onFrame     (features, fps, state) every frame
     * @param {string}  opts.sensitivity
     */
    constructor(videoEl, canvasEl, opts = {}) {
        this.videoEl  = videoEl;
        this.canvasEl = canvasEl;
        this.ctx      = canvasEl.getContext('2d');
        this.onFall      = opts.onFall      || (() => {});
        this.onRecover   = opts.onRecover   || (() => {});
        this.onLongLie   = opts.onLongLie   || (() => {});
        this.onFloorTick = opts.onFloorTick || (() => {});
        this.onFrame     = opts.onFrame     || (() => {});

        this.detector = new FallDetectorTFJS({ sensitivity: opts.sensitivity || 'medium' });

        this._poseDetector = null;
        this._running      = false;
        this._rafId        = null;
        this._stream       = null;
        this._frameTimes = [];
        this._fps = 0;
        this._lastConfidence = 0;
    }

    async start(deviceId = null) {
        if (this._running) return;

        console.log('[TFJSLiveSession] Loading MoveNet Lightning...');
        const model = poseDetection.SupportedModels.MoveNet;
        this._poseDetector = await poseDetection.createDetector(model, {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true,
        });
        console.log('[TFJSLiveSession] MoveNet loaded.');

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

        this.canvasEl.width  = this.videoEl.videoWidth  || 640;
        this.canvasEl.height = this.videoEl.videoHeight || 480;
        if (this.detector) this.detector.frameH = this.canvasEl.height;

        this._running = true;
        this._loop();
        console.log('[TFJSLiveSession] Detection loop started.');
    }

    async _loop() {
        if (!this._running) return;

        const W = this.canvasEl.width;
        const H = this.canvasEl.height;

        let poses = [];
        try {
            poses = await this._poseDetector.estimatePoses(this.videoEl);
        } catch (err) {
            console.error('[TFJSLiveSession] Pose estimation error:', err);
        }

        this.ctx.save();
        this.ctx.scale(-1, 1);
        this.ctx.translate(-W, 0);
        this.ctx.drawImage(this.videoEl, 0, 0, W, H);
        this.ctx.restore();

        let kps = null;
        if (poses && poses.length > 0 && (poses[0].score === undefined || poses[0].score > 0.15)) {
            kps = poses[0].keypoints.map(kp => ({ ...kp, x: W - kp.x }));
        }
        if (kps) {
            const vis = kps.filter(k => k.score >= 0.3);
            if (vis.length) this._lastBody = { x: vis.reduce((a, k) => a + k.x, 0) / vis.length,
                                               y: vis.reduce((a, k) => a + k.y, 0) / vis.length, at: Date.now() };
        }
        this._updateRestZones(W);
        const r = this.detector.detect(kps || []);

        if (kps) drawPose(this.ctx, kps, r.state === 'fallen', 0.2);

        const nowIso = new Date().toISOString();
        if (r.fallDetected) {
            this._lastConfidence = r.confidence;
            this.onFall({ timestamp: nowIso, confidence: r.confidence, features: r.features });
        }
        if (r.state === 'fallen') {
            this.onFloorTick({ secondsOnFloor: r.secondsOnFloor, personVisible: r.personVisible });
            if (r.secondsOnFloor >= LONG_LIE_SEC && !this.detector.longLieNotified) {
                this.detector.longLieNotified = true;
                this.onLongLie({ timestamp: nowIso, secondsOnFloor: r.secondsOnFloor });
            }
        }
        if (r.recovered) {
            this.onRecover({
                timestamp: nowIso,
                secondsOnFloor: r.secondsOnFloor,
                withinLimit: r.secondsOnFloor <= LONG_LIE_SEC,
            });
        }

        const now = performance.now();
        this._frameTimes.push(now);
        if (this._frameTimes.length > 30) this._frameTimes.shift();
        if (this._frameTimes.length >= 2) {
            const elapsed = this._frameTimes[this._frameTimes.length - 1] - this._frameTimes[0];
            this._fps = ((this._frameTimes.length - 1) / elapsed) * 1000;
        }

        drawOverlay(this.ctx, W, H, r.state === 'fallen', this._lastConfidence, r.features, this._fps,
            r.secondsOnFloor, r.personVisible, [], r.resting || null);   // beds / sofas are used silently, not drawn

        this.onFrame(r.features, this._fps, r.state);

        this._rafId = requestAnimationFrame(() => this._loop());
    }

    /**
     * Beds / sofas seen by the room's object detector (shared with medicine tracking).
     * - A bed / sofa counts only after it was seen in 2 checks (one stray detection is ignored).
     * - It is forgotten when the detector looks again ~3 times and it is not there any more,
     *   unless the person is on it (the detector often stops seeing a bed once someone lies on it).
     */
    _updateRestZones(W) {
        const now = Date.now();
        if (!this._zoneMem) this._zoneMem = [];
        if (now - (this._lastZoneCheck || 0) < 1000) return;
        this._lastZoneCheck = now;
        let fresh = [];
        try {
            if (typeof TableWatch !== 'undefined' && TableWatch.furniture) fresh = TableWatch.furniture();
        } catch (e) { fresh = []; }
        const checkedAt = fresh.checkedAt || 0;
        if (!checkedAt || checkedAt === this._zoneCheckedAt) {        // no new look at the room yet
            this.detector.restZones = this._zoneMem.filter(z => z.hits >= 2);
            return;
        }
        this._zoneCheckedAt = checkedAt;
        const matched = new Set();
        for (const f of fresh) {
            const b = { kind: f.kind, x: W - f.x - f.w, y: f.y, w: f.w, h: f.h };   // mirror like the keypoints
            const same = this._zoneMem.find(z => z.kind === b.kind && _iou(z, b) > 0.3);
            if (same) { Object.assign(same, b, { seen: now, hits: same.hits + 1, missed: 0 }); matched.add(same); }
            else { const z = { ...b, seen: now, hits: 1, missed: 0 }; this._zoneMem.push(z); matched.add(z); }
        }
        const body = this._lastBody && now - this._lastBody.at < 5000 ? this._lastBody : null;
        const restingRecently = now - (this.detector._lastRestMs || 0) < 15000;
        for (const z of this._zoneMem) {
            if (matched.has(z)) continue;
            const occupied = restingRecently ||
                (body && body.x >= z.x && body.x <= z.x + z.w && body.y >= z.y && body.y <= z.y + z.h);
            if (!occupied) z.missed = (z.missed || 0) + 1;
        }
        // gone: missed 3 looks in a row (~8 s), a single stray sighting after 1 miss, or unseen for 30 min
        this._zoneMem = this._zoneMem.filter(z =>
            z.missed < 3 && !(z.hits < 2 && z.missed >= 1) && now - z.seen < 30 * 60 * 1000);
        this.detector.restZones = this._zoneMem.filter(z => z.hits >= 2);
    }

    /** Seconds on the floor right now (0 if not fallen). */
    get secondsOnFloor() { return this.detector.secondsOnFloor; }
    get isFallen()       { return this.detector.state === 'fallen'; }

    stop() {
        this._running = false;
        if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
        this.videoEl.srcObject = null;
        this.detector.reset();
        this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
        console.log('[TFJSLiveSession] Stopped.');
    }

    get isRunning() { return this._running; }
    get fps()       { return this._fps; }
}
