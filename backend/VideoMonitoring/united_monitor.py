"""
UnitedMonitor — YOLO pose + Bidirectional LSTM fall detection pipeline.

External API is UNCHANGED:
process_frame(frame) → {
    "fall":       {"fall_detected", "all_persons", "keypoints", "boxes", "features", ...},
    "inactivity": {"person_detected", "time_inactive_seconds", "alert"},
    "global_state": {"fall_detected", "inactivity_alert", ...}
}
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "BodyMovementDetection"))

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

from fall_lstm_model import FallLSTM, FallSequenceBuffer, load_model, DEFAULT_SEQ_LEN
from inactivity_monitor import InactivityMonitor
from prototype_fall_detector import PrototypeFallDetector

_YOLO_MODEL_PATH = os.path.join(_here, "..", "yolov8n-pose.pt")
_LSTM_MODEL_PATH = os.path.join(_here, "fall_lstm.pth")

_SENSITIVITY_PRESETS = {
    "low":    {"lstm_thresh": 0.95, "confirm_frames": 10, "confirm_decay": 1},
    "medium": {"lstm_thresh": 0.90, "confirm_frames": 7,  "confirm_decay": 1},
    "high":   {"lstm_thresh": 0.85, "confirm_frames": 5,  "confirm_decay": 1},
}

def _sound_loud_beep():
    """Plays an urgent, loud audio beep on Windows in a non-blocking background thread."""
    def _beep_worker():
        try:
            import winsound
            # High-decibel emergency siren beep pattern: 3 piercing pulses
            for freq, dur in [(2400, 200), (2800, 200), (3200, 350)]:
                winsound.Beep(freq, dur)
                time.sleep(0.05)
        except Exception:
            pass
    try:
        import threading
        threading.Thread(target=_beep_worker, daemon=True).start()
    except Exception:
        pass

class UnitedMonitor:
    def __init__(self, sensitivity="medium", inactivity_threshold=30, stable_percentage=0.10, is_live=False, process_every_n_frames=2, video_fps: float = 0.0, fall_model: str = "prototype_rf"):
        print(f"[UnitedMonitor] Initializing... live={is_live} model={fall_model}")
        self.fall_model = fall_model
        self.sensitivity = sensitivity
        self.is_live = is_live
        self.process_every_n_frames = process_every_n_frames if is_live else 1
        
        cfg = _SENSITIVITY_PRESETS.get(sensitivity, _SENSITIVITY_PRESETS["medium"])
        self._lstm_thresh = cfg["lstm_thresh"]
        self._confirm_frames = cfg["confirm_frames"]
        self._confirm_decay = cfg["confirm_decay"]

        self._frame_counter = 0
        self._confirm_count = 0
        self._fall_detected = False
        self._is_sleeping = False
        self._fall_confidence = 0
        self._fall_start_time = None
        self._inactivity_alert = False
        self._last_result = None
        self._last_fall_event_time = -9999.0
        self._fall_event_just_fired = False
        self._is_currently_falling = False
        self._reset_confirm_frames = 0
        self._has_been_standing = False
        self._standing_duration = 0.0
        self._fall_anchor_box = None  # bounding box locked at fall-trigger time (visual anchor)

        # Alert Suppression
        self._last_event_frame = -999
        self._EVENT_FRAME_COOLDOWN = 300 # 10 seconds at 30fps

        # Tracking state — once the monitored_id is locked it never auto-switches
        self._monitored_id = None
        self._monitored_id_locked = False   # True after first person is chosen
        self._monitored_centroid = None

        # Pre-disappearance pose state tracking
        # We record what the body was doing just BEFORE it left the frame.
        # If the body was tilting/falling and then vanished →  real fall.
        # If the body was upright and walked out → NOT a fall.
        self._monitored_last_seen = None       # timestamp when monitored was last seen
        self._monitored_was_present = False    # True if monitored was in last processed frame
        self._disappear_fall_fired = False     # one-shot guard per disappearance event
        self._DISAPPEAR_FALL_WINDOW = 2.0      # seconds: how long after disappear to still check
        # Rolling history (last N readings while monitored was visible)
        self._pre_disappear_torso_angles = []  # recent torso angles
        self._pre_disappear_lstm_probs   = []  # recent LSTM fall probabilities
        self._pre_disappear_bbox_horiz   = []  # recent bbox horizontality flags
        self._PRE_HISTORY_LEN = 10             # keep last 10 readings (~0.3–0.4 s at 25fps)

        self._fps_start = time.time()
        self._fps_frames = 0
        # For offline mode, initialise _current_fps from the actual video fps so that
        # dt = 1/_current_fps is correct from frame 1. For live mode, fps is measured
        # in real-time from wall-clock, so start at 0 and let the loop measure it.
        self._video_fps   = video_fps   # stored for offline dt override
        self._current_fps = video_fps if (not is_live and video_fps > 0) else 0.0

        self._yolo = YOLO(_YOLO_MODEL_PATH)
        self._prototype_detector = None
        self._lstm = None
        self._device = None
        self._seq_buf = None

        if self.fall_model.startswith("prototype"):
            variant = "window"
            if "motion" in self.fall_model:
                variant = "motion"
            elif "frame" in self.fall_model:
                variant = "frame"
            self._prototype_detector = PrototypeFallDetector(model_variant=variant)
        else:
            self._lstm, self._device = load_model(_LSTM_MODEL_PATH)
            seq_len = DEFAULT_SEQ_LEN if is_live else 15
            self._seq_buf = FallSequenceBuffer(seq_len=seq_len)
        self._inactivity = InactivityMonitor(safety_threshold_seconds=inactivity_threshold, stability_percentage=stable_percentage)

    def _is_human_pose(self, kps_xy, kps_conf, box):
        """
        Hardened Humanity Test (Furniture Ghost Rejection):
        Checks if the skeleton's geometry and joints are human-like.

        FIX: Stricter thresholds to eliminate chair/furniture false detections.
        Special case: lying/fallen person (horizontal bbox) relaxes head checks
        since face is not visible from top-down cameras after a fall.
        """
        if kps_xy is None or kps_conf is None or box is None: return False

        # 0. Minimum box area — ignore tiny/distant detections (furniture artifacts)
        bbox_w = box[2] - box[0]
        bbox_h = box[3] - box[1]
        box_area = bbox_w * bbox_h
        if box_area < 2000:  # Less than ~45x45 px box → skip
            return False

        # Check if bbox is horizontal (person lying down)
        is_horizontal_bbox = (bbox_w > bbox_h * 1.2) if bbox_h > 0 else False

        if is_horizontal_bbox:
            # Person is lying down — face is not visible from top-down camera.
            # Skip strict head check. Just require shoulders + hips to be detected.
            core_joints = [5, 6, 11, 12]  # Left/right shoulder + left/right hip
            core_conf = sum(1 for idx in core_joints if idx < len(kps_conf) and kps_conf[idx] > 0.3)
            return core_conf >= 3  # At least 3 of 4 core joints visible

        # 1. Geometry Check: Spine Proportionality
        # A human's neck-to-hip distance should be reasonable relative to the box height.
        neck = (kps_xy[5] + kps_xy[6]) / 2.0
        hip  = (kps_xy[11] + kps_xy[12]) / 2.0
        spine_len = math.sqrt((neck[0]-hip[0])**2 + (neck[1]-hip[1])**2)

        if spine_len < 0.15 * bbox_h:  # Raised from 0.12 — chairs collapse skeleton more
            return False

        # 2. Strict Head Check (Eyes/Ears/Nose) — raised confidence threshold
        head_joints = [0, 1, 2, 3, 4]
        head_conf_count = sum(1 for idx in head_joints if idx < len(kps_conf) and kps_conf[idx] > 0.5)
        if head_conf_count < 2:  # Need at least 2 facial keypoints with good confidence
            return False

        # 3. Structural Integrity (Shoulders, Hips, Knees, Ankles) — raised threshold
        major_joints = [5, 6, 11, 12, 13, 14, 15, 16]
        conf_count = sum(1 for idx in major_joints if idx < len(kps_conf) and kps_conf[idx] > 0.45)

        # Need at least 5 major joints with reasonable confidence
        return conf_count >= 5

    def process_frame(self, frame: np.ndarray) -> dict:
        self._frame_counter += 1
        self._fall_event_just_fired = False  # Reset at frame start; set to True below if event fires
        now = time.time()
        ts = now - self._fps_start

        if self.is_live and (self._frame_counter % self.process_every_n_frames != 0):
            if self._last_result: return self._last_result

        # YOLO Detection / Tracking
        # For live mode: use track() for stable person IDs (DeepSORT).
        # For offline video analysis: use predict() — 3-5x faster, no tracker overhead.
        if self.is_live:
            yolo_res = self._yolo.track(frame, persist=True, verbose=False, classes=[0], conf=0.25)[0]
        else:
            yolo_res = self._yolo.predict(frame, verbose=False, classes=[0], conf=0.25)[0]
        boxes_raw = yolo_res.boxes
        kps_data = yolo_res.keypoints
        
        all_persons = []
        possible_humans = []
        monitored_idx = -1
        
        if boxes_raw is not None and len(boxes_raw.xyxy) > 0:
            boxes = boxes_raw.xyxy.cpu().numpy()
            # YOLO IDs (may be None if not tracking)
            yolo_ids = boxes_raw.id.cpu().numpy().astype(int) if boxes_raw.id is not None else None
            
            for i in range(len(boxes)):
                kp_xy = kps_data.xy[i].cpu().numpy()
                kp_conf = kps_data.conf[i].cpu().numpy()
                y_id = yolo_ids[i] if yolo_ids is not None else i
                
                if self._is_human_pose(kp_xy, kp_conf, boxes[i]):
                    box = boxes[i]
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    
                    # Humanity Score
                    h_score = sum(kp_conf[idx] for idx in [5, 6, 11, 12, 13, 14, 15, 16] if idx < len(kp_conf))
                    
                    possible_humans.append({
                        "id": y_id, "box": box, "xy": kp_xy, "conf": kp_conf, 
                        "cx": cx, "cy": cy, "area": area, "h_score": h_score
                    })

            if possible_humans:
                # ── Sticky ID Tracking ──────────────────────────────────────
                # Priority order:
                #  1. Exact YOLO ID match
                #  2. Centroid proximity (handles YOLO tracker ID reassignment)
                #  3. Single person in frame → always monitored (regardless of lock)
                #  4. Never-locked yet → pick best pose score and lock
                target = None

                if self._monitored_id is not None:
                    matches = [p for p in possible_humans if p["id"] == self._monitored_id]
                    if matches:
                        target = matches[0]

                if target is None and self._monitored_centroid is not None:
                    best_dist = 1e6
                    for p in possible_humans:
                        dist = math.sqrt((p["cx"]-self._monitored_centroid[0])**2 + (p["cy"]-self._monitored_centroid[1])**2)
                        if dist < best_dist and dist < 250:
                            best_dist = dist
                            target = p

                if target is None and len(possible_humans) == 1:
                    # Only one person visible — they must be the monitored person
                    target = possible_humans[0]
                    self._monitored_id_locked = True

                if target is None and not self._monitored_id_locked:
                    # First time with multiple people — pick best pose score and lock
                    target = max(possible_humans, key=lambda p: p["h_score"])
                    self._monitored_id_locked = True

                if target is not None:
                    monitored_idx = target["id"]
                    self._monitored_id = monitored_idx
                    self._monitored_centroid = (target["cx"], target["cy"])
                    self._monitored_was_present = True
                    self._monitored_last_seen = now
                    self._disappear_fall_fired = False  # reset once found again
                else:
                    # Monitored person temporarily out of frame
                    monitored_idx = -1

                for p in possible_humans:
                    is_mon = (p["id"] == monitored_idx)
                    all_persons.append({
                        "box": p["box"], "xy": p["xy"], "is_monitored": is_mon,
                        "label": "MONITORED" if is_mon else "VISITOR"
                    })
            else:
                # No humans at all — keep the ID but clear centroid
                monitored_idx = -1
                self._monitored_centroid = None
                self._monitored_was_present = False
                if self._prototype_detector is not None:
                    self._prototype_detector.reset()
        else:
            self._monitored_id = None
            self._monitored_centroid = None

        # Fall Detection on Monitored Person
        lstm_prob = 0.0
        features = {}
        monitored_xy = None
        monitored_box = None
        bbox_is_horizontal = False
        # dt — time between frames in seconds.
        # For LIVE mode: use measured wall-clock processing fps (accurate for real-time).
        # For OFFLINE mode: use the real video fps so that time-based gates (standing
        # duration thresholds, etc.) fire at the correct VIDEO-time instants.
        # Without this fix, processing at 2fps on a 30fps video makes the HUD think
        # 10 seconds have elapsed after just 21 processed frames, clearing _is_currently_falling
        # and re-triggering the same fall 5-6 times in a single clip.
        if not self.is_live and self._video_fps > 0:
            dt = 1.0 / self._video_fps
        elif self._current_fps > 0:
            dt = 1.0 / self._current_fps
        else:
            dt = 1.0 / 30.0  # safe default

        if monitored_idx != -1:
            p = [p for p in possible_humans if p["id"] == monitored_idx][0]
            monitored_xy, monitored_box, conf_kps = p["xy"], p["box"], p["conf"]

            neck = (monitored_xy[5] + monitored_xy[6]) / 2.0
            mid_hip = (monitored_xy[11] + monitored_xy[12]) / 2.0
            torso_angle = self._get_angle(neck, mid_hip, vertical=True)
            is_upright = torso_angle < 35

            # Compute bbox aspect ratio — horizontal bbox = person is lying down
            bw = monitored_box[2] - monitored_box[0]
            bh = monitored_box[3] - monitored_box[1]
            bbox_is_horizontal = (bw > bh * 1.2) if bh > 0 else False

            if is_upright:
                self._standing_duration += dt
                if self._standing_duration > 0.3: self._has_been_standing = True
            else:
                self._standing_duration = max(0.0, self._standing_duration - dt)

            cooldown = self._EVENT_FRAME_COOLDOWN if self.is_live else 60
            fall_triggered = False

            if self._prototype_detector is not None:
                prob, p_event, p_alarm = self._prototype_detector.process_keypoints(now, monitored_xy, monitored_box, frame.shape)
                lstm_prob = prob
                is_sleeping_detected = getattr(self._prototype_detector, 'is_sleeping', False)

                # Re-arm on standing; set sleep mode when lying static
                if is_upright and self._standing_duration > 1.0:
                    self._is_sleeping = False
                    self._inactivity.set_external_sleep_status(False)
                elif is_sleeping_detected:
                    self._is_sleeping = True
                    self._inactivity.set_external_sleep_status(True)

                # 90%+ confirmed fall required; deliberate lie-down / sleeping strictly rejected
                fall_triggered = (
                    p_event and
                    prob >= 0.90 and
                    self._has_been_standing and
                    not self._is_sleeping and
                    not self._is_currently_falling and
                    (self._frame_counter - self._last_event_frame) > cooldown
                )
                if fall_triggered:
                    self._fall_confidence = int(prob * 100)
            else:
                if self._seq_buf is not None:
                    self._seq_buf.push(monitored_xy, conf_kps)
                    if self._seq_buf.ready and self._lstm is not None:
                        with torch.no_grad():
                            logits = self._lstm(self._seq_buf.get_tensor(device=self._device))
                            lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])

                is_significantly_tilted = torso_angle > 45
                if lstm_prob > self._lstm_thresh and self._has_been_standing:
                    if bbox_is_horizontal:
                        self._confirm_count += 2
                    elif is_significantly_tilted:
                        self._confirm_count += 1
                else:
                    self._confirm_count = max(0, self._confirm_count - 1)

                fall_triggered = (
                    self._confirm_count >= self._confirm_frames and
                    not self._is_currently_falling and
                    (self._frame_counter - self._last_event_frame) > cooldown
                )

            # ── Record pose state for pre-disappearance analysis ──
            self._pre_disappear_torso_angles.append(torso_angle)
            self._pre_disappear_lstm_probs.append(lstm_prob)
            self._pre_disappear_bbox_horiz.append(bbox_is_horizontal)
            # Keep only last N readings
            self._pre_disappear_torso_angles = self._pre_disappear_torso_angles[-self._PRE_HISTORY_LEN:]
            self._pre_disappear_lstm_probs   = self._pre_disappear_lstm_probs[-self._PRE_HISTORY_LEN:]
            self._pre_disappear_bbox_horiz   = self._pre_disappear_bbox_horiz[-self._PRE_HISTORY_LEN:]

            features = {"lstm_prob": lstm_prob, "fall_prob": lstm_prob, "is_upright": is_upright, "standing_duration": self._standing_duration,
                        "torso_angle": torso_angle, "fall_model": self.fall_model}
            if fall_triggered:
                self._fall_event_just_fired = True
                self._is_currently_falling = True
                self._fall_detected = True
                self._last_fall_event_time = now
                self._last_event_frame = self._frame_counter
                # Lock the visual indicator to the box where the fall was detected.
                # This prevents the "! FALL DETECTED" label from jumping to other people
                # in multi-person scenes (predict() mode uses sequential IDs that shift).
                self._fall_anchor_box = monitored_box.copy() if monitored_box is not None else None
                _sound_loud_beep()
            
            # HUD Clearing Logic: once clearly upright and stable, clear the fall flag
            if self._is_currently_falling:
                if is_upright and self._standing_duration > 2.0:
                    self._fall_detected = False
                    if self._standing_duration > 10.0:
                        self._is_currently_falling = False
                        self._confirm_count = 0
                        self._fall_anchor_box = None  # release anchor when fully resolved

        # ── Pose-Aware Disappearance Check ───────────────────────────────────
        # Only fires if the body was ALREADY showing fall characteristics
        # (tilting torso, rising LSTM score, or horizontal bbox) before vanishing.
        # Walking out of frame = body stays upright throughout → will NOT trigger.
        if (
            self._monitored_was_present and
            monitored_idx == -1 and
            self._monitored_last_seen is not None and
            not self._disappear_fall_fired and
            self._has_been_standing and
            not self._is_currently_falling and
            (self._frame_counter - self._last_event_frame) > self._EVENT_FRAME_COOLDOWN
        ):
            time_since_seen = now - self._monitored_last_seen
            if time_since_seen < self._DISAPPEAR_FALL_WINDOW and self._pre_disappear_torso_angles:
                # Analyse the last readings BEFORE the person left frame
                avg_torso  = sum(self._pre_disappear_torso_angles) / len(self._pre_disappear_torso_angles)
                max_torso  = max(self._pre_disappear_torso_angles)
                avg_lstm   = sum(self._pre_disappear_lstm_probs)   / len(self._pre_disappear_lstm_probs) if self._pre_disappear_lstm_probs else 0.0
                any_horiz  = any(self._pre_disappear_bbox_horiz)

                # Fall signal present before disappearing:
                #   a) Torso was seriously tilted (>50°) AND lstm elevated — hard fall signal
                #   b) BBox was already horizontal (lying down) before vanishing
                # NOTE: lstm alone (0.40 threshold) was too sensitive — fired on walking away.
                # Now we require EITHER (torso > 50 AND lstm > 0.60) OR (bbox was horizontal).
                was_falling_before = (
                    (max_torso > 65 and avg_lstm > 0.75) or   # tilting hard + lstm elevated
                    (any_horiz and avg_lstm > 0.70)            # horizontal bbox BUT still needs LSTM support — alone it fires on split-screen blobs
                )

                if was_falling_before:
                    self._fall_event_just_fired = True
                    self._fall_detected = True
                    self._is_currently_falling = True
                    self._last_fall_event_time = now
                    self._last_event_frame = self._frame_counter
                    self._disappear_fall_fired = True
                    _sound_loud_beep()
                    print(f"[UnitedMonitor] Pose-aware disappearance fall: torso={avg_torso:.1f}° lstm={avg_lstm:.2f} horiz={any_horiz}")

        # Inactivity
        inactivity_res = self._inactivity.update([monitored_box.tolist()] if monitored_box is not None else [], now)
        self._inactivity_alert = inactivity_res.get("alert", False)

        result = {
            "fall": {
                "fall_detected": self._fall_detected,
                "fall_event_fired": self._fall_event_just_fired,
                "confidence": self._fall_confidence if self._fall_detected else 0,
                "all_persons": all_persons,
                "features": features,
                "keypoints": monitored_xy,
                "boxes": monitored_box,
                "fall_anchor_box": self._fall_anchor_box,
            },
            "inactivity": inactivity_res,
            "global_state": {
                "fall_detected": self._fall_detected,
                "confidence": self._fall_confidence if self._fall_detected else 0,
                "is_sleeping": self._is_sleeping,
                "inactivity_alert": self._inactivity_alert,
                "fps": self._current_fps,
                "timestamp": ts
            }
        }
        
        self._fps_frames += 1
        if now - self._fps_start > 1.0:
            self._current_fps = self._fps_frames / (now - self._fps_start)
            self._fps_start, self._fps_frames = now, 0
        self._last_result = result
        return result

    def _get_angle(self, p1, p2, vertical=False):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        return math.degrees(math.atan2(abs(dx), abs(dy))) if vertical else math.degrees(math.atan2(dy, dx))

    def reset_fall(self):
        self._fall_detected = False
        self._is_currently_falling = False
        self._confirm_count = 0
        if self._seq_buf is not None:
            self._seq_buf.reset()
        if self._prototype_detector is not None:
            self._prototype_detector.reset()

def draw_united_interface(frame, results, draw_skeleton=True):
    frame_copy = frame.copy()
    fall_res = results["fall"]
    gs = results["global_state"]
    all_persons = fall_res.get("all_persons", [])
    fall_detected  = gs["fall_detected"]
    fall_anchor    = fall_res.get("fall_anchor_box")
    h, w = frame_copy.shape[:2]

    # ── Determine which person is the "fallen/monitored" one ──────────────────
    # When a fall is active, we locate the MONITORED person by finding whoever is
    # closest (by centroid) to the fall_anchor position — NOT by trusting the
    # is_monitored flag, which can jump to other people in predict() mode.
    # Before any fall: just use is_monitored as normal.
    if fall_detected and fall_anchor is not None:
        acx = (fall_anchor[0] + fall_anchor[2]) / 2.0
        acy = (fall_anchor[1] + fall_anchor[3]) / 2.0
        best_dist, fall_person_idx = 1e9, -1
        for idx, p in enumerate(all_persons):
            bx1, by1, bx2, by2 = p["box"]
            pcx, pcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = math.sqrt((pcx - acx) ** 2 + (pcy - acy) ** 2)
            if dist < best_dist:
                best_dist, fall_person_idx = dist, idx
        # Accept the closest match only if they're within 300px of the anchor.
        # Beyond that the anchor is stale (unlikely in a short clip) — fall back
        # to is_monitored in that edge case.
        use_anchor_for_person = (best_dist < 300)
    else:
        use_anchor_for_person = False
        fall_person_idx = -1

    # ── Draw bounding boxes and labels for each detected person ──
    for idx, p in enumerate(all_persons):
        box = p["box"]

        # Decide if this person is the monitored/fallen person:
        if fall_detected and use_anchor_for_person:
            is_fall_person = (idx == fall_person_idx)
        else:
            is_fall_person = p["is_monitored"]

        if is_fall_person:
            if fall_detected:
                color = (0, 0, 255)           # Red — fall
                conf_val = fall_res.get("confidence", 95)
                label = f"! FALL DETECTED ({conf_val}%)"
            elif gs.get("is_sleeping"):
                color = (255, 200, 0)         # Soft Cyan/Blue — Resting
                label = ">> RESTING / IN BED"
            elif gs["inactivity_alert"]:
                color = (0, 128, 255)         # Orange — inactivity
                label = ">> INACTIVITY ALERT"
            else:
                color = (0, 200, 0)           # Green — monitored OK
                label = ">> MONITORED"
        else:
            # Everyone else is always VISITOR — no exceptions
            color = (0, 200, 255)             # Yellow
            label = "VISITOR"

        x1, y1, x2, y2 = map(int, box)
        thickness = 3 if is_fall_person else 2
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame_copy, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if draw_skeleton and is_fall_person and p["xy"] is not None:
            kps = p["xy"]
            conn = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
            for s, e in conn:
                if s < len(kps) and e < len(kps):
                    cv2.line(frame_copy, (int(kps[s][0]), int(kps[s][1])), (int(kps[e][0]), int(kps[e][1])), color, 2)


    # ── Status bar at top of frame ── always visible even with no detections ──
    bar_h = 30
    cv2.rectangle(frame_copy, (0, 0), (w, bar_h), (20, 20, 20), -1)

    fps_val = gs.get("fps", 0.0)
    n_persons = len(all_persons)
    monitored_found = any(p["is_monitored"] for p in all_persons)

    # Build status string
    if gs["fall_detected"] and (fall_person_idx != -1 or (fall_anchor is not None and n_persons > 0)):
        conf_val = fall_res.get("confidence", 95)
        status_txt   = f"FALL DETECTED (CONFIRMED {conf_val}%)"
        status_color = (0, 0, 255)
    elif gs.get("is_sleeping"):
        status_txt   = "Resting / Sleeping in bed (Sleep mode active)"
        status_color = (255, 200, 0)
    elif n_persons == 0:
        status_txt   = "Scanning... (step back so full body is visible)"
        status_color = (100, 100, 100)
    elif monitored_found:
        if gs["inactivity_alert"]:
            status_txt   = "INACTIVITY ALERT"
            status_color = (0, 128, 255)
        else:
            status_txt   = "Monitoring active"
            status_color = (0, 200, 0)
    else:
        status_txt   = f"Visitors in frame: {n_persons}"
        status_color = (0, 200, 255)

    fps_txt = f"FPS:{fps_val:.1f}"
    cv2.putText(frame_copy, status_txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)
    cv2.putText(frame_copy, fps_txt, (w - 80, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame_copy

