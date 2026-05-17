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

_YOLO_MODEL_PATH = os.path.join(_here, "..", "yolov8n-pose.pt")
_LSTM_MODEL_PATH = os.path.join(_here, "fall_lstm.pth")

_SENSITIVITY_PRESETS = {
    "low":    {"lstm_thresh": 0.95, "confirm_frames": 10, "confirm_decay": 1},
    "medium": {"lstm_thresh": 0.90, "confirm_frames": 7,  "confirm_decay": 1},
    "high":   {"lstm_thresh": 0.85, "confirm_frames": 5,  "confirm_decay": 1},
}

class UnitedMonitor:
    def __init__(self, sensitivity="medium", inactivity_threshold=30, stable_percentage=0.10, is_live=False, process_every_n_frames=2):
        print(f"[UnitedMonitor] Initializing... live={is_live}")
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
        self._fall_start_time = None
        self._inactivity_alert = False
        self._last_result = None
        self._last_fall_event_time = -9999.0
        self._fall_event_just_fired = False
        self._is_currently_falling = False
        self._reset_confirm_frames = 0
        self._has_been_standing = False
        self._standing_duration = 0.0

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
        self._current_fps = 0.0

        self._yolo = YOLO(_YOLO_MODEL_PATH)
        self._lstm, self._device = load_model(_LSTM_MODEL_PATH)
        self._seq_buf = FallSequenceBuffer(seq_len=DEFAULT_SEQ_LEN)
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
        now = time.time()
        ts = now - self._fps_start

        if self.is_live and (self._frame_counter % self.process_every_n_frames != 0):
            if self._last_result: return self._last_result

        # YOLO Tracking (Stable IDs)
        yolo_res = self._yolo.track(frame, persist=True, verbose=False, classes=[0], conf=0.25)[0]
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
        else:
            self._monitored_id = None
            self._monitored_centroid = None

        # Fall Detection on Monitored Person
        lstm_prob = 0.0
        features = {}
        monitored_xy = None
        monitored_box = None
        bbox_is_horizontal = False
        dt = 1.0 / self._current_fps if self._current_fps > 0 else 0.033

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
                if self._standing_duration > 0.5: self._has_been_standing = True
            else:
                self._standing_duration = max(0.0, self._standing_duration - dt)

            self._seq_buf.push(monitored_xy, conf_kps)
            if self._seq_buf.ready:
                with torch.no_grad():
                    logits = self._lstm(self._seq_buf.get_tensor(device=self._device))
                    lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])

            # ── Record pose state for pre-disappearance analysis ──
            self._pre_disappear_torso_angles.append(torso_angle)
            self._pre_disappear_lstm_probs.append(lstm_prob)
            self._pre_disappear_bbox_horiz.append(bbox_is_horizontal)
            # Keep only last N readings
            self._pre_disappear_torso_angles = self._pre_disappear_torso_angles[-self._PRE_HISTORY_LEN:]
            self._pre_disappear_lstm_probs   = self._pre_disappear_lstm_probs[-self._PRE_HISTORY_LEN:]
            self._pre_disappear_bbox_horiz   = self._pre_disappear_bbox_horiz[-self._PRE_HISTORY_LEN:]

            features = {"lstm_prob": lstm_prob, "is_upright": is_upright, "standing_duration": self._standing_duration}
            
            # Confirmation logic — LSTM score + bbox orientation
            # bbox_is_horizontal: wide bbox means person is lying down
            # FIX: ALWAYS require bbox_is_horizontal — sitting people can score high LSTM
            # but their bbox stays vertical (tall box). Lying on the floor = wide/horizontal box.
            if lstm_prob > self._lstm_thresh and self._has_been_standing and bbox_is_horizontal:
                self._confirm_count += 1
            else:
                self._confirm_count = max(0, self._confirm_count - 1)

            fall_triggered = (
                self._confirm_count >= self._confirm_frames and
                not self._is_currently_falling and
                (self._frame_counter - self._last_event_frame) > self._EVENT_FRAME_COOLDOWN
            )
            if fall_triggered:
                self._fall_event_just_fired = True
                self._is_currently_falling = True
                self._fall_detected = True
                self._last_fall_event_time = now
                self._last_event_frame = self._frame_counter
            
            # HUD Clearing Logic: once clearly upright and stable, clear the fall flag
            if self._is_currently_falling:
                if is_upright and self._standing_duration > 2.0:
                    self._fall_detected = False
                    if self._standing_duration > 10.0:
                        self._is_currently_falling = False
                        self._confirm_count = 0

        # ── Pose-Aware Disappearance Check ───────────────────────────────────
        # Only fires if the body was ALREADY showing fall characteristics
        # (tilting torso, rising LSTM score, or horizontal bbox) before vanishing.
        # Walking out of frame = body stays upright throughout → will NOT trigger.
        self._fall_event_just_fired = False
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
                    (max_torso > 50 and avg_lstm > 0.60) or   # tilting hard + lstm elevated
                    any_horiz                                   # bbox already horizontal = lying
                )

                if was_falling_before:
                    self._fall_event_just_fired = True
                    self._fall_detected = True
                    self._is_currently_falling = True
                    self._last_fall_event_time = now
                    self._last_event_frame = self._frame_counter
                    self._disappear_fall_fired = True
                    print(f"[UnitedMonitor] Pose-aware disappearance fall: torso={avg_torso:.1f}° lstm={avg_lstm:.2f} horiz={any_horiz}")

        # Inactivity
        inactivity_res = self._inactivity.update([monitored_box.tolist()] if monitored_box is not None else [], now)
        self._inactivity_alert = inactivity_res.get("alert", False)

        result = {
            "fall": {
                "fall_detected": self._fall_detected,
                "fall_event_fired": self._fall_event_just_fired,
                "all_persons": all_persons,
                "features": features,
                "keypoints": monitored_xy,
                "boxes": monitored_box
            },
            "inactivity": inactivity_res,
            "global_state": {
                "fall_detected": self._fall_detected,
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
        self._seq_buf.reset()

def draw_united_interface(frame, results, draw_skeleton=True):
    frame_copy = frame.copy()
    fall_res = results["fall"]
    gs = results["global_state"]
    all_persons = fall_res.get("all_persons", [])
    h, w = frame_copy.shape[:2]

    # ── Draw bounding boxes and labels for each detected person ──
    for p in all_persons:
        box, is_mon = p["box"], p["is_monitored"]
        # Use plain ASCII text — OpenCV cannot render Unicode/emoji
        if is_mon:
            if gs["fall_detected"]:
                color = (0, 0, 255)       # Red — fall
                label = "! FALL DETECTED"
            elif gs["inactivity_alert"]:
                color = (0, 128, 255)     # Orange — inactivity
                label = ">> INACTIVITY ALERT"
            else:
                color = (0, 200, 0)       # Green — monitored OK
                label = ">> MONITORED"
        else:
            color = (0, 200, 255)         # Yellow — visitor
            label = "VISITOR"

        x1, y1, x2, y2 = map(int, box)
        thickness = 3 if is_mon else 2
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame_copy, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if draw_skeleton and is_mon and p["xy"] is not None:
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
    if gs["fall_detected"]:
        # Always show fall detected when flag is set — even if person temporarily
        # leaves the frame (e.g. lying flat with face down, rejected by pose check)
        status_txt   = "FALL DETECTED"
        status_color = (0, 0, 255)
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

