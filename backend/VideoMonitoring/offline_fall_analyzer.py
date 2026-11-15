"""
OfflineFallAnalyzer — whole-video fall analysis for UPLOADED recordings.

The live detectors must decide frame-by-frame. For an uploaded video we can look at the
entire timeline afterwards, which makes detection much more reliable for real home videos
(camera close to the person, legs out of view, camera cuts, low frame rates):

    1. Every analysed frame -> posture of the main person: upright / other / lying / none
       (uses RAW pose keypoints, so a half-visible person still counts).
    2. A FALL = the person was visible and NOT lying, then the head dropped quickly
       (relative to body size or frame height) and the body stayed LYING for >= 1 s.
       Lying down slowly onto a bed has no quick drop -> ignored.
       Bending over with straight legs under the hips is not "lying" -> ignored.
    3. GOT UP = upright again for >= 1.5 s after the fall; time on the floor = got-up time - fall time.
       Still down when the video ends -> time on the floor until the end.

Usage:
    an = OfflineFallAnalyzer()
    for each analysed frame: an.add(video_time_sec, people, frame_w, frame_h)
        people = [{"box": [x1,y1,x2,y2], "xy": (17,2) array, "conf": (17,) array}, ...]
    falls = an.finalize(video_duration_sec)
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np

KP_MIN = 0.30          # keypoint confidence to use a joint
STAY_DOWN_SEC = 1.0    # must stay lying this long to count as a fall
UP_HOLD_SEC = 1.5      # must stay upright this long to count as "got up"
PRE_WINDOW_SEC = 4.0   # look for the non-lying person within this time before the fall
DROP_WINDOW_SEC = 1.5  # the head drop must happen within this time
LONG_LIE_SEC = 120     # 2-minute rule


def _pt(xy, conf, idxs):
    pts = [xy[i] for i in idxs if i < len(conf) and conf[i] >= KP_MIN and (xy[i][0] > 0 or xy[i][1] > 0)]
    if not pts:
        return None
    return np.mean(np.array(pts, dtype=float), axis=0)


def classify(person: Dict[str, Any], fw: int, fh: int) -> Dict[str, Any]:
    """Posture of one person in one frame."""
    xy = np.asarray(person["xy"], dtype=float)
    conf = np.asarray(person.get("conf") if person.get("conf") is not None else np.ones(len(xy)), dtype=float)
    x1, y1, x2, y2 = [float(v) for v in person["box"]]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = bw / bh

    head = _pt(xy, conf, [0, 1, 2, 3, 4])
    sho = _pt(xy, conf, [5, 6])
    hip = _pt(xy, conf, [11, 12])
    knee = _pt(xy, conf, [13, 14])
    ankle = _pt(xy, conf, [15, 16])

    angle_v = None
    torso = None
    if sho is not None and hip is not None:
        dx, dy = hip[0] - sho[0], hip[1] - sho[1]
        torso = max(1.0, math.hypot(dx, dy))
        angle_v = math.degrees(math.atan2(abs(dx), abs(dy)))    # 0 = vertical torso, 90 = horizontal

    if angle_v is None and head is None and hip is None:
        return {"state": "none"}

    lying = False
    if angle_v is not None:
        if angle_v >= 60:
            lying = True
        elif angle_v >= 40 and aspect >= 1.2:
            lying = True
        # head at or below hip level with a tilted torso (face-down / on all fours on the floor)
        elif angle_v >= 35 and head is not None and hip is not None and head[1] >= hip[1] - 0.15 * torso:
            lying = True
    else:
        lying = aspect >= 1.5

    # Bending over to pick something up: torso horizontal but straight legs under the hips
    if lying and torso and hip is not None and ankle is not None:
        if (ankle[1] - hip[1]) > 1.4 * torso and abs(ankle[0] - hip[0]) < 0.6 * torso:
            lying = False

    upright = (angle_v is not None and angle_v <= 30 and aspect < 1.0
               and not (head is not None and hip is not None and head[1] > hip[1]))

    ref = head if head is not None else (sho if sho is not None else hip)
    return {
        "state": "lying" if lying else ("upright" if upright else "other"),
        "head_y": float(ref[1]) / fh,               # 0 = top of frame, 1 = bottom
        "box_h": bh / fh,
        "angle_v": angle_v,
        "aspect": aspect,
        "cx": (x1 + x2) / 2.0 / fw,
        "area": bw * bh / (fw * fh),
    }


class OfflineFallAnalyzer:
    def __init__(self):
        self.frames: List[Dict[str, Any]] = []
        self._last_cx: Optional[float] = None

    def add(self, t: float, people: List[Dict[str, Any]], fw: int, fh: int) -> None:
        """Record the main person's posture at video time t (seconds)."""
        best, best_score = None, -1.0
        for p in people or []:
            try:
                c = classify(p, fw, fh)
            except Exception:
                continue
            if c["state"] == "none":
                continue
            # biggest, most confident person; prefer the one near the previous position
            conf = p.get("conf")
            mean_conf = float(np.mean(conf)) if conf is not None and len(conf) else 0.5
            score = c["area"] * (0.5 + mean_conf)
            if self._last_cx is not None:
                score *= 1.0 / (1.0 + 3.0 * abs(c["cx"] - self._last_cx))
            if score > best_score:
                best, best_score = c, score
        if best is None:
            self.frames.append({"t": t, "state": "none"})
            return
        self._last_cx = best["cx"]
        best["t"] = t
        self.frames.append(best)

    # ── timeline analysis ──────────────────────────────────────────────────
    def _smoothed(self) -> List[str]:
        """Majority vote over ~0.5 s so single noisy frames don't flip the state."""
        states = [f["state"] for f in self.frames]
        out = []
        for i, f in enumerate(self.frames):
            window = [s for j, s in enumerate(states)
                      if abs(self.frames[j]["t"] - f["t"]) <= 0.25 and s != "none"]
            if not window:
                out.append("none")
                continue
            out.append(max(set(window), key=window.count))
        return out

    def finalize(self, duration_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        fr = self.frames
        if len(fr) < 3:
            return []
        st = self._smoothed()
        end_t = duration_sec if duration_sec is not None else fr[-1]["t"]
        falls: List[Dict[str, Any]] = []
        i = 0
        n = len(fr)
        while i < n:
            if st[i] != "lying":
                i += 1
                continue
            # lying segment (gaps of "none"/other up to 0.5 s allowed)
            j, last_lying = i, i
            while j + 1 < n:
                if st[j + 1] == "lying":
                    last_lying = j + 1
                elif fr[j + 1]["t"] - fr[last_lying]["t"] > 0.5:
                    break
                j += 1
            # Face-down / on hands and knees on the floor flips between "lying" and "other":
            # extend the down period through neighbouring "other" frames at the same low height.
            low_heads = [fr[k]["head_y"] for k in range(i, last_lying + 1) if st[k] == "lying" and "head_y" in fr[k]]
            if low_heads:
                low = float(np.median(low_heads)) - 0.06
                s = i
                while s - 1 >= 0 and st[s - 1] == "other" and fr[s - 1].get("head_y", 0) >= low \
                        and fr[s]["t"] - fr[s - 1]["t"] <= 0.5:
                    s -= 1
                e = last_lying
                while e + 1 < n and st[e + 1] in ("other", "none", "lying") and \
                        (st[e + 1] == "none" or fr[e + 1].get("head_y", 0) >= low) and fr[e + 1]["t"] - fr[e]["t"] <= 0.5:
                    e += 1
                visible = [k for k in range(s, e + 1) if st[k] != "none"]
                lying_share = (sum(1 for k in visible if st[k] == "lying") / len(visible)) if visible else 0
                if lying_share >= 0.3:
                    i, last_lying = s, max(last_lying, e)
            seg_start, seg_end = fr[i]["t"], fr[last_lying]["t"]

            # person visible and NOT lying shortly before
            pre = [k for k in range(i) if fr[i]["t"] - fr[k]["t"] <= PRE_WINDOW_SEC + 2.0
                   and st[k] in ("upright", "other") and "head_y" in fr[k]]

            # Fell partly out of the camera view: visible -> not visible -> seen lying low.
            # Then the drop is measured from the last visible frame and the fall starts when they vanished.
            hidden = False
            fall_start = seg_start
            drop_window = DROP_WINDOW_SEC
            if pre:
                p = pre[-1]
                between = st[p + 1:i]
                gap = seg_start - fr[p]["t"]
                if between and gap <= 6.0 and between.count("none") / len(between) >= 0.7:
                    hidden = True
                    fall_start = fr[p + 1]["t"]
                    drop_window = gap + DROP_WINDOW_SEC
            pre = [k for k in pre if seg_start - fr[k]["t"] <= (drop_window if hidden else PRE_WINDOW_SEC)]

            stays_down = ((seg_end - seg_start) >= STAY_DOWN_SEC
                          or (end_t - seg_end < 0.6 and seg_end - seg_start >= 0.6)
                          or (hidden and seg_end - seg_start >= 0.3 and seg_end - fall_start >= 1.5))

            fall = None
            if stays_down and len(pre) >= (1 if hidden else 2):
                lying_idx = [k for k in range(i, last_lying + 1) if st[k] == "lying" and "head_y" in fr[k]]
                down_head = float(np.median([fr[k]["head_y"] for k in lying_idx[:6]])) if lying_idx else None
                best_drop_frame, best_drop_body, quick = 0.0, 0.0, False
                for k in pre:
                    dt = seg_start - fr[k]["t"]
                    if dt > drop_window or down_head is None:
                        continue
                    d_frame = down_head - fr[k]["head_y"]
                    d_body = d_frame / max(0.05, fr[k]["box_h"])
                    best_drop_frame = max(best_drop_frame, d_frame)
                    best_drop_body = max(best_drop_body, d_body)
                    if d_frame >= 0.15 or d_body >= 0.30:
                        quick = True
                if quick:
                    conf = 70 + min(15, best_drop_body * 20) + min(10, (seg_end - seg_start) * 3) - (8 if hidden else 0)
                    fall = {"timestamp": round(fall_start, 2), "confidence": int(min(97, conf)),
                            "detection_method": "timeline_offscreen" if hidden else "timeline",
                            "drop_frac": round(best_drop_body, 2)}
                    seg_start = fall_start   # time on the floor counts from the fall itself

            if fall:
                # Did the person get up afterwards?
                k = last_lying + 1
                up_start = None
                while k < n:
                    if st[k] == "upright":
                        if up_start is None:
                            up_start = k
                        if fr[k]["t"] - fr[up_start]["t"] >= UP_HOLD_SEC:
                            break
                    elif st[k] == "lying":
                        up_start = None
                    k += 1
                if up_start is not None and k < n:
                    secs = round(max(0.0, fr[up_start]["t"] - seg_start), 1)
                    fall.update({"got_up": True, "time_to_get_up_sec": secs,
                                 "got_up_at_video_sec": round(fr[up_start]["t"], 2),
                                 "within_limit": secs <= LONG_LIE_SEC})
                    i = k
                else:
                    fall.update({"got_up": False, "time_to_get_up_sec": None,
                                 "time_on_ground_sec": round(max(0.0, end_t - seg_start), 1)})
                    i = n
                falls.append(fall)
            else:
                i = last_lying + 1
        return falls


def merge_falls(primary: List[Dict[str, Any]], timeline: List[Dict[str, Any]], gap_sec: float = 4.0) -> List[Dict[str, Any]]:
    """Union of the frame-by-frame detector's falls and the timeline falls (same fall = within gap_sec)."""
    out = [dict(f) for f in primary]
    for tf in timeline:
        match = next((f for f in out if abs(float(f.get("timestamp", 0)) - tf["timestamp"]) <= gap_sec), None)
        if match:
            if not match.get("got_up") and tf.get("got_up"):
                match["got_up"] = True
                match["time_to_get_up_sec"] = tf["time_to_get_up_sec"]
                match["got_up_at_video_sec"] = tf.get("got_up_at_video_sec")
            continue
        out.append({
            "timestamp": tf["timestamp"], "message": "Fall Detected", "frame": None,
            "confidence": tf["confidence"], "detection_method": "timeline",
            "got_up": tf.get("got_up", False), "time_to_get_up_sec": tf.get("time_to_get_up_sec"),
            "got_up_at_video_sec": tf.get("got_up_at_video_sec"),
            "time_on_ground_sec": tf.get("time_on_ground_sec"),
        })
    out.sort(key=lambda f: float(f.get("timestamp", 0)))
    return out
