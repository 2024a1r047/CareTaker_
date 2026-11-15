"""
Fall event storage: keeps a permanent record (and readable summary) of every detected fall,
including whether the person got up afterwards and how long they were on the ground.

Used by:
  - uploaded video analysis   (routes/video_monitoring.py -> process_video_direct)
  - live camera streams       (VideoMonitoring/stream_manager.py)
  - in-browser TF.js live     (POST /api/video-monitoring/log-event)

Detection itself is NOT done here; this module only records what the detector reported.
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

_METHOD_LABELS = {
    "pose_model": "pose model",
    "lstm": "motion (LSTM) model",
    "disappearance": "fall while leaving the camera view",
    "tfjs": "in-browser live detector",
    "timeline": "whole-video posture analysis",
    "timeline_offscreen": "whole-video analysis, fell partly out of view",
}


def _fmt_secs(sec: Optional[float]) -> str:
    if sec is None:
        return "unknown time"
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f} s"
    m, s = divmod(int(round(sec)), 60)
    return f"{m} min {s} s"


def _fmt_video_time(sec: Optional[float]) -> str:
    if sec is None:
        return ""
    m, s = divmod(max(0.0, float(sec)), 60)
    return f"{int(m):02d}:{s:04.1f}"


def build_fall_summary(ev: Dict[str, Any]) -> str:
    """One readable line, e.g.
    'Fall at 00:12.4 in the video (93% confidence, pose model). The person got up after 8.4 s.'"""
    parts = []
    where = f"at {_fmt_video_time(ev.get('video_time_sec'))} in the video" if ev.get("video_time_sec") is not None \
        else f"at {(ev.get('occurred_at') or datetime.utcnow()).strftime('%d %b %Y %H:%M:%S')} UTC"
    detail = []
    if ev.get("confidence"):
        detail.append(f"{round(float(ev['confidence']))}% confidence")
    if ev.get("detection_method"):
        detail.append(_METHOD_LABELS.get(ev["detection_method"], ev["detection_method"]))
    parts.append(f"Fall {where}" + (f" ({', '.join(detail)})" if detail else "") + ".")
    if ev.get("got_up"):
        parts.append(f"The person got up after {_fmt_secs(ev.get('time_to_get_up_sec'))}.")
        t_up = ev.get("time_to_get_up_sec")
        if t_up is not None:
            parts.append("Within the 2-minute limit." if float(t_up) <= 120
                         else "That is longer than 2 minutes (long lie) — check for injury.")
    elif ev.get("time_on_ground_sec") is not None:
        parts.append(f"The person did NOT get up — still on the ground {_fmt_secs(ev.get('time_on_ground_sec'))} "
                     f"after the fall when {'the video ended' if ev.get('source') == 'upload' else 'monitoring stopped'}. "
                     f"Check on them.")
    else:
        parts.append("Waiting to see if the person gets up.")
    return " ".join(parts)


def build_video_summary(falls: List[Dict[str, Any]], duration_sec: float) -> str:
    """Overall summary for one analysed video."""
    if not falls:
        return f"No falls detected in {_fmt_secs(duration_sec)} of video."
    up = [f for f in falls if f.get("got_up")]
    down = [f for f in falls if not f.get("got_up")]
    s = f"{len(falls)} fall{'s' if len(falls) != 1 else ''} detected in {_fmt_secs(duration_sec)} of video. "
    if up:
        s += f"Got up after {len(up)} of them (" + ", ".join(_fmt_secs(f.get('time_to_get_up_sec')) for f in up) + "). "
    if down:
        s += f"{len(down)} fall{'s' if len(down) != 1 else ''} without getting up before the video ended — needs attention."
    return s.strip()


def _caretaker_id_for(db, username: Optional[str]) -> Optional[int]:
    if not username:
        return None
    from tables.users import CareTaker
    ct = db.query(CareTaker).filter(CareTaker.username == username).first()
    return ct.id if ct else None


def save_fall_event(data: Dict[str, Any]) -> Optional[int]:
    """Insert one fall row. `data` may hold caretaker_id or username. Returns the new id."""
    from config import SessionLocal
    from tables.video_analysis import FallEvent
    db = SessionLocal()
    try:
        cid = data.get("caretaker_id") or _caretaker_id_for(db, data.get("username"))
        row = FallEvent(
            caretaker_id=cid,
            recipient_id=data.get("recipient_id"),
            video_analysis_id=data.get("video_analysis_id"),
            source=data.get("source", "upload"),
            video_filename=data.get("video_filename"),
            camera_name=data.get("camera_name"),
            occurred_at=data.get("occurred_at") or datetime.utcnow(),
            video_time_sec=data.get("video_time_sec"),
            confidence=data.get("confidence"),
            detection_method=data.get("detection_method"),
            got_up=bool(data.get("got_up")),
            time_to_get_up_sec=data.get("time_to_get_up_sec"),
            time_on_ground_sec=data.get("time_on_ground_sec"),
            got_up_at=data.get("got_up_at"),
        )
        row.summary = data.get("summary") or build_fall_summary({**data, "occurred_at": row.occurred_at})
        db.add(row)
        db.commit()
        return row.id
    except Exception as e:
        db.rollback()
        logger.error(f"[FallEvents] could not save fall event: {e}")
        return None
    finally:
        db.close()


def update_fall_event(event_id: Optional[int], **changes) -> None:
    """Update a fall row (e.g. got_up / time_to_get_up_sec) and refresh its summary."""
    if not event_id:
        return
    from config import SessionLocal
    from tables.video_analysis import FallEvent
    db = SessionLocal()
    try:
        row = db.query(FallEvent).filter(FallEvent.id == event_id).first()
        if not row:
            return
        for k, v in changes.items():
            setattr(row, k, v)
        row.summary = build_fall_summary({
            "video_time_sec": row.video_time_sec, "occurred_at": row.occurred_at, "confidence": row.confidence,
            "detection_method": row.detection_method, "got_up": row.got_up, "source": row.source,
            "time_to_get_up_sec": row.time_to_get_up_sec, "time_on_ground_sec": row.time_on_ground_sec,
        })
        row.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"[FallEvents] could not update fall event {event_id}: {e}")
    finally:
        db.close()


def run_in_background(fn, *args, **kwargs) -> None:
    """DB writes from the live camera loop must not slow down frame processing."""
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


def fall_event_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "caretaker_id": row.caretaker_id,
        "recipient_id": row.recipient_id,
        "video_analysis_id": row.video_analysis_id,
        "source": row.source,
        "video_filename": row.video_filename,
        "camera_name": row.camera_name,
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
        "video_time_sec": row.video_time_sec,
        "confidence": row.confidence,
        "detection_method": row.detection_method,
        "got_up": row.got_up,
        "time_to_get_up_sec": row.time_to_get_up_sec,
        "time_on_ground_sec": row.time_on_ground_sec,
        "got_up_at": row.got_up_at.isoformat() if row.got_up_at else None,
        "summary": row.summary,
    }
