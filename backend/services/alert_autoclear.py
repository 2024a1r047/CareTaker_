"""
Auto-clear for EVENT alerts on the patient's dashboard (the red "High Alert" banner).

Event alerts are about a moment in time: medicines not taken, an emergency raised from the
voice assistant, a long lie after a fall, water not touched. They are cleared (marked read):
  1. as soon as the patient talks NORMALLY to the voice assistant (no pain / distress words),
     for alerts that are at least 2 minutes old, and
  2. automatically 4 hours after the alert was raised.

Clinical alerts (lab worsening, critical values from reports) are NOT touched here; they stay
until the caretaker handles them.
"""
import datetime
import re
from typing import Optional

from sqlalchemy.orm import Session

EXPIRE_HOURS = 4
MIN_AGE_SEC = 120          # don't clear an alert raised during this very conversation

_EVENT_PREFIXES = ("MEDICINES NOT TAKEN", "WATER NOT TOUCHED", "EMERGENCY triggered via Saathi Voice",
                   "EMERGENCY triggered via", "LONG LIE")

# words that mean the patient is NOT fine (Hinglish, English, Devanagari)
_DISTRESS = re.compile(
    r"(dard|pain|takleef|takleef|chakkar|ghabrahat|bechaini|ulti|vomit|bukhar|fever|saans|breath|"
    r"gir gay|gir gayi|giri|fell|fall|chot|injur|bachao|help me|emergency|ambulance|hospital|"
    r"behosh|faint|kamzori|weak|khoon|blood|seene|chest|sar dard|headache|bp high|sugar low|"
    r"accha nahi|achha nahi|theek nahi|thik nahi|not well|unwell|bimar|"
    r"दर्द|तकलीफ|चक्कर|घबराहट|उल्टी|बुखार|सांस|साँस|गिर|चोट|बचाओ|मदद|बेहोश|कमजोरी|खून|सीने|ठीक नहीं|अच्छा नहीं|बीमार)",
    re.I)


def is_event_alert(message: Optional[str]) -> bool:
    return bool(message) and message.strip().upper().startswith(tuple(p.upper() for p in _EVENT_PREFIXES))


def is_normal_talk(text: Optional[str], mood: Optional[str] = None) -> bool:
    """A real sentence from the patient with no sign of pain or distress."""
    t = (text or "").strip()
    if len(re.findall(r"\w+", t)) < 2:
        return False            # "haan", "..." — too short to tell
    if (mood or "").lower() in ("distressed", "sad", "anxious", "angry", "pain", "scared", "fearful"):
        return False
    return not _DISTRESS.search(t)


def _unread_event_alerts(db: Session, recipient_id: int):
    from tables.medical_conditions import MedicalAlert
    rows = (db.query(MedicalAlert)
            .filter(MedicalAlert.care_recipient_id == recipient_id, MedicalAlert.is_read == False)  # noqa: E712
            .all())
    return [a for a in rows if is_event_alert(a.message)]


def expire_old_event_alerts(db: Session, recipient_id: int, hours: int = EXPIRE_HOURS) -> int:
    """Event alerts older than `hours` are marked read. Returns how many were cleared."""
    if not recipient_id:
        return 0
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        n = 0
        for a in _unread_event_alerts(db, recipient_id):
            if a.created_at and a.created_at < cutoff:
                a.is_read = True
                n += 1
        if n:
            db.commit()
        return n
    except Exception as e:
        db.rollback()
        print(f"[alert_autoclear] expire failed: {e}")
        return 0


def clear_on_normal_talk(db: Session, recipient_id: int, text: Optional[str], mood: Optional[str] = None) -> int:
    """The patient spoke normally to the assistant: clear their (older) event alerts."""
    # any reply after the "aap gir gaye the, chot to nahi lagi?" question counts as answered
    try:
        from services.voice_health_brief import fall_checkin_answered
        if text and text.strip():
            fall_checkin_answered(recipient_id, text)
    except Exception:
        pass
    # remember lasting personal facts from what they just said (job, family, likes...) right away
    try:
        from services.saathi_memory_service import remember_from_turn
        remember_from_turn(recipient_id, text or "")
    except Exception:
        pass
    if not recipient_id or not is_normal_talk(text, mood):
        return 0
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=MIN_AGE_SEC)
        n = 0
        for a in _unread_event_alerts(db, recipient_id):
            if a.created_at and a.created_at <= cutoff:
                a.is_read = True
                n += 1
        if n:
            db.commit()
            print(f"[alert_autoclear] patient {recipient_id} talking normally -> cleared {n} event alert(s)")
        return n
    except Exception as e:
        db.rollback()
        print(f"[alert_autoclear] clear failed: {e}")
        return 0
