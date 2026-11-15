"""
Voice Health Brief — one place that turns the patient's records into spoken answers.

Used by:
  * the built-in Saathi engine (services/voice_bot_engine.py) to answer questions like
    "meri neend kaisi thi?", "agli dawai kaunsi hai?", "meri tabiyat kaisi hai?",
    "kya main giri thi?", "bahar mausam kaisa hai?" from the database instead of filler lines;
  * Asha (ElevenLabs) via services/agent_medical_context.py, which adds the monitoring /
    weather / sleep text below to the MEDICAL RECORD briefing sent at the start of a call.

Every function is read-only and fails soft (returns "" / None), so a missing table or an
API error never breaks a conversation.
"""
import os
import re
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(os.getenv("PATIENT_TIMEZONE", "Asia/Kolkata"))
except Exception:  # pragma: no cover - zoneinfo/tzdata missing
    _TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


# ─── helpers ────────────────────────────────────────────────────────────────
def now_local() -> datetime.datetime:
    return datetime.datetime.now(_TZ)


def _to_local(dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """DB timestamps are naive UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(_TZ)


def _fmt_when(dt: Optional[datetime.datetime]) -> str:
    """'aaj subah 8:15', 'kal raat 9:30', '6 Nov dopahar 3:45'."""
    d = _to_local(dt)
    if not d:
        return ""
    h = d.hour
    part = "subah" if 4 <= h < 12 else "dopahar" if 12 <= h < 17 else "shaam" if 17 <= h < 20 else "raat"
    clock = f"{(h % 12) or 12}:{d.minute:02d}"
    today = now_local().date()
    if d.date() == today:
        day = "aaj"
    elif d.date() == today - datetime.timedelta(days=1):
        day = "kal"
    else:
        day = d.strftime("%-d %b") if os.name != "nt" else d.strftime("%#d %b")
    return f"{day} {part} {clock} baje"


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _g(v) -> str:
    """Pretty number: 97.70 -> 97.7, 70.0 -> 70."""
    n = _num(v)
    if n is None:
        return str(v)
    return f"{n:g}" if abs(n) < 1000 else str(int(round(n)))


def _recipient(db: Session, rid: int):
    from tables.users import CareRecipient
    return db.query(CareRecipient).filter(CareRecipient.id == rid).first()


def _gender(recip) -> str:
    if not recip or not recip.gender:
        return ""
    return (recip.gender.value if hasattr(recip.gender, "value") else str(recip.gender)).lower()


# ─── vitals & sleep ─────────────────────────────────────────────────────────
_VITAL_FIELDS = ("heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation",
                 "temperature", "sleep_score", "bmi", "weight")


def latest_vitals(db: Session, rid: int) -> Dict[str, Any]:
    """Latest non-empty value of each vital (rows can be partially filled), plus sleep history."""
    from tables.vital_signs import VitalSign
    rows = (db.query(VitalSign).filter(VitalSign.care_recipient_id == rid)
            .order_by(VitalSign.recorded_at.desc()).limit(60).all())
    out: Dict[str, Any] = {"sleep_history": []}
    for r in rows:
        for f in _VITAL_FIELDS:
            val = getattr(r, f, None)
            if val is not None and f not in out:
                out[f] = val
                out[f + "_at"] = r.recorded_at
        if r.sleep_score is not None:
            out["sleep_history"].append((r.recorded_at, float(r.sleep_score)))
    return out


def bp_category(s, d) -> str:
    if s is None or d is None:
        return ""
    if s >= 180 or d >= 120:
        return "bahut zyada (khatarnak)"
    if s >= 140 or d >= 90:
        return "zyada (stage 2)"
    if s >= 130 or d >= 80:
        return "thoda zyada (stage 1)"
    if s >= 120:
        return "thoda badha hua"
    if s < 90 or d < 60:
        return "kam"
    return "normal"


def sleep_text(db: Session, rid: int, address: str) -> str:
    v = latest_vitals(db, rid)
    score = v.get("sleep_score")
    if score is None:
        return (f"{address}, aapka sleep score abhi record mein nahi hai. Aapke caretaker ise app mein "
                f"'Log Vitals' se daal sakte hain. Waise kal raat neend kaisi aayi?")
    s = float(score)
    quality = ("bahut achhi" if s >= 85 else "achhi" if s >= 75 else "theek-thaak, thodi kam"
               if s >= 60 else "kam, neend poori nahi hui")
    text = f"{address}, aapka sabse naya sleep score {_g(s)} out of 100 hai ({_fmt_when(v.get('sleep_score_at'))}). Yeh {quality} hai."
    hist = [x[1] for x in v.get("sleep_history", [])[:7]]
    if len(hist) >= 2:
        avg = sum(hist) / len(hist)
        prev = hist[1]
        trend = "pehle se behtar" if s > prev + 3 else "pehle se kam" if s < prev - 3 else "pehle jaisa hi"
        text += f" Pichhli baar {_g(prev)} tha, yaani {trend}. Pichhle {len(hist)} records ka average {_g(round(avg))} hai."
    if s < 75:
        text += (" Achhi neend ke liye raat ko jaldi khana, sone se pehle chai-coffee na lena, aur ek hi samay "
                 "par sona madad karta hai. Chahein to main sone se pehle halka bhajan laga sakti hoon.")
    return text


def vital_text(db: Session, rid: int, address: str, which: str) -> str:
    v = latest_vitals(db, rid)
    when = lambda f: _fmt_when(v.get(f + "_at"))
    if which == "bp":
        s, d = v.get("systolic_bp"), v.get("diastolic_bp")
        if s is None or d is None:
            return f"{address}, aapka BP abhi record mein nahi hai. Agar machine hai to abhi naap kar mujhe bataiye, main samjha dungi."
        cat = bp_category(s, d)
        tip = ""
        if s >= 140 or d >= 90:
            tip = " 5 minute aaram se baith kar dobara naapiye. Namak kam lijiye aur BP ki dawai samay par lijiye."
        if s >= 180 or d >= 120:
            tip = " Yeh bahut zyada hai. 5 minute baad dobara naapiye; agar phir bhi itna hai ya seene mein dard, chakkar ya saans ki dikkat ho, to main turant caretaker ko call karti hoon."
        return f"{address}, aapka aakhri BP {s}/{d} tha ({when('systolic_bp')}). Normal 120/80 se kam hota hai, to yeh {cat} hai.{tip}"
    if which == "pulse":
        hr = v.get("heart_rate")
        if hr is None:
            return f"{address}, aapki dhadkan (pulse) abhi record mein nahi hai."
        st = "zyada" if hr > 100 else "kam" if hr < 60 else "normal"
        return f"{address}, aapki aakhri pulse {hr} thi ({when('heart_rate')}). Normal 60 se 100 hoti hai, to yeh {st} hai." + (
            " Thoda aaram kijiye aur dobara check kijiye." if st != "normal" else "")
    if which == "oxygen":
        o2 = v.get("oxygen_saturation")
        if o2 is None:
            return f"{address}, aapka oxygen level (SpO2) abhi record mein nahi hai."
        o2 = float(o2)
        st = ("normal" if o2 >= 95 else "thoda kam, doctor ko batana chahiye" if o2 >= 92
              else "kam, jaldi doctor ko dikhana chahiye" if o2 >= 90 else "bahut kam, yeh emergency hai")
        return f"{address}, aapka aakhri oxygen level {_g(o2)}% tha ({when('oxygen_saturation')}). Normal 95 se 100 hota hai, yeh {st}." + (
            " Kya main abhi caretaker ko call karun?" if o2 < 90 else "")
    if which == "temperature":
        t = v.get("temperature")
        if t is None:
            return f"{address}, aapka temperature abhi record mein nahi hai."
        t = float(t)
        f = t if t > 50 else t * 9 / 5 + 32
        st = "normal" if f < 99.5 else "halka bukhar" if f < 100.4 else "bukhar"
        return f"{address}, aapka aakhri temperature {_g(t)}{'°F' if t > 50 else '°C'} tha ({when('temperature')}). Yeh {st} hai."
    if which == "weight":
        parts = []
        if v.get("weight") is not None:
            parts.append(f"wajan {_g(v['weight'])} kg")
        if v.get("bmi") is not None:
            b = float(v["bmi"])
            cat = "kam" if b < 18.5 else "normal" if b < 25 else "thoda zyada (overweight)" if b < 30 else "zyada (obese)"
            parts.append(f"BMI {_g(b)}, jo {cat} hai (normal 18.5 se 24.9)")
        if not parts:
            return f"{address}, aapka wajan aur BMI abhi record mein nahi hai."
        return f"{address}, aapka " + " aur ".join(parts) + "."
    return ""


def vitals_overview(db: Session, rid: int) -> List[str]:
    """Short Hinglish concerns from the latest vitals (empty list if all fine)."""
    v = latest_vitals(db, rid)
    out = []
    s, d = v.get("systolic_bp"), v.get("diastolic_bp")
    if s is not None and d is not None and (s >= 130 or d >= 80 or s < 90):
        out.append(f"BP {s}/{d} {bp_category(s, d)}")
    hr = v.get("heart_rate")
    if hr is not None and (hr > 100 or hr < 60):
        out.append(f"pulse {hr} {'zyada' if hr > 100 else 'kam'}")
    o2 = _num(v.get("oxygen_saturation"))
    if o2 is not None and o2 < 95:
        out.append(f"oxygen {_g(o2)}% kam")
    sl = _num(v.get("sleep_score"))
    if sl is not None and sl < 70:
        out.append(f"sleep score {_g(sl)} kam")
    return out


# ─── medicines ──────────────────────────────────────────────────────────────
def _parse_times(m) -> List[datetime.time]:
    raw = f"{m.schedule_time or ''}"
    times = []
    for hh, mm in re.findall(r"\b(\d{1,2}):(\d{2})\b", raw):
        h, mi = int(hh), int(mm)
        if 0 <= h < 24 and 0 <= mi < 60:
            times.append(datetime.time(h, mi))
    return sorted(set(times))


def _active_meds(db: Session, rid: int):
    from tables.medications import Medication
    try:
        from tables.medications import MedicationStatus
        statuses = [MedicationStatus.active, "active"]
    except Exception:
        statuses = ["active"]
    return db.query(Medication).filter(Medication.care_recipient_id == rid,
                                       Medication.status.in_(statuses)).all()


def _clock(t: datetime.time) -> str:
    h = t.hour
    part = "subah" if 4 <= h < 12 else "dopahar" if 12 <= h < 17 else "shaam" if 17 <= h < 20 else "raat"
    return f"{part} {(h % 12) or 12}{'' if t.minute == 0 else ':%02d' % t.minute} baje"


def medicine_schedule(db: Session, rid: int) -> Dict[str, Any]:
    now = now_local()
    now_min = now.hour * 60 + now.minute
    slots: Dict[datetime.time, List[str]] = {}
    sos = []
    for m in _active_meds(db, rid):
        label = f"{m.medicine_name}" + (f" {m.dosage}" if m.dosage else "")
        times = _parse_times(m)
        if not times:
            if "need" in (m.frequency or "").lower() or "sos" in (m.frequency or "").lower():
                sos.append(label)
            continue
        for t in times:
            slots.setdefault(t, []).append(label)
    due, nxt = [], None
    for t in sorted(slots):
        diff = (t.hour * 60 + t.minute) - now_min
        if -90 <= diff <= 45:
            due.append((t, slots[t]))
    upcoming = [t for t in sorted(slots) if (t.hour * 60 + t.minute) - now_min > 45]
    if upcoming:
        nxt = (upcoming[0], slots[upcoming[0]], False)
    elif slots:
        first = sorted(slots)[0]
        nxt = (first, slots[first], True)  # tomorrow
    return {"due": due, "next": nxt, "sos": sos, "slots": slots}


def next_medicine_text(db: Session, rid: int, address: str) -> str:
    sch = medicine_schedule(db, rid)
    if not sch["slots"] and not sch["sos"]:
        return f"{address}, aapke profile mein abhi koi samay wali dawai darj nahi hai."
    parts = []
    if sch["due"]:
        for t, names in sch["due"]:
            parts.append(f"{_clock(t)} ki dawai abhi leni hai: {', '.join(names)}")
        msg = f"{address}, " + "; ".join(parts) + ". Abhi le lijiye, aur lene ke baad mujhe bata dijiye."
    else:
        msg = f"{address}, abhi koi dawai due nahi hai."
    if sch["next"]:
        t, names, tomorrow = sch["next"]
        msg += f" Agli dawai {'kal ' if tomorrow else ''}{_clock(t)} hai: {', '.join(names)}."
    if sch["sos"]:
        msg += f" Zaroorat padne par: {', '.join(sch['sos'])}."
    return msg


# ─── labs, conditions, allergies, reports ───────────────────────────────────
_LAB_WORDS = {
    "sugar": ["glucose", "hba1c", "sugar"],
    "cholesterol": ["cholesterol", "ldl", "hdl", "triglyceride"],
    "hemoglobin": ["hemoglobin", "haemoglobin", "rbc", "iron", "ferritin"],
    "kidney": ["creatinine", "egfr", "urea", "bun", "microalbumin", "uric"],
    "liver": ["alt", "ast", "bilirubin", "alp", "sgpt", "sgot", "albumin"],
    "vitamin": ["vitamin"],
    "thyroid": ["tsh", "t3", "t4", "thyroid"],
}


def lab_lines(db: Session, rid: int) -> Tuple[List[str], str]:
    """Plausible latest lab values (from the same source Asha uses) + abnormal summary."""
    from services.agent_medical_context import _labs_section
    recip = _recipient(db, rid)
    lines, abnormal = _labs_section(rid, db, _gender(recip))
    return [l.lstrip("- ").strip() for l in (lines or "").splitlines() if l.strip()], abnormal or ""


def _hinglish_lab(line: str) -> str:
    # "Fasting Glucose 168 mg/dL (normal 70-99) HIGH, on 2026-10-20"
    line = re.sub(r"\s*\[[^\]]*\]", "", line)  # drop English footnotes like "[2 hours after a meal]"
    line = re.sub(r",?\s*on (\d{4}-\d{2}-\d{2})", r" (\1 ki report)", line)
    return (line.replace(" HIGH", " — zyada hai").replace(" LOW", " — kam hai")
                .replace(" NORMAL", " — normal hai").replace("(normal ", "(normal range "))


def lab_topic_text(db: Session, rid: int, address: str, topic: str) -> str:
    words = _LAB_WORDS.get(topic, [topic])
    lines, _ = lab_lines(db, rid)
    hits = [l for l in lines if any(w in l.lower() for w in words)]
    if not hits:
        return (f"{address}, aapke record mein abhi {topic} ki koi jaanch nahi hai. "
                f"Jab report upload hogi, main aapko bata dungi.")
    return f"{address}, aapki {topic} jaanch: " + "; ".join(_hinglish_lab(h) for h in hits[:4]) + "."


def abnormal_labs_text(db: Session, rid: int, limit: int = 4) -> str:
    _, abnormal = lab_lines(db, rid)
    if not abnormal:
        return ""
    items = [a.strip() for a in abnormal.split(",") if a.strip()]
    # "Fasting Glucose high (168 mg/dL)" -> "Fasting Glucose 168 mg/dL zyada"
    out = []
    for it in items[:limit]:
        m = re.match(r"(.+?)\s+(high|low)\s*\((.+)\)", it, re.I)
        out.append(f"{m.group(1)} {m.group(3)} {'zyada' if m.group(2).lower() == 'high' else 'kam'}" if m else it)
    more = len(items) - limit
    return ", ".join(out) + (f", aur {more} aur" if more > 0 else "")


def conditions_list(db: Session, rid: int) -> List[str]:
    from tables.medical_conditions import PatientCondition
    try:
        from tables.medical_conditions import ConditionStatus
        flt = PatientCondition.status == ConditionStatus.active
    except Exception:
        flt = PatientCondition.status == "active"
    return [c.disease_name for c in db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid, flt).all() if c.disease_name]


def allergies_list(db: Session, rid: int) -> List[str]:
    from tables.allergies import Allergy, AllergyStatus
    return [a.allergen + (f" ({a.reaction})" if a.reaction else "") for a in db.query(Allergy).filter(
        Allergy.care_recipient_id == rid, Allergy.status == AllergyStatus.active).all()]


def active_recommendations(db: Session, rid: int, limit: int = 3) -> List[str]:
    from tables.medical_recommendations import MedicalRecommendation
    order = {"critical": 0, "high": 1, "medium": 2, "suggestion": 3}
    recs = (db.query(MedicalRecommendation)
            .filter(MedicalRecommendation.care_recipient_id == rid,
                    MedicalRecommendation.resolved_at.is_(None),
                    (MedicalRecommendation.archived == False) | (MedicalRecommendation.archived.is_(None)))  # noqa: E712
            .order_by(MedicalRecommendation.created_at.desc()).limit(20).all())
    recs.sort(key=lambda r: order.get((r.severity or "").lower(), 5))
    out, seen = [], set()
    for r in recs:
        title = (r.title or r.metric or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        act = (r.do_this_now or "").strip().rstrip(".!")
        out.append(f"{title.rstrip('.')}" + (f": {act}" if act else ""))
        if len(out) >= limit:
            break
    return out


def reports_text(db: Session, rid: int, address: str) -> str:
    from tables.medical_reports import MedicalReport
    reps = (db.query(MedicalReport).filter(MedicalReport.care_recipient_id == rid)
            .order_by(MedicalReport.report_date.desc().nullslast(), MedicalReport.uploaded_at.desc()).limit(3).all())
    if not reps:
        return f"{address}, aapke profile mein abhi koi medical report upload nahi hui hai."
    latest = reps[0]
    date = latest.report_date or (latest.uploaded_at.date() if latest.uploaded_at else None)
    name = re.sub(r"[_\-]+", " ", re.sub(r"\.\w+$", "", latest.filename or "report")).strip()
    msg = f"{address}, aapki sabse nayi report {date.strftime('%d %b %Y') if date else ''} ki hai ({name})."
    ab = abnormal_labs_text(db, rid)
    if ab:
        msg += f" Reports ke hisaab se jo values normal se alag hain: {ab}."
    else:
        msg += " Isme koi khaas value normal range se bahar nahi hai."
    recs = active_recommendations(db, rid, 1)
    if recs:
        msg += f" Sabse zaroori salah: {recs[0]}."
    return msg


# ─── monitoring: falls, sounds, alerts ──────────────────────────────────────
def _scope_filter(model, rid_col, care_col, recip):
    """Events tagged with this patient, or untagged events from the same caretaker's camera/mic."""
    from sqlalchemy import or_, and_
    if recip and recip.caretaker_id:
        return or_(rid_col == recip.id, and_(rid_col.is_(None), care_col == recip.caretaker_id))
    return rid_col == (recip.id if recip else -1)


def falls_info(db: Session, rid: int, days: int = 30) -> Dict[str, Any]:
    try:
        from tables.video_analysis import FallEvent
    except Exception:
        return {}
    recip = _recipient(db, rid)
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    try:
        q = db.query(FallEvent).filter(FallEvent.occurred_at >= since,
                                       _scope_filter(FallEvent, FallEvent.recipient_id, FallEvent.caretaker_id, recip))
        evs = q.order_by(FallEvent.occurred_at.desc()).all()
    except Exception:
        db.rollback()
        return {}
    # Merge detections < 90 s apart into one fall (older detector logged the same fall repeatedly)
    merged = []
    for e in sorted([e for e in evs if e.occurred_at], key=lambda x: x.occurred_at):
        if merged and (e.occurred_at - merged[-1]["last"]).total_seconds() <= 90:
            merged[-1]["last"] = e.occurred_at
            if e.got_up:
                merged[-1]["ev"] = e if not merged[-1]["ev"].got_up else merged[-1]["ev"]
            continue
        merged.append({"ev": e, "last": e.occurred_at})
    falls = [m["ev"] for m in merged]
    falls.sort(key=lambda x: x.occurred_at, reverse=True)
    week = [e for e in falls if e.occurred_at >= datetime.datetime.utcnow() - datetime.timedelta(days=7)]
    return {"count": len(falls), "week": len(week), "last": falls[0] if falls else None, "days": days}


def falls_text(db: Session, rid: int, address: str) -> str:
    info = falls_info(db, rid)
    if not info:
        return f"{address}, girne ki jaankari ke liye camera monitoring abhi judi nahi hai."
    if not info["count"]:
        return f"{address}, pichhle {info['days']} din mein camera ne girne ka koi record nahi dekha. Bahut achhi baat hai, dhyan se chaliye."
    e = info["last"]
    msg = f"{address}, pichhle {info['days']} din mein camera ne {info['count']} baar girne jaisa record kiya hai"
    msg += f" (is hafte {info['week']} baar)." if info["week"] else "."
    msg += f" Aakhri baar {_fmt_when(e.occurred_at)}"
    if e.got_up:
        t = e.time_to_get_up_sec
        msg += f", aur aap {('%g second' % round(t, 1)) if t is not None and t < 60 else 'kuch der'} mein uth gaye the."
    else:
        msg += ", uske baad uthne ka record nahi mila."
    msg += " Kya us waqt kahin chot lagi thi? Agar dard ho to main caretaker ko bata deti hoon."
    return msg


# Falls already asked about in a conversation (recipient id, fall id) - kept for older callers
_FALL_CHECKIN_ASKED: set = set()
# rid -> fall id the patient has ANSWERED about (they spoke after being asked). Until then every new
# conversation opens with the fall question again.
_FALL_CHECKIN_ANSWERED: Dict[int, int] = {}
_FALL_CHECKIN_PENDING: Dict[int, Any] = {}      # rid -> (fall id, when, address) asked about, waiting for a reply


def fall_checkin_answered(rid: Optional[int], reply: str = "") -> None:
    """The patient spoke after the fall question: never open with it again for this fall, and
    remember the fall (and what they said) permanently in the assistant's memory."""
    if not rid or rid not in _FALL_CHECKIN_PENDING:
        return
    fall_id, when, address = _FALL_CHECKIN_PENDING.pop(rid)
    _FALL_CHECKIN_ANSWERED[rid] = fall_id
    try:
        from services.saathi_memory_service import remember_fact
        said = re.sub(r"\s+", " ", reply or "").strip()[:140]
        remember_fact(rid, "health",
                      f"{address} {when} ghar par gir gaye the (fall #{fall_id}); chot ke baare mein poochhne par "
                      f"unhone kaha: \"{said}\". Is baare mein dobara mat poochhna, bas yaad rakhna aur dhyan rakhna.")
    except Exception:
        pass


def recent_fall_checkin(db: Session, rid: int, address: str, hours: int = 12,
                        mark: bool = True) -> Optional[Dict[str, Any]]:
    """The camera saw a fall in the last `hours` that the assistant has not asked about yet.

    Returns {"hi": opening line in Hinglish, "en": English, "context": note for the agent} or None.
    With mark=True the fall counts as asked, so the next conversation does not repeat it.
    """
    info = falls_info(db, rid, 1)
    e = info.get("last") if info else None
    if not e or not e.occurred_at:
        return None
    if e.occurred_at < datetime.datetime.utcnow() - datetime.timedelta(hours=hours):
        return None
    if _FALL_CHECKIN_ANSWERED.get(rid) == e.id:
        return None                                   # already talked about this fall
    try:                                              # answered in an earlier run of the server
        from services.saathi_memory_service import has_fact_containing
        if has_fact_containing(rid, f"(fall #{e.id})"):
            _FALL_CHECKIN_ANSWERED[rid] = e.id
            return None
    except Exception:
        pass
    if mark:
        _FALL_CHECKIN_ASKED.add((rid, e.id))
        _FALL_CHECKIN_PENDING[rid] = (e.id, _fmt_when(e.occurred_at), address)
    when = _fmt_when(e.occurred_at)
    t = e.time_to_get_up_sec
    if e.got_up:
        up_hi = (f" Achha hua aap {round(t)} second mein uth gaye." if t is not None and t < 120
                 else " Aap uth to gaye, par kaafi der lagi.")
        up_en = (f" Good that you got up within {round(t)} seconds." if t is not None and t < 120
                 else " You got up, but it took a while.")
    else:
        up_hi, up_en = "", ""
    hi = (f"{address}, maine dekha ki {when} aap gir gaye the.{up_hi} "
          f"Aap theek hain? Kahin chot to nahi lagi? Dard ho to batayiye, main caretaker ko bata deti hoon.")
    en = (f"{address}, I saw that you fell {when}.{up_en} Are you okay? Did you get hurt anywhere? "
          f"If anything hurts, tell me and I will let your caretaker know.")
    got = (f"got up after {round(t)} s" if e.got_up and t is not None else "got up" if e.got_up else "no record of getting up")
    ctx = (f"RECENT FALL: the home camera saw {address} fall {when} (local time) and they {got}. "
           f"Gently ask whether they are hurt (\"Kahin chot to nahi lagi?\"), ask about pain, dizziness or "
           f"bleeding; if they are hurt or in pain, offer to inform the caretaker.")
    return {"hi": hi, "en": en, "context": ctx, "fall_id": e.id}


def sounds_info(db: Session, rid: int, days: int = 7) -> Dict[str, int]:
    from tables.audio_events import AudioEvent
    from sqlalchemy import func
    recip = _recipient(db, rid)
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    try:
        rows = (db.query(AudioEvent.event_type, func.count())
                .filter(AudioEvent.detected_at >= since,
                        _scope_filter(AudioEvent, AudioEvent.care_recipient_id, AudioEvent.caretaker_id, recip))
                .group_by(AudioEvent.event_type).all())
    except Exception:
        db.rollback()
        return {}
    return {(k.value if hasattr(k, "value") else str(k)).lower(): n for k, n in rows}


def sounds_text(db: Session, rid: int, address: str) -> str:
    c = sounds_info(db, rid)
    cough, sneeze = c.get("cough", 0), c.get("sneeze", 0)
    if not cough and not sneeze:
        return f"{address}, pichhle 7 din mein microphone ne khansi ya chheenk ka koi record nahi kiya."
    msg = f"{address}, pichhle 7 din mein {cough} baar khansi aur {sneeze} baar chheenk record hui hai."
    if cough >= 10:
        msg += " Khansi zyada hai; garam paani pijiye, aur agar balgam, bukhar ya saans ki dikkat ho to doctor ko dikhana chahiye."
    return msg


def recent_alerts(db: Session, rid: int, days: int = 7) -> List[Any]:
    from tables.medical_conditions import MedicalAlert
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    try:
        return (db.query(MedicalAlert).filter(MedicalAlert.care_recipient_id == rid, MedicalAlert.created_at >= since)
                .order_by(MedicalAlert.created_at.desc()).limit(5).all())
    except Exception:
        db.rollback()
        return []


# ─── weather ────────────────────────────────────────────────────────────────
_WEATHER_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}


def weather_info(city: Optional[str]) -> Optional[dict]:
    key = os.getenv("WEATHER_API_KEY")
    if not key:
        return None
    city = (city or os.getenv("DEFAULT_CITY") or "Jammu").strip()
    hit = _WEATHER_CACHE.get(city.lower())
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    data = None
    try:
        from weather import WeatherPredictionModel
        raw = WeatherPredictionModel(key, city).fetch_data(city)
        if raw and raw.get("current"):
            cur, day = raw["current"], ((raw.get("forecast") or {}).get("forecastday") or [{}])[0].get("day", {})
            aqi = (cur.get("air_quality") or {}).get("us-epa-index")
            data = {
                "city": (raw.get("location") or {}).get("name", city),
                "temp_c": cur.get("temp_c"), "feels_c": cur.get("feelslike_c"),
                "humidity": cur.get("humidity"), "condition": (cur.get("condition") or {}).get("text", ""),
                "aqi": aqi, "aqi_label": {1: "achhi", 2: "theek (moderate)", 3: "sensitive logon ke liye kharab",
                                          4: "kharab", 5: "bahut kharab", 6: "khatarnak"}.get(aqi, ""),
                "max_c": day.get("maxtemp_c"), "min_c": day.get("mintemp_c"),
                "rain_chance": day.get("daily_chance_of_rain"),
            }
    except Exception as e:
        print(f"[voice_health_brief] weather failed: {e}")
    _WEATHER_CACHE[city.lower()] = (time.time(), data)
    return data


def weather_text(db: Session, rid: int, address: str) -> str:
    recip = _recipient(db, rid)
    w = weather_info(getattr(recip, "city", None))
    if not w:
        return f"{address}, mausam ki taaza jaankari abhi nahi mil paa rahi. Thodi der baad poochiye."
    msg = (f"{address}, abhi {w['city']} mein {_g(w['temp_c'])}°C hai, {w['condition']}. "
           f"Humidity {w['humidity']}%" + (f", aur hawa ki quality {w['aqi_label']} hai" if w.get("aqi_label") else "") + ".")
    if w.get("max_c") is not None:
        msg += f" Aaj din mein {_g(w['min_c'])} se {_g(w['max_c'])}°C tak rahega"
        msg += f", baarish ki sambhavna {w['rain_chance']}%." if w.get("rain_chance") is not None else "."
    conds = " ".join(conditions_list(db, rid)).lower()
    tips = []
    if (w.get("aqi") or 0) >= 3 and any(k in conds for k in ("copd", "asthma", "lung", "pulmonary")):
        tips.append("hawa kharab hai, aur aapko saans ki takleef hai, isliye aaj bahar kam jaiye aur inhaler paas rakhiye")
    if (w.get("temp_c") or 0) >= 35:
        tips.append("garmi zyada hai, paani zyada pijiye aur dopahar mein bahar mat jaiye")
    if (w.get("temp_c") or 99) <= 12:
        tips.append("thand hai, garam kapde pehen kar rakhiye")
    if tips:
        msg += " " + "; ".join(tips).capitalize() + "."
    return msg


# ─── whole-picture answers ──────────────────────────────────────────────────
def health_summary_text(db: Session, rid: int, address: str, caretaker_name: str = "") -> str:
    parts = []
    concerns = vitals_overview(db, rid)
    if concerns:
        parts.append("Aakhri vitals mein: " + ", ".join(concerns))
    ab = abnormal_labs_text(db, rid, 3)
    if ab:
        parts.append("Reports mein: " + ab)
    f = falls_info(db, rid, 7)
    if f.get("count"):
        parts.append(f"is hafte {f['count']} baar girne ka record")
    if not parts:
        msg = f"{address}, aapke aakhri vitals aur reports sab normal range mein hain. Bahut badhiya!"
    else:
        msg = f"{address}, aapki sehat ka haal: " + ". ".join(parts) + "."
    recs = active_recommendations(db, rid, 2)
    if recs:
        msg += " Abhi ke liye salah: " + "; ".join(recs) + "."
    if parts and caretaker_name:
        msg += f" Kya main {caretaker_name} ji ko yeh bata doon?"
    return msg


def profile_text(db: Session, rid: int, address: str) -> str:
    r = _recipient(db, rid)
    if not r:
        return f"{address}, aapki profile abhi nahi mil rahi."
    bits = [f"aapka naam {r.full_name}"]
    if r.age:
        bits.append(f"umar {r.age} saal")
    if getattr(r, "city", None):
        bits.append(f"shehar {r.city}")
    if getattr(r, "blood_group", None):
        bits.append(f"blood group {r.blood_group}")
    msg = f"{address}, " + ", ".join(bits) + "."
    conds = conditions_list(db, rid)
    if conds:
        msg += f" Aapki bimariyan record mein: {', '.join(conds)}."
    alg = allergies_list(db, rid)
    msg += f" Allergy: {', '.join(alg)}." if alg else " Koi allergy darj nahi hai."
    meds = _active_meds(db, rid)
    if meds:
        msg += f" Aap {len(meds)} dawaiyan le rahe hain."
    if getattr(r, "doctor_remarks", None):
        msg += f" Doctor ki salah: {r.doctor_remarks}."
    if r.caretaker:
        msg += f" Aapke caretaker {r.caretaker.full_name or r.caretaker.username} hain."
    return msg


def conditions_text(db: Session, rid: int, address: str) -> str:
    conds = conditions_list(db, rid)
    if not conds:
        return f"{address}, aapke record mein abhi koi bimari darj nahi hai."
    return (f"{address}, aapke record mein yeh bimariyan darj hain: {', '.join(conds)}. "
            f"Koi nayi bimari hai ya nahi, yeh doctor jaanch karke batayenge.")


# ─── where are my medicines? (room camera) ─────────────────────────────────
def medicine_locations_list(db: Session, rid: Optional[int], caretaker_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Medicines the room camera looks for, with where/when each was last seen (newest first)."""
    try:
        from tables.medicine_ref import MedicineRef
        from sqlalchemy import or_
    except Exception:
        return []
    try:
        q = db.query(MedicineRef)
        if rid:
            recip = _recipient(db, rid)
            cid = caretaker_id or (recip.caretaker_id if recip else None)
            q = q.filter(or_(MedicineRef.recipient_id == rid,
                             (MedicineRef.recipient_id.is_(None)) & (MedicineRef.caretaker_id == cid)))
        elif caretaker_id:
            q = q.filter(MedicineRef.caretaker_id == caretaker_id)
        rows = q.all()
    except Exception:
        db.rollback()
        return []
    out = []
    now = datetime.datetime.utcnow()
    for m in rows:
        mins = int((now - m.last_seen_at).total_seconds() // 60) if m.last_seen_at else None
        out.append({"id": m.id, "name": m.name, "location": m.last_location, "location_hi": m.last_location_hi,
                    "last_seen_at": (m.last_seen_at.isoformat() + "Z") if m.last_seen_at else None,
                    "minutes_ago": mins})
    out.sort(key=lambda x: (x["minutes_ago"] is None, x["minutes_ago"] or 0))
    # the same medicine may be saved twice (photo + camera teach): keep its most recent sighting
    merged, seen_names = [], set()
    for m in out:
        key = (m["name"] or "").strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        merged.append(m)
    return merged


def _ago_hi(mins: int) -> str:
    if mins < 2:
        return "abhi"
    if mins < 60:
        return f"{mins} minute pehle"
    h = mins // 60
    return f"{h} ghante pehle" if h < 24 else f"{h // 24} din pehle"


def medicine_location_text(db: Session, rid: Optional[int], address: str, caretaker_id: Optional[int] = None,
                           name_hint: str = "") -> str:
    meds = medicine_locations_list(db, rid, caretaker_id)
    if not meds:
        return (f"{address}, kamre ke camera ko abhi kisi dawai ki photo nahi di gayi hai. Aapke caretaker "
                f"Video Monitoring mein dawai ki photo upload kar sakte hain, phir main bata paungi ki dawai kahan rakhi hai.")
    hint = (name_hint or "").lower()
    named = [m for m in meds if hint and any(w for w in hint.split() if len(w) > 3 and w in m["name"].lower())]
    pick = named or meds
    if not named:
        # a general "where is my medicine?": talk about the ones the camera has actually seen
        seen = [m for m in pick if m["minutes_ago"] is not None]
        pick = seen or pick
    parts = []
    for m in pick[:3]:
        if m["minutes_ago"] is None:
            parts.append(f"{m['name']} abhi tak camera mein nahi dikhi")
        else:
            where = m.get("location_hi") or m.get("location") or "kamre mein"
            if m["minutes_ago"] < 2:
                # seen by the room camera right now: "rifagut 400 abhi takiye par, bed par ... rakhi hai"
                parts.append(f"{m['name']} abhi {where} rakhi hai")
            else:
                parts.append(f"{m['name']} {_ago_hi(m['minutes_ago'])} {where} dikhi thi")
    return f"{address}, " + "; ".join(parts) + "."


_MED_WHERE = ["dawai kahan", "dawai kaha", "dawa kahan", "dawaai kahan", "dawaiyan kahan", "goli kahan", "tablet kahan",
              "medicine kahan", "medicine kaha", "where is my medicine", "where are my medicines", "where is the medicine",
              "meri dawai kidhar", "dawai kidhar", "दवाई कहाँ", "दवा कहाँ", "दवाई कहां", "दवा कहां", "गोली कहाँ"]


def is_medicine_where_question(t: str) -> bool:
    t = (t or "").lower()
    if any(p in t for p in _MED_WHERE):
        return True
    where = ("kahan" in t or "kaha " in t or t.rstrip(" ?.!").endswith("kaha") or "kidhar" in t or "where" in t
             or "कहाँ" in t or "कहां" in t or "कहा " in t)
    # spoken / typed spellings: dawai, davai, dvayi, dwai, dava, medicine, tablet, goli...
    med = re.search(r"\b(d[aw]?[vw]a+y?i?|d[vw]ayi|medicine|medicin|tablet|goli|strip)\b", t) or \
          any(w in t for w in ("दवा", "दवाई", "गोली"))
    return bool(where and med)


# ─── router for the built-in Saathi engine ──────────────────────────────────
_FALL_HISTORY = ["giri thi", "gira tha", "gir gayi thi", "gir gaya tha", "girne ka", "kitni baar gir", "kab giri",
                 "kab gira", "fall hua tha", "fall history", "did i fall", "kya main giri", "kya main gira",
                 "गिरी थी", "गिरा था", "गिरने"]


def is_fall_history_question(t: str) -> bool:
    return any(p in t for p in _FALL_HISTORY)


def _has(t: str, words) -> bool:
    return any(w in t for w in words)


def answer_health_question(text: str, rid: int, db: Session, address: str,
                           caretaker_name: str = "") -> Optional[Dict[str, str]]:
    """Returns {"reply", "intent"} when the question is about the patient's data, else None."""
    t = (text or "").lower().strip()
    if not t:
        return None
    try:
        if is_fall_history_question(t):
            return {"reply": falls_text(db, rid, address), "intent": "chat"}
        if is_medicine_where_question(t):
            return {"reply": medicine_location_text(db, rid, address, name_hint=t), "intent": "chat"}
        if any(w in t for w in ("kahan", "kaha ", "kidhar", "where", "कहाँ", "कहां")):
            # "Metformin kahan rakhi hai?" -> a saved medicine's name + a where-word
            names = [m["name"].lower() for m in medicine_locations_list(db, rid)]
            if any(any(tok in t for tok in n.split() if len(tok) > 3) for n in names):
                return {"reply": medicine_location_text(db, rid, address, name_hint=t), "intent": "chat"}
        if _has(t, ["mausam", "weather", "bahar kaisa", "aqi", "pradushan", "pollution", "hawa kaisi", "barish hogi",
                    "baarish", "garmi kitni", "thand kitni", "मौसम"]):
            return {"reply": weather_text(db, rid, address), "intent": "chat"}
        if _has(t, ["sleep", "neend", "nind", "soyi", "soya", "नींद"]):
            return {"reply": sleep_text(db, rid, address), "intent": "chat"}
        if _has(t, ["agli dawai", "agla dose", "next medicine", "next dose", "kaunsi dawai lun", "kaun si dawai lun",
                    "abhi kaunsi dawai", "abhi kaun si dawai", "dawai ka time", "dawai kab", "kab leni", "kitne baje dawai",
                    "which medicine now", "अगली दवा", "कौन सी दवा"]):
            return {"reply": next_medicine_text(db, rid, address), "intent": "show_medicines"}
        if _has(t, ["tabiyat", "tabiyet", "sehat", "health kaisi", "meri health", "clinical", "analysis", "kya problem",
                    "health summary", "sab theek", "meri condition kaisi", "तबीयत", "सेहत"]):
            return {"reply": health_summary_text(db, rid, address, caretaker_name), "intent": "chat"}
        if _has(t, ["blood pressure", " bp", "bp ", "pressure", "बीपी"]) or t == "bp":
            return {"reply": vital_text(db, rid, address, "bp"), "intent": "chat"}
        if _has(t, ["pulse", "dhadkan", "heart rate", "dil ki dhadkan", "धड़कन"]):
            return {"reply": vital_text(db, rid, address, "pulse"), "intent": "chat"}
        if _has(t, ["oxygen", "spo2", "saturation", "ऑक्सीजन"]):
            return {"reply": vital_text(db, rid, address, "oxygen"), "intent": "chat"}
        if _has(t, ["temperature", "bukhar", "taapman", "बुखार"]):
            return {"reply": vital_text(db, rid, address, "temperature"), "intent": "chat"}
        if _has(t, ["weight", "wajan", "vajan", "bmi", "वजन"]):
            return {"reply": vital_text(db, rid, address, "weight"), "intent": "chat"}
        for topic, words in (("sugar", ["sugar", "shugar", "glucose", "hba1c", "शुगर"]),
                             ("cholesterol", ["cholesterol", "ldl", "hdl", "triglyceride"]),
                             ("hemoglobin", ["hemoglobin", "haemoglobin", "khoon ki kami", "anemia", "anaemia"]),
                             ("kidney", ["kidney", "gurde", "creatinine", "egfr"]),
                             ("liver", ["liver", "jigar", "sgpt", "sgot"]),
                             ("vitamin", ["vitamin"]),
                             ("thyroid", ["thyroid", "tsh"])):
            if _has(t, words):
                return {"reply": lab_topic_text(db, rid, address, topic), "intent": "chat"}
        if _has(t, ["khansi kitni", "kitni baar khansi", "kitni khansi", "cough count", "chheenk kitni", "khansi ka record"]):
            return {"reply": sounds_text(db, rid, address), "intent": "chat"}
        if _has(t, ["report", "jaanch", "test result", "रिपोर्ट"]):
            return {"reply": reports_text(db, rid, address), "intent": "show_reports"}
        if _has(t, ["kya bimari", "meri bimari", "bimariyan", "my conditions", "kaun si bimari", "बीमारी"]):
            return {"reply": conditions_text(db, rid, address), "intent": "chat"}
        if _has(t, ["mere baare mein", "meri profile", "meri jaankari", "about me", "meri details", "मेरे बारे"]):
            return {"reply": profile_text(db, rid, address), "intent": "chat"}
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[voice_health_brief] answer failed: {e}")
    return None


# ─── extra briefing text for Asha (ElevenLabs) ──────────────────────────────
def monitoring_context_text(db: Session, rid: int) -> str:
    """English block appended to Asha's MEDICAL RECORD: sleep, BMI, falls, sounds, alerts, weather."""
    lines = []
    try:
        v = latest_vitals(db, rid)
        if v.get("sleep_score") is not None:
            hist = [x[1] for x in v.get("sleep_history", [])[:7]]
            avg = f", average of last {len(hist)} records {round(sum(hist) / len(hist))}" if len(hist) > 1 else ""
            lines.append(f"SLEEP SCORE: latest {_g(v['sleep_score'])}/100 (85+ very good, 75-84 good, 60-74 fair, below 60 poor){avg}.")
        if v.get("bmi") is not None or v.get("weight") is not None:
            lines.append(f"BODY: weight {_g(v.get('weight')) if v.get('weight') is not None else 'not recorded'} kg, "
                         f"BMI {_g(v.get('bmi')) if v.get('bmi') is not None else 'not recorded'} (normal 18.5-24.9).")
    except Exception:
        db.rollback()
    try:
        f = falls_info(db, rid, 30)
        if f:
            if f.get("count"):
                e = f["last"]
                d = _to_local(e.occurred_at)
                got = (f"got up after {round(e.time_to_get_up_sec, 1)} s" if e.got_up and e.time_to_get_up_sec is not None
                       else "got up" if e.got_up else "no record of getting up")
                lines.append(f"FALLS (home camera): {f['count']} in the last 30 days ({f['week']} in the last 7 days); "
                             f"last on {d.strftime('%d %b %Y at %I:%M %p')} local time, {got}.")
            else:
                lines.append("FALLS (home camera): none in the last 30 days.")
    except Exception:
        db.rollback()
    try:
        s = sounds_info(db, rid, 7)
        if s:
            lines.append(f"SOUNDS (home microphone, last 7 days): {s.get('cough', 0)} coughs, {s.get('sneeze', 0)} sneezes.")
    except Exception:
        db.rollback()
    try:
        al = recent_alerts(db, rid, 7)
        if al:
            lines.append("ALERTS (last 7 days): " + " | ".join(
                f"{_to_local(a.created_at).strftime('%d %b %I:%M %p')}: {(a.message or '')[:140]}" for a in al[:3]))
    except Exception:
        db.rollback()
    try:
        w = weather_info(getattr(_recipient(db, rid), "city", None))
        if w:
            lines.append(f"WEATHER NOW ({w['city']}): {_g(w['temp_c'])} C, {w['condition']}, humidity {w['humidity']}%, "
                         f"air quality index {w.get('aqi')} ({w.get('aqi_label')}); today {_g(w.get('min_c'))}-{_g(w.get('max_c'))} C, "
                         f"rain chance {w.get('rain_chance')}%. Use this when they ask about the weather or going outside.")
    except Exception:
        pass
    try:
        meds = medicine_locations_list(db, rid)
        if meds:
            lines.append("MEDICINE LOCATIONS (room camera; use when they ask where a medicine is): " + "; ".join(
                f"{m['name']}: " + (f"seen {m['minutes_ago']} min ago {m['location']}" if m["minutes_ago"] is not None
                                    else "not seen by the camera yet") for m in meds[:6]))
    except Exception:
        db.rollback()
    return "\n".join(lines)
