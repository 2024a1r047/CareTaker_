"""
Report -> Medication section sync.

Reads the medicines prescribed in an uploaded medical report and adds them to the
patient's medication list (table `medications`), the same table the Medication section,
reminders and the Saathi voice agent use.

How medicines are found
  1. Rule-based reader (always runs, no AI needed) that understands typical Indian
     prescription / discharge-summary formats:
        "Tab. Paracetamol 500 mg, thrice daily after meals for 3 days"
        "Metformin SR 500 mg   1 tablet   After breakfast and dinner   Continue"
        "7:30 AM   Tiotropium inhaler (1 capsule)"        (daily schedule tables)
     Table rows are only read inside medication sections (Prescription / Rx /
     Discharge Medications / Current Medication ...), so drugs that are only mentioned
     in narrative text (e.g. IV drugs given in hospital) are not imported.
  2. AI extraction (`medication_details` from services.report_ingestion), used when a
     valid Gemini key is configured. Both results are merged.

Safety rules
  - Never deletes or edits an existing medicine.
  - Skips a medicine when the patient already has an ACTIVE one with the same name.
  - Medicines marked STOPPED / discontinued in the report are skipped.
  - A course that has already ended (e.g. "for 3 days" in an old report) is added as
    'completed', so it shows as past medication and creates no reminders.
  - auto_order_enabled = False: importing a report never places a pharmacy order.
  - schedule_time is only set when the report gives a time or a time of day
    (morning / after breakfast / night / 8 AM ...), so reminders are never guessed.
  - Each report is imported once (flag in extracted_data["medications_synced_at"]),
    so a medicine the caretaker later removes is not re-added.
"""

import datetime
import re
import threading
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────
_UNIT = r"(?:mg|mcg|µg|ug|g|gm|ml|iu|units?|%)"
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"          # 500 | 0.5 | 60,000
_STRENGTH = re.compile(rf"({_NUM}(?:\s*/\s*{_NUM})?)\s*({_UNIT})\b", re.I)
# Dosage-form prefix ("Tab. Metformin"); must be followed by a word, not "(morning)"
_FORM_PREFIX = re.compile(r"^\s*(?:[-*••●]|\(cid:\d+\)|\d+[.)]?)?\s*(tab|tablet|cap|capsule|syp|syrup|inj|injection|oint|drops?)\.?\s+(?=[A-Za-z])", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_LEADING_JUNK = re.compile(r"^\s*(?:[-*••●#]|\(cid:\d+\)|\d+[.)]?(?=\s))\s*")
_SECTION_START = re.compile(
    r"(prescription|^\s*rx\s*$|discharge medications?|medications? (?:on discharge|advised|prescribed|summary|schedule|plan|list)|"
    r"current medications?|medicines? (?:prescribed|advised)|treatment advised|daily\s+medication\s+schedule|^\s*advice\s*:?\s*$)", re.I)
_SECTION_END = re.compile(
    r"^\s*(tip:|note:|notes:|investigations?|follow[- ]?up|diet|lab(?:oratory)? results?|impression|diagnosis|"
    r"history|vitals?|examination|signature|dr\.\s|consultant|warnings?|avoid|allerg|precautions?|"
    r"emergency|when to|red flags?|instructions?)", re.I)
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_STOPPED = re.compile(r"\b(stopped|stop|discontinued?|withheld|hold)\b", re.I)

_NOT_A_MEDICINE = {
    "medicine", "medicines", "medication", "medications", "dose", "drug", "name", "time", "timing",
    "frequency", "duration", "purpose", "how", "spirometry", "parameter", "test", "result", "units",
    "reference", "range", "iv", "total", "blood", "glucose", "hba1c", "patient", "age", "weight",
    "height", "date", "report", "self", "self-medication", "warm", "saltwater",
    # time-of-day / instruction words (wrapped table cells like "tablet (morning)")
    "morning", "mid-morning", "afternoon", "evening", "night", "bedtime", "breakfast", "lunch", "dinner",
    "daily", "once", "twice", "weekly", "before", "after", "with", "via", "orally", "inhaled", "mix",
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
}
_FORM_WORDS = {"tab", "tablet", "tablets", "cap", "capsule", "capsules", "inhaler", "rotacap", "syrup",
               "syp", "inj", "injection", "sr", "er", "xr", "cr", "mr", "od", "respules", "spray"}


def _norm_key(name: str) -> str:
    words = [w for w in re.findall(r"[a-z+]+", (name or "").lower()) if w not in _FORM_WORDS]
    return "".join(words)


def _clean_name(raw: str) -> str:
    name = _FORM_PREFIX.sub("", raw)
    name = _LEADING_JUNK.sub("", name)
    name = re.sub(r"\s{2,}", " ", name).strip(" ,;:-()")
    return name


def _frequency(text: str) -> (Optional[str], int, bool):
    """Returns (frequency label, doses_per_day, as_needed)."""
    t = text.lower()
    if re.search(r"\b(sos|as needed|as required|prn|when breathless|if needed|when needed)\b", t):
        return "As needed (SOS)", 1, True
    if re.search(r"\b(weekly|once a week|once weekly|every week|per week)\b", t):
        day = re.search(r"\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b", t)
        # Reminders in this app are daily, so weekly medicines get no daily reminder time.
        return "Once weekly" + (f" ({day.group(1).title()})" if day else ""), 1, True
    m = re.search(r"\b([01])\s*-\s*([01])\s*-\s*([01])\b", t)
    if m:
        n = sum(int(x) for x in m.groups())
        return f"{m.group(0)} ({n} times daily)", max(1, n), False
    m = re.search(r"\b(\d{1,2})\s*-?\s*hourly\b", t)
    if m and int(m.group(1)) > 0:
        n = max(1, 24 // int(m.group(1)))
        return f"Every {m.group(1)} hours", n, False
    if re.search(r"\b(four times|qid|qds)\b", t):
        return "Four times daily", 4, False
    if re.search(r"\b(thrice|three times|tds|tid)\b", t):
        return "Thrice daily", 3, False
    if re.search(r"\b(twice|two times|bd|bid)\b", t) or re.search(r"\band\b.*\b(pm|night|dinner|evening)\b", t):
        return "Twice daily", 2, False
    if re.search(r"\b(once|od|daily|every day|qd|hs|at night|at bedtime)\b", t):
        return "Once daily", 1, False
    return None, 1, False


def _schedule_times(text: str, doses_per_day: int) -> Optional[str]:
    """Only from explicit times or time-of-day words in the report; never guessed."""
    t = text.lower()
    times = []
    for h, mnt, ap in _TIME.findall(t):
        hour = int(h) % 12 + (12 if ap == "pm" else 0)
        times.append(f"{hour:02d}:{int(mnt or 0):02d}")
    if not times:
        words = [
            (r"before breakfast|empty stomach", "07:30"),
            (r"morning|after breakfast|with breakfast|breakfast", "08:00"),
            (r"afternoon|lunch", "13:00"),
            (r"evening", "18:00"),
            (r"dinner|night|bedtime|\bhs\b", "21:00"),
        ]
        rest = t
        for pat, hhmm in words:
            if re.search(pat, rest):
                rest = re.sub(pat, " ", rest)  # "before breakfast" must not also count as "breakfast"
                if hhmm not in times:
                    times.append(hhmm)
        # "after meals" with a known count -> standard meal times
        if not times and re.search(r"after meals|with meals|after food|before food|before meals", t):
            times = {1: ["08:00"], 2: ["08:00", "21:00"], 3: ["08:00", "13:00", "21:00"]}.get(doses_per_day, [])
    times = sorted(set(times))
    return ", ".join(times) if times else None


def _duration_days(text: str) -> Optional[int]:
    t = text.lower()
    if re.search(r"\b(ongoing|continue|continued|long[- ]term|lifelong|active)\b", t):
        return None
    m = re.search(r"(?:for|x|×)\s*(\d+)\s*(day|days|week|weeks|month|months)\b", t) or \
        re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        return n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)
    if re.search(r"\bonly\b", t):
        return 1
    return None


def _end_date_from_text(text: str, start: datetime.date) -> Optional[datetime.date]:
    """'26 Aug only' -> that date (course ends on a named day)."""
    m = re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b[^.]*\bonly\b", text.lower())
    if not m:
        return None
    try:
        d = datetime.date(start.year, _MONTHS[m.group(2)], int(m.group(1)))
        return d if d >= start - datetime.timedelta(days=180) else d.replace(year=d.year + 1)
    except ValueError:
        return None


def _parse_line(line: str, in_section: bool, schedule_section: bool) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    if len(raw) < 4:
        return None
    has_prefix = bool(_FORM_PREFIX.match(raw))
    strength = _STRENGTH.search(raw)

    # Daily schedule rows: "7:30 AM  Tiotropium inhaler (1 capsule)"
    if schedule_section:
        tm = _TIME.match(raw.strip())
        if tm:
            rest = raw[tm.end():].strip()
            name = re.split(r"\(|\d", rest, maxsplit=1)[0]
            name = _clean_name(name)
            if name and name.split()[0].lower() not in _NOT_A_MEDICINE:
                return {"name": name, "dosage": f"{strength.group(1)} {strength.group(2)}" if strength else "",
                        "timing_text": tm.group(0), "line": raw, "from_schedule": True}
        return None

    if not (has_prefix or (in_section and strength)):
        return None
    if not strength and not has_prefix:
        return None

    before = raw[:strength.start()] if strength else re.split(r",|\s{2,}", _FORM_PREFIX.sub("", raw), maxsplit=1)[0]
    name = _clean_name(before)
    if not name or not re.match(r"[A-Za-z]", name):
        return None
    if name.split()[0].lower() in _NOT_A_MEDICINE or len(name.split()) > 6:
        return None
    if re.search(r"\b(iv|intravenous)\b", raw, re.I) and not has_prefix:
        return None  # hospital-only drug in a table row
    dosage = f"{strength.group(1)} {strength.group(2)}".replace(" / ", "/") if strength else ""
    return {"name": name, "dosage": dosage, "timing_text": raw[strength.end():] if strength else raw, "line": raw}


def extract_medicines_from_text(text: str) -> List[Dict[str, Any]]:
    """Rule-based extraction. Returns merged medicine dicts (one per medicine)."""
    if not text:
        return []
    found: Dict[str, Dict[str, Any]] = {}
    schedule_rows: List = []
    in_section = False
    schedule_section = False
    blank_run = 0
    for line in text.splitlines():
        if not line.strip():
            blank_run += 1
            if blank_run >= 2:
                in_section = schedule_section = False
            continue
        blank_run = 0
        if _SECTION_START.search(line):
            in_section = True
            schedule_section = bool(re.search(r"schedule", line, re.I))
            continue
        if in_section and _SECTION_END.search(line):
            in_section = schedule_section = False
        item = _parse_line(line, in_section, schedule_section)
        if not item:
            continue
        key = _norm_key(item["name"])
        if not key:
            continue
        if item.get("from_schedule"):
            # Schedule-table rows only add times to medicines found in the medicine list,
            # so diet/activity schedules ("11:00 AM fruit") never become medicines.
            schedule_rows.append((key, item["timing_text"]))
            continue
        cur = found.setdefault(key, {"name": item["name"], "dosage": "", "texts": [], "stopped": False})
        if item["dosage"] and not cur["dosage"]:
            cur["dosage"] = item["dosage"]
        if len(item["name"]) > len(cur["name"]) and len(item["name"].split()) <= 4:
            cur["name"] = item["name"]
        cur["texts"].append(item["timing_text"])
        if _STOPPED.search(item["line"]):
            cur["stopped"] = True

    # Exact times from the schedule table (e.g. "7:30 AM") replace time-of-day words
    sched_times: Dict[str, List[str]] = {}
    for key, ttext in schedule_rows:
        match = next((k for k in found if k == key or k.startswith(key) or key.startswith(k)), None)
        if match:
            sched_times.setdefault(match, []).append(ttext)

    meds = []
    for key, m in found.items():
        joined = " ".join(m["texts"])
        freq, per_day, as_needed = _frequency(joined)
        times = None
        if not as_needed:
            times = _schedule_times(" ".join(sched_times[key]), per_day) if key in sched_times else None
            times = times or _schedule_times(joined, per_day)
            if not freq and times:  # "Morning 8 AM" -> once daily; "8 AM and 8 PM" -> twice daily
                n = len(times.split(","))
                freq, per_day = {1: "Once daily", 2: "Twice daily", 3: "Thrice daily"}.get(n, f"{n} times daily"), n
        meds.append({
            "name": m["name"],
            "dosage": m["dosage"],
            "frequency": freq or "",
            "doses_per_day": per_day,
            "schedule_time": times,
            "duration_days": _duration_days(joined),
            "end_text": joined,
            "stopped": m["stopped"],
        })
    return meds


def _from_ai(details: Any) -> List[Dict[str, Any]]:
    """Normalises AI `medication_details` (list of dicts) to the same shape."""
    out = []
    for d in details or []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        text = " ".join(str(d.get(k) or "") for k in ("frequency", "timing", "duration", "status"))
        freq, per_day, as_needed = _frequency(text)
        out.append({
            "name": str(d["name"]).strip(),
            "dosage": str(d.get("dosage") or "").strip(),
            "frequency": str(d.get("frequency") or freq or "").strip(),
            "doses_per_day": per_day,
            "schedule_time": None if as_needed else _schedule_times(text, per_day),
            "duration_days": _duration_days(text),
            "stopped": bool(_STOPPED.search(str(d.get("status") or ""))),
        })
    return out


def sync_report_medications(report, db: Session, text: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Imports medicines from one MedicalReport into the medications table.
    Commits its own changes. Never raises (returns an error dict instead).
    """
    from tables.medications import Medication, MedicationStatus

    summary = {"report_id": getattr(report, "id", None), "added": [], "completed": [], "skipped": []}
    try:
        extracted = dict(report.extracted_data) if isinstance(report.extracted_data, dict) else {}
        if extracted.get("medications_synced_at") and not force:
            summary["skipped"].append("already imported")
            return summary

        if text is None:
            try:
                from utils.summarizer import extract_text_from_bytes
                text = extract_text_from_bytes(report.data, report.filename) if report.data else ""
            except Exception as e:
                print(f"[report_meds] could not read report {report.id}: {e}")
                text = ""

        meds: Dict[str, Dict[str, Any]] = {}
        for m in extract_medicines_from_text(text) + _from_ai(extracted.get("medication_details")):
            key = _norm_key(m["name"])
            if not key:
                continue
            if key in meds:  # merge: fill missing fields
                for f in ("dosage", "frequency", "schedule_time", "duration_days"):
                    if not meds[key].get(f) and m.get(f):
                        meds[key][f] = m[f]
                meds[key]["stopped"] = meds[key]["stopped"] or m["stopped"]
            else:
                meds[key] = dict(m)

        start = report.report_date or (report.uploaded_at.date() if report.uploaded_at else datetime.date.today())
        today = datetime.date.today()
        all_existing = db.query(Medication).filter(Medication.care_recipient_id == report.care_recipient_id).all()
        existing_active = {_norm_key(x.medicine_name) for x in all_existing if x.status == MedicationStatus.active}
        # Same medicine already imported with the same start date (e.g. the same report uploaded twice)
        existing_same_start = {(_norm_key(x.medicine_name), x.start_date) for x in all_existing}

        for key, m in meds.items():
            if m["stopped"]:
                summary["skipped"].append(f"{m['name']} (stopped in report)")
                continue
            if key in existing_active:
                summary["skipped"].append(f"{m['name']} (already active)")
                continue
            if (key, start) in existing_same_start:
                summary["skipped"].append(f"{m['name']} (already imported)")
                continue
            end = _end_date_from_text(m.get("end_text") or "", start)
            if not end and m.get("duration_days"):
                end = start + datetime.timedelta(days=m["duration_days"])
            finished = bool(end and end < today)
            # A course that ended before the report date (e.g. "26 Aug only" in a summary
            # uploaded later) must not start after it ends.
            med_start = min(start, end) if end else start
            db.add(Medication(
                care_recipient_id=report.care_recipient_id,
                medicine_name=m["name"][:120],
                dosage=m.get("dosage") or None,
                frequency=m.get("frequency") or None,
                schedule_time=m.get("schedule_time"),
                start_date=med_start,
                end_date=end,
                status=MedicationStatus.completed if finished else MedicationStatus.active,
                current_stock=0,
                doses_per_day=m.get("doses_per_day") or 1,
                auto_order_enabled=False,
            ))
            (summary["completed"] if finished else summary["added"]).append(m["name"])
            existing_same_start.add((key, start))
            if not finished:
                existing_active.add(key)

        extracted["medication_details"] = extracted.get("medication_details") or [dict(m) for m in meds.values()]
        extracted["medications_synced_at"] = datetime.datetime.utcnow().isoformat()
        report.extracted_data = extracted  # reassign so SQLAlchemy saves the JSON change
        db.commit()
        if summary["added"] or summary["completed"]:
            print(f"[report_meds] report {report.id} (recipient {report.care_recipient_id}): "
                  f"added {summary['added']}, past courses {summary['completed']}, skipped {summary['skipped']}")
    except Exception as e:
        db.rollback()
        print(f"[report_meds] report {getattr(report, 'id', '?')} failed: {e}")
        summary["error"] = str(e)
    return summary


def sync_report_medications_by_id(report_id: int, text: Optional[str] = None) -> Dict[str, Any]:
    """Opens its own DB session; safe to call from pipelines and background threads."""
    from config import SessionLocal
    from tables.medical_reports import MedicalReport
    db = SessionLocal()
    try:
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report:
            return {"report_id": report_id, "error": "not found"}
        return sync_report_medications(report, db, text=text)
    finally:
        db.close()


def backfill_all_reports() -> Dict[str, int]:
    """Imports medicines from every processed report that has not been imported yet."""
    from config import SessionLocal
    from tables.medical_reports import MedicalReport, ReportProcessingStatus
    db = SessionLocal()
    totals = {"reports": 0, "added": 0, "completed": 0}
    try:
        reports = (
            db.query(MedicalReport)
            .filter(MedicalReport.processing_status == ReportProcessingStatus.completed)
            .order_by(MedicalReport.report_date.asc().nullsfirst(), MedicalReport.id.asc())
            .all()
        )
        for r in reports:
            if isinstance(r.extracted_data, dict) and r.extracted_data.get("medications_synced_at"):
                continue
            res = sync_report_medications(r, db)
            totals["reports"] += 1
            totals["added"] += len(res.get("added", []))
            totals["completed"] += len(res.get("completed", []))
    except Exception as e:
        print(f"[report_meds] backfill failed: {e}")
    finally:
        db.close()
    print(f"[report_meds] backfill done: {totals}")
    return totals


def start_backfill_in_background() -> None:
    threading.Thread(target=backfill_all_reports, name="report-meds-backfill", daemon=True).start()
