"""
Medical context for the ElevenLabs "Asha" voice agent.

Builds a spoken-friendly briefing from what is already stored for a care recipient:
  - uploaded medical reports and their AI analysis (medical_reports)
  - latest lab results, classified LOW / NORMAL / HIGH against standard adult ranges (lab_values)
  - latest recorded vitals (vital_signs)
  - open care recommendations (medical_recommendations)
  - general guidance: normal ranges for BP, heart rate, SpO2, temperature, blood sugar,
    and what to do when a reading the patient shares is high or low.

Read-only: it never writes to the database. Every section is wrapped so a failure in one
table only drops that section and never blocks the voice call.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

# ─────────────────────────────────────────────────────────────────────────────
# Standard adult reference ranges (general guidance, not lab-specific).
# unit = unit the value must be in for the comparison; (low, high), None = open-ended.
# ─────────────────────────────────────────────────────────────────────────────
LAB_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "HbA1c":             {"unit": "%",      "range": (4.0, 5.6),   "note": "5.7-6.4 prediabetes, 6.5 or more diabetes; diabetic target usually below 7"},
    "Fasting Glucose":   {"unit": "mg/dL",  "range": (70, 99),     "note": "100-125 prediabetes, 126 or more diabetes"},
    "Random Glucose":    {"unit": "mg/dL",  "range": (70, 139),    "note": "200 or more suggests diabetes"},
    "Postprandial Glucose": {"unit": "mg/dL", "range": (70, 139),  "note": "2 hours after a meal"},
    "Creatinine":        {"unit": "mg/dL",  "range": (0.7, 1.3),   "note": "higher means the kidneys are filtering less"},
    "eGFR":              {"unit": "mL/min", "range": (90, None),   "note": "60-89 mildly reduced, 30-59 moderate, 15-29 severe, below 15 kidney failure"},
    "BUN":               {"unit": "mg/dL",  "range": (7, 20),      "note": ""},
    "Urea":              {"unit": "mg/dL",  "range": (15, 45),     "note": ""},
    "Uric Acid":         {"unit": "mg/dL",  "range": (3.4, 7.0),   "note": ""},
    "Potassium":         {"unit": "mEq/L",  "range": (3.5, 5.0),   "note": "above 5.5 can affect the heart rhythm"},
    "Sodium":            {"unit": "mEq/L",  "range": (135, 145),   "note": ""},
    "Calcium":           {"unit": "mg/dL",  "range": (8.5, 10.5),  "note": ""},
    "ALT":               {"unit": "U/L",    "range": (7, 56),      "note": "liver enzyme"},
    "AST":               {"unit": "U/L",    "range": (10, 40),     "note": "liver enzyme"},
    "Total Cholesterol": {"unit": "mg/dL",  "range": (None, 199),  "note": "200-239 borderline, 240 or more high"},
    "LDL":               {"unit": "mg/dL",  "range": (None, 99),   "note": "'bad' cholesterol; 160 or more high"},
    "HDL":               {"unit": "mg/dL",  "range": (40, None),   "note": "'good' cholesterol; higher is better"},
    "Triglycerides":     {"unit": "mg/dL",  "range": (None, 149),  "note": "200 or more high"},
    "Hemoglobin":        {"unit": "g/dL",   "range": None,         "note": "low hemoglobin means anemia"},  # gender-specific below
    "WBC":               {"unit": "",       "range": (4.0, 11.0),  "note": "thousand per microlitre; high can mean infection"},
    "Platelets":         {"unit": "",       "range": (150, 450),   "note": "thousand per microlitre"},
    "CRP":               {"unit": "mg/L",   "range": (None, 5.0),  "note": "inflammation marker"},
    "TSH":               {"unit": "",       "range": (0.4, 4.0),   "note": "thyroid"},
    "Vitamin D":         {"unit": "ng/mL",  "range": (30, 100),    "note": "below 20 deficient"},
    "Vitamin B12":       {"unit": "pg/mL",  "range": (200, 900),   "note": ""},
    "Systolic BP":       {"unit": "mmHg",   "range": (90, 119),    "note": ""},
    "Diastolic BP":      {"unit": "mmHg",   "range": (60, 79),     "note": ""},
    "Heart Rate":        {"unit": "bpm",    "range": (60, 100),    "note": ""},
    "Oxygen Saturation": {"unit": "%",      "range": (95, 100),    "note": ""},
}

HEMOGLOBIN_RANGE = {"male": (13.5, 17.5), "female": (12.0, 15.5)}

# Spoken guidance the agent uses when the patient shares a reading or asks about ranges.
VITALS_GUIDANCE = """NORMAL RANGES AND WHAT TO DO (general adult guidance; the patient's own doctor's targets come first):
BLOOD PRESSURE (top/bottom number, mmHg):
- Normal: below 120 and below 80.
- Elevated: 120-129 and below 80. Advice: less salt, regular walks, recheck in a few days.
- High stage 1: 130-139 or 80-89. Advice: rest, recheck, keep a log, take prescribed BP medicine on time, tell the caretaker/doctor if it stays this high.
- High stage 2: 140 or more, or 90 or more. Advice: sit quietly 5 minutes and measure again; if still 140/90 or more, inform the caretaker and doctor soon. Do not skip or double medicines on your own.
- Dangerous: 180 or more, or 120 or more. Recheck after 5 minutes; if still this high, or with chest pain, breathlessness, severe headache, weakness, confusion or vision change, this is an EMERGENCY: call the caretaker and emergency services (112/108) immediately.
- Low: below 90 or below 60. Advice: sit or lie down, drink water, stand up slowly. Urgent help if they faint, feel confused, have chest pain or a very fast heartbeat.
HEART RATE (pulse at rest): normal 60-100 bpm. Above 100 at rest is fast, below 60 is slow (can be normal for fit people or with some BP medicines). Advice: rest and recheck; urgent help if with chest pain, dizziness, fainting or breathlessness.
OXYGEN (SpO2): normal 95-100%. 92-94% inform the doctor; below 92% seek medical help promptly; below 90% is an emergency.
TEMPERATURE: normal 97-99 F (36.1-37.2 C). 100.4 F (38 C) or more is fever: rest, fluids, inform the doctor; urgent if 103 F or more, or with confusion or breathlessness.
BLOOD SUGAR: fasting 70-99 normal (diabetic target 80-130); 2 hours after a meal below 140 normal (diabetic target below 180).
- Low sugar, below 70: take 15 g fast sugar now (half glass juice, 3-4 glucose tablets or 1 spoon sugar/honey), recheck in 15 minutes, then eat a snack. If confused or not able to swallow, it is an EMERGENCY.
- High sugar, above 250, or above 300 at any time: drink water, check medicines were taken, inform the doctor; urgent if vomiting, very drowsy or breathing fast.
HOW TO GIVE ADVICE: say whether the reading is normal, high or low and the normal range, relate it to their reports and medicines, give simple safe steps (rest, recheck, water, low salt, take prescribed medicine on time), and say clearly when to call the doctor or emergency services. Never tell them to start, stop or change the dose of a medicine; that is for their doctor."""

# Junk "metrics" that report parsers sometimes extract from headers.
_JUNK_METRIC = re.compile(r"(report|block|sector|age|reported|collected|lab no|page|patient)", re.I)


def _range_for(metric: str, unit: str, gender: str) -> Tuple[Optional[Tuple], str]:
    ref = LAB_REFERENCE_RANGES.get(metric)
    if not ref:
        return None, ""
    rng = ref["range"]
    if metric == "Hemoglobin":
        rng = HEMOGLOBIN_RANGE["female" if gender.startswith("f") else "male"]
    expected = ref["unit"].lower()
    u = unit.lower()
    if metric in ("Potassium", "Sodium") and u.startswith("mmol"):
        u = "meq/l"  # mmol/L == mEq/L for monovalent ions
    if expected and u and not u.startswith(expected.split("/")[0]):
        return None, ref["note"]  # different unit: don't classify a value we can't compare
    return rng, ref["note"]


def _fmt_range(rng: Tuple) -> str:
    lo, hi = rng
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}"
    if lo is not None:
        return f"{lo:g} or more"
    return f"below {hi:g}" if hi is not None else ""


def _classify(value: float, rng: Optional[Tuple]) -> str:
    if not rng:
        return ""
    lo, hi = rng
    if lo is not None and value < lo:
        return "LOW"
    if hi is not None and value > hi:
        return "HIGH"
    return "NORMAL"


def _clean(text: str, limit: int) -> str:
    t = re.sub(r"START_CLINICAL_SUMMARY|END_CLINICAL_SUMMARY", " ", text or "")
    t = re.sub(r"[^\x20-\x7E\n]", " ", t)          # drop broken bullet glyphs
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:limit].rsplit(" ", 1)[0] + "...") if len(t) > limit else t


def _reports_section(recipient_id: int, db: Session) -> str:
    from tables.medical_reports import MedicalReport, ReportProcessingStatus
    reps = (
        db.query(MedicalReport)
        .filter(MedicalReport.care_recipient_id == recipient_id,
                MedicalReport.processing_status == ReportProcessingStatus.completed)
        .order_by(MedicalReport.report_date.desc().nullslast(), MedicalReport.uploaded_at.desc())
        .limit(12).all()
    )
    lines, seen = [], set()
    for r in reps:
        ex = r.extracted_data if isinstance(r.extracted_data, dict) else {}
        key = (r.filename, str(r.report_date))
        if key in seen:
            continue
        seen.add(key)
        date = r.report_date.isoformat() if r.report_date else (r.uploaded_at.date().isoformat() if r.uploaded_at else "date unknown")
        parts = [f"{date} - {r.filename}"]
        diagnoses = [d for d in (ex.get("diagnoses") or []) if isinstance(d, str)]
        if diagnoses:
            parts.append("findings: " + ", ".join(diagnoses[:6]))
        notes = ex.get("doctor_notes")
        if isinstance(notes, str) and notes.strip():
            parts.append("doctor notes: " + _clean(notes, 220))
        if not diagnoses and r.analysis_summary:
            parts.append("summary: " + _clean(r.analysis_summary, 220))
        lines.append("; ".join(parts))
        if len(lines) >= 6:
            break
    return "\n".join(f"- {l}" for l in lines)


def _labs_section(recipient_id: int, db: Session, gender: str) -> Tuple[str, str]:
    from tables.medical_conditions import LabValue
    rows = (
        db.query(LabValue)
        .filter(LabValue.care_recipient_id == recipient_id, LabValue.is_mapped == True)  # noqa: E712
        .order_by(LabValue.recorded_date.desc(), LabValue.id.desc())
        .all()
    )
    history: Dict[str, List[Any]] = {}
    for lv in rows:
        if not lv.metric_name or _JUNK_METRIC.search(lv.metric_name):
            continue
        history.setdefault(lv.metric_name, []).append(lv)

    lines, abnormal = [], []
    for metric in sorted(history):
        latest = history[metric][0]
        value = latest.normalized_value if latest.normalized_value is not None else latest.metric_value
        value = round(float(value), 2)
        unit =(latest.normalized_unit or latest.unit or "").replace("²", "2").replace("³", "3")
        rng, note = _range_for(metric, unit, gender)
        if not rng and latest.reference_range_low is not None and latest.reference_range_high is not None:
            rng = (latest.reference_range_low, latest.reference_range_high)
        status = _classify(value, rng) or ("ABNORMAL" if latest.is_abnormal else "")
        date = latest.recorded_date.isoformat() if latest.recorded_date else ""
        text = f"{metric} {value:g} {unit}".strip()
        if rng:
            text += f" (normal {_fmt_range(rng)})"
        if status:
            text += f" {status}"
        if len(history[metric]) > 1:
            prev = history[metric][1]
            pv = prev.normalized_value if prev.normalized_value is not None else prev.metric_value
            text += f", previous {round(float(pv), 2):g}"
        if date:
            text += f", on {date}"
        if note and status in ("HIGH", "LOW", "ABNORMAL"):
            text += f" [{note}]"
        lines.append(text)
        if status in ("HIGH", "LOW", "ABNORMAL"):
            abnormal.append(f"{metric} {status.lower()} ({value:g} {unit})".strip())
    return "\n".join(f"- {l}" for l in lines), ", ".join(abnormal)


def _vitals_section(recipient_id: int, db: Session) -> str:
    from tables.vital_signs import VitalSign
    v = (
        db.query(VitalSign)
        .filter(VitalSign.care_recipient_id == recipient_id)
        .order_by(VitalSign.recorded_at.desc())
        .first()
    )
    if not v:
        return ""
    items = []
    if v.systolic_bp and v.diastolic_bp:
        s, d = v.systolic_bp, v.diastolic_bp
        cat = ("dangerously high" if s >= 180 or d >= 120 else "high (stage 2)" if s >= 140 or d >= 90
               else "high (stage 1)" if s >= 130 or d >= 80 else "elevated" if s >= 120
               else "low" if s < 90 or d < 60 else "normal")
        items.append(f"BP {s}/{d} mmHg ({cat})")
    if v.heart_rate:
        items.append(f"pulse {v.heart_rate} bpm ({_classify(v.heart_rate, (60, 100)).lower()})")
    if v.oxygen_saturation:
        items.append(f"SpO2 {v.oxygen_saturation}% ({_classify(v.oxygen_saturation, (95, 100)).lower()})")
    if v.temperature:
        items.append(f"temperature {v.temperature:g} F")
    if v.weight:
        items.append(f"weight {v.weight:g}")
    when = v.recorded_at.strftime("%Y-%m-%d %H:%M") if v.recorded_at else ""
    return ", ".join(items) + (f" (recorded {when})" if when and items else "")


def _recommendations_section(recipient_id: int, db: Session) -> str:
    from tables.medical_recommendations import MedicalRecommendation
    recs = (
        db.query(MedicalRecommendation)
        .filter(MedicalRecommendation.care_recipient_id == recipient_id,
                MedicalRecommendation.resolved_at.is_(None),
                (MedicalRecommendation.archived == False) | (MedicalRecommendation.archived.is_(None)))  # noqa: E712
        .order_by(MedicalRecommendation.created_at.desc())
        .limit(15).all()
    )
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "suggestion": 4}
    recs.sort(key=lambda r: order.get((r.severity or "").lower(), 5))
    lines, seen_words = [], []
    for r in recs:
        msg = _clean(r.do_this_now or r.message or "", 200)
        if not msg:
            continue
        # Skip near-duplicates (same advice re-generated with slightly different wording)
        words = set(re.findall(r"[a-z0-9.]+", msg.lower()))
        if any(len(words & w) / max(1, len(words | w)) > 0.6 for w in seen_words):
            continue
        seen_words.append(words)
        title = r.title or r.metric
        lines.append(f"{title} ({r.severity}): {msg}")
        if len(lines) >= 5:
            break
    return "\n".join(f"- {l}" for l in lines)


def build_medical_context(recipient_id: int, db: Session) -> Dict[str, str]:
    """Returns string sections (empty string when nothing is on record)."""
    from tables.users import CareRecipient

    gender = ""
    try:
        recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if recip and recip.gender:
            gender = (recip.gender.value if hasattr(recip.gender, "value") else str(recip.gender)).lower()
    except Exception:
        db.rollback()

    out = {"medical_reports": "", "lab_results": "", "abnormal_findings": "",
           "latest_vitals": "", "care_recommendations": "", "vitals_guidance": VITALS_GUIDANCE}
    sections = [
        ("medical_reports", lambda: _reports_section(recipient_id, db)),
        ("lab_results", lambda: _labs_section(recipient_id, db, gender)),
        ("latest_vitals", lambda: _vitals_section(recipient_id, db)),
        ("care_recommendations", lambda: _recommendations_section(recipient_id, db)),
    ]
    for key, fn in sections:
        try:
            val = fn()
            if key == "lab_results":
                out["lab_results"], out["abnormal_findings"] = val
            else:
                out[key] = val
        except Exception as e:
            db.rollback()
            print(f"[Agent medical context] {key} skipped for recipient {recipient_id}: {e}")
    return out


def format_medical_context(ctx: Dict[str, str]) -> str:
    """One text block for the agent (sent as an ElevenLabs contextual_update)."""
    na = "none on record"
    return (
        "MEDICAL RECORD from the CareTaker platform (uploaded reports and their analysis, lab results, vitals, "
        "care plan). You DO have access to this; use it to answer any question about their reports, tests, "
        "BP, heart rate, sugar, kidney, liver, cholesterol or other health values, explaining in simple words "
        "what each value means, whether it is normal, high or low, and what they can do. Do not read it all out "
        "unprompted.\n"
        f"ABNORMAL FINDINGS (latest): {ctx.get('abnormal_findings') or na}\n"
        f"LATEST VITALS: {ctx.get('latest_vitals') or na}\n"
        f"LAB RESULTS (latest value per test):\n{ctx.get('lab_results') or na}\n"
        f"UPLOADED REPORTS (newest first):\n{ctx.get('medical_reports') or na}\n"
        f"CARE PLAN RECOMMENDATIONS:\n{ctx.get('care_recommendations') or na}\n\n"
        f"{ctx.get('vitals_guidance') or VITALS_GUIDANCE}"
    )
