"""
Weekly Report Generator
-----------------------
Aggregates all 9 data domains for a care recipient over a 7-day window,
runs a 2-pass Gemini triage+narrative, computes a wellness score,
and persists the result in the WeeklyReport table.
"""

import datetime
import os
import re
import json
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from utils.gemini_client import call_gemini


# ── Helpers ──────────────────────────────────────────────────────────────────

def _week_bounds(reference: datetime.date = None):
    """Return (monday, sunday) of the week containing `reference`."""
    ref = reference or datetime.date.today()
    monday = ref - datetime.timedelta(days=ref.weekday())
    sunday = monday + datetime.timedelta(days=6)
    return monday, sunday


def _prev_week_bounds(week_start: datetime.date):
    prev_end = week_start - datetime.timedelta(days=1)
    prev_start = prev_end - datetime.timedelta(days=6)
    return prev_start, prev_end


# ── 1. Data Aggregation (pure SQL, zero AI calls) ───────────────────────────

def aggregate_weekly_data(recipient_id: int, db: Session,
                          week_start: datetime.date, week_end: datetime.date) -> Dict[str, Any]:
    start_dt = datetime.datetime.combine(week_start, datetime.time.min)
    end_dt = datetime.datetime.combine(week_end, datetime.time(23, 59, 59))

    sections = {
        "vitals_summary": _agg_vitals(recipient_id, db, start_dt, end_dt),
        "lab_summary": _agg_labs(recipient_id, db, week_start, week_end),
        "medication_adherence": _agg_meds(recipient_id, db, start_dt, end_dt),
        "condition_updates": _agg_conditions(recipient_id, db, start_dt, end_dt),
        "respiratory_activity": _agg_audio(recipient_id, db, start_dt, end_dt),
        "physical_activity": _agg_video(recipient_id, db, start_dt, end_dt),
        "environment": _agg_environment(recipient_id, db, start_dt, end_dt),
        "mood_wellbeing": _agg_mood(recipient_id, db, start_dt, end_dt),
        "alerts_recap": _agg_alerts(recipient_id, db, start_dt, end_dt),
    }
    # present state (latest vitals, BP / sugar / labs, medicines) - shown even in a week without new readings
    sections["current_health"] = _current_snapshot(recipient_id, db)
    # real values for rows that would otherwise show 0 / N/A (camera medicine intake, latest lab
    # report, today's weather) - never invented numbers
    sections["report_extras"] = _report_extras(recipient_id, db, start_dt, end_dt, sections["current_health"])
    return sections


def _doses_per_day(m: Dict) -> int:
    sched = [t for t in re.split(r"[,;]", m.get("schedule") or "") if t.strip()]
    if sched:
        return len(sched)
    f = (m.get("frequency") or "").lower()
    if "sos" in f or "needed" in f:
        return 0
    for word, n in (("four", 4), ("thrice", 3), ("three", 3), ("twice", 2), ("two", 2), ("once", 1)):
        if word in f:
            return n
    return 1


def _report_extras(rid, db, start, end, current: Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    meds = (current or {}).get("medicines") or []
    out["medicines_prescribed"] = len(meds)
    out["doses_per_day"] = sum(_doses_per_day(m) for m in meds)
    out["sos_medicines"] = sum(1 for m in meds if _doses_per_day(m) == 0)

    # medicine intake / sightings seen by the room camera this week
    try:
        from tables.table_watch import TableWatchEvent
        evs = db.query(TableWatchEvent).filter(
            TableWatchEvent.recipient_id == rid, TableWatchEvent.item == "medicine",
            TableWatchEvent.occurred_at >= start, TableWatchEvent.occurred_at <= end).all()
        out["camera_medicine_taken"] = sum(1 for e in evs if e.event == "taken")
        out["camera_missed_alerts"] = sum(1 for e in evs if e.event == "untouched_alert")
    except Exception:
        db.rollback()
    try:
        from tables.medicine_ref import MedicineSighting
        out["medicine_sightings"] = db.query(MedicineSighting).filter(
            MedicineSighting.recipient_id == rid,
            MedicineSighting.first_seen_at >= start, MedicineSighting.first_seen_at <= end).count()
    except Exception:
        db.rollback()

    # latest lab report summary (from the key labs already collected)
    labs = (current or {}).get("key_labs") or []
    if labs:
        dates = [l.get("date") for l in labs if l.get("date")]
        out["latest_lab_date"] = dates[0] if dates else None
        out["latest_lab_tests"] = len(labs)
        out["latest_lab_out_of_range"] = sum(1 for l in labs if l.get("status") in ("High", "Low", "Abnormal"))

    # today's weather where the patient lives (the room has no sensor readings)
    try:
        from weather import WeatherPredictionModel
        from tables.users import CareRecipient
        recip = db.query(CareRecipient).filter(CareRecipient.id == rid).first()
        wm = WeatherPredictionModel(os.getenv("WEATHER_API_KEY"), os.getenv("DEFAULT_CITY", "Jammu"))
        data = wm.fetch_data(city=(recip.city if recip and recip.city else None)) or wm.fetch_data()
        cur = (data or {}).get("current") or {}
        if cur:
            aqi = (cur.get("air_quality") or {}).get("us-epa-index")
            out["weather"] = {
                "place": (data.get("location") or {}).get("name"),
                "temp_c": cur.get("temp_c"), "humidity": cur.get("humidity"),
                "condition": (cur.get("condition") or {}).get("text"),
                "aqi": {1: "Good", 2: "Moderate", 3: "Unhealthy for sensitive people", 4: "Unhealthy",
                        5: "Very unhealthy", 6: "Hazardous"}.get(aqi),
            }
    except Exception as e:
        print(f"[weekly_report] weather skipped: {e}")
    return out


def _safe_avg(values):
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else None


def _agg_vitals(rid, db, start, end):
    from tables.vital_signs import VitalSign
    rows = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == rid,
        VitalSign.recorded_at >= start, VitalSign.recorded_at <= end
    ).all()
    if not rows:
        return {"readings": 0, "avg_heart_rate": None, "avg_systolic_bp": None,
                "avg_spo2": None, "avg_sleep_score": None, "avg_temperature": None, "alerts": []}
    alerts = []
    hr_vals = [r.heart_rate for r in rows if r.heart_rate]
    bp_vals = [r.systolic_bp for r in rows if r.systolic_bp]
    spo2_vals = [r.oxygen_saturation for r in rows if r.oxygen_saturation]
    avg_bp = _safe_avg(bp_vals)
    if avg_bp and avg_bp > 130:
        alerts.append(f"Avg systolic BP was {avg_bp} mmHg (above 130)")
    avg_spo2 = _safe_avg(spo2_vals)
    if avg_spo2 and avg_spo2 < 95:
        alerts.append(f"Avg SpO2 was {avg_spo2}% (below 95)")
    return {
        "readings": len(rows),
        "avg_heart_rate": _safe_avg(hr_vals),
        "avg_systolic_bp": avg_bp,
        "avg_spo2": avg_spo2,
        "avg_sleep_score": _safe_avg([r.sleep_score for r in rows if r.sleep_score]),
        "avg_temperature": _safe_avg([r.temperature for r in rows if r.temperature]),
        "alerts": alerts,
    }


def _agg_labs(rid, db, ws, we):
    from tables.medical_conditions import LabValue
    labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == rid,
        LabValue.recorded_date >= ws, LabValue.recorded_date <= we
    ).all()
    abnormal = [l for l in labs if l.is_abnormal]
    changes = []
    for l in labs:
        pct = l.pct_change_from_previous
        arrow = "↑" if pct and pct > 0 else "↓" if pct and pct < 0 else ""
        if pct and abs(pct) >= 1:
            changes.append(f"{l.metric_name} {arrow} {abs(round(pct, 1))}% to {l.normalized_value} {l.normalized_unit or ''}")
    return {"new_labs_this_week": len(labs), "abnormal_count": len(abnormal), "key_changes": changes[:5]}


def _agg_meds(rid, db, start, end):
    from tables.medication_dose_logs import MedicationDoseLog, DoseStatusEnum
    from tables.medications import Medication
    logs = db.query(MedicationDoseLog).filter(
        MedicationDoseLog.care_recipient_id == rid,
        MedicationDoseLog.scheduled_time >= start, MedicationDoseLog.scheduled_time <= end
    ).all()
    total = len(logs)
    taken = sum(1 for l in logs if l.status == DoseStatusEnum.TAKEN)
    missed = sum(1 for l in logs if l.status == DoseStatusEnum.MISSED)
    # Find problem meds
    missed_by_med = {}
    for l in logs:
        if l.status == DoseStatusEnum.MISSED:
            missed_by_med[l.medication_id] = missed_by_med.get(l.medication_id, 0) + 1
    problem_meds = []
    for med_id, count in sorted(missed_by_med.items(), key=lambda x: -x[1])[:3]:
        med = db.query(Medication).filter(Medication.medication_id == med_id).first()
        if med:
            problem_meds.append(f"{med.medicine_name} - missed {count} doses")
    # If no logs exist, adherence is UNKNOWN (not perfect) — use None to signal missing data
    adherence_pct = round(taken / total * 100, 1) if total > 0 else None
    return {
        "total_doses": total, "taken": taken, "missed": missed,
        "adherence_pct": adherence_pct,
        "no_data": total == 0,
        "problem_meds": problem_meds,
    }


# ── Current health snapshot (latest data, not limited to this week) ─────────
_KEY_LABS = [
    # (label, metric names as stored, unit hint)
    ("Fasting blood sugar", ("Fasting Glucose", "Fasting Blood Sugar", "FBS"), "mg/dL"),
    ("Post-meal blood sugar", ("Postprandial Glucose", "PP Glucose", "PPBS", "Post Prandial Glucose"), "mg/dL"),
    ("Random blood sugar", ("Random Glucose", "Random Blood Sugar", "RBS"), "mg/dL"),
    ("HbA1c (3-month sugar)", ("HbA1c",), "%"),
    ("Estimated average glucose", ("Estimated Average Glucose",), "mg/dL"),
    ("Hemoglobin", ("Hemoglobin",), "g/dL"),
    ("Creatinine", ("Creatinine",), "mg/dL"),
    ("eGFR (kidney)", ("eGFR", "confirm. eGFR"), "mL/min"),
    ("BUN / Urea", ("BUN", "Urea", "Blood Urea"), "mg/dL"),
    ("Total cholesterol", ("Total Cholesterol", "Cholesterol"), "mg/dL"),
    ("LDL cholesterol", ("LDL", "LDL Cholesterol"), "mg/dL"),
    ("HDL cholesterol", ("HDL", "HDL Cholesterol"), "mg/dL"),
    ("Triglycerides", ("Triglycerides",), "mg/dL"),
    ("TSH (thyroid)", ("TSH",), "uIU/mL"),
]


# standard limits for the sugar tests (used only when the report gave no reference range)
_SUGAR_LIMITS = {
    "Fasting blood sugar": (70, 100), "Post-meal blood sugar": (None, 140), "Random blood sugar": (None, 140),
    "HbA1c (3-month sugar)": (None, 5.7), "Estimated average glucose": (None, 117),
}


def _current_snapshot(rid, db) -> Dict[str, Any]:
    """Latest vitals (with date), 30-day vital trends, latest sugar / key lab values and the
    current medicines - so the weekly report always shows the patient's present state, even in a
    week with no new readings."""
    from tables.vital_signs import VitalSign
    out: Dict[str, Any] = {"latest_vitals": None, "vital_trends": [], "key_labs": [], "medicines": []}

    # latest vitals: newest value of each vital (they may come from different readings)
    try:
        rows = db.query(VitalSign).filter(VitalSign.care_recipient_id == rid)                  .order_by(VitalSign.recorded_at.desc()).limit(60).all()
        if rows:
            def newest(attr):
                for r in rows:
                    v = getattr(r, attr, None)
                    if v not in (None, 0):
                        return v, r.recorded_at
                return None, None
            lv = {}
            for key, attr in [("heart_rate", "heart_rate"), ("systolic_bp", "systolic_bp"),
                              ("diastolic_bp", "diastolic_bp"), ("spo2", "oxygen_saturation"),
                              ("temperature", "temperature"), ("sleep_score", "sleep_score"),
                              ("weight", "weight"), ("bmi", "bmi")]:
                v, _ = newest(attr)
                lv[key] = round(float(v), 1) if v is not None else None
            last = rows[0].recorded_at
            lv["recorded_at"] = last.strftime("%d %b %Y") if last else None
            lv["days_ago"] = (datetime.datetime.utcnow() - last).days if last else None
            out["latest_vitals"] = lv

            # trends over the last 30 days: first vs latest reading
            since = datetime.datetime.utcnow() - datetime.timedelta(days=30)
            recent = [r for r in rows if r.recorded_at and r.recorded_at >= since][::-1]   # oldest first
            for label, attr, unit in [("Heart rate", "heart_rate", "bpm"), ("Systolic BP", "systolic_bp", "mmHg"),
                                      ("Diastolic BP", "diastolic_bp", "mmHg"), ("SpO2", "oxygen_saturation", "%"),
                                      ("Sleep score", "sleep_score", ""), ("Weight", "weight", "kg")]:
                vals = [float(getattr(r, attr)) for r in recent if getattr(r, attr, None) not in (None, 0)]
                if len(vals) >= 2:
                    ch = vals[-1] - vals[0]
                    direction = "rising" if ch > 0.5 else "falling" if ch < -0.5 else "stable"
                    out["vital_trends"].append({"name": label, "first": round(vals[0], 1), "last": round(vals[-1], 1),
                                                "change": round(ch, 1), "direction": direction,
                                                "readings": len(vals), "unit": unit})
    except Exception:
        db.rollback()

    # latest sugar + key lab values (misread / impossible values skipped)
    try:
        from tables.medical_conditions import LabValue
        try:
            from services.agent_medical_context import _plausible, _num
        except Exception:
            _plausible = lambda m, v: v is not None   # noqa: E731
            _num = lambda lv: lv.normalized_value if lv.normalized_value is not None else lv.metric_value  # noqa: E731
        labs = db.query(LabValue).filter(LabValue.care_recipient_id == rid)                  .order_by(LabValue.recorded_date.desc(), LabValue.id.desc()).all()
        for label, names, unit_hint in _KEY_LABS:
            lower = {n.lower() for n in names}
            for lv in labs:
                if (lv.metric_name or "").strip().lower() not in lower:
                    continue
                val = _num(lv)
                if val is None or not _plausible(lv.metric_name, float(val)):
                    continue
                lo, hi = lv.reference_range_low, lv.reference_range_high
                if lo is None and hi is None and label in _SUGAR_LIMITS:
                    lo, hi = _SUGAR_LIMITS[label]
                if lo is None and hi is None:
                    status = "Abnormal" if lv.is_abnormal else "-"        # no range on record: do not guess
                else:
                    status = ("High" if hi is not None and val > hi else "Low" if lo is not None and val < lo
                              else "Abnormal" if lv.is_abnormal else "Normal")
                unit = next((u for u in (lv.normalized_unit, lv.unit)
                             if u and re.fullmatch(r"[A-Za-z%/µμ.\d^ ]{1,12}", u.strip())
                             and re.search(r"(%|/|g|l|mol|iu|u|mm)", u.strip(), re.I)), unit_hint)
                out["key_labs"].append({
                    "name": label, "value": round(float(val), 2),
                    "unit": unit,
                    "date": lv.recorded_date.strftime("%d %b %Y") if lv.recorded_date else None,
                    "status": status,
                    "change_pct": round(lv.pct_change_from_previous, 1) if lv.pct_change_from_previous else None,
                })
                break
    except Exception:
        db.rollback()

    # current medicines
    try:
        from tables.medications import Medication
        meds = db.query(Medication).filter(Medication.care_recipient_id == rid).all()
        for m in meds:
            st = m.status.value if hasattr(m.status, "value") else str(m.status or "")
            if st.lower() != "active":
                continue
            out["medicines"].append({"name": m.medicine_name, "dosage": m.dosage or "",
                                     "frequency": m.frequency or "", "schedule": m.schedule_time or ""})
    except Exception:
        db.rollback()
    return out


def _agg_conditions(rid, db, start, end):
    from tables.medical_conditions import ConditionHistory, PatientCondition, ConditionStatus
    changes = db.query(ConditionHistory).join(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid,
        ConditionHistory.recorded_at >= start, ConditionHistory.recorded_at <= end
    ).all()
    status_changes = [f"{c.condition.disease_name}: {c.previous_status or '?'} to {c.new_status}" for c in changes if c.condition]
    new_diag = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid,
        PatientCondition.created_at >= start, PatientCondition.created_at <= end
    ).all()
    # Count ALL active conditions (not just this week) for burden scoring
    active_conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid,
        PatientCondition.status == ConditionStatus.active
    ).all()
    return {
        "status_changes": status_changes[:5],
        "new_diagnoses": [d.disease_name for d in new_diag],
        "active_condition_count": len(active_conditions),
        "active_condition_names": [c.disease_name for c in active_conditions[:5]],
    }


def _agg_audio(rid, db, start, end):
    from tables.audio_events import AudioEvent, AudioEventType
    coughs = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == rid, AudioEvent.event_type == AudioEventType.cough,
        AudioEvent.detected_at >= start, AudioEvent.detected_at <= end
    ).scalar() or 0
    sneezes = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == rid, AudioEvent.event_type == AudioEventType.sneeze,
        AudioEvent.detected_at >= start, AudioEvent.detected_at <= end
    ).scalar() or 0
    return {"cough_count_7d": coughs, "sneeze_count_7d": sneezes, "cough_daily_avg": round(coughs / 7, 1)}


def _agg_video(rid, db, start, end):
    from tables.video_analysis import VideoAnalysis
    vids = db.query(VideoAnalysis).filter(
        VideoAnalysis.recipient_id == rid,
        VideoAnalysis.timestamp >= start, VideoAnalysis.timestamp <= end
    ).all()
    falls = sum(v.fall_count for v in vids)
    out = {
        "fall_count": falls,
        "avg_activity_score": _safe_avg([v.activity_score for v in vids]),
        "avg_mobility_score": _safe_avg([v.mobility_score for v in vids]),
    }
    # Falls from the room camera's fall log (live monitoring + analysed videos), counted exactly like
    # Video Monitoring > History: detections < 90 s apart = one fall. The old count above only saw
    # uploaded-video analyses, so live-camera falls were missing from the report.
    try:
        inc = _fall_incidents(rid, db, start, end)
        if inc is not None:
            ups = [i["up"] for i in inc if i["got_up"] and i["up"] is not None]
            out.update({
                "fall_count": len(inc),
                "falls_got_up_in_time": sum(1 for i in inc if i["outcome"] == "ok"),
                "long_lies": sum(1 for i in inc if i["outcome"] == "long_lie"),
                "falls_outcome_unknown": sum(1 for i in inc if i["outcome"] == "unknown"),
                "avg_get_up_sec": round(sum(ups) / len(ups), 1) if ups else None,
                "fall_times": [i["when"] for i in inc][:10],
            })
    except Exception:
        db.rollback()
    return out


def _fall_incidents(rid, db, start, end):
    from tables.video_analysis import FallEvent
    from tables.users import CareRecipient
    from sqlalchemy import or_, and_
    recip = db.query(CareRecipient).filter(CareRecipient.id == rid).first()
    scope = FallEvent.recipient_id == rid
    if recip and recip.caretaker_id and \
            db.query(CareRecipient).filter(CareRecipient.caretaker_id == recip.caretaker_id).count() == 1:
        scope = or_(scope, and_(FallEvent.recipient_id.is_(None), FallEvent.caretaker_id == recip.caretaker_id))
    rows = (db.query(FallEvent).filter(scope, FallEvent.occurred_at >= start, FallEvent.occurred_at <= end)
            .order_by(FallEvent.occurred_at.asc(), FallEvent.id.asc()).all())
    LONG_LIE = 120
    inc = []
    for r in rows:
        if not r.occurred_at:
            continue
        last = inc[-1] if inc else None
        same_video = last and r.source == "upload" and last["video"] == (r.video_filename or "")
        if last and (r.occurred_at - last["last"]).total_seconds() <= 90 and (r.source != "upload" or same_video):
            last["last"] = r.occurred_at
            if r.got_up and (last["up"] is None or not last["got_up"]):
                last["got_up"], last["up"] = True, r.time_to_get_up_sec
            if r.time_on_ground_sec:
                last["ground"] = max(last["ground"] or 0, r.time_on_ground_sec)
            continue
        inc.append({"at": r.occurred_at, "last": r.occurred_at, "video": r.video_filename or "",
                    "got_up": bool(r.got_up), "up": r.time_to_get_up_sec, "ground": r.time_on_ground_sec})
    ist = datetime.timedelta(hours=5, minutes=30)
    for i in inc:
        if i["got_up"] and i["up"] is not None:
            i["outcome"] = "ok" if i["up"] <= LONG_LIE else "long_lie"
        elif i["got_up"]:
            i["outcome"] = "ok"
        elif (i["ground"] or 0) > LONG_LIE:
            i["outcome"] = "long_lie"
        else:
            i["outcome"] = "unknown"
        i["when"] = (i["at"] + ist).strftime("%a %d %b, %I:%M %p")
    return inc


def _agg_environment(rid, db, start, end):
    from tables.environment import EnvironmentSensor
    rows = db.query(EnvironmentSensor).filter(
        EnvironmentSensor.care_recipient_id == rid,
        EnvironmentSensor.timestamp >= start, EnvironmentSensor.timestamp <= end
    ).all()
    extreme = sum(1 for r in rows if (r.temperature_c and (r.temperature_c > 32 or r.temperature_c < 15)))
    return {
        "avg_room_temp_c": _safe_avg([r.temperature_c for r in rows]),
        "avg_humidity": _safe_avg([r.humidity_percent for r in rows]),
        "readings": len(rows), "extreme_readings": extreme,
    }


def _agg_mood(rid, db, start, end):
    from tables.conversation_history import ConversationMessage, MoodEnum
    msgs = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == rid,
        ConversationMessage.created_at >= start, ConversationMessage.created_at <= end
    ).all()
    moods = {}
    for m in msgs:
        if m.mood_detected:
            key = m.mood_detected.value if hasattr(m.mood_detected, 'value') else str(m.mood_detected)
            moods[key] = moods.get(key, 0) + 1
    dominant = max(moods, key=moods.get) if moods else "neutral"
    sessions = set(m.conversation_session_id for m in msgs)
    return {
        "conversations": len(sessions), "messages": len(msgs),
        "dominant_mood": dominant, "mood_distribution": moods,
    }


def _agg_alerts(rid, db, start, end):
    from tables.medical_conditions import MedicalAlert
    from tables.medical_recommendations import MedicalRecommendation
    alerts = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == rid,
        MedicalAlert.created_at >= start, MedicalAlert.created_at <= end
    ).all()
    crit = sum(1 for a in alerts if a.severity and a.severity.value == "critical")
    high = sum(1 for a in alerts if a.severity and a.severity.value == "high")
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == rid,
        MedicalRecommendation.created_at >= start, MedicalRecommendation.created_at <= end
    ).order_by(desc(MedicalRecommendation.created_at)).limit(5).all()
    themes = list(set(r.condition_group for r in recs if r.condition_group))
    return {"critical_count": crit, "high_count": high, "total_alerts": len(alerts), "top_themes": themes[:5]}


# ── 2. Week-over-Week Comparison ────────────────────────────────────────────

def build_comparison(current: Dict, previous: Optional[Dict]) -> Dict:
    if not previous:
        return {"has_previous": False}
    comp = {"has_previous": True}
    # Adherence
    curr_adh = (current.get("medication_adherence") or {}).get("adherence_pct")
    prev_adh = (previous.get("medication_adherence") or {}).get("adherence_pct")
    # no doses logged in one of the weeks -> adherence_pct is None: no change can be computed
    comp["adherence_change"] = round(curr_adh - prev_adh, 1) if curr_adh is not None and prev_adh is not None else None
    # Vitals
    for key in ["avg_heart_rate", "avg_systolic_bp", "avg_spo2", "avg_sleep_score"]:
        c = current.get("vitals_summary", {}).get(key)
        p = previous.get("vitals_summary", {}).get(key)
        comp[f"{key}_change"] = round(c - p, 1) if c is not None and p is not None else None
    # Cough trend
    curr_c = (current.get("respiratory_activity") or {}).get("cough_count_7d") or 0
    prev_c = (previous.get("respiratory_activity") or {}).get("cough_count_7d") or 0
    comp["cough_trend"] = "increasing" if curr_c > prev_c * 1.2 else "decreasing" if curr_c < prev_c * 0.8 else "stable"
    # Activity
    curr_a = current.get("physical_activity", {}).get("avg_activity_score")
    prev_a = previous.get("physical_activity", {}).get("avg_activity_score")
    comp["activity_change"] = round(curr_a - prev_a, 1) if curr_a and prev_a else None
    return comp


# ── 3. Wellness Score (deterministic, no AI) ────────────────────────────────

def compute_wellness_score(sections: Dict) -> float:
    score = 100.0

    # ── Disease burden (up to -30): active chronic conditions ─────────────────
    conditions = sections.get("condition_updates", {})
    active_count = conditions.get("active_condition_count", 0)
    # Each active condition subtracts points; caps at 30
    # 1 condition = -8, 2 = -14, 3 = -20, 4+ = -28+
    if active_count >= 1:
        score -= min(30, active_count * 7)

    # ── Data coverage penalty (-15 if no vitals recorded this week) ───────────
    vitals = sections.get("vitals_summary", {})
    if vitals.get("readings", 0) == 0:
        score -= 15   # No monitoring data = cannot confirm health

    # ── Medication adherence (up to -25) ──────────────────────────────────────
    med = sections.get("medication_adherence", {})
    if med.get("no_data"):
        # No medication logs — if patient has conditions, assume unmonitored gap
        if active_count > 0:
            score -= 10
    else:
        adh = med.get("adherence_pct")
        if adh is not None:
            score -= max(0, (100 - adh)) * 0.25

    # ── Vitals instability ────────────────────────────────────────────────────
    if vitals.get("alerts"):
        score -= len(vitals["alerts"]) * 5
    if vitals.get("avg_spo2") and vitals["avg_spo2"] < 94:
        score -= 10
    if vitals.get("avg_systolic_bp") and vitals["avg_systolic_bp"] > 140:
        score -= 8

    # ── Lab trends (up to -20) ────────────────────────────────────────────────
    labs = sections.get("lab_summary", {})
    score -= min(20, labs.get("abnormal_count", 0) * 4)

    # ── Activity / falls (-8 per fall) ───────────────────────────────────────
    act = sections.get("physical_activity", {})
    if act.get("fall_count", 0) > 0:
        score -= min(25, act["fall_count"] * 8)   # capped like the other categories (6 falls no longer = score 0)
    if act.get("avg_activity_score") and act["avg_activity_score"] < 3:
        score -= 5

    # ── Mood (up to -10) ─────────────────────────────────────────────────────
    mood = sections.get("mood_wellbeing", {})
    neg_moods = sum(mood.get("mood_distribution", {}).get(m, 0)
                    for m in ["sad", "anxious", "distressed", "angry", "lonely"])
    if neg_moods > 3:
        score -= min(10, neg_moods * 2)

    # ── Environment (up to -10) ───────────────────────────────────────────────
    env = sections.get("environment", {})
    score -= min(10, env.get("extreme_readings", 0) * 3)

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts = sections.get("alerts_recap", {})
    score -= alerts.get("critical_count", 0) * 5
    score -= alerts.get("high_count", 0) * 2

    return max(0, min(100, round(score, 1)))


def _determine_risk_trend(score: float, prev_score: Optional[float]) -> str:
    if prev_score is None:
        return "stable"
    diff = score - prev_score
    if diff > 5:
        return "improving"
    elif diff < -5:
        return "declining"
    return "stable"


# ── 4. Gemini AI Narrative ──────────────────────────────────────────────────

TRIAGE_PROMPT = """You are Caretaker.ai weekly report triage. Given this patient week summary,
decide if a full AI narrative is needed. Return ONLY valid JSON:
{{"needs_deep_analysis": true/false, "severity": "low|moderate|critical", "reason": "one sentence"}}

Data: {data}"""

NARRATIVE_PROMPT = """You are Caretaker.ai writing a weekly health report for a family caregiver.
Write in warm, plain English. No medical jargon. Reference specific numbers.

PATIENT: {name} (Age {age})
WEEK: {week_start} to {week_end}
{comparison_note}

WEEKLY DATA:
{data}

Return ONLY valid JSON:
{{
  "executive_summary": "2-3 sentences summarizing the week",
  "highlights": ["positive thing 1", "positive thing 2"],
  "concerns": ["concern 1 with specific numbers"],
  "caregiver_actions": ["specific action 1", "specific action 2"],
  "doctor_note": "One sentence for the doctor, or empty string if not needed",
  "emotional_wellbeing": "1-2 sentences about mood patterns and emotional state this week"
}}"""


def _run_triage(sections: Dict) -> Dict:
    compact = json.dumps({k: v for k, v in sections.items()}, default=str)
    if len(compact) > 4000:
        compact = compact[:4000]
    resp = call_gemini({
        "contents": [{"parts": [{"text": TRIAGE_PROMPT.format(data=compact)}]}],
        "generationConfig": {"maxOutputTokens": 256, "temperature": 0.1}
    }, timeout=15, caller="[weekly_report_triage]", model_override="gemini-2.5-flash-lite")
    if resp and "candidates" in resp:
        raw = ""
        for p in resp["candidates"][0]["content"]["parts"]:
            if "text" in p and not p.get("thought"):
                raw = p["text"].strip()
        if raw:
            import re
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    return {"needs_deep_analysis": False, "severity": "low", "reason": "triage_failed"}


def _run_narrative(sections: Dict, comparison: Dict, recipient, week_start, week_end) -> tuple:
    comp_note = ""
    if comparison.get("has_previous"):
        adh_chg = comparison.get('adherence_change')
        adh_txt = f"{adh_chg:+.1f}%" if adh_chg is not None else "not comparable (no doses logged)"
        comp_note = f"vs Last Week: adherence {adh_txt}, cough trend: {comparison.get('cough_trend', 'stable')}"
    severity = "low"
    # Determine model
    model = "gemini-3.1-flash-lite"
    alerts = sections.get("alerts_recap", {})
    if alerts.get("critical_count", 0) > 0 or sections.get("physical_activity", {}).get("fall_count", 0) > 0:
        model = "gemini-2.5-flash"
        severity = "critical"
    prompt = NARRATIVE_PROMPT.format(
        name=recipient.full_name, age=recipient.age or "?",
        week_start=str(week_start), week_end=str(week_end),
        comparison_note=comp_note,
        data=json.dumps(sections, default=str)[:6000]
    )
    resp = call_gemini({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3}
    }, timeout=45, caller="[weekly_report_narrative]", model_override=model)
    if resp and "candidates" in resp:
        raw = ""
        for p in resp["candidates"][0]["content"]["parts"]:
            if "text" in p and not p.get("thought"):
                raw = p["text"].strip()
        if raw:
            import re
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            try:
                return json.loads(cleaned), model
            except json.JSONDecodeError:
                pass
    return None, model


def _natural_narrative(sections: Dict, recipient, score: float, comparison: Dict) -> Dict:
    """A natural, specific weekly summary written from the data (used when the AI model is not
    available): current BP / sugar / labs, medicines, falls, mood, plus actions and a doctor note."""
    name = recipient.full_name or "the patient"
    first = name.split()[0]
    cur = sections.get("current_health") or {}
    lv = cur.get("latest_vitals") or {}
    labs = {l["name"]: l for l in (cur.get("key_labs") or [])}
    meds = cur.get("medicines") or []
    pa = sections.get("physical_activity") or {}
    mood = sections.get("mood_wellbeing") or {}
    adh = sections.get("medication_adherence") or {}
    cond = sections.get("condition_updates") or {}
    vit_week = sections.get("vitals_summary") or {}

    summary, highs, concerns, actions, doc = [], [], [], [], []

    # vitals
    sbp, dbp, hr, spo2 = lv.get("systolic_bp"), lv.get("diastolic_bp"), lv.get("heart_rate"), lv.get("spo2")
    when = lv.get("recorded_at")
    ago = lv.get("days_ago")
    when_txt = (f"on {when}" + (f" ({ago} days ago)" if ago else " (today)" if ago == 0 else "")) if when else ""
    vit_bits = []
    if sbp and dbp:
        bp_state = ("high" if sbp >= 140 or dbp >= 90 else "slightly raised" if sbp >= 130 or dbp >= 80
                    else "low" if sbp < 90 or dbp < 60 else "normal")
        vit_bits.append(f"blood pressure {sbp:g}/{dbp:g} mmHg ({bp_state})")
        if bp_state in ("high", "slightly raised"):
            concerns.append(f"Blood pressure was {sbp:g}/{dbp:g} mmHg at the last reading - above the healthy 120/80.")
            doc.append(f"BP {sbp:g}/{dbp:g}")
        elif bp_state == "normal":
            highs.append(f"Blood pressure was in the normal range ({sbp:g}/{dbp:g} mmHg).")
    if hr:
        vit_bits.append(f"heart rate {hr:g} bpm")
        if hr > 100:
            concerns.append(f"Heart rate was fast at {hr:g} bpm (normal 60-100).")
            doc.append(f"HR {hr:g}")
    if spo2:
        vit_bits.append(f"oxygen {spo2:g}%")
        if spo2 < 95:
            concerns.append(f"Oxygen level (SpO2) was low at {spo2:g}% (normal 95-100%).")
            doc.append(f"SpO2 {spo2:g}%")
    if vit_bits:
        summary.append(f"{first}'s latest vitals {when_txt}: " + ", ".join(vit_bits) + ".")
    if (vit_week.get("readings") or 0) == 0:
        actions.append("Check and log blood pressure, sugar and oxygen at least 2-3 times this week - "
                       "no vitals were recorded in the last 7 days.")

    # sugar and labs
    sugar = [labs[k] for k in ("Fasting blood sugar", "Post-meal blood sugar", "Random blood sugar",
                               "HbA1c (3-month sugar)", "Estimated average glucose") if k in labs]
    if sugar:
        txt = ", ".join(f"{s['name'].lower()} {s['value']:g} {s['unit']}" for s in sugar[:3])
        high_sugar = [s for s in sugar if s["status"] == "High"]
        date = sugar[0].get("date")
        summary.append(f"Her last sugar tests ({date}) showed {txt}" +
                       (" - higher than the healthy range." if high_sugar else " - within range."))
        if high_sugar:
            concerns.append("Blood sugar is high: " + ", ".join(f"{s['name'].lower()} {s['value']:g} {s['unit']}"
                                                              for s in high_sugar[:3]) + ".")
            doc.append("raised glucose")
            actions.append("Check fasting and after-meal sugar this week; give diabetes medicines on time and "
                           "keep sweets and white rice / maida low.")
    other_bad = [l for k, l in labs.items() if l not in sugar and l["status"] in ("High", "Low")]
    for l in other_bad[:4]:
        concerns.append(f"{l['name']} was {l['status'].lower()} at {l['value']:g} {l['unit']} ({l.get('date')}).")
    other_ok = [l for k, l in labs.items() if l not in sugar and l["status"] == "Normal"]
    if other_ok:
        highs.append("Normal results for " + ", ".join(l["name"].split(" (")[0].lower() for l in other_ok[:3]) + ".")
    lab_dates = [l.get("date") for l in labs.values() if l.get("date")]
    if lab_dates:
        try:
            newest = max(datetime.datetime.strptime(d, "%d %b %Y") for d in lab_dates)
            if (datetime.datetime.now() - newest).days > 90:
                actions.append(f"The last blood tests are from {newest.strftime('%b %Y')} - ask the doctor about "
                               "repeating HbA1c, lipid profile and hemoglobin.")
        except Exception:
            pass

    # medicines
    if meds:
        summary.append(f"{first} is on {len(meds)} medicines, including " +
                       ", ".join(m["name"] for m in meds[:4]) + ("." if len(meds) <= 4 else " and others."))
    if adh.get("no_data"):
        actions.append("Mark medicines as taken in the app each day so the weekly adherence can be tracked.")
    elif adh.get("adherence_pct") is not None:
        if adh["adherence_pct"] >= 90:
            highs.append(f"Took {adh['adherence_pct']:g}% of scheduled medicine doses.")
        else:
            concerns.append(f"Only {adh['adherence_pct']:g}% of medicine doses were taken ({adh.get('missed', 0)} missed).")

    # falls
    nf = pa.get("fall_count") or 0
    if nf:
        ok, ll = pa.get("falls_got_up_in_time") or 0, pa.get("long_lies") or 0
        avg = pa.get("avg_get_up_sec")
        summary.append(f"The home camera recorded {nf} fall{'s' if nf > 1 else ''} this week" +
                       (f"; she got up within 2 minutes after {ok} of them" if ok else "") +
                       (f" (average {avg:g} s)" if avg else "") + ".")
        concerns.append(f"{nf} fall{'s' if nf > 1 else ''} this week" +
                        (f", including {ll} long lie{'s' if ll > 1 else ''} (on the floor over 2 minutes)" if ll else "") + ".")
        actions.append("Reduce fall risk: clear the floor, use night lights, non-slip footwear, and help her "
                       "stand up slowly (BP and sugar medicines can cause dizziness).")
        doc.append(f"{nf} fall(s) this week")
    else:
        highs.append("No falls were recorded by the home camera this week.")

    # mood
    sessions = mood.get("conversations") or mood.get("total_sessions") or mood.get("sessions")
    dm = mood.get("dominant_mood")
    mood_txt = ""
    if dm:
        mood_txt = f"Her mood in conversations with the voice assistant was mostly {dm}"
        mood_txt += f" across {sessions} sessions." if sessions else "."
        if dm in ("sad", "anxious", "lonely", "stressed"):
            actions.append("Spend some extra time talking with her this week - her mood was low.")

    conds = cond.get("active_condition_names") or []
    if conds:
        doc.insert(0, "Active: " + ", ".join(conds))
    summary.append(f"Overall wellness score this week: {score:g}/100.")

    return {
        "executive_summary": " ".join(summary),
        "highlights": highs[:4],
        "concerns": concerns[:6],
        "caregiver_actions": actions[:5],
        "doctor_note": "; ".join(doc) + "." if doc else "",
        "emotional_wellbeing": mood_txt,
    }


# ── 5. Orchestrator ─────────────────────────────────────────────────────────

def generate_weekly_report(recipient_id: int, db: Session,
                           reference_date: datetime.date = None) -> Dict:
    from tables.users import CareRecipient
    from tables.weekly_reports import WeeklyReport

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {"error": "Recipient not found"}

    week_start, week_end = _week_bounds(reference_date or (datetime.date.today() - datetime.timedelta(days=1)))

    # Check if report already exists
    existing = db.query(WeeklyReport).filter(
        WeeklyReport.care_recipient_id == recipient_id,
        WeeklyReport.week_start == week_start
    ).first()
    if existing:
        return {"report_id": existing.id, "status": "already_exists",
                "report": existing.report_data, "narrative": existing.ai_narrative,
                "score": existing.overall_score, "comparison": existing.comparison}

    # Aggregate current week
    sections = aggregate_weekly_data(recipient_id, db, week_start, week_end)

    # Get previous week for comparison
    prev_start, prev_end = _prev_week_bounds(week_start)
    prev_report = db.query(WeeklyReport).filter(
        WeeklyReport.care_recipient_id == recipient_id,
        WeeklyReport.week_start == prev_start
    ).first()
    prev_sections = prev_report.report_data if prev_report else None
    comparison = build_comparison(sections, prev_sections)

    # Wellness score
    prev_score = prev_report.overall_score if prev_report else None
    score = compute_wellness_score(sections)
    risk_trend = _determine_risk_trend(score, prev_score)
    if prev_score is not None:
        comparison["score_change"] = round(score - prev_score, 1)

    # AI triage + narrative
    narrative = None
    model_used = None
    triage = _run_triage(sections)
    if triage.get("needs_deep_analysis"):
        narrative, model_used = _run_narrative(sections, comparison, recipient, week_start, week_end)
    if not narrative:
        try:
            narrative = _natural_narrative(sections, recipient, score, comparison)
        except Exception as _ne:
            print(f"[weekly_report] natural summary failed: {_ne}")
            narrative = None
    if not narrative:
        # Lightweight rule-based summary
        narrative = {
            "executive_summary": f"A generally stable week for {recipient.full_name}. Wellness score: {score}/100.",
            "highlights": [], "concerns": [], "caregiver_actions": [],
            "doctor_note": "", "emotional_wellbeing": ""
        }
        adh = sections.get("medication_adherence", {})
        adh_pct = adh.get("adherence_pct")   # may be None if no logs
        if adh_pct is not None and adh_pct >= 90:
            narrative["highlights"].append(f"Good medication adherence at {adh_pct}%")
        elif adh_pct is None and adh.get("no_data"):
            narrative["concerns"].append("Medication adherence could not be tracked this week")
        if adh.get("missed", 0) > 0:
            narrative["concerns"].append(f"{adh['missed']} medication doses were missed")
        labs = sections.get("lab_summary", {})
        if labs.get("abnormal_count", 0) > 0:
            narrative["concerns"].extend(labs.get("key_changes", [])[:2])
        mood = sections.get("mood_wellbeing", {})
        if mood.get("dominant_mood"):
            narrative["emotional_wellbeing"] = f"Dominant mood this week: {mood['dominant_mood']}."
        # Mention active conditions in summary if present
        cond = sections.get("condition_updates", {})
        if cond.get("active_condition_count", 0) > 0:
            names = ", ".join(cond.get("active_condition_names", []))
            narrative["doctor_note"] = f"Patient has {cond['active_condition_count']} active condition(s): {names}."

    # Persist
    report = WeeklyReport(
        care_recipient_id=recipient_id,
        week_start=week_start, week_end=week_end,
        report_data=sections, ai_narrative=narrative,
        comparison=comparison, model_used=model_used,
        overall_score=score, risk_trend=risk_trend,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    print(f"[weekly_report] Generated for {recipient.full_name} "
          f"(score={score}, trend={risk_trend}, model={model_used or 'rule-based'})")

    return {
        "report_id": report.id,
        "recipient_name": recipient.full_name,
        "week_start": str(week_start), "week_end": str(week_end),
        "sections": sections, "ai_narrative": narrative,
        "comparison": comparison, "overall_score": score,
        "risk_trend": risk_trend, "model_used": model_used,
    }


def generate_all_weekly_reports(db: Session):
    from tables.users import CareRecipient
    recipients = db.query(CareRecipient).all()
    results = []
    for r in recipients:
        try:
            result = generate_weekly_report(r.id, db)
            results.append({"id": r.id, "name": r.full_name, "status": "ok", "score": result.get("overall_score")})
        except Exception as e:
            print(f"[weekly_report] Failed for {r.full_name}: {e}")
            results.append({"id": r.id, "name": r.full_name, "status": "error", "error": str(e)})
    return results
