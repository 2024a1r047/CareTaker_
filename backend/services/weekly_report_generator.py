"""
Weekly Report Generator
-----------------------
Aggregates all 9 data domains for a care recipient over a 7-day window,
runs a 2-pass Gemini triage+narrative, computes a wellness score,
and persists the result in the WeeklyReport table.
"""

import datetime
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
    return sections


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
            problem_meds.append(f"{med.medicine_name} — missed {count} doses")
    return {
        "total_doses": total, "taken": taken, "missed": missed,
        "adherence_pct": round(taken / total * 100, 1) if total > 0 else 100.0,
        "problem_meds": problem_meds,
    }


def _agg_conditions(rid, db, start, end):
    from tables.medical_conditions import ConditionHistory, PatientCondition
    changes = db.query(ConditionHistory).join(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid,
        ConditionHistory.recorded_at >= start, ConditionHistory.recorded_at <= end
    ).all()
    status_changes = [f"{c.condition.disease_name}: {c.previous_status or '?'} → {c.new_status}" for c in changes if c.condition]
    new_diag = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == rid,
        PatientCondition.created_at >= start, PatientCondition.created_at <= end
    ).all()
    return {"status_changes": status_changes[:5], "new_diagnoses": [d.disease_name for d in new_diag]}


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
    return {
        "fall_count": falls,
        "avg_activity_score": _safe_avg([v.activity_score for v in vids]),
        "avg_mobility_score": _safe_avg([v.mobility_score for v in vids]),
    }


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
    curr_adh = current.get("medication_adherence", {}).get("adherence_pct", 0)
    prev_adh = previous.get("medication_adherence", {}).get("adherence_pct", 0)
    comp["adherence_change"] = round(curr_adh - prev_adh, 1)
    # Vitals
    for key in ["avg_heart_rate", "avg_systolic_bp", "avg_spo2", "avg_sleep_score"]:
        c = current.get("vitals_summary", {}).get(key)
        p = previous.get("vitals_summary", {}).get(key)
        comp[f"{key}_change"] = round(c - p, 1) if c is not None and p is not None else None
    # Cough trend
    curr_c = current.get("respiratory_activity", {}).get("cough_count_7d", 0)
    prev_c = previous.get("respiratory_activity", {}).get("cough_count_7d", 0)
    comp["cough_trend"] = "increasing" if curr_c > prev_c * 1.2 else "decreasing" if curr_c < prev_c * 0.8 else "stable"
    # Activity
    curr_a = current.get("physical_activity", {}).get("avg_activity_score")
    prev_a = previous.get("physical_activity", {}).get("avg_activity_score")
    comp["activity_change"] = round(curr_a - prev_a, 1) if curr_a and prev_a else None
    return comp


# ── 3. Wellness Score (deterministic, no AI) ────────────────────────────────

def compute_wellness_score(sections: Dict) -> float:
    score = 100.0
    # Medication adherence (25%)
    adh = sections.get("medication_adherence", {}).get("adherence_pct", 100)
    score -= max(0, (100 - adh)) * 0.25
    # Vitals stability (20%)
    vitals = sections.get("vitals_summary", {})
    if vitals.get("alerts"):
        score -= len(vitals["alerts"]) * 5
    if vitals.get("avg_spo2") and vitals["avg_spo2"] < 94:
        score -= 10
    # Lab trends (20%)
    labs = sections.get("lab_summary", {})
    score -= labs.get("abnormal_count", 0) * 4
    # Activity/mobility (15%)
    act = sections.get("physical_activity", {})
    if act.get("fall_count", 0) > 0:
        score -= act["fall_count"] * 8
    if act.get("avg_activity_score") and act["avg_activity_score"] < 3:
        score -= 5
    # Mood (10%)
    mood = sections.get("mood_wellbeing", {})
    neg_moods = sum(mood.get("mood_distribution", {}).get(m, 0) for m in ["sad", "anxious", "distressed", "angry", "lonely"])
    if neg_moods > 3:
        score -= min(10, neg_moods * 2)
    # Environment (10%)
    env = sections.get("environment", {})
    score -= min(10, env.get("extreme_readings", 0) * 3)
    # Alerts
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
        comp_note = f"vs Last Week: adherence {comparison.get('adherence_change', 0):+.1f}%, cough trend: {comparison.get('cough_trend', 'stable')}"
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
    else:
        # Lightweight rule-based summary
        narrative = {
            "executive_summary": f"A generally stable week for {recipient.full_name}. Wellness score: {score}/100.",
            "highlights": [], "concerns": [], "caregiver_actions": [],
            "doctor_note": "", "emotional_wellbeing": ""
        }
        adh = sections.get("medication_adherence", {})
        if adh.get("adherence_pct", 100) >= 90:
            narrative["highlights"].append(f"Good medication adherence at {adh['adherence_pct']}%")
        if adh.get("missed", 0) > 0:
            narrative["concerns"].append(f"{adh['missed']} medication doses were missed")
        labs = sections.get("lab_summary", {})
        if labs.get("abnormal_count", 0) > 0:
            narrative["concerns"].extend(labs.get("key_changes", [])[:2])
        mood = sections.get("mood_wellbeing", {})
        if mood.get("dominant_mood"):
            narrative["emotional_wellbeing"] = f"Dominant mood this week: {mood['dominant_mood']}."

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
