"""
Caretaker.ai — Sensor Fusion Insights Engine
=============================================
Aggregates ALL data sources (vitals, environment, medical reports, audio events,
video analysis) and generates cross-domain health conclusions using:
  1. Deterministic rule-based sensor fusion (always fast, always reproducible)
  2. Gemini AI interpretation (deep clinical insights, natural language)

Architecture:
  Body Sensors (ESP32) → Vitals (HR, BP, SpO2, Temp)
  Room Sensors         → Environment (Temp, Humidity, AQI)
  Camera               → Video Analysis (Falls, Activity, Mobility)
  Microphone           → Audio Events (Cough, Sneeze frequency)
  Medical Reports      → Conditions, Lab Values, Risk Score
       ↓ ALL MERGE HERE ↓
  [Sensor Fusion Engine] → Cross-Domain Conclusions → Dashboard + Alerts
"""

import os
import json
import datetime
import requests
from utils.gemini_client import call_gemini
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from tables.users import CareRecipient
from tables.vital_signs import VitalSign
from tables.audio_events import AudioEvent, AudioEventType
from tables.video_analysis import VideoAnalysis
from tables.medical_conditions import PatientCondition, LabValue, ConditionStatus
from tables.medical_recommendations import MedicalRecommendation, severity_rank
from services.recommendation_engine import (
    LITE_MODEL, FLASH_MODEL, PRO_MODEL, build_context_payload, 
    route_to_model, merge_ai_output, is_duplicate,
    safety_filter, PRIORITY, get_daily_usage, should_run_flash,
    group_triggered_rules, get_triggered_rules
)


# ═══════════════════════════════════════════════════════════════════════
#  THRESHOLDS — Clinical reference ranges for sensor fusion rules
# ═══════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "heart_rate":         {"low": 60, "high": 100, "critical_low": 50, "critical_high": 130},
    "systolic_bp":        {"low": 90, "high": 130, "critical_low": 80, "critical_high": 180},
    "diastolic_bp":       {"low": 60, "high": 85, "critical_low": 50, "critical_high": 120},
    "oxygen_saturation":  {"low": 95, "critical_low": 90},
    "temperature_f":      {"low": 97.0, "high": 99.5, "critical_low": 95.0, "critical_high": 103.0},
    "sleep_score":        {"low": 50, "good": 70},
    "bmi":                {"underweight": 18.5, "normal_high": 24.9, "overweight": 30.0},
    "room_temp_c":        {"low": 18, "high": 26, "critical_low": 15, "critical_high": 32},
    "humidity":           {"low": 30, "high": 60, "critical_high": 80},
    "aqi_epa":            {"moderate": 2, "unhealthy": 4, "hazardous": 5},
    "cough_daily":        {"elevated": 5, "high": 15, "critical": 30},
    "fall_risk_days":     7,
}

# Severity Levels
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MODERATE = "moderate"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"


LITE_SYSTEM_PROMPT = """
You are a health monitoring assistant. Scan this patient snapshot and
decide if anything needs a caregiver's attention today.

ONLY flag something if it is genuinely actionable — do not invent advice.
If everything looks stable, return an empty recommendations list.

Return ONLY valid JSON:
{
  "needs_full_analysis": true | false,
  "reason": "one sentence if true, empty string if false",
  "quick_nudges": [
    {
      "caregiver_message": "one specific sentence a family member can act on now",
  "quick_nudges": [
    {
      "caregiver_message": "one specific sentence a family member can act on now",
      "action": "diet | lifestyle | medication_check | environment | monitor",
      "urgency": "suggestion"
    }
  ],
  "health_state": "stable | monitoring"
}

If needs_full_analysis is true, the system will escalate to a more
powerful model — you do not need to provide the full recommendation.
Keep quick_nudges to suggestion-level only. Max 2 nudges.
"""

FLASH_SYSTEM_PROMPT = """
You are Caretaker — a warm health companion helping a family
member look after their elderly loved one at home.

You receive grouped health findings. Your job is to write
1-3 plain-English action cards the caregiver can act on TODAY.

━━━━ BANNED WORDS (never use) ━━━━
endocrinologist, cardiologist, dyslipidemia, creatinine, HbA1c,
ALT, AST, eGFR, systolic, diastolic, NSAIDs, lipid, clinical
correlation, aggressive management, consult physician immediately

━━━━ WRITING RULES ━━━━
- Write like a caring friend who happens to know medicine
- Always anchor to a real number: not "sugar is high" but
  "blood sugar reading has been above the healthy range this month"
- Every action must be doable today: not "increase fiber" but
  "add a small bowl of oats or lentils to lunch today"
- One card per condition group — never one card per lab metric
- The next_check field is mandatory — it tells the caregiver
  when they can stop worrying about this today

━━━━ GOOD EXAMPLE ━━━━
{
  "title": "Blood sugar today",
  "caregiver_message": "His blood sugar control reading has been
    above the healthy range for a few weeks, and his cholesterol
    is also slightly high. Both improve with the same changes —
    so one focused day of eating well makes a real difference.",
  "do_this_now": "Serve a meal with no white rice, bread, or
    sweet drinks today — plain dal, sabzi, and roti is perfect.",
  "why_this_matters": "High blood sugar and cholesterol together
    put extra pressure on the heart — small daily changes add up
    significantly over weeks.",
  "today_actions": [
    "Offer water or plain lassi instead of juice or tea with sugar",
    "A 10-minute walk after lunch helps bring blood sugar down",
    "Avoid fried snacks today — a handful of nuts is a good swap"
  ],
  "next_check": {
    "when": "tomorrow morning",
    "look_for": "Whether he ate well today — that is the only
      thing that matters right now, not the numbers.",
    "if_worse": "If he feels dizzy or unusually tired today,
      sit him down, give him water, and call his doctor."
  },
  "condition_group": "cardiovascular",
  "urgency": "today",
  "confidence": 0.82
}

━━━━ OUTPUT FORMAT ━━━━
Return ONLY valid JSON, no markdown fences:
{
  "cards": [ ...1 to 3 card objects as shown above... ],
  "overall_today": "One sentence about his overall state today.",
  "health_trend": "stable | monitoring | declining"
}
"""

PRO_SYSTEM_PROMPT = """
You are a senior clinical decision support engine embedded in Caretaker.ai.
You are called only when a serious or multi-domain health risk is flagged.
Output will be shown to a family caregiver. It must be clinically rigorous 
internally but translated into plain, urgent English for the reader.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR REASONING PROCESS — follow this exactly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PATTERN: What is the most dangerous chain across ALL domains?
   (e.g., Poor sleep → reduced activity → elevated HR → SpO2 declining)
2. CAUSE: What is the most likely trigger (AQI, medication, infection)?
3. ACTION: What must the caregiver do in the NEXT 60 MINUTES?
4. ESCALATION: Does this need a doctor call or ambulance?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEFORE YOU WRITE — run this check on every sentence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself: "Could a worried parent at 2am follow this instruction?"
✗ VAGUE: "Patient exhibiting hypoxic symptoms. Recommend assessment."
✓ SPECIFIC: "Call the doctor and say: oxygen is at 91% and air is poor."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA ANCHORING — mandatory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every caregiver_message MUST reference specific values (e.g., SpO2 91%, HR 108).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "recommendations": [
    {
      "caregiver_message": "2-3 sentences. Plain English. Warm but urgent tone.",
      "do_this_now": "Single imperative sentence. If ambulance needed, start with Call 112.",
      "why_this_matters": "One sentence. What happens if ignored in the next hour.",
      "action": "emergency | doctor_visit | test | lifestyle",
      "urgency": "critical | high",
      "time_window": "now | next_15_min | next_hour",
      "call_doctor": true | false,
      "call_ambulance": true | false,
      "confidence": 0.0-1.0,
      "reasoning": "Internal clinical chain-of-thought (audit only).",
      "root_cause": "Likely trigger of this episode.",
      "next_check": {
          "when": "next 15 mins | 30 mins | 1 hour",
          "look_for": "Specific clinical sign to watch.",
          "if_worse": "Immediate next step if deteriorating."
      }
    }
  ],
  "cross_domain_pattern": "Describe the multi-domain chain you found.",
  "health_state": "at_risk | critical",
  "snapshot_summary": "One sentence describing overall state right now."
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every caregiver_message must end with: "Do not leave them alone right now."
- If call_ambulance is true, call_doctor must also be true.
- Max 2 recommendations, ranked by severity.
"""


# ═══════════════════════════════════════════════════════════════════════
#  DATA AGGREGATOR — Collects all data for a recipient
# ═══════════════════════════════════════════════════════════════════════

def aggregate_patient_data(recipient_id: int, db: Session) -> dict:
    """
    Collect ALL available data for a recipient from every subsystem.
    This is the single source of truth for the insights engine.
    """
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {"error": "Recipient not found"}

    # 1. Latest Vitals (from ESP32 sensors)
    latest_vital = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id
    ).order_by(desc(VitalSign.recorded_at)).first()

    # Recent vitals for trend analysis (last 24h)
    vitals_24h = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id,
        VitalSign.recorded_at >= datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    ).order_by(desc(VitalSign.recorded_at)).all()

    # 2. Audio Events (cough/sneeze in last 7 days)
    audio_cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    cough_count_7d = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.event_type == AudioEventType.cough,
        AudioEvent.detected_at >= audio_cutoff
    ).scalar() or 0

    sneeze_count_7d = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.event_type == AudioEventType.sneeze,
        AudioEvent.detected_at >= audio_cutoff
    ).scalar() or 0

    # Daily average coughs
    cough_daily_avg = round(cough_count_7d / 7, 1)

    # 3. Video Analysis (falls, activity in last 7 days)
    video_cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=THRESHOLDS["fall_risk_days"])
    recent_videos = db.query(VideoAnalysis).filter(
        VideoAnalysis.recipient_id == recipient_id,
        VideoAnalysis.timestamp >= video_cutoff
    ).all()

    fall_count_7d = sum(v.fall_count for v in recent_videos)
    has_recent_fall = any(v.has_fall for v in recent_videos)
    avg_activity = round(sum(v.activity_score for v in recent_videos) / max(len(recent_videos), 1), 1)
    avg_mobility = round(sum(v.mobility_score for v in recent_videos) / max(len(recent_videos), 1), 1)

    # 4. Medical Conditions
    active_conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()

    # 5. Recent Lab Values
    recent_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id
    ).order_by(desc(LabValue.recorded_date)).limit(20).all()

    abnormal_labs = [l for l in recent_labs if l.is_abnormal]

    # 6. Environmental Data (from local room sensors)
    env_data = _fetch_environment_data(recipient, db)

    return {
        "recipient": {
            "id": recipient.id,
            "name": recipient.full_name,
            "age": recipient.age,
            "gender": recipient.gender.value if recipient.gender else None,
            "respiratory_condition": recipient.respiratory_condition_status,
            "risk_score": recipient.risk_score,
        },
        "vitals": {
            "heart_rate": latest_vital.heart_rate if latest_vital else None,
            "systolic_bp": latest_vital.systolic_bp if latest_vital else None,
            "diastolic_bp": latest_vital.diastolic_bp if latest_vital else None,
            "oxygen_saturation": latest_vital.oxygen_saturation if latest_vital else None,
            "temperature": latest_vital.temperature if latest_vital else None,
            "sleep_score": latest_vital.sleep_score if latest_vital else None,
            "bmi": latest_vital.bmi if latest_vital else None,
            "recorded_at": latest_vital.recorded_at.isoformat() if latest_vital and latest_vital.recorded_at else None,
            "readings_24h": len(vitals_24h),
        },
        "audio": {
            "cough_count_7d": cough_count_7d,
            "sneeze_count_7d": sneeze_count_7d,
            "cough_daily_avg": cough_daily_avg,
        },
        "video": {
            "fall_count_7d": fall_count_7d,
            "has_recent_fall": has_recent_fall,
            "avg_activity_score": avg_activity,
            "avg_mobility_score": avg_mobility,
        },
        "medical": {
            "active_conditions": [{"name": c.disease_name, "status": c.status.value, "severity": c.severity} for c in active_conditions],
            "condition_count": len(active_conditions),
            "abnormal_lab_count": len(abnormal_labs),
            "abnormal_labs": [{"metric": l.metric_name, "value": l.normalized_value, "unit": l.normalized_unit} for l in abnormal_labs[:5]],
        },
        "environment": env_data,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


def _fetch_environment_data(recipient, db: Session) -> dict:
    """Fetch current environmental data — tries local sensors first, then Weather API (same as profile.html)."""
    # Try local environment sensors first
    try:
        from tables.environment import EnvironmentSensor
        latest_sensor = db.query(EnvironmentSensor).filter(
            EnvironmentSensor.care_recipient_id == recipient.id
        ).order_by(desc(EnvironmentSensor.timestamp)).first()
        
        if latest_sensor:
            return {
                "room_temp_c": latest_sensor.temperature_c,
                "humidity": latest_sensor.humidity_percent,
                "aqi_epa": latest_sensor.aqi,
                "pm25": None,
                "location": f"{getattr(recipient, 'full_name', 'Patient')}'s Room",
            }
    except Exception as e:
        print(f"[insights_engine] Environment DB fetch failed: {e}")
    
    # Fallback: use Weather API directly (same logic as /api/weather/current endpoint in main.py)
    try:
        from weather import WeatherPredictionModel
        # Use the same default API key and city as main.py
        api_key = os.environ.get("WEATHER_API_KEY")
        default_city = os.environ.get("DEFAULT_CITY", "Jammu")
        
        if api_key:
            # Use recipient's city (same as profile.html weather widget), fallback to default
            target_city = getattr(recipient, 'city', None) or default_city
            weather = WeatherPredictionModel(api_key, default_city)
            data = weather.fetch_data(city=target_city)
            
            if data and data.get("current"):
                current = data["current"]
                location_name = data.get("location", {}).get("name", target_city)
                aqi_index = current.get("air_quality", {}).get("us-epa-index")
                env_result = {
                    "room_temp_c": current.get("temp_c"),
                    "humidity": current.get("humidity"),
                    "aqi_epa": aqi_index,
                    "pm25": current.get("air_quality", {}).get("pm2_5"),
                    "location": location_name,
                }
                print(f"[insights_engine] Weather data: {location_name} — {env_result['room_temp_c']}°C, {env_result['humidity']}% humidity, AQI {aqi_index}")
                return env_result
    except Exception as e:
        print(f"[insights_engine] Weather API fetch failed: {e}")
        
    return {"room_temp_c": None, "humidity": None, "aqi_epa": None, "pm25": None, "location": None}


# ═══════════════════════════════════════════════════════════════════════
#  RULE-BASED SENSOR FUSION — Cross-domain conclusions
# ═══════════════════════════════════════════════════════════════════════

def generate_fusion_insights(data: dict) -> list:
    """
    Generate cross-domain health insights by combining multiple sensor readings.
    Each insight has: severity, category, title, detail, sensors_involved.
    """
    insights = []
    vitals = data.get("vitals", {})
    audio = data.get("audio", {})
    video = data.get("video", {})
    medical = data.get("medical", {})
    env = data.get("environment", {})
    recipient = data.get("recipient", {})

    hr = vitals.get("heart_rate")
    sbp = vitals.get("systolic_bp")
    dbp = vitals.get("diastolic_bp")
    spo2 = vitals.get("oxygen_saturation")
    temp = vitals.get("temperature")
    sleep = vitals.get("sleep_score")
    bmi = vitals.get("bmi")
    cough_avg = audio.get("cough_daily_avg", 0)
    cough_7d = audio.get("cough_count_7d", 0)
    sneeze_7d = audio.get("sneeze_count_7d", 0)
    fall_count = video.get("fall_count_7d", 0)
    has_fall = video.get("has_recent_fall", False)
    activity = video.get("avg_activity_score", 0)
    mobility = video.get("avg_mobility_score", 0)
    room_temp = env.get("room_temp_c")
    humidity = env.get("humidity")
    aqi = env.get("aqi_epa")
    has_respiratory = recipient.get("respiratory_condition", False)
    conditions = [c["name"].lower() for c in medical.get("active_conditions", [])]
    age = recipient.get("age") or 0

    # ── CROSS-DOMAIN FUSION RULES ──────────────────────────────────

    # 1. HIGH BP + LOW ACTIVITY → Dizziness / Fall Risk
    if sbp and sbp > 140 and activity < 4:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Fall Risk",
            "title": "High Blood Pressure + Low Activity",
            "detail": f"Systolic BP is {sbp} mmHg (elevated) and activity score is only {activity}/10. This combination increases dizziness and fall risk significantly.",
            "recommendation": "Monitor closely. Avoid sudden position changes. Consider BP medication review.",
            "sensors": ["Blood Pressure", "Activity Camera"],
        })

    # 2. LOW SpO2 + HIGH AQI → Respiratory Emergency
    if spo2 and spo2 < 94 and aqi and aqi >= 4:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Respiratory Risk",
            "title": "Low Oxygen + Poor Air Quality",
            "detail": f"SpO2 is {spo2}% (dangerously low) while AQI index is {aqi} (unhealthy). Combined effect poses severe breathing risk.",
            "recommendation": "Move patient to clean air. Use air purifier. Consider supplemental oxygen. Alert caretaker immediately.",
            "sensors": ["SpO2 Sensor", "Air Quality Sensor"],
        })

    # 3. SpO2 LOW + RESPIRATORY CONDITION → Breathing Risk
    if spo2 and spo2 < 95 and has_respiratory:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Respiratory Risk",
            "title": "Low Oxygen with Respiratory Condition",
            "detail": f"SpO2 at {spo2}% in a patient with known respiratory condition. This is below the safe threshold.",
            "recommendation": "Check nebulizer availability. Monitor breathing rate. Consider emergency intervention if SpO2 drops below 90%.",
            "sensors": ["SpO2 Sensor", "Medical History"],
        })

    # 4. HIGH COUGH FREQUENCY + LOW SpO2 → Infection/Pneumonia Risk
    if cough_avg >= THRESHOLDS["cough_daily"]["elevated"] and spo2 and spo2 < 96:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Infection Risk",
            "title": "Persistent Cough + Declining Oxygen",
            "detail": f"Daily cough average is {cough_avg} over 7 days with SpO2 at {spo2}%. Pattern suggests possible respiratory infection or pneumonia.",
            "recommendation": "Schedule chest X-ray. Monitor temperature for fever. Consider antibiotic consultation.",
            "sensors": ["Audio Monitor (Cough)", "SpO2 Sensor"],
        })

    # 5. FEVER + ELEVATED COUGH → Infection Alert
    if temp and temp > 99.5 and cough_7d > 10:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Infection Risk",
            "title": "Fever with Persistent Cough",
            "detail": f"Temperature is {temp}F (elevated) with {cough_7d} cough events in 7 days. Possible upper respiratory infection.",
            "recommendation": "Check for COVID/flu. Monitor temperature trend. Ensure hydration. Consult physician if fever persists > 48h.",
            "sensors": ["Temperature Sensor", "Audio Monitor (Cough)"],
        })

    # 6. FALL DETECTED + HIGH HR → Emergency
    if has_fall and hr and hr > 110:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Emergency",
            "title": "Fall Detected + Elevated Heart Rate",
            "detail": f"Fall detected with heart rate at {hr} bpm (tachycardia). Patient may be in distress or experiencing cardiac event post-fall.",
            "recommendation": "IMMEDIATE CHECK REQUIRED. Verify patient consciousness. Check for injuries. Call emergency services if unresponsive.",
            "sensors": ["Fall Detection Camera", "Heart Rate Sensor"],
        })

    # 7. LOW ACTIVITY + POOR SLEEP → Weakness/Depression Risk
    if sleep and sleep < 50 and activity < 3:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Wellness Risk",
            "title": "Poor Sleep + Very Low Activity",
            "detail": f"Sleep score is {sleep}/100 (poor) with activity at {activity}/10 (very low). Pattern suggests fatigue, weakness, or possible depression.",
            "recommendation": "Encourage gentle movement. Assess sleep environment. Screen for depression. Review sleep medications.",
            "sensors": ["Sleep Sensor", "Activity Camera"],
        })

    # 8. ROOM TOO HOT/COLD + ELDERLY → Heat/Cold Stress
    if room_temp and age >= 60:
        if room_temp > THRESHOLDS["room_temp_c"]["critical_high"]:
            insights.append({
                "severity": SEVERITY_HIGH,
                "category": "Environmental Risk",
                "title": "Room Temperature Dangerously High",
                "detail": f"Room temperature is {room_temp}C ({round(room_temp * 9/5 + 32, 1)}F). Elderly patients are highly susceptible to heat stress and dehydration.",
                "recommendation": "Turn on AC/fan immediately. Ensure patient is hydrated. Move to cooler area if possible.",
                "sensors": ["Room Temperature Sensor"],
            })
        elif room_temp < THRESHOLDS["room_temp_c"]["critical_low"]:
            insights.append({
                "severity": SEVERITY_HIGH,
                "category": "Environmental Risk",
                "title": "Room Temperature Dangerously Low",
                "detail": f"Room temperature is {room_temp}C ({round(room_temp * 9/5 + 32, 1)}F). Risk of hypothermia in elderly patients.",
                "recommendation": "Turn on heating. Add blankets. Monitor patient body temperature closely.",
                "sensors": ["Room Temperature Sensor"],
            })

    # 9. HIGH HUMIDITY + RESPIRATORY → Breathing Difficulty
    if humidity and humidity > 70 and has_respiratory:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Environmental Risk",
            "title": "High Humidity + Respiratory Condition",
            "detail": f"Humidity is {humidity}% (high) for a patient with respiratory issues. High humidity worsens breathing difficulty.",
            "recommendation": "Use dehumidifier. Ensure ventilation. Monitor breathing patterns and SpO2.",
            "sensors": ["Humidity Sensor", "Medical History"],
        })

    # 10. ABNORMAL LABS + FALL HISTORY → Complicated Fall Risk
    if medical.get("abnormal_lab_count", 0) > 2 and fall_count > 0:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Compound Risk",
            "title": "Abnormal Labs + Recent Falls",
            "detail": f"{medical['abnormal_lab_count']} abnormal lab values combined with {fall_count} fall(s) in 7 days. Multiple systems showing stress.",
            "recommendation": "Comprehensive medical review needed. Consider inpatient evaluation. Increase monitoring frequency.",
            "sensors": ["Medical Reports", "Fall Detection Camera"],
        })

    # 11. LOW MOBILITY + HIGH BMI → Pressure Sore Risk
    if bmi and bmi > 30 and mobility < 3:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Mobility Risk",
            "title": "High BMI + Low Mobility",
            "detail": f"BMI is {bmi} (obese) with mobility score of {mobility}/10. High risk of pressure sores, DVT, and muscle atrophy.",
            "recommendation": "Reposition patient every 2 hours. Gentle range-of-motion exercises. Consider physiotherapy.",
            "sensors": ["BMI Data", "Activity Camera"],
        })

    # 12. MULTIPLE CONDITIONS + ELDERLY → Polypharmacy Risk
    if len(conditions) >= 3 and age >= 65:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Medication Risk",
            "title": "Multiple Conditions in Elderly Patient",
            "detail": f"{len(conditions)} active medical conditions in a {age}-year-old patient. High risk of medication interactions (polypharmacy).",
            "recommendation": "Review all medications for interactions. Consider geriatric medication optimization. Monitor for adverse drug reactions.",
            "sensors": ["Medical Reports", "Age Data"],
        })

    # 13. NO VITALS DATA → Monitoring Gap
    if not vitals.get("recorded_at"):
        insights.append({
            "severity": SEVERITY_LOW,
            "category": "Monitoring Gap",
            "title": "No Recent Vital Signs Data",
            "detail": "No vital signs have been recorded. Sensor connection may be lost or ESP32 device is offline.",
            "recommendation": "Check ESP32 device connection. Verify sensor wiring. Ensure WiFi connectivity.",
            "sensors": ["ESP32 Controller"],
        })

    # 14. TACHYCARDIA + LOW SpO2 → Cardiac Stress
    if hr and hr > 100 and spo2 and spo2 < 94:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Cardiac Risk",
            "title": "Tachycardia with Low Oxygen",
            "detail": f"Heart rate elevated at {hr} bpm with SpO2 at {spo2}%. Heart is compensating for low oxygen — possible cardiac or pulmonary event.",
            "recommendation": "URGENT: Monitor continuously. Prepare for emergency intervention. Alert emergency contacts.",
            "sensors": ["Heart Rate Sensor", "SpO2 Sensor"],
        })

    # 15. HIGH BP + DIABETES → Stroke Risk
    has_diabetes = any("diabetes" in c for c in conditions) or any("sugar" in c or "glucose" in c for c in conditions)
    if sbp and sbp > 150 and has_diabetes:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Stroke Risk",
            "title": "Hypertension + Diabetes Combination",
            "detail": f"Systolic BP is {sbp} mmHg with active diabetes. This combination dramatically increases stroke and cardiovascular event risk.",
            "recommendation": "Strict BP control needed. Review diabetic medication. Monitor for signs of stroke (slurred speech, weakness, confusion).",
            "sensors": ["Blood Pressure", "Medical Reports (Diabetes)"],
        })

    # Sort by severity
    severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MODERATE: 2, SEVERITY_LOW: 3, SEVERITY_INFO: 4}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 5))

    return insights


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-SENSOR INSIGHTS — Individual readings outside normal range
# ═══════════════════════════════════════════════════════════════════════

def generate_individual_insights(data: dict) -> list:
    """Generate insights from individual sensor readings that are outside normal ranges."""
    insights = []
    vitals = data.get("vitals", {})
    env = data.get("environment", {})
    audio = data.get("audio", {})

    # Heart Rate
    hr = vitals.get("heart_rate")
    if hr:
        if hr > THRESHOLDS["heart_rate"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critical Heart Rate", "detail": f"Heart rate is {hr} bpm — significantly elevated.", "sensors": ["Heart Rate"]})
        elif hr < THRESHOLDS["heart_rate"]["critical_low"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critically Low Heart Rate", "detail": f"Heart rate is {hr} bpm — bradycardia detected.", "sensors": ["Heart Rate"]})
        elif hr > THRESHOLDS["heart_rate"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Elevated Heart Rate", "detail": f"Heart rate is {hr} bpm — above normal range.", "sensors": ["Heart Rate"]})

    # Blood Pressure
    sbp = vitals.get("systolic_bp")
    if sbp:
        if sbp > THRESHOLDS["systolic_bp"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Hypertensive Crisis", "detail": f"Systolic BP is {sbp} mmHg — emergency level.", "sensors": ["Blood Pressure"]})
        elif sbp > THRESHOLDS["systolic_bp"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "High Blood Pressure", "detail": f"Systolic BP is {sbp} mmHg — elevated.", "sensors": ["Blood Pressure"]})
        elif sbp < THRESHOLDS["systolic_bp"]["critical_low"]:
            insights.append({"severity": SEVERITY_HIGH, "category": "Vitals", "title": "Hypotension", "detail": f"Systolic BP is {sbp} mmHg — dangerously low.", "sensors": ["Blood Pressure"]})

    # SpO2
    spo2 = vitals.get("oxygen_saturation")
    if spo2:
        if spo2 < THRESHOLDS["oxygen_saturation"]["critical_low"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critical Oxygen Level", "detail": f"SpO2 is {spo2}% — requires immediate attention.", "sensors": ["SpO2"]})
        elif spo2 < THRESHOLDS["oxygen_saturation"]["low"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Low Oxygen Saturation", "detail": f"SpO2 is {spo2}% — below optimal threshold.", "sensors": ["SpO2"]})

    # Temperature
    temp = vitals.get("temperature")
    if temp:
        if temp > THRESHOLDS["temperature_f"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "High Fever", "detail": f"Temperature is {temp}F — high fever. Investigate infection.", "sensors": ["Temperature"]})
        elif temp > THRESHOLDS["temperature_f"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Mild Fever", "detail": f"Temperature is {temp}F — above normal.", "sensors": ["Temperature"]})

    # Cough Frequency
    cough_avg = audio.get("cough_daily_avg", 0)
    if cough_avg >= THRESHOLDS["cough_daily"]["critical"]:
        insights.append({"severity": SEVERITY_HIGH, "category": "Respiratory", "title": "Very High Cough Frequency", "detail": f"Average {cough_avg} coughs/day over 7 days — investigate respiratory infection.", "sensors": ["Audio Monitor"]})
    elif cough_avg >= THRESHOLDS["cough_daily"]["elevated"]:
        insights.append({"severity": SEVERITY_LOW, "category": "Respiratory", "title": "Elevated Cough Frequency", "detail": f"Average {cough_avg} coughs/day — monitor for worsening.", "sensors": ["Audio Monitor"]})

    return insights


# ═══════════════════════════════════════════════════════════════════════
#  GEMINI AI — Deep clinical interpretation of fused data
# ═══════════════════════════════════════════════════════════════════════

def gemini_fused_analysis(data: dict, fusion_insights: list) -> dict:
    """
    Send ALL aggregated patient data + rule-based insights to Gemini
    for concise, alert-style clinical insights for the caregiver dashboard.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[insights_engine] GEMINI_API_KEY not set - returning rule-based insights only")
        return _fallback_summary(data, fusion_insights)

    # Build the Clinical Correlation Engine prompt
    recipient = data.get("recipient", {})
    conditions = [c["name"] for c in data.get("medical", {}).get("active_conditions", [])]

    prompt = f"""You are the Caretaker.ai Clinical Correlation Engine.

Your job is to analyze multimodal patient data and generate short actionable insights for the caregiver dashboard.

PATIENT PROFILE:
- Name: {recipient.get('name', 'Unknown')}
- Age: {recipient.get('age', 'Unknown')}
- Known Conditions: {', '.join(conditions) if conditions else 'None recorded'}
- Risk Score: {recipient.get('risk_score', 'N/A')}/100

CURRENT MULTIMODAL DATA:
{json.dumps(data, indent=2, default=str)}

RULE-BASED ALERTS (already generated by deterministic engine):
{json.dumps(fusion_insights, indent=2, default=str)}

YOUR RESPONSIBILITY:
Identify CORRELATIONS between data sources and convert them into clear caregiver insights.

Do NOT report raw sensor data.
Instead, analyze the RELATIONSHIPS between signals:
- Environment + Disease → respiratory risk
- Immobility + time duration → bed sore risk
- Heart rate + medical history → cardiac alert
- Cough + poor air quality → breathing risk
- Missed medication + disease → treatment risk

Combine: current signals + historical conditions + environmental context + medication adherence
to detect meaningful health correlations.

INSIGHT TYPES:
1. Critical Alert — Immediate health risk (e.g., "Fall detected", "Severe coughing detected")
2. Risk Warning — Condition that may worsen (e.g., "Poor air quality — asthma risk", "Prolonged immobility — bed sore risk")
3. Care Suggestion — Recommended action (e.g., "Encourage movement", "Check breathing")

OUTPUT RULES (STRICT):
- Maximum 10-15 words per insight
- Prefer short phrases
- Avoid long explanations
- Avoid repeating similar insights
- Prioritize clinically relevant correlations
- Every insight must answer: "What should the caregiver know right now?"

GOOD EXAMPLES:
"Fall detected"
"Immobile 3 hours"
"Poor air quality — asthma risk"
"Elevated heart rate"
"Missed BP medication"
"Encourage movement"
"Cough frequency increasing — check breathing"
"High humidity affecting respiratory condition"

BAD EXAMPLES (DO NOT DO THIS):
"The patient's heart rate is elevated at 102 bpm which could indicate..." (TOO LONG)
"Based on the data, we recommend..." (TOO VERBOSE)
"Heart rate: 64 bpm" (RAW DATA, NO CORRELATION)

OUTPUT FORMAT:
Return a JSON object with EXACTLY these keys:
{{
  "insights": [
    {{"text": "Short correlation insight.", "category": "Critical Alert", "priority": "critical"}},
    {{"text": "Risk warning here.", "category": "Risk Warning", "priority": "high"}},
    {{"text": "Care suggestion here.", "category": "Care Suggestion", "priority": "moderate"}}
  ],
  "risk_level": "low/moderate/high/critical",
  "overall_status": "One sentence max — patient status summary.",
  "immediate_actions": ["Short action 1", "Short action 2"],
  "monitoring_plan": "One sentence monitoring recommendation."
}}

Return ONLY valid JSON. No markdown, no code fences, no extra text."""

    try:
        result_data = call_gemini({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.2,
                "thinkingConfig": {"thinkingBudget": 1024}
            }
        }, timeout=90, caller="[insights_engine]")

        if result_data and "candidates" in result_data and result_data["candidates"]:
                parts = result_data["candidates"][0]["content"]["parts"]
                # Gemini 2.5 Flash with thinking: the actual text is in the LAST non-thought part
                raw = ""
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        raw = part["text"].strip()
                # Fallback: just use the last part's text if no non-thought part found
                if not raw:
                    raw = parts[-1].get("text", "").strip()
                
                print(f"[insights_engine] Gemini response: {len(raw)} chars")
                import re
                cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.DOTALL)
                cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()

                try:
                    parsed = json.loads(cleaned)
                    return _normalize_gemini_response(parsed)
                except json.JSONDecodeError as e:
                    print(f"[insights_engine] JSON parse error: {e}")
                    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                            return _normalize_gemini_response(parsed)
                        except json.JSONDecodeError:
                            pass
                    print(f"[insights_engine] Failed to parse Gemini response")

    except Exception as e:
        print(f"[insights_engine] Gemini call failed: {e}")

    return _fallback_summary(data, fusion_insights)


def _normalize_gemini_response(parsed: dict) -> dict:
    """Normalize Gemini's response to ensure all expected keys exist for the frontend."""
    insights = parsed.get("insights", [])

    # Build key_concerns from concise insights for backward compatibility with frontend
    key_concerns = []
    for item in insights:
        text = item if isinstance(item, str) else item.get("text", "")
        category = item.get("category", "General") if isinstance(item, dict) else "General"
        priority = item.get("priority", "moderate") if isinstance(item, dict) else "moderate"
        key_concerns.append({
            "concern": text,
            "explanation": text,
            "action": f"Actively tracking — {category}",
            "category": category,
            "priority": priority,
        })

    return {
        "overall_status": parsed.get("overall_status", "Analysis complete."),
        "risk_level": parsed.get("risk_level", "moderate"),
        "key_concerns": key_concerns,
        "insights": insights,
        "immediate_actions": parsed.get("immediate_actions", ["Continue routine monitoring"]),
        "monitoring_plan": parsed.get("monitoring_plan", "Continue current monitoring frequency."),
        "prognosis": parsed.get("prognosis", ""),
    }


def _fallback_summary(data: dict, insights: list) -> dict:
    """Deterministic fallback when Gemini is unavailable — produces concise alert-style output."""
    critical = [i for i in insights if i["severity"] == SEVERITY_CRITICAL]
    high = [i for i in insights if i["severity"] == SEVERITY_HIGH]

    if critical:
        level = "critical"
        status = f"CRITICAL: {len(critical)} critical issue(s) requiring immediate attention."
    elif high:
        level = "high"
        status = f"HIGH RISK: {len(high)} significant concern(s) identified."
    elif insights:
        level = "moderate"
        status = "Moderate concerns detected. Continue monitoring."
    else:
        level = "low"
        status = "All monitored parameters within acceptable ranges."

    # Build concise key_concerns from rule-based insights
    key_concerns = []
    for i in insights[:8]:
        key_concerns.append({
            "concern": i["title"],
            "explanation": i["detail"],
            "action": i.get("recommendation", "Monitor closely."),
            "category": i.get("category", "General"),
            "priority": i["severity"],
        })

    # Build concise insights list (alert-style)
    concise_insights = []
    for i in insights[:8]:
        concise_insights.append({
            "text": i["title"] + ".",
            "category": i.get("category", "General"),
            "priority": i["severity"],
        })

    return {
        "overall_status": status,
        "risk_level": level,
        "key_concerns": key_concerns,
        "insights": concise_insights,
        "immediate_actions": [i.get("recommendation", i["detail"]) for i in critical[:3]] if critical else ["Continue routine monitoring"],
        "monitoring_plan": "Rule-based analysis active. AI service unavailable.",
        "prognosis": "Unable to generate detailed prognosis (AI service unavailable). Rule-based analysis active.",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — Dual-Model Pipeline
# ═══════════════════════════════════════════════════════════════════════

async def run_recommendation_pipeline(
    patient_id: str,
    db: Session,
) -> list:
    """
    Full end-to-end pipeline with 3-tier routing (Lite → Flash/Pro).
    """
    pid = int(patient_id)
    print(f"[insights_engine] Running 3-tier recommendation pipeline for patient {pid}")
    
    # 1. Get Triggered Rules
    raw_rules = get_triggered_rules(pid, db)
    triggered_rules = group_triggered_rules(raw_rules)
    print(f"[insights_engine] Grouped {len(raw_rules)} rules into {len(triggered_rules)} groups.")

    # 2. Get Daily Usage & Target Model
    daily_usage = get_daily_usage(db)
    target_model = route_to_model(triggered_rules, daily_usage)
    
    # 3. Build Context
    context = build_context_payload(pid, db)
    # Add grouped rules to context so Gemini knows what triggered the alert
    context["triggered_groups"] = triggered_rules
    context_str = json.dumps(context, indent=2, default=str)
    
    all_generated = []
    
    # ── TIER 1: LITE (Pre-screener) ──────────────────────────────────────────
    try:
        lite_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": LITE_SYSTEM_PROMPT}]},
                {"role": "user", "parts": [{"text": f"PATIENT DATA:\n{context_str}"}]}
            ]
        }
        lite_resp = call_gemini(lite_payload, caller="[Lite]", model_override=LITE_MODEL)
        if lite_resp:
            from utils.gemini_client import safe_json_parse
            raw_lite = lite_resp["candidates"][0]["content"]["parts"][0]["text"]
            parsed_lite = safe_json_parse(raw_lite)
            
            if parsed_lite:
                # Store Lite nudges if any
                if parsed_lite.get("quick_nudges"):
                    lite_recs = merge_ai_output(parsed_lite, LITE_MODEL, pid, triggered_rules)
                    for rec in lite_recs:
                        # Convert ORM to dict for is_duplicate or just use attributes
                        rec_dict = {
                            "condition_group": rec.condition_group,
                            "urgency": rec.severity
                        }
                        if not is_duplicate(rec_dict, pid, db):
                            db.add(rec)
                            all_generated.append(rec)
                
                # Check if we should proceed to higher tier
                should_escalate = parsed_lite.get("needs_full_analysis", False)
                if not should_escalate and target_model == LITE_MODEL:
                    db.commit()
                    return all_generated
    except Exception as e:
        print(f"[insights_engine] Lite failed: {e}")

    # ── TIER 2 & 3: FLASH / PRO ──────────────────────────────────────────────
    if target_model in [FLASH_MODEL, PRO_MODEL]:
        if target_model == FLASH_MODEL and not should_run_flash(pid, db):
            print(f"[insights_engine] Skipping Flash for {pid} - already ran today.")
            db.commit()
            return all_generated

        current_prompt = PRO_SYSTEM_PROMPT if target_model == PRO_MODEL else FLASH_SYSTEM_PROMPT
        
        try:
            full_payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": current_prompt}]},
                    {"role": "user", "parts": [{"text": f"PATIENT DATA:\n{context_str}"}]}
                ],
                "generationConfig": {"temperature": 0.1}
            }
            
            resp = call_gemini(full_payload, caller=f"[{target_model}]", model_override=target_model)
            if resp:
                from utils.gemini_client import safe_json_parse
                raw_text = resp["candidates"][0]["content"]["parts"][0]["text"]
                parsed = safe_json_parse(raw_text)
                
                if parsed:
                    final_recs = merge_ai_output(parsed, target_model, pid, triggered_rules)
                    for rec in final_recs:
                        rec_dict = {
                            "condition_group": rec.condition_group,
                            "urgency": rec.severity
                        }
                        if not is_duplicate(rec_dict, pid, db):
                            db.add(rec)
                            all_generated.append(rec)
                    
        except Exception as e:
            print(f"[insights_engine] {target_model} failed: {e}")

    # ── FALLBACK: deterministic rule-engine cards ────────────────────────────
    # If Gemini is unavailable (bad/missing key, quota, network) or returned nothing,
    # turn the already-triggered clinical rules into recommendation cards so the
    # Clinical AI tab never sits empty while real abnormalities exist.
    if not all_generated and triggered_rules:
        try:
            fallback = build_rule_based_cards(triggered_rules)
            if fallback.get("cards"):
                for rec in merge_ai_output(fallback, "rule_engine", pid, triggered_rules):
                    rec.source = "rule_engine"
                    rec_dict = {"condition_group": rec.condition_group, "urgency": rec.severity}
                    if not is_duplicate(rec_dict, pid, db):
                        db.add(rec)
                        all_generated.append(rec)
                print(f"[insights_engine] Rule-engine fallback produced {len(all_generated)} card(s) for {pid}")
        except Exception as e:
            print(f"[insights_engine] Rule fallback failed: {e}")

    db.commit()
    return all_generated


_GROUP_LABELS = {
    "cardiovascular": "Heart & Blood Pressure",
    "diabetes": "Blood Sugar",
    "kidney": "Kidney Function",
    "liver": "Liver Function",
    "vitals": "Vital Signs",
    "environment": "Home Environment",
    "general": "General Health",
}


def _strip_disclaimer(msg: str) -> str:
    msg = (msg or "").strip()
    cut = msg.lower().find("please consult a qualified doctor")
    return msg[:cut].strip() if cut > 0 else msg


def build_rule_based_cards(triggered_groups: list) -> dict:
    """Convert grouped static-rule hits into the v3 'cards' schema used by merge_ai_output."""
    sev_to_urgency = {"critical": "act_now", "high": "today", "medium": "medium", "low": "suggestion", "suggestion": "suggestion"}
    sev_to_window = {"critical": "Now", "high": "Today", "medium": "This week", "low": "This month", "suggestion": "This month"}
    sev_to_next = {
        "critical": "Within the next few hours",
        "high": "Later today",
        "medium": "In 2–3 days",
        "low": "At the next routine check",
        "suggestion": "At the next routine check",
    }
    action_order = {"doctor_visit": 0, "medication": 1, "diet": 2, "lifestyle": 3, "monitor": 4, "home_remedy": 9}
    cards = []

    for g in triggered_groups:
        rules = sorted(g.get("rules") or [], key=lambda r: PRIORITY.get(r.get("severity", "low"), 0), reverse=True)
        if not rules:
            continue
        severity = g.get("severity") or rules[0].get("severity", "medium")
        group_key = str(g.get("metric", "general")).lower()
        label = _GROUP_LABELS.get(group_key, group_key.title())
        top = rules[0]

        findings = []
        for r in rules:
            val = r.get("value")
            ref = r.get("reference")
            ref_s = str(ref).strip() if ref is not None else ""
            if not ref_s or ref_s in ("N/A", "-", "None"):
                ref_txt = ""
            elif ref_s.startswith("-"):
                ref_txt = f" (normal below {ref_s.lstrip('-').replace('.0', '')})"
            else:
                ref_txt = f" (normal {ref_s})"
            findings.append(f"{r.get('metric')}: {val}{ref_txt}" if val is not None else str(r.get("metric")))

        actions, seen = [], set()
        for r in rules:
            for a in sorted(r.get("actions") or [], key=lambda a: action_order.get(a.get("type"), 5)):
                text = (a.get("text") or "").strip()
                if not text or text.lower() in seen:
                    continue
                if a.get("type") == "home_remedy":
                    text = f"{text} (traditional remedy — check with doctor first)"
                seen.add(text.lower())
                actions.append(text)
        doctor_first = next((t for t in actions if "doctor" in t.lower() or "physician" in t.lower()), None)

        names = ", ".join(dict.fromkeys(str(r.get("metric")) for r in rules))
        cards.append({
            "condition_group": label,
            "urgency": sev_to_urgency.get(severity, "medium"),
            "title": f"{_strip_disclaimer(top.get('message')).rstrip('.') or label + ' needs attention'}",
            "caregiver_message": " ".join(_strip_disclaimer(r.get("message")) for r in rules),
            "do_this_now": doctor_first or (actions[0] if actions else f"Review {names} with the doctor."),
            "today_actions": actions[:5],
            "why_this_matters": "Flagged from latest readings — " + "; ".join(findings),
            "time_window": sev_to_window.get(severity, "This week"),
            "call_doctor": severity in ("critical", "high"),
            "call_ambulance": False,
            "confidence": 0.9,
            "action": "consult" if severity in ("critical", "high") else "monitor",
            "next_check": {
                "when": sev_to_next.get(severity, "In 2–3 days"),
                "look_for": f"Updated {names} values and how they feel",
                "if_worse": "Call their doctor" if severity != "critical" else "Seek urgent medical care",
            },
        })

    return {"cards": cards, "overall_today": f"{len(cards)} health area(s) need attention based on recent readings."}


def get_active_recommendations(
    patient_id: str,
    db: Session,
    hours: int = 48,
) -> list:
    """
    Returns all unresolved recommendations for the patient from the last `hours` hours, 
    ordered by severity: critical → high → medium → suggestion.
    """
    pid = int(patient_id)
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
    
    # Fetch all non-archived recs
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == pid,
        MedicalRecommendation.created_at >= since,
        MedicalRecommendation.archived == False
    ).all()
    
    # Sort by SEVERITY_RANK (highest rank first)
    # severity_rank is imported from recommendation_engine (which gets it from medical_recommendations)
    recs.sort(key=lambda x: severity_rank(x.severity), reverse=True)
    
    return recs

