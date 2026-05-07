"""Recommendation Engine — Actionable clinical intelligence.

Evaluates longitudinal lab history against deterministic clinical rules.
Produces safe, structured recommendations including lifestyle, diet, and clinical actions.
"""

from typing import List, Dict, Any, Optional
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.medical_conditions import LabValue, PatientCondition
from tables.medical_recommendations import MedicalRecommendation, severity_rank, SEVERITY_RANK
from tables.vital_signs import VitalSign
from tables.audio_events import AudioEvent, AudioEventType
from tables.thresholds import ThresholdConfig
from tables.users import CareRecipient
from tables.medications import Medication
from tables.medication_dose_logs import MedicationDoseLog
from tables.environment import EnvironmentSensor
from tables.video_analysis import VideoAnalysis
from tables.conversation_history import ConversationMessage
from tables.allergies import Allergy

# ─── Model Constants ────────────────────────────────────────────────────────# 3-Tier AI constants
LITE_MODEL  = "gemini-2.5-flash-lite"  # Light triage (20 RPD)
FLASH_MODEL = "gemini-3.1-flash-lite"  # Primary reasoning model (500 RPD)
PRO_MODEL   = "gemini-2.5-flash"       # Deep clinical reasoning (fallback/critical)

# ─── v3 Pipeline Gates ──────────────────────────────────────────────────────
LEGACY_RULE_ENGINE_WRITES_ENABLED = False  # STOP direct DB writes from old rules

CONDITION_GROUPS = {
    "diabetes": {
        "rules": {"HIGH_HBA1C", "HIGH_FASTING_GLUCOSE", "HIGH_PP_GLUCOSE", "GLUCOSE_SPIKE"},
        "label": "Blood sugar management",
    },
    "cardiovascular": {
        "rules": {"HIGH_LDL", "LOW_HDL", "HIGH_TRIGLYCERIDES", "HIGH_BP_SYSTOLIC", "DIABETIC_DYSLIPIDEMIA", "TACHYCARDIA"},
        "label": "Heart health",
    },
    "kidney": {
        "rules": {"HIGH_CREATININE", "LOW_EGFR"},
        "label": "Kidney health",
    },
    "liver": {
        "rules": {"HIGH_ALT", "HIGH_AST"},
        "label": "Liver health",
    },
    "vitals": {
        "rules": {"LOW_SPO2", "FEVER", "BRADYCARDIA"},
        "label": "General vitals",
    },
    "environment": {
        "rules": {"HIGH_PM25", "HIGH_AQI", "TEMP_EXTREME"},
        "label": "Home environment",
    }
}

# Daily budget guard — never exceed these
DAILY_BUDGET = {
    LITE_MODEL:  18,    # leave 2 buffer from 20 RPD
    FLASH_MODEL: 490,   # leave 10 buffer from 500 RPD
    PRO_MODEL:   15,    # leave 5 buffer from 20 RPD
}

EMBEDDING_MODEL = "text-embedding-004"
DUPLICATE_SIMILARITY_THRESHOLD = 0.92

# ─── Cross-domain critical rules ────────────────────────────────────────────
CROSS_DOMAIN_CRITICAL_RULES = {
    "LOW_SPO2_HIGH_AQI",
    "FALL_PLUS_TACHYCARDIA",
    "HIGH_BP_LOW_ACTIVITY",
    "WORSENING_CREATININE_RISING_HR",
    "Diabetic Dyslipidemia", # From evaluate_combinations
    "FALL_PLUS_TACHYCARDIA",
    "LOW_SPO2_HIGH_AQI"
}

# ─── Clinical Rules ─────────────────────────────────────────────────────────
# Each rule maps an actionable condition to severity and concrete actions.

RULES = [
    {
        "metric": "HbA1c",
        "thresholds": [
            {"min": 8.0, "severity": "critical", "message": "Blood sugar control is significantly impaired.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult physician for medication review immediately."},
                 {"type": "test", "text": "Monitor daily blood glucose fasting and PP."}
             ]},
            {"min": 6.5, "severity": "high", "message": "Blood sugar levels indicate diabetes.",
             "actions": [
                 {"type": "doctor_visit", "text": "Discuss diabetes management plan with your doctor."},
                 {"type": "diet", "text": "Strictly reduce refined sugars and carbohydrates."},
                 {"type": "lifestyle", "text": "Increase aerobic physical activity."},
                 {"type": "home_remedy", "text": "Fenugreek (Methi) can complement medical treatment for sugar control.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 5.7, "severity": "medium", "message": "Blood sugar levels indicate prediabetes risk.",
             "actions": [
                 {"type": "diet", "text": "Reduce intake of high glycemic index foods."},
                 {"type": "lifestyle", "text": "Maintain a healthy weight."},
                 {"type": "home_remedy", "text": "Consume Fenugreek (Methi) seeds soaked in water to help lower blood sugar.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "Fasting Glucose",
        "thresholds": [
            {"min": 200, "severity": "critical", "message": "Extremely high fasting blood glucose.",
             "actions": [{"type": "doctor_visit", "text": "Seek immediate medical attention."}]},
            {"min": 126, "severity": "high", "message": "Fasting blood sugar is in the diabetic range.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult doctor for formal diagnosis."},
                 {"type": "home_remedy", "text": "Incorporate Cinnamon and Fenugreek into your daily routine.", "evidence": "traditional", "confidence": "low"}
             ]},
            {"min": 100, "severity": "medium", "message": "Fasting blood sugar is mildly elevated.",
             "actions": [
                 {"type": "diet", "text": "Monitor simple carbohydrate intake."},
                 {"type": "home_remedy", "text": "A pinch of Cinnamon in warm water can improve insulin sensitivity.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "LDL",
        "thresholds": [
            {"min": 190, "severity": "critical", "message": "LDL cholesterol is dangerously high.",
             "actions": [{"type": "doctor_visit", "text": "Consult cardiologist. Statin therapy may be required."}]},
            {"min": 160, "severity": "high", "message": "LDL cholesterol is very high.",
             "actions": [
                 {"type": "diet", "text": "Significantly reduce saturated and trans fats."},
                 {"type": "doctor_visit", "text": "Discuss lipid-lowering options with your doctor."},
                 {"type": "home_remedy", "text": "Garlic and Oats are highly effective at this stage for lipid management.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 130, "severity": "medium", "message": "LDL cholesterol is borderline high.",
             "actions": [
                 {"type": "diet", "text": "Increase soluble dietary fiber (oats, beans)."},
                 {"type": "lifestyle", "text": "Incorporate regular cardio exercise."},
                 {"type": "home_remedy", "text": "Consume 1 clove of raw Garlic daily to naturally lower cholesterol.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "HDL",
        "thresholds": [
            {"max": 35, "severity": "high", "message": "HDL (good) cholesterol is very low.",
             "actions": [{"type": "lifestyle", "text": "Quit smoking if applicable."}, {"type": "diet", "text": "Consume omega-3 rich foods (fish, walnuts)."}]},
            {"max": 40, "severity": "medium", "message": "HDL (good) cholesterol is below optimal.",
             "actions": [{"type": "lifestyle", "text": "Increase moderate aerobic exercise to boost HDL."}]},
        ]
    },
    {
        "metric": "Triglycerides",
        "thresholds": [
            {"min": 500, "severity": "critical", "message": "Triglycerides are extremely high (Pancreatitis risk).",
             "actions": [{"type": "doctor_visit", "text": "Seek immediate medical attention for triglyceride reduction."}]},
            {"min": 200, "severity": "high", "message": "Triglycerides are high.",
             "actions": [{"type": "diet", "text": "Eliminate alcohol and refined sugars."}, {"type": "test", "text": "Re-test lipid profile in 3 months."}]},
            {"min": 150, "severity": "medium", "message": "Triglycerides are borderline high.",
             "actions": [{"type": "diet", "text": "Limit sugary drinks and excess carbohydrates."}]},
        ]
    },
    {
        "metric": "Vitamin D",
        "thresholds": [
            {"max": 10, "severity": "high", "message": "Severe Vitamin D deficiency.",
             "actions": [{"type": "doctor_visit", "text": "Consult physician for prescription-strength Vitamin D therapy."}]},
            {"max": 20, "severity": "medium", "message": "Vitamin D deficiency.",
             "actions": [{"type": "diet", "text": "Consider OTC Vitamin D supplementation (consult pharmacist)."}, {"type": "lifestyle", "text": "Increase safe sunlight exposure."}]},
            {"max": 30, "severity": "low", "message": "Vitamin D insufficiency.",
             "actions": [{"type": "diet", "text": "Increase dietary intake (fortified dairy, fatty fish, eggs)."}]},
        ]
    },
    {
        "metric": "Vitamin B12",
        "thresholds": [
            {"max": 150, "severity": "high", "message": "Severe Vitamin B12 deficiency (Risk of neuropathy/anemia).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician for potential B12 injections."}]},
            {"max": 300, "severity": "medium", "message": "Suboptimal Vitamin B12 levels.",
             "actions": [{"type": "diet", "text": "Increase intake of meat, dairy, or B12 fortified foods."}, {"type": "diet", "text": "Consider B12 supplementation if vegetarian/vegan."}]},
        ]
    },
    {
        "metric": "Creatinine",
        "thresholds": [
            {"min": 2.5, "severity": "critical", "message": "Creatinine is dangerously elevated indicating renal injury.",
             "actions": [{"type": "doctor_visit", "text": "Immediate nephrology consultation required."}]},
            {"min": 1.4, "severity": "high", "message": "Creatinine is elevated (Kidney function impaired).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician immediately."}, {"type": "diet", "text": "Avoid NSAIDs (like ibuprofen) and excessive protein."}]},
        ]
    },
    {
        "metric": "eGFR",
        "thresholds": [
            {"max": 30, "severity": "critical", "message": "eGFR is critically low (Severe kidney disease).",
             "actions": [{"type": "doctor_visit", "text": "Urgent nephrology evaluation."}]},
            {"max": 60, "severity": "high", "message": "eGFR indicates moderate kidney disease.",
             "actions": [{"type": "doctor_visit", "text": "Consult your doctor to review all medications."}, {"type": "diet", "text": "Limit sodium and consult about potassium intake."}]},
        ]
    },
    {
        "metric": "Hemoglobin",
        "thresholds": [
            {"max": 7.0, "severity": "critical", "message": "Critically low Hemoglobin (Severe Anemia).",
             "actions": [{"type": "doctor_visit", "text": "Immediate medical attention required! Go to ER or consult doctor NOW."}]},
            {"max": 10.0, "severity": "high", "message": "Low Hemoglobin indicating moderate anemia.",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor to investigate cause of anemia."}]},
            {"max": 12.0, "severity": "medium", "message": "Hemoglobin is slightly below normal.",
             "actions": [{"type": "diet", "text": "Ensure adequate dietary iron, Vitamin C, and B12."}]},
        ]
    },
    {
        "metric": "Platelets",
        "thresholds": [
            {"max": 50, "severity": "critical", "message": "Critically low platelet count. High risk of bleeding.",
             "actions": [{"type": "doctor_visit", "text": "Seek urgent medical attention avoiding any physical trauma."}]},
            {"max": 100, "severity": "high", "message": "Low platelet count (Thrombocytopenia).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician immediately to determine root cause."}]},
            {"min": 600, "severity": "high", "message": "Elevated platelet count (Thrombocytosis).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor for further hematology workup."}]},
        ]
    },
    {
        "metric": "TSH",
        "thresholds": [
            {"min": 10.0, "severity": "high", "message": "TSH is significantly elevated (Hypothyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor for probable thyroid hormone replacement."}]},
            {"min": 5.0, "severity": "medium", "message": "TSH is mildly elevated (Subclinical hypothyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Monitor for symptoms like fatigue/weight gain and consult doctor."}]},
            {"max": 0.1, "severity": "high", "message": "TSH is highly suppressed (Hyperthyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor. Avoid excess iodine."}]},
        ]
    },
    {
        "metric": "ALT",
        "thresholds": [
            {"min": 200, "severity": "critical", "message": "ALT is extremely elevated (Acute liver injury warning).",
             "actions": [{"type": "doctor_visit", "text": "Seek urgent medical care and hepatology review."}]},
            {"min": 60, "severity": "high", "message": "ALT is elevated indicating liver inflammation.",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor."}, {"type": "lifestyle", "text": "Strictly avoid alcohol and unauthorized supplements."}]},
            {"min": 45, "severity": "medium", "message": "ALT is borderline high.",
             "actions": [{"type": "lifestyle", "text": "Limit alcohol consumption. Monitor weight."}]},
        ]
    },
    {
        "metric": "Systolic BP",
        "thresholds": [
            {"min": 180, "severity": "critical", "message": "Hypertensive Crisis detected.",
             "actions": [{"type": "doctor_visit", "text": "Seek emergency medical care immediately!"}]},
            {"min": 140, "severity": "high", "message": "Stage 2 Hypertension.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult doctor for medication management."},
                 {"type": "diet", "text": "Strictly restrict sodium intake."},
                 {"type": "home_remedy", "text": "Hibiscus tea and Garlic can provide additional support for BP reduction.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 130, "severity": "medium", "message": "Stage 1 Hypertension.",
             "actions": [
                 {"type": "lifestyle", "text": "Engage in regular aerobic exercise and manage stress."},
                 {"type": "home_remedy", "text": "Drink Hibiscus tea daily to help manage blood pressure.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "Diastolic BP",
        "thresholds": [
            {"min": 120, "severity": "critical", "message": "Extremely high diastolic blood pressure (Crisis).",
             "actions": [{"type": "doctor_visit", "text": "Seek emergency medical care immediately!"}]},
            {"min": 90, "severity": "high", "message": "Stage 2 Hypertension (Diastolic).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor for medication review."}, {"type": "diet", "text": "Strictly reduce sodium intake."}]},
            {"min": 80, "severity": "medium", "message": "Stage 1 Hypertension (Diastolic).",
             "actions": [{"type": "lifestyle", "text": "Monitor BP daily and manage weight."}]}
        ]
    },
    {
        "metric": "Heart Rate",
        "thresholds": [
            {"min": 120, "severity": "critical", "message": "Severe Tachycardia detected.",
             "actions": [{"type": "doctor_visit", "text": "Seek immediate medical attention if persistent."}]},
            {"min": 100, "severity": "high", "message": "High resting heart rate (Tachycardia).",
             "actions": [{"type": "lifestyle", "text": "Rest and monitor; consult doctor if persistent."}]},
            {"max": 50, "severity": "high", "message": "Low heart rate (Bradycardia).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician to rule out cardiac issues."}]}
        ]
    },
    {
        "metric": "SpO2",
        "thresholds": [
            {"max": 88, "severity": "critical", "message": "Critically low oxygen saturation.",
             "actions": [{"type": "doctor_visit", "text": "URGENT: Requires immediate oxygen evaluation."}]},
            {"max": 92, "severity": "high", "message": "Low oxygen saturation detected.",
             "actions": [{"type": "doctor_visit", "text": "Monitor respiratory health closely; consult doctor."}]},
            {"max": 94, "severity": "medium", "message": "Mildly low oxygen saturation.",
             "actions": [{"type": "lifestyle", "text": "Ensure proper ventilation and check if activity-induced."}]}
        ]
    },
]

# ─── Sensor Fusion Urgency Multipliers ───────────────────────────────────────
# We apply a priority multiplier to recommendations if sensor signals show 
# adverse trends.
URGENCY_CONFIG = {
    "vitals_deteriorating": 1.5,
    "high_fall_risk": 2.0,
    "respiratory_distress": 1.8,
    "low_activity": 1.3
}

# ─── Combination Rules ──────────────────────────────────────────────────────

def evaluate_combinations(latest_labs: Dict[str, Dict]) -> List[Dict]:
    combos = []
    
    # Combo 1: Diabetic Dyslipidemia (High HbA1c + High LDL/Triglycerides)
    hba1c = latest_labs.get("HbA1c")
    ldl = latest_labs.get("LDL")
    
    if hba1c and ldl and hba1c["value"] > 6.5 and ldl["value"] > 130:
        combos.append({
            "metric": "Diabetic Dyslipidemia",
            "severity": "high",
            "message": "Combined high blood sugar and high LDL cholesterol significantly increases cardiovascular risk.",
            "actions": [
                {"type": "doctor_visit", "text": "Consult cardiologist/endocrinologist for aggressive lipid management."},
                {"type": "diet", "text": "Adopt a strict heart-healthy, low-carb diet."}
            ],
            "value": None,
            "reference": None,
            "source": "hybrid"
        })
        
    return combos

# ─── Next Test Recommender ──────────────────────────────────────────────────

def evaluate_missing_tests(history: Dict[str, List[Dict]], now: datetime.datetime) -> List[Dict]:
    missing = []
    
    # Rule 1: If diabetic (HbA1c > 6.5 historically), should test every 6 months.
    hba1labs = history.get("HbA1c", [])
    if any(lab["value"] > 6.5 for lab in hba1labs):
        # find most recent
        if hba1labs:
            latest = max(hba1labs, key=lambda x: x["date"])
            latest_date = latest["date"]
            if isinstance(latest_date, datetime.date) and not isinstance(latest_date, datetime.datetime):
                latest_date = datetime.datetime.combine(latest_date, datetime.time.min)
                
            days_since = (now - latest_date).days
            if days_since > 180:
                missing.append({
                    "metric": "HbA1c",
                    "severity": "suggestion",
                    "message": f"It has been {days_since // 30} months since your last HbA1c test. Diabetic guidelines recommend testing every 3-6 months.",
                    "actions": [{"type": "test", "text": "Schedule an HbA1c test."}],
                    "value": None,
                    "reference": None,
                    "source": "rule"
                })
    
    return missing

# ─── Trend Detection ────────────────────────────────────────────────────────

def detect_trend(metric: str, labs: List[Dict]) -> Optional[str]:
    """Analyzes last 90 days of labs to detect worsening trends."""
    if len(labs) < 2:
        return None
        
    # Sort chronologically
    sorted_labs = sorted(labs, key=lambda x: x["date"])
    recent = [l for l in sorted_labs if (datetime.datetime.utcnow() - l["date"]).days <= 90]
    
    if len(recent) < 3:
        return None
        
    # Look at last 3 recent values
    last3 = recent[-3:]
    v1, v2, v3 = last3[0]["value"], last3[1]["value"], last3[2]["value"]
    
    # ── Trend Smoothing (Architect Rule #4) ──────────
    # Ignore tiny fluctuations (noise) < 5%
    avg = (abs(v1) + abs(v2) + abs(v3)) / 3
    if avg > 0:
        d1 = abs(v2 - v1) / avg
        d2 = abs(v3 - v2) / avg
        if d1 < 0.03 and d2 < 0.03: # Tight smoothing for individual steps
            return None
        # Overall trend delta
        total_delta = abs(v3 - v1) / avg
        if total_delta < 0.05:
            return None

    # "Up is bad" metrics
    if metric in ["HbA1c", "LDL", "Fasting Glucose", "Creatinine", "ALT", "TSH", "Systolic BP"]:
        if v3 > v2 and v2 > v1:
            return "increasing_bad"
            
    # "Down is bad" metrics
    if metric in ["eGFR", "HDL", "Vitamin D", "Vitamin B12", "Hemoglobin"]:
        if v3 < v2 and v2 < v1:
            return "decreasing_bad"
            
    return None

# ─── Safety Filter ──────────────────────────────────────────────────────────

BLOCKED_WORDS = ["prescribe", "dose", "dosage"]

def safety_filter(alerts: List[Dict]) -> List[Dict]:
    safe_alerts = []
    for a in alerts:
        msg = a["message"].lower()
        if any(w in msg for w in BLOCKED_WORDS):
            continue  # Drop unsafe system generation
            
        # Architect Rule #1: Sanitize medicine suggestions
        # Architect Rule #3: Hide home remedies in critical cases
        if a["severity"] == "critical":
            a["actions"] = [act for act in a["actions"] if act["type"] != "home_remedy"]
        
        # Guarantee safety disclaimer
        if not a["message"].endswith("."):
            a["message"] += "."
        
        # Proactively add 'Consult Doctor' if not present
        has_advice = any("doctor" in str(act).lower() or act["type"] == "doctor_visit" for act in a["actions"])
        if not has_advice:
            a["actions"].append({"type": "doctor_visit", "text": "Consult a doctor for clinical correlation."})
            
        a["message"] += " Please consult a qualified doctor before making health changes or taking new medications."
        
        safe_alerts.append(a)
    return safe_alerts

def route_to_model(triggered_rules: List[Dict], daily_usage: Dict = None) -> str:
    """
    Intelligently routes to Flash Lite (routine), Flash (high risk), or Pro (critical).
    Uses a fallback chain to respect daily quotas.
    """
    if daily_usage is None: daily_usage = {}
    
    severities = {r.get("severity", "low") for r in triggered_rules}
    rule_ids = {r.get("rule_id", "none") for r in triggered_rules}

    # 1. Critical path → Pro (if quota available)
    if "critical" in severities or (rule_ids & CROSS_DOMAIN_CRITICAL_RULES):
        if daily_usage.get(PRO_MODEL, 0) < DAILY_BUDGET[PRO_MODEL]:
            return PRO_MODEL
        # Pro budget exhausted — fall back to Flash with escalation flag
        return FLASH_MODEL

    # 2. High severity or multi-domain → Flash
    if "high" in severities or len(triggered_rules) >= 2:
        if daily_usage.get(FLASH_MODEL, 0) < DAILY_BUDGET[FLASH_MODEL]:
            return FLASH_MODEL
        # Flash budget exhausted — fall back to Lite
        return LITE_MODEL

    # 3. Everything else → Lite
    return LITE_MODEL


def get_daily_usage(db: Session) -> Dict[str, int]:
    """Calculate usage per model for the current day."""
    from sqlalchemy import func
    from datetime import datetime
    today = datetime.utcnow().date()
    
    usage = db.query(
        MedicalRecommendation.model_used,
        func.count(MedicalRecommendation.id)
    ).filter(
        func.date(MedicalRecommendation.created_at) == today
    ).group_by(MedicalRecommendation.model_used).all()
    
    return {model: count for model, count in usage}


def should_run_flash(patient_id: int, db: Session) -> bool:
    """
    Returns False if Flash already ran for this patient today
    at the same or higher severity — prevents burning daily quota.
    """
    from sqlalchemy import func
    from datetime import datetime
    today = datetime.utcnow().date()
    
    existing = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == patient_id,
        MedicalRecommendation.model_used == FLASH_MODEL,
        func.date(MedicalRecommendation.created_at) == today,
    ).count()
    
    return existing == 0 


def deduplicate(new_recs: List[MedicalRecommendation], patient_id: int, db: Session) -> List[MedicalRecommendation]:
    """
    Uses Semantic Similarity (Embeddings) to find duplicates from the last 24h.
    Falls back to string match if embedding service fails.
    """
    from datetime import datetime, timedelta
    from utils.gemini_client import get_embedding
    import numpy as np

    since = datetime.utcnow() - timedelta(hours=24)
    existing = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == patient_id,
        MedicalRecommendation.created_at >= since
    ).all()
    
    if not existing:
        return new_recs

    # Get embeddings for existing messages
    existing_embeddings = []
    for r in existing:
        msg = r.caregiver_message or r.message
        emb = get_embedding(msg, model=EMBEDDING_MODEL)
        if emb:
            existing_embeddings.append((r, np.array(emb)))

    kept = []
    for n in new_recs:
        new_msg = n.caregiver_message or n.message
        new_emb = np.array(get_embedding(new_msg, model=EMBEDDING_MODEL)) if existing_embeddings else None
        
        is_duplicate = False
        
        if new_emb is not None:
            # Vector comparison
            for old_rec, old_emb in existing_embeddings:
                similarity = np.dot(new_emb, old_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(old_emb))
                if similarity > DUPLICATE_SIMILARITY_THRESHOLD:
                    # It's a semantic duplicate. 
                    # Only keep if the NEW one is more severe.
                    if PRIORITY.get(n.severity, 0) > PRIORITY.get(old_rec.severity, 0):
                        # Escalation! We actually want to replace/add the new one
                        is_duplicate = False
                        break
                    is_duplicate = True
                    break
        else:
            # Fallback to fuzzy string match
            for r in existing:
                if n.message[:50] == r.message[:50]:
                    if PRIORITY.get(n.severity, 0) <= PRIORITY.get(r.severity, 0):
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            kept.append(n)
            
    return kept

# ─── Prioritization & Deduplication ─────────────────────────────────────────

PRIORITY = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "suggestion": 1,
    "low": 0
}

def is_duplicate(new_rec: Dict[str, Any], patient_id: int, db: Session) -> bool:
    """
    One active card per condition_group per patient per 8-hour window.
    Only allow through if severity has escalated.
    """
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=8)

    # Step 1: exact match check (fast)
    exact = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == patient_id,
        MedicalRecommendation.condition_group == new_rec.get("condition_group"),
        MedicalRecommendation.resolved_at == None,
        MedicalRecommendation.created_at >= cutoff,
    ).first()

    if exact:
        # Only allow through if severity has escalated
        current_rank = SEVERITY_RANK.get(exact.severity, 0)
        # Handle both 'urgency' from AI and 'severity' from rule engine
        new_sev = new_rec.get("urgency") or new_rec.get("severity", "suggestion")
        new_rank = SEVERITY_RANK.get(new_sev, 0)
        if new_rank <= current_rank:
            return True  # duplicate, suppress it

    return False  # new, allow it


def group_triggered_rules(rules: List[Dict]) -> List[Dict]:
    """
    Groups individual triggered rules into thematic condition groups.
    Example: [Glucose Alert, HbA1c Alert] -> [Diabetes Group]
    """
    groups = {
        "diabetes": ["glucose", "hba1c", "blood sugar", "insulin"],
        "cardiovascular": ["blood pressure", "systolic", "diastolic", "heart rate", "hr", "cholesterol", "ldl", "hdl"],
        "kidney": ["creatinine", "egfr", "kidney"],
        "liver": ["alt", "ast", "liver"],
        "vitals": ["oxygen", "spo2", "temperature", "temp", "respiration"],
        "environment": ["aqi", "humidity", "room temp"],
        "general": []
    }
    
    grouped = {}
    
    for r in rules:
        metric = str(r.get("metric", "")).lower()
        found_group = "general"
        for g_name, keywords in groups.items():
            if any(k in metric for k in keywords):
                found_group = g_name
                break
        
        if found_group not in grouped:
            grouped[found_group] = {
                "metric": found_group.capitalize(),
                "severity": r.get("severity", "low"),
                "description": r.get("description", ""),
                "rules": [r]
            }
        else:
            # Update severity to highest in group
            if PRIORITY.get(r.get("severity", "low"), 0) > PRIORITY.get(grouped[found_group]["severity"], 0):
                grouped[found_group]["severity"] = r["severity"]
            
            # Append to description if unique
            if r.get("description") and r["description"] not in grouped[found_group]["description"]:
                grouped[found_group]["description"] += " | " + r["description"]
            
            grouped[found_group]["rules"].append(r)
            
    return list(grouped.values())


def fetch_sensor_fusion_context(recipient_id: int, db: Session, days: int = 7) -> Dict[str, Any]:
    """Fetches trends from vitals and audio events for fusion."""
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    
    # 1. Fetch Vitals
    vitals = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id,
        VitalSign.recorded_at >= since
    ).order_by(VitalSign.recorded_at.asc()).all()
    
    # 2. Fetch Audio Events (Coughs, Sneezes)
    audio = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.detected_at >= since
    ).all()
    
    # Simple slope calculation for HR and SpO2 with None-checks (Safety Pattern #5)
    hr_values = [v.heart_rate for v in vitals if getattr(v, "heart_rate", None) is not None]
    spo2_values = [v.oxygen_saturation for v in vitals if getattr(v, "oxygen_saturation", None) is not None]
    
    hr_trend = "stable"
    if len(hr_values) > 3:
        if hr_values[-1] > hr_values[0] * 1.1: hr_trend = "rising"
        elif hr_values[-1] < hr_values[0] * 0.9: hr_trend = "falling"

    spo2_trend = "stable"
    if len(spo2_values) > 3:
        if spo2_values[-1] < spo2_values[0] * 0.95: spo2_trend = "falling"

    # Audio event safety check
    respiratory_count = 0
    for a in audio:
        e_type = str(getattr(a, "event_type", "")).lower() # Safety Pattern #4
        if "cough" in e_type or "sneeze" in e_type:
            respiratory_count += 1

    context = {
        "vitals_count": len(vitals),
        "hr_trend": hr_trend,
        "spo2_trend": spo2_trend,
        "audio_event_count": len(audio),
        "is_respiratory_active": respiratory_count > 5
    }
    
    # Debug telemetry (Safety Pattern 🧪 Step 3)
    print(f"[sensor_fusion] Context built: HR={hr_trend}, SpO2={spo2_trend}, RespEvents={respiratory_count}")
    return context


def lookup_dynamic_threshold(metric: str, recipient: CareRecipient, db: Session) -> Optional[List[Dict]]:
    """Look up patient-specific thresholds from the DB based on age and comorbidities."""
    if not recipient:
        return None
        
    age = recipient.age or 65
    configs = db.query(ThresholdConfig).filter(
        ThresholdConfig.metric == metric,
        ThresholdConfig.age_min <= age,
        ThresholdConfig.age_max >= age
    ).all()
    
    # FUTURE: filter by comorbidity_tag if recipient has conditions
    
    if not configs:
        return None
        
    return [
        {
            "min": getattr(c, "min_value", -float('inf')) if getattr(c, "min_value", None) is not None else -float('inf'),
            "max": getattr(c, "max_value", float('inf')) if getattr(c, "max_value", None) is not None else float('inf'),
            "severity": getattr(c, "severity", "medium"),
            "message": getattr(c, "message_template", "Abnormal value detected"),
            "actions": getattr(c, "actions", []) or [],
            "action_payload": getattr(c, "action_payload", None)
        }
        for c in configs
    ]


def get_state_of_health(recipient_id: int, db: Session) -> Dict[str, Any]:
    """Computes a high-level health category (Architect Rule #11)."""
    # Fetch recent recommendations
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id
    ).order_by(desc(MedicalRecommendation.created_at)).limit(10).all()
    
    if not recs:
        return {"category": "Stable", "color": "var(--success)", "icon": "fa-check-circle"}
        
    highest_sev = "low"
    for r in recs:
        r_sev = getattr(r, "severity", "low").lower()
        if PRIORITY.get(r_sev, 0) > PRIORITY.get(highest_sev, 0):
            highest_sev = r_sev
            
    if highest_sev == "critical":
        return {"category": "Critical Risk", "color": "var(--danger)", "icon": "fa-exclamation-triangle", "label": "Immediate attention required"}
    if highest_sev == "high":
        return {"category": "High Concern", "color": "#f97316", "icon": "fa-bolt", "label": "Rising health risks detected"}
    if highest_sev == "medium":
        return {"category": "Moderate Risk", "color": "var(--warning)", "icon": "fa-info-circle", "label": "Health monitoring advised"}
        
    return {"category": "Good", "color": "var(--success)", "icon": "fa-check-circle", "label": "Maintain current lifestyle"}


# ─── Context Builder & Output Merging ─────────────────────────────────────────

def build_context_payload(patient_id: int, db: Session) -> Dict[str, Any]:
    """
    Assembles the full structured JSON payload sent to Gemini.
    Pulls data from Patient Profile, Vitals, Trends, Environment, Labs, Medications, and Behavioral sources.
    """
    now = datetime.datetime.utcnow()
    recipient = db.query(CareRecipient).filter(CareRecipient.id == patient_id).first()
    if not recipient:
        return {"error": "Patient not found"}

    # 1. Profile
    profile = {
        "age": recipient.age,
        "gender": recipient.gender.value if recipient.gender else "unknown",
        "conditions": [c.disease_name for c in db.query(PatientCondition).filter(PatientCondition.care_recipient_id == patient_id).all()],
        "allergies": [a.allergen for a in db.query(Allergy).filter(Allergy.care_recipient_id == patient_id).all()]
    }

    # 2. Vitals & Trends (last 7 days)
    vitals_raw = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == patient_id,
        VitalSign.recorded_at >= now - datetime.timedelta(days=7)
    ).order_by(VitalSign.recorded_at.desc()).all()

    latest_vitals = {}
    trends = {}
    
    VITAL_ATTRS = ["heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation", "temperature", "sleep_score", "bmi"]
    
    if vitals_raw:
        v0 = vitals_raw[0]
        latest_vitals = {attr: getattr(v0, attr) for attr in VITAL_ATTRS}
        
        for attr in VITAL_ATTRS:
            readings = [getattr(v, attr) for v in vitals_raw if getattr(v, attr) is not None][:7]
            if len(readings) < 2:
                trends[attr] = "stable"
                continue
            
            # Simple trend detection
            first = readings[-1]
            last = readings[0]
            
            diff = last - first
            if abs(diff) < (first * 0.05 if first != 0 else 1):
                trends[attr] = "stable"
            elif diff > 0:
                trends[attr] = "rising"
            else:
                trends[attr] = "declining"
            
            # Check for high fluctuations
            if len(readings) >= 3:
                variance = sum((r - sum(readings)/len(readings))**2 for r in readings) / len(readings)
                if variance > (sum(readings)/len(readings) * 0.2)**2:
                    trends[attr] = "fluctuating"

    # 3. Environment
    latest_env_obj = db.query(EnvironmentSensor).filter(EnvironmentSensor.care_recipient_id == patient_id).order_by(EnvironmentSensor.timestamp.desc()).first()
    environment = {
        "aqi": getattr(latest_env_obj, "aqi", None),
        "pm25": getattr(latest_env_obj, "pm25", None) if hasattr(latest_env_obj, "pm25") else None,
        "room_temperature": getattr(latest_env_obj, "temperature_c", None),
        "humidity": getattr(latest_env_obj, "humidity_percent", None)
    }

    # 4. Lab Results
    all_labs = db.query(LabValue).filter(LabValue.care_recipient_id == patient_id).order_by(LabValue.recorded_date.desc()).all()
    labs = {}
    for lab in all_labs:
        if lab.metric_name not in labs:
            # Find previous value for this metric
            prev = next((l for l in all_labs if l.metric_name == lab.metric_name and l.recorded_date < lab.recorded_date), None)
            labs[lab.metric_name] = {
                "current": lab.normalized_value or lab.metric_value,
                "unit": lab.normalized_unit or lab.unit,
                "previous": prev.normalized_value or prev.metric_value if prev else None,
                "date": lab.recorded_date.isoformat()
            }

    # 5. Medications & Adherence
    meds_objs = db.query(Medication).filter(Medication.care_recipient_id == patient_id).all()
    medications = []
    for m in meds_objs:
        # Calculate 7-day adherence
        logs = db.query(MedicationDoseLog).filter(
            MedicationDoseLog.medication_id == m.medication_id,
            MedicationDoseLog.scheduled_time >= now - datetime.timedelta(days=7)
        ).all()
        
        taken = sum(1 for l in logs if l.status == "TAKEN")
        total = len(logs)
        adherence = round(taken / total, 2) if total > 0 else 1.0
        
        medications.append({
            "name": m.medicine_name,
            "dose": m.dosage,
            "frequency": m.frequency,
            "adherence_rate_7d": adherence
        })

    # 6. Behavioral
    latest_video = db.query(VideoAnalysis).filter(VideoAnalysis.recipient_id == patient_id).order_by(VideoAnalysis.timestamp.desc()).first()
    recent_falls = db.query(VideoAnalysis).filter(
        VideoAnalysis.recipient_id == patient_id,
        VideoAnalysis.timestamp >= now - datetime.timedelta(days=7),
        VideoAnalysis.has_fall == True
    ).count()
    
    recent_audio = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == patient_id,
        AudioEvent.detected_at >= now - datetime.timedelta(days=1),
        AudioEvent.event_type == AudioEventType.cough
    ).count()

    behavioral = {
        "activity_score": getattr(latest_video, "activity_score", 0),
        "mobility_score": getattr(latest_video, "mobility_score", 0),
        "fall_events_7d": recent_falls,
        "cough_events_24h": recent_audio
    }

    # 7. Conversation
    latest_msg = db.query(ConversationMessage).filter(ConversationMessage.care_recipient_id == patient_id).order_by(ConversationMessage.created_at.desc()).first()
    conversation = {
        "latest_sentiment": latest_msg.mood_detected.value if latest_msg and latest_msg.mood_detected else "neutral",
        "intent": latest_msg.trigger_type.value if latest_msg and latest_msg.trigger_type else "unknown"
    }

    return {
        "patient_profile": profile,
        "vitals": latest_vitals,
        "vital_trends": trends,
        "environment": environment,
        "labs": labs,
        "medications": medications,
        "behavioral": behavioral,
        "conversation": conversation,
        "timestamp": now.isoformat()
    }


def merge_ai_output(
    raw_json: Dict[str, Any],
    model_used: str,
    patient_id: int,
    triggered_rules: List[Dict],
    escalated_from_id: Optional[int] = None,
) -> List[MedicalRecommendation]:
    """
    Normalizes Gemini output into MedicalRecommendation ORM objects.
    Supports v3 schema (cards) and legacy schema (recommendations).
    """
    recommendations = []
    
    # Determine highest severity rule for trigger_context
    highest_rule = None
    if triggered_rules:
        highest_rule = max(triggered_rules, key=lambda x: PRIORITY.get(x.get("severity", "low"), 0))
    
    trigger_context = highest_rule.get("description") if highest_rule else "Multimodal AI Analysis"
    
    # Support both v3 "cards" and legacy "recommendations" keys
    recs_list = raw_json.get("cards") or raw_json.get("recommendations", [])
    if not isinstance(recs_list, list):
        recs_list = [recs_list]
        
    for rec in recs_list:
        msg = rec.get("caregiver_message", "")
        # Enforce clinical disclaimer
        disclaimer = "Please consult a doctor before making any health changes."
        if disclaimer.lower() not in msg.lower():
            msg = msg.rstrip(".") + f". {disclaimer}"
            
        # Extract next_check fields with sensible defaults
        next_check = rec.get("next_check") or {}
        if not isinstance(next_check, dict): next_check = {}
        
        # Urgency mapping
        urgency = rec.get("urgency", "today")
        severity = urgency.replace("act_now", "critical").replace("today", "high")
        if severity not in SEVERITY_RANK:
            severity = "medium"

        orm_rec = MedicalRecommendation(
            care_recipient_id=patient_id,
            metric=rec.get("condition_group", "Multimodal Insight"),
            severity=severity,
            message=msg,
            caregiver_message=msg,
            do_this_now=rec.get("do_this_now") or "",
            action_type=rec.get("action", "monitor"),
            confidence_score=rec.get("confidence", 0.7),
            reasoning=rec.get("reasoning") if model_used == PRO_MODEL else None,
            call_doctor=str(rec.get("call_doctor", "false")).lower(),
            call_ambulance=str(rec.get("call_ambulance", "false")).lower(),
            health_state=raw_json.get("health_state") or raw_json.get("overall_today"),
            model_used=model_used,
            escalated_from=escalated_from_id,
            trigger_context=trigger_context,
            source=model_used,
            
            # v3 Enhanced Fields
            title=rec.get("title") or "Health Update",
            condition_group=rec.get("condition_group") or "general",
            today_actions=rec.get("today_actions") or [],
            why_this_matters=rec.get("why_this_matters") or "",
            time_window=rec.get("time_window") or urgency,
            root_cause=rec.get("root_cause"),
            snapshot_summary=raw_json.get("overall_today"),
            
            # Loop Closure
            next_check_when=next_check.get("when") or "Check back this evening",
            next_check_look_for=next_check.get("look_for") or "How they are feeling overall",
            next_check_if_worse=next_check.get("if_worse") or "Call their doctor",
            
            created_at=datetime.datetime.utcnow()
        )
        recommendations.append(orm_rec)
        
    return recommendations


# ─── Main Execution Pipeline ──────────────────────────────────────────────────

def generate_recommendations(recipient_id: int, db: Session):
    now = datetime.datetime.utcnow()
    
    # 0. Fetch Context & Recipient Profile
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    sensor_context = fetch_sensor_fusion_context(recipient_id, db)
    
    # 1. Fetch entire lab history + Recent Vitals
    all_labs = db.query(LabValue).filter(LabValue.care_recipient_id == recipient_id).all()
    all_vitals = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id).order_by(VitalSign.recorded_at.desc()).limit(100).all()
    
    # Group by metric
    history = {}
    latest_labs = {} # Using this for both Labs and Vitals for unified threshold evaluation
    
    # A. Process Labs
    for lab in all_labs:
        name = lab.metric_name
        if name not in history:
            history[name] = []
        # Normalize date to datetime for calculations
        recorded_at = lab.recorded_date or lab.created_at
        if isinstance(recorded_at, datetime.date) and not isinstance(recorded_at, datetime.datetime):
            recorded_at = datetime.datetime.combine(recorded_at, datetime.time.min)
 
        entry = {
            "value": lab.normalized_value or lab.metric_value,
            "unit": lab.normalized_unit or lab.unit,
            "date": recorded_at,
            "ref": f"{lab.reference_range_low or ''}-{lab.reference_range_high or ''}"
        }
        history[name].append(entry)
        
        if name not in latest_labs or entry["date"] >= latest_labs[name]["date"]:
            latest_labs[name] = entry
            
    # B. Process Vitals
    VITAL_MAP = {
        "measured_hr": "Heart Rate",
        "measured_spo2": "SpO2",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "measured_temp": "Temperature"
    }

    for v in all_vitals:
        for attr, metric_name in VITAL_MAP.items():
            val = getattr(v, attr, None)
            if val is not None:
                if metric_name not in history: history[metric_name] = []
                entry = {
                    "value": val,
                    "unit": "",
                    "date": v.recorded_at,
                    "ref": "N/A"
                }
                history[metric_name].append(entry)
                if metric_name not in latest_labs or entry["date"] >= latest_labs[metric_name]["date"]:
                    latest_labs[metric_name] = entry
 
    alerts = []
 
    # 2. Evaluate Clinical Thresholds (Dynamic then Static)
    for name, latest in latest_labs.items():
        val = latest["value"]
        
        # Priority 1: Dynamic DB Thresholds
        thresholds = lookup_dynamic_threshold(name, recipient, db)
        is_dynamic = True
        
        # Priority 2: Static Fallback rules
        if not thresholds:
            is_dynamic = False
            for rule in RULES:
                if rule["metric"] == name:
                    thresholds = rule["thresholds"]
                    break
                    
        if thresholds:
            for t in thresholds:
                hit = False
                if "min" in t and val >= t["min"]: hit = True
                if "max" in t and val <= t["max"]: hit = True
                
                if hit:
                    # Sensor Fusion Multiplier
                    severity = t.get("severity", "medium")
                    prio_score = PRIORITY.get(severity, 0)
                    
                    if sensor_context.get("hr_trend") == "rising" and name in ["Systolic BP", "HbA1c"]:
                        prio_score *= URGENCY_CONFIG.get("vitals_deteriorating", 1.2)
                        print(f"[recommendation_engine] URGENCY BOOST: {name} severity boosted due to rising HR trend")
                    if sensor_context.get("is_respiratory_active") and name == "Hemoglobin":
                        prio_score *= URGENCY_CONFIG.get("respiratory_distress", 1.3)
                        print(f"[recommendation_engine] URGENCY BOOST: {name} severity boosted due to respiratory events")
                        
                    # Re-map back to severity string if boosted
                    if prio_score >= 4: severity = "critical"
                    elif prio_score >= 3: severity = "high"
                    
                    alert = {
                        "metric": name,
                        "severity": severity,
                        "message": t.get("message") or t.get("message_template", ""),
                        "actions": t.get("actions", []),
                        "action_payload": t.get("action_payload"),
                        "value": val,
                        "reference": latest.get("ref", "N/A"),
                        "source": "dynamic_config" if is_dynamic else "static_rule"
                    }
                    
                    # Trend inspection
                    trend = detect_trend(name, history[name])
                    if trend:
                        alert["message"] += f" (Condition shows worsening trend over 90 days)"
                        alert["source"] = "trend"
                        
                    alerts.append(alert)
                    break 

    # 3. Evaluate Combo Rules
    alerts.extend(evaluate_combinations(latest_labs))
    
    # 4. Evaluate Missing Tests
    alerts.extend(evaluate_missing_tests(history, now))
 
    # 5. Safety Filter
    safe_alerts = safety_filter(alerts)
    
    # 6. Return raw rules (deduping happens in AI pipeline)
    return safe_alerts
    
    # 7. Return triggered rules instead of writing directly to DB
    return safe_alerts

def get_triggered_rules(recipient_id: int, db: Session) -> List[Dict]:
    """Helper to call generate_recommendations and get raw rules."""
    return generate_recommendations(recipient_id, db)
