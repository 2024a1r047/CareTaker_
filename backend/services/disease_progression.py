"""Disease Progression Engine — Core clinical intelligence.

Compares reports longitudinally. Detects trends, updates condition statuses,
generates clinical interpretations, calculates volatility, and manages
condition lifecycle (active → resolved → past).
"""

import datetime
import math
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.medical_conditions import (
    PatientCondition, ConditionHistory, LabValue,
    ConditionStatus, ConditionSeverity
)
from tables.disease_dictionary import DiseaseDictionary
from services.disease_detection import normalize_lab_value, STANDARD_UNITS


# ---------- Unit Normalization Wrapper ----------

def _normalize_and_store_lab(metric_name, raw_value, raw_unit):
    """Normalize a lab value and return (normalized_value, normalized_unit)."""
    norm_val, norm_unit = normalize_lab_value(raw_value, raw_unit, metric_name)
    return norm_val, norm_unit


# ---------- Volatility Calculation ----------

def calculate_volatility(values: list) -> float:
    """Calculate coefficient of variation (std_dev / mean) for a series of values.

    Returns value between 0.0 (stable) and 1.0+ (highly volatile).
    Uses last 5 readings max.
    """
    recent = values[-5:] if len(values) > 5 else values
    if len(recent) < 2:
        return 0.0

    mean = sum(recent) / len(recent)
    if mean == 0:
        return 0.0

    variance = sum((v - mean) ** 2 for v in recent) / len(recent)
    std_dev = math.sqrt(variance)
    return round(std_dev / abs(mean), 4)


def volatility_label(cv: float) -> str:
    """Human-readable volatility label."""
    if cv < 0.05:
        return "Low"
    elif cv < 0.15:
        return "Medium"
    else:
        return "High"


# ---------- Trend Detection ----------

def detect_trend(values: list) -> str:
    """Detect trend direction from a series of values.

    Returns: 'increasing', 'decreasing', 'stable', or 'fluctuating'
    """
    if len(values) < 2:
        return "stable"

    recent = values[-5:] if len(values) > 5 else values
    diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]

    if not diffs:
        return "stable"

    # Introduction of a noise threshold (e.g., 2% of the average value)
    avg_val = sum(recent) / len(recent)
    noise_threshold = abs(avg_val) * 0.02 if avg_val != 0 else 0.1

    # Filter significant diffs
    sig_diffs = [d for d in diffs if abs(d) > noise_threshold]
    
    if not sig_diffs:
        # Check if there is a slow but steady trend even if each step is small
        total_diff = recent[-1] - recent[0]
        if abs(total_diff) > noise_threshold * 2:
            return "increasing" if total_diff > 0 else "decreasing"
        return "stable"

    pos = sum(1 for d in sig_diffs if d > 0)
    neg = sum(1 for d in sig_diffs if d < 0)
    total = len(sig_diffs)

    if pos == total:
        return "increasing"
    elif neg == total:
        return "decreasing"
    elif pos >= total * 0.7:
        return "increasing"
    elif neg >= total * 0.7:
        return "decreasing"

    # Check for fluctuation (alternating significant directions)
    alternations = sum(1 for i in range(len(sig_diffs)-1) if (sig_diffs[i] > 0) != (sig_diffs[i+1] > 0))
    if alternations >= 1: # Any significant reversal in a small window is a fluctuation
        return "fluctuating"

    return "stable"


# ---------- Status Determination ----------

def _determine_status_from_rules(disease_code: str, current_labs: dict, db: Session) -> tuple:
    """Determine condition status using disease-specific rules from the dictionary.

    Args:
        disease_code: ICD code
        current_labs: {metric_name: normalized_value}
        db: database session

    Returns:
        (new_status, new_severity) or (None, None) if no rules matched
    """
    disease = db.query(DiseaseDictionary).filter(
        DiseaseDictionary.code == disease_code
    ).first()

    if not disease or not disease.status_rules:
        return None, None

    rules = disease.status_rules
    monitoring_metrics = disease.monitoring_metrics or []

    # Check which metrics are available
    available = {m: current_labs[m] for m in monitoring_metrics if m in current_labs}
    if not available:
        return None, None

    # Check rules in order of severity: worsening → moderate → controlled → resolved
    for status_key in ["worsening", "moderate", "controlled", "resolved"]:
        if status_key not in rules:
            continue
        rule = rules[status_key]

        # Skip "consecutive_normal" checks here (handled separately)
        metric_rules = {k: v for k, v in rule.items() if k != "consecutive_normal"}

        all_match = True
        for metric, conditions in metric_rules.items():
            if metric not in available:
                all_match = False
                break
            value = available[metric]
            for op, threshold in conditions.items():
                if op == ">" and not (value > threshold):
                    all_match = False
                elif op == ">=" and not (value >= threshold):
                    all_match = False
                elif op == "<" and not (value < threshold):
                    all_match = False
                elif op == "<=" and not (value <= threshold):
                    all_match = False

        if all_match and metric_rules:
            status_map = {
                "worsening": (ConditionStatus.worsening, ConditionSeverity.severe),
                "moderate": (ConditionStatus.active, ConditionSeverity.moderate),
                "controlled": (ConditionStatus.controlled, ConditionSeverity.mild),
                "resolved": (ConditionStatus.controlled, ConditionSeverity.mild),
            }
            return status_map.get(status_key, (None, None))

    return None, None


# ---------- Clinical Interpretation Generator ----------

def generate_clinical_interpretation(
    condition: PatientCondition,
    current_labs: dict,
    trend: str,
    volatility: float,
    pct_from_baseline: float = None,
    consecutive_uncontrolled: int = 0
) -> str:
    """Generate doctor-friendly clinical interpretation text.

    Examples:
        "Diabetes control has deteriorated compared to last 2 visits. HbA1c increased 15% from baseline."
        "Hypertension remains uncontrolled for 3 consecutive visits."
        "BP variability increasing — medication adherence should be reviewed."
    """
    parts = []
    disease = condition.disease_name

    # Status-specific language
    if condition.status == ConditionStatus.worsening:
        parts.append(f"{disease} control has deteriorated.")
    elif condition.status == ConditionStatus.improving:
        parts.append(f"{disease} is showing improvement.")
    elif condition.status == ConditionStatus.controlled:
        parts.append(f"{disease} is now controlled.")
    elif condition.status == ConditionStatus.resolved:
        parts.append(f"{disease} appears resolved — normal readings for 3+ consecutive reports.")
    elif condition.status == ConditionStatus.chronic_stable:
        parts.append(f"{disease} remains chronically stable.")

    # Uncontrolled streak
    if consecutive_uncontrolled >= 3:
        parts.append(f"Condition has been uncontrolled for {consecutive_uncontrolled} consecutive visits.")
    elif consecutive_uncontrolled == 2:
        parts.append("Condition uncontrolled for 2 consecutive visits.")

    # Baseline comparison
    if pct_from_baseline is not None and abs(pct_from_baseline) > 5:
        direction = "increased" if pct_from_baseline > 0 else "decreased"
        parts.append(f"Primary metric has {direction} {abs(pct_from_baseline):.1f}% from baseline.")

    # Trend
    if trend == "increasing":
        parts.append("Trend: ↑ Increasing over recent reports.")
    elif trend == "decreasing":
        parts.append("Trend: ↓ Decreasing over recent reports.")
    elif trend == "fluctuating":
        parts.append("Trend: ↕ Values fluctuating — consistency needed.")

    # Volatility warning
    if volatility > 0.15:
        parts.append("High variability detected — medication adherence should be reviewed.")
    elif volatility > 0.1:
        parts.append("Moderate variability in readings noted.")

    return " ".join(parts)


# ---------- Master Analysis Function ----------

def analyze_progression(
    recipient_id: int,
    extracted_data: dict,
    report_id: int,
    report_date_str: str,
    db: Session
) -> dict:
    """Master function: orchestrates the full progression analysis pipeline.

    Steps:
    1. Store new lab values (normalized)
    2. For each existing condition, compare labs and update status
    3. Create history entries for status transitions
    4. Check for resolution (3+ normal readings)
    5. Return progression results for alert engine

    Returns:
        dict with: status_changes[], new_lab_values[], trend_info{}
    """
    result = {
        "status_changes": [],
        "new_lab_values": [],
        "trend_info": {},
        "clinical_interpretations": [],
    }

    try:
        report_date = datetime.date.fromisoformat(report_date_str) if report_date_str else datetime.date.today()
    except (ValueError, TypeError):
        report_date = datetime.date.today()

    lab_values_raw = extracted_data.get("lab_values", {})

    # Step 1: Store normalized lab values
    for metric_name, val_info in lab_values_raw.items():
        if not isinstance(val_info, dict):
            continue
        raw_value = val_info.get("value")
        if raw_value is None:
            continue
        try:
            raw_value = float(raw_value)
        except (ValueError, TypeError):
            continue

        raw_unit = val_info.get("unit", "")
        norm_val, norm_unit = _normalize_and_store_lab(metric_name, raw_value, raw_unit)

        # Get previous values for this metric
        prev_labs = db.query(LabValue).filter(
            LabValue.care_recipient_id == recipient_id,
            LabValue.metric_name == metric_name,
            LabValue.report_id != report_id
        ).order_by(desc(LabValue.recorded_date)).limit(10).all()

        prev_values = [l.normalized_value for l in reversed(prev_labs)]

        # Calculate % changes
        pct_prev = None
        pct_baseline = None
        if prev_values:
            last_val = prev_values[-1]
            if last_val != 0:
                pct_prev = round(((norm_val - last_val) / abs(last_val)) * 100, 2)

        # Find baseline for this metric (from earliest condition using it)
        conditions = db.query(PatientCondition).filter(
            PatientCondition.care_recipient_id == recipient_id,
            PatientCondition.baseline_value.isnot(None)
        ).all()
        for cond in conditions:
            disease_dict = db.query(DiseaseDictionary).filter(
                DiseaseDictionary.code == cond.disease_code
            ).first()
            if disease_dict and metric_name in (disease_dict.monitoring_metrics or []):
                if cond.baseline_value and cond.baseline_value != 0:
                    pct_baseline = round(((norm_val - cond.baseline_value) / abs(cond.baseline_value)) * 100, 2)
                break

        # Determination of abnormal (simplified: check against known thresholds)
        is_abnormal = _is_abnormal(metric_name, norm_val)

        # Get or compute reference range
        ref_low, ref_high = _get_reference_range(metric_name)

        # Update existing LabValue record (saved by ingestion) with calculated trends
        existing_lv = db.query(LabValue).filter(
            LabValue.report_id == report_id,
            LabValue.metric_name == metric_name
        ).first()
        if existing_lv:
            existing_lv.pct_change_from_previous = pct_prev
            existing_lv.pct_change_from_baseline = pct_baseline
            # Safety Net: If ingestion didn't have a range, use system standard range to flag
            if not existing_lv.is_abnormal and is_abnormal:
                existing_lv.is_abnormal = True

        result["new_lab_values"].append({
            "metric": metric_name,
            "value": norm_val,
            "unit": norm_unit,
            "pct_change_prev": pct_prev,
            "pct_change_baseline": pct_baseline,
            "is_abnormal": is_abnormal,
        })

        # Trend calculation
        all_values = prev_values + [norm_val]
        trend = detect_trend(all_values)
        vol = calculate_volatility(all_values)
        result["trend_info"][metric_name] = {
            "trend": trend,
            "volatility": vol,
            "volatility_label": volatility_label(vol),
            "data_points": len(all_values),
        }

    # Step 2: Update existing conditions
    existing_conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()

    current_labs_normalized = {}
    for metric_name, val_info in lab_values_raw.items():
        if isinstance(val_info, dict) and val_info.get("value") is not None:
            try:
                raw_v = float(val_info["value"])
                norm_v, _ = _normalize_and_store_lab(metric_name, raw_v, val_info.get("unit", ""))
                current_labs_normalized[metric_name] = norm_v
            except (ValueError, TypeError):
                continue

    for condition in existing_conditions:
        old_status = condition.status
        old_severity = condition.severity

        # Determine new status from rules
        new_status, new_severity = _determine_status_from_rules(
            condition.disease_code, current_labs_normalized, db
        )

        if new_status is None:
            continue  # No relevant labs in this report for this condition

        # Check for resolution
        if new_status == ConditionStatus.controlled:
            condition.consecutive_normal_count += 1
            disease_dict = db.query(DiseaseDictionary).filter(
                DiseaseDictionary.code == condition.disease_code
            ).first()
            consecutive_needed = 3
            if disease_dict and disease_dict.status_rules:
                resolved_rule = disease_dict.status_rules.get("resolved", {})
                consecutive_needed = resolved_rule.get("consecutive_normal", 3)

            if condition.consecutive_normal_count >= consecutive_needed:
                new_status = ConditionStatus.resolved
                condition.resolved_date = report_date
        else:
            condition.consecutive_normal_count = 0

        # Determine if improving (was worse, now better)
        status_priority = {
            ConditionStatus.worsening: 4,
            ConditionStatus.active: 3,
            ConditionStatus.controlled: 1,
            ConditionStatus.improving: 2,
            ConditionStatus.chronic_stable: 2,
            ConditionStatus.resolved: 0,
        }
        if (status_priority.get(new_status, 3) < status_priority.get(old_status, 3)
                and new_status != ConditionStatus.resolved):
            new_status = ConditionStatus.improving

        # Only record change if status or severity actually changed
        status_changed = (new_status != old_status or new_severity != old_severity)

        if status_changed:
            condition.status = new_status
            condition.severity = new_severity
            condition.status_version += 1
            condition.last_updated = report_date

            # Get trend and volatility for this condition's primary metric
            disease_dict = db.query(DiseaseDictionary).filter(
                DiseaseDictionary.code == condition.disease_code
            ).first()
            primary_metric = (disease_dict.monitoring_metrics or [None])[0] if disease_dict else None
            trend = result["trend_info"].get(primary_metric, {}).get("trend", "stable")
            vol = result["trend_info"].get(primary_metric, {}).get("volatility", 0)
            pct_baseline = None
            for lv in result["new_lab_values"]:
                if lv["metric"] == primary_metric:
                    pct_baseline = lv.get("pct_change_baseline")
                    break

            # Count consecutive uncontrolled visits
            recent_history = db.query(ConditionHistory).filter(
                ConditionHistory.condition_id == condition.id
            ).order_by(desc(ConditionHistory.recorded_at)).limit(5).all()
            uncontrolled_count = 0
            for h in recent_history:
                if h.new_status in ("worsening", "active"):
                    uncontrolled_count += 1
                else:
                    break
            if new_status in (ConditionStatus.worsening, ConditionStatus.active):
                uncontrolled_count += 1

            interpretation = generate_clinical_interpretation(
                condition, current_labs_normalized, trend, vol,
                pct_baseline, uncontrolled_count
            )

            # Create history entry
            history = ConditionHistory(
                condition_id=condition.id,
                report_id=report_id,
                previous_status=old_status.value if old_status else None,
                new_status=new_status.value,
                previous_severity=old_severity.value if old_severity else None,
                new_severity=new_severity.value if new_severity else None,
                status_version=condition.status_version,
                clinical_interpretation=interpretation,
                change_reason=f"Report analysis on {report_date}",
                recorded_at=datetime.datetime.utcnow()
            )
            db.add(history)

            result["status_changes"].append({
                "condition_id": condition.id,
                "disease_name": condition.disease_name,
                "disease_code": condition.disease_code,
                "old_status": old_status.value if old_status else None,
                "new_status": new_status.value,
                "old_severity": old_severity.value if old_severity else None,
                "new_severity": new_severity.value if new_severity else None,
                "clinical_interpretation": interpretation,
            })
            result["clinical_interpretations"].append(interpretation)

    db.flush()
    print(f"[disease_progression] Progression: {len(result['status_changes'])} changes, {len(result['new_lab_values'])} lab values")
    return result


# ---------- Helper Functions ----------

def _is_abnormal(metric_name: str, value: float) -> bool:
    """Check if a normalized value is abnormal based on standard reference ranges."""
    ranges = {
        "HbA1c": (4.0, 5.7),
        "Fasting Glucose": (70, 100),
        "Systolic BP": (90, 120),
        "Diastolic BP": (60, 80),
        "Total Cholesterol": (0, 200),
        "LDL": (0, 100),
        "HDL": (40, 999),
        "Triglycerides": (0, 150),
        "Creatinine": (0.6, 1.2),
        "eGFR": (60, 999),
        "Hemoglobin": (12.0, 17.5),
        "Hematocrit": (36, 52),
        "TSH": (0.4, 4.5),
        "Uric Acid": (3.0, 7.0),
        "BUN": (7, 20),
    }
    if metric_name in ranges:
        low, high = ranges[metric_name]
        return value < low or value > high
    return False


def _get_reference_range(metric_name: str) -> tuple:
    """Get standard reference range for a metric."""
    ranges = {
        "HbA1c": (4.0, 5.7),
        "Fasting Glucose": (70, 100),
        "Systolic BP": (90, 120),
        "Diastolic BP": (60, 80),
        "Total Cholesterol": (0, 200),
        "LDL": (0, 100),
        "HDL": (40, 60),
        "Triglycerides": (0, 150),
        "Creatinine": (0.6, 1.2),
        "eGFR": (60, 120),
        "Hemoglobin": (12.0, 17.5),
        "Hematocrit": (36, 52),
        "TSH": (0.4, 4.5),
        "Uric Acid": (3.0, 7.0),
        "BUN": (7, 20),
        "Ferritin": (12, 300),
    }
    return ranges.get(metric_name, (None, None))
