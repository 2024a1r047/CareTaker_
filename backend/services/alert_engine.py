"""Alert & Insight Engine — Generates alerts with throttling.

Produces clinically relevant alerts based on condition status transitions.
Implements cooldown periods and minimum change thresholds to prevent alert fatigue.
Supports monitoring gap detection using disease-specific frequencies.
"""

import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.medical_conditions import (
    MedicalAlert, PatientCondition, LabValue,
    AlertType, AlertSeverity, ConditionStatus
)
from tables.disease_dictionary import DiseaseDictionary
from tables.users import CareRecipient


def generate_alerts(
    recipient_id: int,
    progression_result: dict,
    db: Session
) -> list:
    """Generate alerts based on progression analysis results.

    Implements:
    - Alert throttling via cooldown_until
    - Minimum change threshold checking
    - Monitoring gap detection (disease-aware)

    Args:
        recipient_id: Care recipient ID
        progression_result: Output from disease_progression.analyze_progression()
        db: Database session

    Returns:
        List of created alert dicts
    """
    created_alerts = []
    now = datetime.datetime.utcnow()

    # Process status changes
    for change in progression_result.get("status_changes", []):
        condition_id = change.get("condition_id")
        new_status = change.get("new_status")
        disease_name = change.get("disease_name", "Unknown condition")
        disease_code = change.get("disease_code")
        interpretation = change.get("clinical_interpretation", "")

        # Check cooldown
        if condition_id and _is_in_cooldown(condition_id, db, now):
            print(f"[alert_engine] Skipping alert for condition {condition_id} — in cooldown")
            continue

        # Check minimum change threshold
        if condition_id and not _meets_change_threshold(condition_id, disease_code, db):
            print(f"[alert_engine] Skipping alert for condition {condition_id} — below threshold")
            continue

        # Determine alert type and severity
        alert_type, alert_severity, message = _build_alert(
            new_status, disease_name, interpretation
        )

        if alert_type is None:
            continue

        # Get cooldown duration
        cooldown_days = _get_cooldown_days(disease_code, db)
        cooldown_until = now + datetime.timedelta(days=cooldown_days)

        alert = MedicalAlert(
            care_recipient_id=recipient_id,
            condition_id=condition_id,
            alert_type=alert_type,
            message=message,
            severity=alert_severity,
            is_read=False,
            cooldown_until=cooldown_until,
            created_at=now
        )
        db.add(alert)
        created_alerts.append({
            "alert_type": alert_type.value,
            "severity": alert_severity.value,
            "message": message,
            "disease_name": disease_name,
        })

    # Check for critical lab values (independent of status changes)
    for lab in progression_result.get("new_lab_values", []):
        # A physically impossible value (e.g. Hemoglobin 1.0 from a misread report) is a reading
        # error, not a medical emergency: never raise a critical alert for it.
        try:
            from services.agent_medical_context import _plausible
            if not _plausible(lab["metric"], float(lab["value"])):
                print(f"[alert_engine] skipped critical alert for implausible {lab['metric']} = {lab['value']}")
                continue
        except (TypeError, ValueError, ImportError):
            pass
        if _is_critical_value(lab["metric"], lab["value"]):
            critical_alert = MedicalAlert(
                care_recipient_id=recipient_id,
                condition_id=None,
                alert_type=AlertType.critical,
                message=f"CRITICAL: {lab['metric']} at {lab['value']} {lab.get('unit', '')} — immediate attention needed.",
                severity=AlertSeverity.critical,
                is_read=False,
                cooldown_until=now + datetime.timedelta(days=1),  # Short cooldown for critical
                created_at=now
            )
            db.add(critical_alert)
            created_alerts.append({
                "alert_type": "critical",
                "severity": "critical",
                "message": critical_alert.message,
                "disease_name": None,
            })

    db.flush()
    print(f"[alert_engine] Generated {len(created_alerts)} alerts")
    return created_alerts


def check_monitoring_gaps(recipient_id: int, db: Session) -> list:
    """Check for monitoring gaps based on disease-specific frequencies.

    Returns list of gap alerts to create.
    """
    created_alerts = []
    now = datetime.datetime.utcnow()

    # Get active conditions
    conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()

    for condition in conditions:
        # Get monitoring frequency from disease dictionary
        disease = db.query(DiseaseDictionary).filter(
            DiseaseDictionary.code == condition.disease_code
        ).first()

        freq_months = disease.monitoring_frequency_months if disease else 6

        # Check last report date for this condition's metrics
        last_lab = db.query(LabValue).filter(
            LabValue.care_recipient_id == recipient_id,
            LabValue.metric_name.in_(disease.monitoring_metrics or [])
        ).order_by(desc(LabValue.recorded_date)).first() if disease else None

        if last_lab:
            months_since = (now.date() - last_lab.recorded_date).days / 30
            if months_since > freq_months:
                # Check if we already have a recent monitoring_gap alert for this condition
                existing_gap = db.query(MedicalAlert).filter(
                    MedicalAlert.care_recipient_id == recipient_id,
                    MedicalAlert.condition_id == condition.id,
                    MedicalAlert.alert_type == AlertType.monitoring_gap,
                    MedicalAlert.created_at > now - datetime.timedelta(days=30)
                ).first()

                if not existing_gap:
                    alert = MedicalAlert(
                        care_recipient_id=recipient_id,
                        condition_id=condition.id,
                        alert_type=AlertType.monitoring_gap,
                        message=(
                            f"Monitoring gap detected for {condition.disease_name}. "
                            f"Last relevant lab results were {months_since:.0f} months ago. "
                            f"Recommended monitoring frequency: every {freq_months} months."
                        ),
                        severity=AlertSeverity.medium,
                        is_read=False,
                        created_at=now
                    )
                    db.add(alert)
                    created_alerts.append({
                        "alert_type": "monitoring_gap",
                        "severity": "medium",
                        "message": alert.message,
                        "disease_name": condition.disease_name,
                    })

    if created_alerts:
        db.flush()
        print(f"[alert_engine] Generated {len(created_alerts)} monitoring gap alerts")
    return created_alerts


# ---------- Internal Helpers ----------

def _is_in_cooldown(condition_id: int, db: Session, now: datetime.datetime) -> bool:
    """Check if the most recent alert for this condition is still in cooldown."""
    last_alert = db.query(MedicalAlert).filter(
        MedicalAlert.condition_id == condition_id,
    ).order_by(desc(MedicalAlert.created_at)).first()

    if last_alert and last_alert.cooldown_until and last_alert.cooldown_until > now:
        return True
    return False


def _meets_change_threshold(condition_id: int, disease_code: str, db: Session) -> bool:
    """Check if the lab value change meets the minimum threshold for alerting."""
    disease = db.query(DiseaseDictionary).filter(
        DiseaseDictionary.code == disease_code
    ).first()

    if not disease:
        return True  # No rules → allow alert

    threshold = disease.minimum_change_threshold or 5.0
    metrics = disease.monitoring_metrics or []

    # Check if any monitored metric changed enough
    condition = db.query(PatientCondition).filter(
        PatientCondition.id == condition_id
    ).first()
    if not condition:
        return True

    for metric in metrics:
        last_lab = db.query(LabValue).filter(
            LabValue.care_recipient_id == condition.care_recipient_id,
            LabValue.metric_name == metric
        ).order_by(desc(LabValue.recorded_date)).first()

        if last_lab and last_lab.pct_change_from_previous is not None:
            if abs(last_lab.pct_change_from_previous) >= threshold:
                return True

    return False


def _build_alert(new_status: str, disease_name: str, interpretation: str) -> tuple:
    """Build alert type, severity, and message based on status change."""
    if new_status == "worsening":
        return (
            AlertType.worsening,
            AlertSeverity.high,
            f"⚠️ {disease_name} control has worsened. {interpretation}"
        )
    elif new_status == "improving":
        return (
            AlertType.improving,
            AlertSeverity.low,
            f"✅ {disease_name} is showing improvement. {interpretation}"
        )
    elif new_status == "resolved":
        return (
            AlertType.resolved,
            AlertSeverity.low,
            f"🎉 {disease_name} appears controlled and stable for 3+ consecutive reports. {interpretation}"
        )
    elif new_status == "active":
        # Only alert if previously was controlled or improving
        return (
            AlertType.worsening,
            AlertSeverity.medium,
            f"⚠️ {disease_name} has returned to active status. {interpretation}"
        )
    elif new_status == "controlled":
        return (
            AlertType.improving,
            AlertSeverity.low,
            f"✅ {disease_name} is now controlled. {interpretation}"
        )
    return (None, None, None)


def _get_cooldown_days(disease_code: str, db: Session) -> int:
    """Get cooldown days from disease dictionary, default 1 (for testing/high responsiveness)."""
    # Industry standard is 7-30 days, but for this demo/setup we want immediate feedback
    return 1


def _is_critical_value(metric_name: str, value: float) -> bool:
    """Check if a lab value is at critical (emergency) levels."""
    critical_thresholds = {
        "Fasting Glucose": {"low": 40, "high": 400},
        "Systolic BP": {"low": 70, "high": 200},
        "Diastolic BP": {"low": 40, "high": 130},
        "Heart Rate": {"low": 40, "high": 150},
        "Hemoglobin": {"low": 7.0, "high": None},
        "eGFR": {"low": 15, "high": None},
        "Creatinine": {"low": None, "high": 4.0},
        "TSH": {"low": None, "high": 20.0},
        "HbA1c": {"low": None, "high": 12.0},
    }
    if metric_name not in critical_thresholds:
        return False
    t = critical_thresholds[metric_name]
    if t.get("low") is not None and value < t["low"]:
        return True
    if t.get("high") is not None and value > t["high"]:
        return True
    return False
