"""Medical History API Routes.

Endpoints for patient medical state, conditions, trends, alerts, and AI analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from config import get_db
from typing import List, Optional
from pydantic import BaseModel
from services.lab_analytics_ai import analyze_lab_trends
from services.weekly_report_generator import generate_weekly_report

from repository import medical_history as repo
from services.disease_progression import detect_trend, calculate_volatility, volatility_label
from services.medical_history_ai import analyze_patient_health, calculate_risk_score
from services.alert_engine import check_monitoring_gaps
from models.medical_history import (
    ConditionSchema, ConditionWithHistory, ConditionHistorySchema,
    LabValueSchema, AlertSchema, MetricTrend, TrendDataPoint,
    TrendSummary, PatientMedicalState, RiskScoreBreakdown, RiskFactor,
    HealthAnalysisResult
)


router = APIRouter(
    prefix="/api/recipients",
    tags=["Medical History"]
)


# ---------- Full Patient Medical State ----------

@router.get("/{recipient_id}/medical-state")
def get_medical_state(recipient_id: int, db: Session = Depends(get_db)):
    """Get full patient medical state including conditions, latest labs, risk, and alerts."""
    active = repo.get_active_conditions(db, recipient_id)
    past = repo.get_past_conditions(db, recipient_id)
    latest_labs = repo.get_latest_lab_values(db, recipient_id)
    unread_alerts = repo.get_unread_alerts(db, recipient_id)
    recent_alerts = repo.get_all_alerts(db, recipient_id, limit=10)

    # Get risk score
    from tables.users import CareRecipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient not found")

    risk_breakdown = None
    if recipient.risk_factors_breakdown:
        breakdown = recipient.risk_factors_breakdown
        risk_breakdown = {
            "risk_score": breakdown.get("risk_score", 0),
            "risk_category": breakdown.get("risk_category", "Unknown"),
            "risk_trajectory": breakdown.get("risk_trajectory", "stable"),
            "factors": breakdown.get("factors", [])
        }

    return {
        "active_conditions": [_serialize_condition(c) for c in active],
        "past_conditions": [_serialize_condition(c) for c in past],
        "latest_labs": [_serialize_lab(l) for l in latest_labs],
        "risk_score": risk_breakdown,
        "unread_alert_count": len(unread_alerts),
        "recent_alerts": [_serialize_alert(a) for a in recent_alerts],
    }


# ---------- Conditions ----------

@router.get("/{recipient_id}/conditions")
def get_conditions(recipient_id: int, db: Session = Depends(get_db)):
    """Get all conditions (active + past) for a recipient."""
    active = repo.get_active_conditions(db, recipient_id)
    past = repo.get_past_conditions(db, recipient_id)
    return {
        "active": [_serialize_condition(c) for c in active],
        "past": [_serialize_condition(c) for c in past],
    }


@router.get("/{recipient_id}/conditions/{condition_id}/timeline")
def get_condition_timeline(
    recipient_id: int,
    condition_id: int,
    db: Session = Depends(get_db)
):
    """Get full timeline/history for a specific condition."""
    condition = repo.get_condition_by_id(db, condition_id)
    if not condition or condition.care_recipient_id != recipient_id:
        raise HTTPException(status_code=404, detail="Condition not found")

    history = repo.get_condition_timeline(db, condition_id)
    return {
        "condition": _serialize_condition(condition),
        "timeline": [
            {
                "id": h.id,
                "previous_status": h.previous_status,
                "new_status": h.new_status,
                "previous_severity": h.previous_severity,
                "new_severity": h.new_severity,
                "status_version": h.status_version,
                "clinical_interpretation": h.clinical_interpretation,
                "change_reason": h.change_reason,
                "recorded_at": str(h.recorded_at),
                "report_id": h.report_id,
            }
            for h in history
        ]
    }


# ---------- Trends ----------

@router.get("/{recipient_id}/trends")
def get_trends(recipient_id: int, db: Session = Depends(get_db)):
    """Get trend summary for all metrics."""
    available = repo.get_available_metrics(db, recipient_id)
    metrics = []

    for metric_name in available:
        labs = repo.get_lab_time_series(db, recipient_id, metric_name)
        if not labs:
            continue

        values = [l.normalized_value for l in labs]
        trend_dir = detect_trend(values)
        vol = calculate_volatility(values)

        # Get baseline from conditions
        baseline_val = None
        conditions = repo.get_active_conditions(db, recipient_id) + repo.get_past_conditions(db, recipient_id)
        for c in conditions:
            if c.baseline_value is not None:
                from tables.disease_dictionary import DiseaseDictionary
                dd = db.query(DiseaseDictionary).filter(DiseaseDictionary.code == c.disease_code).first()
                if dd and metric_name in (dd.monitoring_metrics or []):
                    baseline_val = c.baseline_value
                    break

        metrics.append({
            "metric_name": metric_name,
            "data_points": [
                {
                    "date": str(l.recorded_date),
                    "value": l.normalized_value,
                    "unit": l.normalized_unit,
                    "is_abnormal": l.is_abnormal,
                    "pct_change_from_previous": l.pct_change_from_previous,
                    "pct_change_from_baseline": l.pct_change_from_baseline,
                    "confidence_score":   round(l.confidence_score, 3) if l.confidence_score is not None else None,
                    "extraction_source":  l.extraction_source or "regex",
                    "source_text":        l.source_text or "",
                }
                for l in labs
            ],
            "reference_range_low": labs[0].reference_range_low if labs else None,
            "reference_range_high": labs[0].reference_range_high if labs else None,
            "trend_direction": trend_dir,
            "volatility": vol,
            "volatility_label": volatility_label(vol),
            "latest_value": values[-1] if values else None,
            "baseline_value": baseline_val,
        })

    return {
        "metrics": metrics,
        "available_metrics": available,
    }


@router.get("/{recipient_id}/trends/{metric_name}")
def get_metric_trend(
    recipient_id: int,
    metric_name: str,
    db: Session = Depends(get_db)
):
    """Get time-series trend for a specific metric."""
    labs = repo.get_lab_time_series(db, recipient_id, metric_name)
    if not labs:
        return {"data_points": [], "trend_direction": "stable", "volatility": 0}

    values = [l.normalized_value for l in labs]
    return {
        "metric_name": metric_name,
        "data_points": [
            {
                "date": str(l.recorded_date),
                "value": l.normalized_value,
                "unit": l.normalized_unit,
                "is_abnormal": l.is_abnormal,
                "pct_change_from_previous": l.pct_change_from_previous,
                "pct_change_from_baseline": l.pct_change_from_baseline,
                "confidence_score":  round(l.confidence_score, 3) if l.confidence_score is not None else None,
                "extraction_source": l.extraction_source or "regex",
                "source_text":       l.source_text or "",
            }
            for l in labs
        ],
        "reference_range_low": labs[0].reference_range_low if labs else None,
        "reference_range_high": labs[0].reference_range_high if labs else None,
        "trend_direction": detect_trend(values),
        "volatility": calculate_volatility(values),
        "volatility_label": volatility_label(calculate_volatility(values)),
        "latest_value": values[-1] if values else None,
    }


class LabAnalysisRequest(BaseModel):
    metric_names: Optional[List[str]] = None

@router.post("/{recipient_id}/lab-analysis")
def analyze_lab_data(
    recipient_id: int,
    request: LabAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Analyze lab trends using Gemini AI."""
    trends_result = get_trends(recipient_id, db)
    metrics_to_analyze = trends_result.get("metrics", [])
    
    if request.metric_names:
        metrics_to_analyze = [m for m in metrics_to_analyze if m["metric_name"] in request.metric_names]
        
    analysis_results = analyze_lab_trends(metrics_to_analyze)
    return {
        "analysis": analysis_results
    }


# ---------- Weekly Reports ----------

@router.get("/{recipient_id}/weekly-reports")
def list_weekly_reports(
    recipient_id: int,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List past weekly reports for a recipient."""
    from tables.weekly_reports import WeeklyReport
    from sqlalchemy import desc
    reports = db.query(WeeklyReport).filter(
        WeeklyReport.care_recipient_id == recipient_id
    ).order_by(desc(WeeklyReport.week_start)).offset(offset).limit(limit).all()
    return {
        "reports": [
            {
                "report_id": r.id,
                "week_start": str(r.week_start),
                "week_end": str(r.week_end),
                "overall_score": r.overall_score,
                "risk_trend": r.risk_trend,
                "model_used": r.model_used,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in reports
        ]
    }


@router.get("/{recipient_id}/weekly-reports/latest_json")
def get_latest_weekly_report(
    recipient_id: int,
    db: Session = Depends(get_db)
):
    """Get the most recent weekly report."""
    from tables.weekly_reports import WeeklyReport
    from sqlalchemy import desc
    report = db.query(WeeklyReport).filter(
        WeeklyReport.care_recipient_id == recipient_id
    ).order_by(desc(WeeklyReport.week_start)).first()
    if not report:
        raise HTTPException(status_code=404, detail="No weekly reports found")
    return {
        "report_id": report.id,
        "week_start": str(report.week_start),
        "week_end": str(report.week_end),
        "sections": report.report_data,
        "ai_narrative": report.ai_narrative,
        "comparison": report.comparison,
        "overall_score": report.overall_score,
        "risk_trend": report.risk_trend,
        "model_used": report.model_used,
    }


@router.post("/{recipient_id}/weekly-reports/generate_json")
def generate_report_on_demand(
    recipient_id: int,
    db: Session = Depends(get_db)
):
    """Manually generate a weekly report (for testing or on-demand refresh)."""
    result = generate_weekly_report(recipient_id, db)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{recipient_id}/weekly-reports/{report_id}/download")
def download_weekly_report(
    recipient_id: int,
    report_id: int,
    db: Session = Depends(get_db)
):
    """Download a weekly report as a JSON file for the dashboard."""
    from tables.weekly_reports import WeeklyReport
    from fastapi.responses import JSONResponse
    report = db.query(WeeklyReport).filter(
        WeeklyReport.id == report_id,
        WeeklyReport.care_recipient_id == recipient_id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    download_payload = {
        "report_id": report.id,
        "recipient_id": recipient_id,
        "week_start": str(report.week_start),
        "week_end": str(report.week_end),
        "overall_score": report.overall_score,
        "risk_trend": report.risk_trend,
        "sections": report.report_data,
        "ai_narrative": report.ai_narrative,
        "comparison": report.comparison,
        "model_used": report.model_used,
        "generated_at": report.created_at.isoformat() if report.created_at else None,
    }
    filename = f"weekly_report_{recipient_id}_{report.week_start}.json"
    return JSONResponse(
        content=download_payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ---------- Alerts ----------

@router.get("/{recipient_id}/alerts")
def get_alerts(
    recipient_id: int,
    unread_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get alerts for a recipient."""
    if unread_only:
        alerts = repo.get_unread_alerts(db, recipient_id)
    else:
        alerts = repo.get_all_alerts(db, recipient_id)
    return {
        "alerts": [_serialize_alert(a) for a in alerts],
        "unread_count": len(repo.get_unread_alerts(db, recipient_id)),
    }


@router.patch("/{recipient_id}/alerts/{alert_id}/read")
def mark_alert_as_read(
    recipient_id: int,
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Mark an alert as read."""
    alert = repo.mark_alert_read(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.commit()
    return {"status": "ok", "alert_id": alert_id}


# ---------- AI Analysis ----------

@router.post("/{recipient_id}/analyze")
def trigger_analysis(
    recipient_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger full health analysis (deterministic scoring + Gemini interpretation)."""
    # Run synchronously for now (risk scoring is fast, Gemini call is the slow part)
    try:
        result = analyze_patient_health(recipient_id, db)
        db.commit()

        # Also check for monitoring gaps
        gap_alerts = check_monitoring_gaps(recipient_id, db)
        if gap_alerts:
            db.commit()

        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------- Report Reprocessing ----------

@router.post("/{recipient_id}/reports/{report_id}/reprocess")
def reprocess_report(
    recipient_id: int,
    report_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Re-run structured extraction on a report."""
    from tables.medical_reports import MedicalReport, ReportProcessingStatus

    report = db.query(MedicalReport).filter(
        MedicalReport.id == report_id,
        MedicalReport.care_recipient_id == recipient_id
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    report.processing_status = ReportProcessingStatus.processing
    db.commit()

    # Run in background
    def _reprocess():
        from config import SessionLocal
        bg_db = SessionLocal()
        try:
            from services.report_ingestion import extract_structured_report
            from services.disease_detection import detect_diseases_from_report
            from services.disease_progression import analyze_progression
            from services.alert_engine import generate_alerts
            from utils.summarizer import extract_text_from_bytes

            bg_report = bg_db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
            if not bg_report:
                return

            # Extract text
            text = ""
            if bg_report.data:
                text = extract_text_from_bytes(bg_report.data, bg_report.filename)

            if not text:
                bg_report.processing_status = ReportProcessingStatus.failed
                bg_db.commit()
                return

            # Structured extraction
            extracted = extract_structured_report(text)
            bg_report.extracted_data = extracted
            bg_report.report_date = None
            if extracted.get("report_date"):
                try:
                    from datetime import date as dt_date
                    bg_report.report_date = dt_date.fromisoformat(extracted["report_date"])
                except (ValueError, TypeError):
                    pass

            # Disease detection
            existing = repo.get_all_conditions(bg_db, recipient_id)
            new_diseases = detect_diseases_from_report(extracted, existing, bg_db, report_id, extracted.get("report_date"))

            for disease in new_diseases:
                from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity, SourceType
                cond = PatientCondition(
                    care_recipient_id=recipient_id,
                    disease_code=disease["disease_code"],
                    disease_name=disease["disease_name"],
                    status=ConditionStatus.active,
                    severity=ConditionSeverity.moderate,
                    first_detected=disease.get("first_detected"),
                    last_updated=disease.get("first_detected"),
                    baseline_value=disease.get("baseline_value"),
                    baseline_date=disease.get("baseline_date"),
                    confidence_score=disease.get("confidence_score", 0.5),
                    source_type=SourceType(disease.get("source_type", "lab_inferred")),
                    source_report_id=report_id,
                )
                bg_db.add(cond)
            bg_db.flush()

            # Progression analysis
            progression = analyze_progression(
                recipient_id, extracted, report_id,
                extracted.get("report_date", ""), bg_db
            )

            # Generate alerts
            generate_alerts(recipient_id, progression, bg_db)

            bg_report.processing_status = ReportProcessingStatus.completed
            bg_db.commit()
            print(f"[reprocess] Report {report_id} reprocessed successfully")

        except Exception as e:
            print(f"[reprocess] Error: {e}")
            bg_db.rollback()
            try:
                bg_report = bg_db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
                if bg_report:
                    bg_report.processing_status = ReportProcessingStatus.failed
                    bg_db.commit()
            except Exception:
                pass
        finally:
            bg_db.close()

    background_tasks.add_task(_reprocess)
    return {"status": "processing", "report_id": report_id}


# ---------- Serialization Helpers ----------

def _serialize_condition(c):
    return {
        "id": c.id,
        "disease_code": c.disease_code,
        "disease_name": c.disease_name,
        "status": c.status.value if c.status else None,
        "severity": c.severity.value if c.severity else None,
        "status_version": c.status_version,
        "first_detected": str(c.first_detected) if c.first_detected else None,
        "last_updated": str(c.last_updated) if c.last_updated else None,
        "resolved_date": str(c.resolved_date) if c.resolved_date else None,
        "baseline_value": c.baseline_value,
        "baseline_date": str(c.baseline_date) if c.baseline_date else None,
        "consecutive_normal_count": c.consecutive_normal_count,
        "confidence_score": c.confidence_score,
        "source_type": c.source_type.value if c.source_type else None,
    }


def _serialize_lab(l):
    return {
        "id": l.id,
        "metric_name": l.metric_name,
        "metric_value": l.metric_value,
        "unit": l.unit,
        "normalized_value": l.normalized_value,
        "normalized_unit": l.normalized_unit,
        "reference_range_low": l.reference_range_low,
        "reference_range_high": l.reference_range_high,
        "is_abnormal": l.is_abnormal,
        "pct_change_from_previous": l.pct_change_from_previous,
        "pct_change_from_baseline": l.pct_change_from_baseline,
        "recorded_date": str(l.recorded_date),
    }


def _serialize_alert(a):
    return {
        "id": a.id,
        "alert_type": a.alert_type.value if a.alert_type else None,
        "message": a.message,
        "severity": a.severity.value if a.severity else None,
        "is_read": a.is_read,
        "condition_id": a.condition_id,
        "created_at": str(a.created_at),
    }
