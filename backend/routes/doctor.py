"""Doctor Dashboard API Routes.

Provides aggregated patient data specifically for doctors:
- Patient list with risk scores and summary stats
- Detailed clinical summary for individual patients
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from config import get_db
from datetime import datetime, timedelta, date
from pydantic import BaseModel

class RemarksModel(BaseModel):
    remarks: str

class PrescriptionModel(BaseModel):
    medicine_name: str
    dosage: str
    frequency: str
    schedule_time: str
    duration_days: int
    notes: str = ""
    current_stock: int = 0
    doses_per_day: int = 1

class LabOrderModel(BaseModel):
    test_name: str
    doctor_notes: str = ""

router = APIRouter(tags=["Doctor Dashboard"])


@router.get("/doctor/patients")
def get_all_patients(db: Session = Depends(get_db)):
    """
    List all care recipients with summary stats for the doctor sidebar.
    Returns: id, name, age, gender, risk_score, condition count, medication count, last vitals.
    """
    from tables.users import CareRecipient
    from tables.medical_conditions import PatientCondition, ConditionStatus
    from tables.medications import Medication
    from tables.vital_signs import VitalSign
    from tables.medical_conditions import MedicalAlert

    recipients = db.query(CareRecipient).all()
    patients = []

    for r in recipients:
        # Count active conditions
        active_conditions = db.query(func.count(PatientCondition.id)).filter(
            PatientCondition.care_recipient_id == r.id,
            PatientCondition.status != ConditionStatus.resolved
        ).scalar() or 0

        # Count active medications
        active_meds = db.query(func.count(Medication.medication_id)).filter(
            Medication.care_recipient_id == r.id,
            Medication.status == 'active'
        ).scalar() or 0

        # Get latest vitals
        latest_vital = db.query(VitalSign).filter(
            VitalSign.care_recipient_id == r.id
        ).order_by(desc(VitalSign.recorded_at)).first()

        # Count unread alerts
        unread_alerts = db.query(func.count(MedicalAlert.id)).filter(
            MedicalAlert.care_recipient_id == r.id,
            MedicalAlert.is_read == False
        ).scalar() or 0

        vitals_snapshot = None
        if latest_vital:
            vitals_snapshot = {
                "heart_rate": latest_vital.heart_rate,
                "systolic_bp": latest_vital.systolic_bp,
                "diastolic_bp": latest_vital.diastolic_bp,
                "oxygen_saturation": latest_vital.oxygen_saturation,
                "temperature": latest_vital.temperature,
                "recorded_at": str(latest_vital.recorded_at) if latest_vital.recorded_at else None
            }

        patients.append({
            "id": r.id,
            "name": r.full_name,
            "age": r.age,
            "gender": r.gender.value if r.gender else None,
            "city": r.city,
            "blood_group": r.blood_group,
            "risk_score": r.risk_score or 0,
            "active_conditions_count": active_conditions,
            "active_medications_count": active_meds,
            "unread_alerts_count": unread_alerts,
            "latest_vitals": vitals_snapshot,
            "respiratory_condition": r.respiratory_condition_status,
        })

    # Sort by risk score descending (highest risk first)
    patients.sort(key=lambda p: p["risk_score"], reverse=True)
    return {"patients": patients, "total": len(patients)}


@router.get("/doctor/patients/{recipient_id}/summary")
def get_patient_clinical_summary(recipient_id: int, db: Session = Depends(get_db)):
    """
    Full clinical summary for a specific patient.
    Returns: general info, vitals history, conditions, medications with adherence,
    allergies, alerts, lab trends, AI summary.
    """
    from tables.users import CareRecipient
    from tables.medical_conditions import PatientCondition, ConditionStatus, LabValue, MedicalAlert
    from tables.medications import Medication, MedicationHistory
    from tables.vital_signs import VitalSign
    from tables.allergies import Allergy
    from tables.audio_events import AudioEvent

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # 1. General Info
    general_info = {
        "id": recipient.id,
        "name": recipient.full_name,
        "age": recipient.age,
        "gender": recipient.gender.value if recipient.gender else None,
        "height": recipient.height,
        "weight": recipient.weight,
        "blood_group": recipient.blood_group,
        "phone_number": recipient.phone_number,
        "email": recipient.email,
        "city": recipient.city,
        "emergency_contact": recipient.emergency_contact,
        "respiratory_condition": recipient.respiratory_condition_status,
        "registration_date": str(recipient.registration_date) if recipient.registration_date else None,
        "risk_score": recipient.risk_score or 0,
        "risk_factors": recipient.risk_factors_breakdown.get("factors", []) if isinstance(recipient.risk_factors_breakdown, dict) else [],
        "risk_category": recipient.risk_factors_breakdown.get("risk_category", "Unknown") if isinstance(recipient.risk_factors_breakdown, dict) else "Unknown",
        "doctor_remarks": recipient.doctor_remarks,
    }

    # 2. Active Conditions
    active_conditions_raw = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).order_by(desc(PatientCondition.last_updated)).all()

    conditions = [{
        "id": c.id,
        "disease_name": c.disease_name,
        "disease_code": c.disease_code,
        "status": c.status.value,
        "severity": c.severity.value if c.severity else None,
        "first_detected": str(c.first_detected) if c.first_detected else None,
        "last_updated": str(c.last_updated) if c.last_updated else None,
        "confidence_score": c.confidence_score,
    } for c in active_conditions_raw]

    # 3. Active Medications with stock info
    meds_raw = db.query(Medication).filter(
        Medication.care_recipient_id == recipient_id,
        Medication.status == 'active'
    ).all()

    medications = []
    for m in meds_raw:
        days_left = None
        if m.current_stock and m.doses_per_day and m.doses_per_day > 0:
            days_left = m.current_stock // m.doses_per_day

        medications.append({
            "id": m.medication_id,
            "medicine_name": m.medicine_name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "schedule_time": m.schedule_time,
            "start_date": str(m.start_date) if m.start_date else None,
            "end_date": str(m.end_date) if m.end_date else None,
            "current_stock": m.current_stock or 0,
            "doses_per_day": m.doses_per_day or 1,
            "days_left": days_left,
            "auto_order_enabled": getattr(m, 'auto_order_enabled', True),
        })

    # 4. Medication adherence (simple: count completed vs active)
    total_meds_ever = db.query(func.count(Medication.medication_id)).filter(
        Medication.care_recipient_id == recipient_id
    ).scalar() or 0
    completed_meds = db.query(func.count(Medication.medication_id)).filter(
        Medication.care_recipient_id == recipient_id,
        Medication.status == 'completed'
    ).scalar() or 0

    medication_adherence = {
        "total_prescribed": total_meds_ever,
        "completed": completed_meds,
        "active": len(meds_raw),
        "adherence_rate": round((completed_meds / total_meds_ever * 100), 1) if total_meds_ever > 0 else 100.0,
    }

    # 5. Allergies
    allergies_raw = db.query(Allergy).filter(
        Allergy.care_recipient_id == recipient_id,
        Allergy.status == 'active'
    ).all()
    allergies = [{
        "id": a.allergy_id,
        "allergen": a.allergen,
        "type": a.allergy_type.value if a.allergy_type else None,
        "reaction": a.reaction,
        "severity": a.severity,
    } for a in allergies_raw]

    # 6. Recent Vitals (last 30)
    vitals_raw = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id
    ).order_by(desc(VitalSign.recorded_at)).limit(30).all()

    vitals = [{
        "recorded_at": str(v.recorded_at) if v.recorded_at else None,
        "heart_rate": v.heart_rate,
        "systolic_bp": v.systolic_bp,
        "diastolic_bp": v.diastolic_bp,
        "oxygen_saturation": v.oxygen_saturation,
        "temperature": v.temperature,
        "sleep_score": v.sleep_score,
    } for v in reversed(vitals_raw)]  # Chronological order

    # 7. Alerts (all, most recent first)
    alerts_raw = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == recipient_id
    ).order_by(desc(MedicalAlert.created_at)).limit(20).all()

    alerts = [{
        "id": a.id,
        "severity": a.severity.value if a.severity else None,
        "message": a.message,
        "is_read": a.is_read,
        "created_at": str(a.created_at) if a.created_at else None,
    } for a in alerts_raw]

    # 8. Lab Trends
    labs_raw = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id
    ).order_by(LabValue.recorded_date).all()

    # Group by metric name
    lab_metrics = {}
    for l in labs_raw:
        name = l.metric_name
        if name not in lab_metrics:
            lab_metrics[name] = []
        lab_metrics[name].append({
            "date": str(l.recorded_date) if l.recorded_date else None,
            "value": l.normalized_value,
            "unit": l.normalized_unit,
            "is_abnormal": l.is_abnormal,
            "reference_low": l.reference_range_low,
            "reference_high": l.reference_range_high,
        })

    # 9. Audio Events (recent cough/sneeze counts for last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    try:
        audio_events = db.query(AudioEvent).filter(
            AudioEvent.care_recipient_id == recipient_id,
            AudioEvent.detected_at >= seven_days_ago
        ).all()
        cough_count = sum(1 for e in audio_events if e.event_type and 'cough' in e.event_type.lower())
        sneeze_count = sum(1 for e in audio_events if e.event_type and 'sneeze' in e.event_type.lower())
    except Exception:
        cough_count = 0
        sneeze_count = 0

    audio_summary = {
        "period": "7 days",
        "cough_count": cough_count,
        "sneeze_count": sneeze_count,
        "total_events": cough_count + sneeze_count,
    }

    # 10. AI Clinical Summary (use existing report_summary if available)
    clinical_summary = recipient.report_summary or "No clinical summary available. Upload medical reports to generate AI-powered insights."

    return {
        "general_info": general_info,
        "conditions": conditions,
        "medications": medications,
        "medication_adherence": medication_adherence,
        "allergies": allergies,
        "vitals": vitals,
        "alerts": alerts,
        "lab_trends": lab_metrics,
        "audio_summary": audio_summary,
        "clinical_summary": clinical_summary,
    }

class AIChatModel(BaseModel):
    message: str
    history: list = []

@router.post("/doctor/patients/{recipient_id}/ai_chat")
def patient_ai_chat(recipient_id: int, model: AIChatModel, db: Session = Depends(get_db)):
    # 1. Get patient context using existing method
    try:
        context = get_patient_clinical_summary(recipient_id, db)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Patient not found")

    # 2. Call Gemini
    from utils.summarizer import chat_with_patient_ai
    
    answer = chat_with_patient_ai(
        patient_context=context,
        chat_history=model.history,
        user_message=model.message
    )
    
    return {"reply": answer}

@router.post("/doctor/patients/{recipient_id}/remarks")
def update_doctor_remarks(recipient_id: int, model: RemarksModel, db: Session = Depends(get_db)):
    from tables.users import CareRecipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Patient not found")
    recipient.doctor_remarks = model.remarks
    db.commit()
    return {"message": "Remarks updated successfully"}

@router.post("/doctor/patients/{recipient_id}/prescribe")
def prescribe_medication(recipient_id: int, model: PrescriptionModel, db: Session = Depends(get_db)):
    from tables.users import CareRecipient
    from tables.medications import Medication, MedicationStatus
    from tables.medical_conditions import MedicalAlert, AlertType, AlertSeverity

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # 1. Create Medication
    new_med = Medication(
        care_recipient_id=recipient_id,
        medicine_name=model.medicine_name,
        dosage=model.dosage,
        frequency=model.frequency,
        schedule_time=model.schedule_time,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=model.duration_days),
        status=MedicationStatus.active,
        current_stock=model.current_stock,
        doses_per_day=model.doses_per_day,
        auto_order_enabled=True
    )
    db.add(new_med)

    # 2. Create Notification/Alert for the Patient
    alert = MedicalAlert(
        care_recipient_id=recipient_id,
        alert_type=AlertType.new_diagnosis, # Using new_diagnosis as a filler for now, or just a generic one
        message=f"Dr. has prescribed new medication: {model.medicine_name} ({model.dosage}). Please follow the schedule: {model.frequency}.",
        severity=AlertSeverity.medium,
        is_read=False
    )
    db.add(alert)

    db.commit()
    return {"message": "Prescription added and notification sent successfully"}

@router.post("/doctor/patients/{recipient_id}/lab_order")
def order_lab_test(recipient_id: int, model: LabOrderModel, db: Session = Depends(get_db)):
    from tables.users import CareRecipient
    from tables.medical_conditions import LabOrderDetail, MedicalAlert, AlertType, AlertSeverity
    
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # 1. Create Lab Order record
    new_order = LabOrderDetail(
        care_recipient_id=recipient_id,
        test_name=model.test_name,
        doctor_notes=model.doctor_notes,
        status="pending"
    )
    db.add(new_order)

    # 2. Create Notification/Alert for the Caretaker
    alert = MedicalAlert(
        care_recipient_id=recipient_id,
        alert_type=AlertType.monitoring_gap, 
        message=f"Dr. has ordered a new lab test: {model.test_name}. Please book an appointment with Dr. Lal PathLabs.",
        severity=AlertSeverity.medium,
        is_read=False
    )
    db.add(alert)

    db.commit()
    return {
        "message": "Lab order recorded. Redirecting to booking service...",
        "redirect_url": "https://www.lalpathlabs.com/"
    }


@router.get("/doctors")
def get_all_doctors(db: Session = Depends(get_db)):
    """
    Returns all registered doctors for display on the caretaker main dashboard.
    Includes name, specialization, email, and patient count.
    """
    from tables.users import Doctor
    doctors = db.query(Doctor).all()
    result = []
    for d in doctors:
        result.append({
            "id": d.id,
            "name": d.full_name or d.username,
            "email": d.email,
            "specialization": d.specialization or "General Practitioner",
            "phone": d.phone_number,
        })
    return {"doctors": result, "total": len(result)}
