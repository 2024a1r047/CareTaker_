from fastapi import APIRouter, Depends, File, UploadFile, Header, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, Dict
import datetime

from models.users import ResponseSchema, RecipientUpdate, CareRecipientCreate, ConditionInput, MedicationInput, AllergyInput
from tables.users import CareRecipient, CareTaker
from config import get_db
from repository.users import UsersRepo
from repository.medical_reports import create_medical_report, list_reports_for_recipient, delete_medical_report
from utils.summarizer import extract_text_from_bytes, summarize_text_via_gemini, summarize_report_insights_via_gemini
from fastapi.responses import StreamingResponse
from io import BytesIO
from tables.medical_reports import MedicalReport
from pydantic import BaseModel


router = APIRouter(tags=["Recipients"])


def _run_medical_history_pipeline(recipient_id: int, report_id: int, filename: str):
    """Async background task. Runs V2 hybrid pipeline; falls back to V1 on failure."""
    try:
        _run_medical_history_pipeline_v2(recipient_id, report_id, filename)
    except Exception:
        _run_medical_history_pipeline_v1_legacy(recipient_id, report_id, filename)


def _run_medical_history_pipeline_v2(recipient_id: int, report_id: int, filename: str):
    """V2 hybrid pipeline: deterministic lab extraction + Gemini for clinical context."""
    from config import SessionLocal
    db = SessionLocal()
    try:
        from tables.medical_reports import MedicalReport, ReportProcessingStatus
        from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity, SourceType
        from services.report_ingestion import run_hybrid_lab_extraction, extract_structured_report
        from services.disease_detection import detect_diseases_from_report
        from services.disease_progression import analyze_progression
        from services.alert_engine import generate_alerts, check_monitoring_gaps
        from services.medical_history_ai import calculate_risk_score
        from repository import medical_history as repo
        import datetime

        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report:
            return

        report.processing_status = ReportProcessingStatus.processing
        db.flush()

        # ── 1. Strategy Decision: Multimodal Vision vs Text Map-Reduce ────────
        text = ""
        is_scanned = False
        if report.data:
            try:
                # Try fast extraction first
                text = extract_text_from_bytes(report.data, filename)
                # If text is extremely short for a medical report, assume it's scanned/image-based
                if len(text.strip()) < 250:
                    print(f"[pipeline_v3] Text too sparse ({len(text)} chars). Switching to VISION.")
                    is_scanned = True
            except Exception as e:
                print(f"[pipeline_v3] Baseline extraction failed: {e}. Falling back to VISION.")
                is_scanned = True

        upload_date = report.uploaded_at.date() if report.uploaded_at else datetime.date.today()
        
        if is_scanned and report.data:
            # ── 2A. VISION PIPELINE (Multimodal) ──────────────────────────────
            from services.report_ingestion import run_vision_lab_extraction
            hybrid_result = run_vision_lab_extraction(
                image_bytes=report.data,
                mime_type=report.mime_type or "image/jpeg",
                report_id=report_id,
                care_recipient_id=recipient_id,
                upload_date=upload_date,
                db=db
            )
            report_date_obj = hybrid_result["report_date"]
            
            # Since we have no text for the legacy clinical extractor, we rely on Vision's notes
            extracted = {
                "report_date": str(report_date_obj),
                "diagnoses": [],
                "resolved_diagnoses": [],
                "lab_values": {
                    row["metric_name"]: {"value": row["normalized_value"], "unit": row["normalized_unit"]}
                    for row in hybrid_result["lab_rows"]
                },
                "medications": [],
                "doctor_notes": hybrid_result.get("clinical_notes", "Vision-extracted report"),
                "symptoms": [],
                "extraction_template": "VISION_MULTIMODAL"
            }
        else:
            # ── 2B. TEXT PIPELINE (Map-Reduce Hybrid) ─────────────────────────
            from services.report_ingestion import run_hybrid_lab_extraction
            hybrid_result = run_hybrid_lab_extraction(
                text=text, report_id=report_id,
                care_recipient_id=recipient_id, upload_date=upload_date, db=db,
            )
            report_date_obj = hybrid_result["report_date"]
            
            # Clinical context map-reduce
            report_date_str = str(report_date_obj) if report_date_obj else None
            extracted = extract_structured_report(text, report_date_hint=report_date_str)
            extracted["lab_values"] = {
                row["metric_name"]: {"value": row["normalized_value"], "unit": row["normalized_unit"]}
                for row in hybrid_result["lab_rows"]
            }
            extracted["extraction_template"] = hybrid_result["template"]

        if report_date_obj:
            report.report_date = report_date_obj
        report.extracted_data = extracted

        # ── 4. Disease detection ──────────────────────────────────────────────
        existing = repo.get_all_conditions(db, recipient_id)
        new_diseases = detect_diseases_from_report(
            extracted, existing, db, report_id,
            report_date_str or extracted.get("report_date")
        )
        for disease in new_diseases:
            cond = PatientCondition(
                care_recipient_id=recipient_id,
                disease_code=disease["disease_code"],
                disease_name=disease["disease_name"],
                status=ConditionStatus.active,
                severity=ConditionSeverity.moderate,
                first_detected=disease.get("first_detected", datetime.date.today()),
                last_updated=disease.get("first_detected", datetime.date.today()),
                baseline_value=disease.get("baseline_value"),
                baseline_date=disease.get("baseline_date"),
                confidence_score=disease.get("confidence_score", 0.5),
                source_type=SourceType(disease.get("source_type", "lab_inferred")),
                source_report_id=report_id,
            )
            db.add(cond)
        db.flush()

        # ── 5–7. Progression, alerts, risk score ──────────────────────────────
        progression = analyze_progression(
            recipient_id, extracted, report_id,
            report_date_str or extracted.get("report_date", ""), db
        )
        generate_alerts(recipient_id, progression, db)
        check_monitoring_gaps(recipient_id, db)
        calculate_risk_score(recipient_id, db)

        # ── 8. Update last_report_date ────────────────────────────────────────
        from tables.users import CareRecipient
        rec = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if rec:
            rec.last_report_date = datetime.datetime.utcnow()

        report.processing_status = ReportProcessingStatus.completed
        db.commit()
        print(f"[pipeline_v2] ✅ Report {report_id} — "
              f"{hybrid_result['saved_count']} lab values, {len(new_diseases)} new conditions")

        # ── 9. Aggregate summary ──────────────────────────────────────────────
        try:
            _recalculate_aggregate_summary(db, recipient_id)
        except Exception as e:
            print(f"[pipeline_v2] summary failed: {e}")

        # ── 10. Clinical Recommendations ──────────────────────────────────────
        try:
            from services.insights_engine import run_recommendation_pipeline
            import asyncio
            # Since this is a background task thread, we use asyncio.run
            asyncio.run(run_recommendation_pipeline(str(recipient_id), db))
            _process_recommendation_actions(recipient_id, db)

        except Exception as e:
            print(f"[pipeline_v2] recommendations failed: {e}")

    except Exception as e:
        print(f"[pipeline_v2] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        db.rollback()
        try:
            rep = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
            if rep:
                rep.processing_status = ReportProcessingStatus.failed
                db.commit()
        except Exception:
            pass
        raise   # Re-raise so the outer dispatcher triggers v1 fallback
    finally:
        db.close()


def _run_medical_history_pipeline_v1_legacy(recipient_id: int, report_id: int, filename: str):
    """V1 legacy fallback — pure Gemini extraction (used only if v2 fails entirely)."""
    from config import SessionLocal
    db = SessionLocal()
    try:
        from tables.medical_reports import MedicalReport, ReportProcessingStatus
        from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity, SourceType
        from services.report_ingestion import extract_structured_report
        from services.disease_detection import detect_diseases_from_report
        from services.disease_progression import analyze_progression
        from services.alert_engine import generate_alerts, check_monitoring_gaps
        from services.medical_history_ai import calculate_risk_score
        from repository import medical_history as repo
        import datetime

        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report:
            return

        text = extract_text_from_bytes(report.data, filename) if report.data else ""
        if not text:
            report.processing_status = ReportProcessingStatus.failed
            db.commit()
            return

        extracted = extract_structured_report(text)
        report.extracted_data = extracted
        if extracted.get("report_date"):
            try:
                report.report_date = datetime.date.fromisoformat(extracted["report_date"])
            except (ValueError, TypeError):
                pass

        existing = repo.get_all_conditions(db, recipient_id)
        new_diseases = detect_diseases_from_report(extracted, existing, db, report_id, extracted.get("report_date"))
        for disease in new_diseases:
            db.add(PatientCondition(
                care_recipient_id=recipient_id,
                disease_code=disease["disease_code"],
                disease_name=disease["disease_name"],
                status=ConditionStatus.active,
                severity=ConditionSeverity.moderate,
                first_detected=disease.get("first_detected", datetime.date.today()),
                last_updated=disease.get("first_detected", datetime.date.today()),
                confidence_score=disease.get("confidence_score", 0.5),
                source_type=SourceType(disease.get("source_type", "lab_inferred")),
                source_report_id=report_id,
            ))
        db.flush()
        progression = analyze_progression(recipient_id, extracted, report_id, extracted.get("report_date", ""), db)
        generate_alerts(recipient_id, progression, db)
        check_monitoring_gaps(recipient_id, db)
        calculate_risk_score(recipient_id, db)

        from tables.users import CareRecipient
        rec = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if rec:
            rec.last_report_date = datetime.datetime.utcnow()

        report.processing_status = ReportProcessingStatus.completed
        db.commit()
        print(f"[pipeline_v1] ✅ Report {report_id} complete (legacy mode)")
        try:
            _recalculate_aggregate_summary(db, recipient_id)
        except Exception:
            pass
    except Exception as e:
        print(f"[pipeline_v1] ❌ Error: {e}")
        db.rollback()
        try:
            rep = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
            if rep:
                rep.processing_status = ReportProcessingStatus.failed
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        parts = auth_header.split()
        if len(parts) != 2:
            return None
        token = parts[1]
        from repository.users import JWTRepo
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None


def _recalculate_aggregate_summary(db: Session, recipient_id: int):
    """Helper to ensure longitudinal history is ALWAYS up to date after any change."""
    try:
        recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if not recipient:
            return

        reports = list_reports_for_recipient(db, recipient_id)
        if not reports:
            recipient.report_summary = None
            db.add(recipient)
            db.commit()
            return

        insights = []
        print(f'[recipients] Recalculating history for {len(reports)} reports...')
        for r in reports:
            if not r.analysis_summary:
                txt = extract_text_from_bytes(r.data or b'', r.mime_type)
                if txt:
                    r.analysis_summary = summarize_text_via_gemini(txt, target_words=150)
                    db.add(r)
            
            if r.analysis_summary:
                insights.append(r.analysis_summary)
        
        if insights:
            longitudinal = summarize_report_insights_via_gemini(insights)
            recipient.report_summary = longitudinal
            db.add(recipient)
            db.commit()
            print(f"[recipients] Longitudinal history updated for recipient {recipient_id}")
    except Exception as e:
        print(f"[recipients] Error during aggregate recalculation: {e}")


@router.post('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
async def upload_medical_report(recipient_id: int, background_tasks: BackgroundTasks, file: UploadFile = File(...), authorization: Optional[str] = Header(None), db: Session = Depends(get_db), request: Request = None):
    # Auth
    # Debug: log some request header info
    try:
        hdrs = dict(request.headers)
        print(f"[recipients] Request headers: Authorization={'present' if hdrs.get('authorization') else 'missing'}, Content-Type={hdrs.get('content-type')}")
    except Exception:
        pass

    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Ensure recipient belongs to caretaker
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    # Basic validation
    content = await file.read()
    max_size = 5 * 1024 * 1024  # 5 MB
    if len(content) > max_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # include application/octet-stream to handle some browsers that don't set a specific mime
    allowed = ['application/pdf', 'image/png', 'image/jpeg', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/octet-stream']
    mime = file.content_type or 'application/octet-stream'
    if mime not in allowed:
        # allow some generic types but prefer strict list
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {mime}")

    # Save to DB
    try:
        # Debug: log mime and size
        try:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} mime={mime} size={len(content)} bytes")
        except Exception:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} (could not determine size)")

        report = create_medical_report(db, recipient_id, file.filename, mime, content)
        print(f"[recipients] Uploaded report id={report.id} recipient_id={recipient_id} filename={file.filename}")

        # Launch async medical history pipeline (v3)
        report_id = report.id
        if background_tasks:
            background_tasks.add_task(
                _run_medical_history_pipeline,
                recipient_id=recipient_id,
                report_id=report_id,
                filename=file.filename
            )
            print(f"[recipients] Medical history pipeline queued for report {report_id}")

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename, 'processing_status': 'processing'})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")


@router.delete('/recipients/{recipient_id}/reports/{report_id}', response_model=ResponseSchema)
def remove_medical_report(recipient_id: int, report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete a medical report."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    try:
        if delete_medical_report(db, report_id):
            # Auto-update summary after deletion
            _recalculate_aggregate_summary(db, recipient_id)
            # Auto-recalculate risk score and analysis after deletion
            try:
                from services.medical_history_ai import calculate_risk_score
                calculate_risk_score(recipient_id, db)
                db.commit()
                print(f"[recipients] Risk score recalculated after report {report_id} deletion")
            except Exception as re:
                print(f"[recipients] Risk recalculation after delete failed: {re}")
            return ResponseSchema(code=200, status='success', message='Report deleted successfully')
        else:
            raise HTTPException(status_code=500, detail="Failed to delete report")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion error: {e}")


@router.get('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
def list_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    reports = list_reports_for_recipient(db, recipient_id)
    print(f"[recipients] Listing {len(reports)} reports for recipient_id={recipient_id} (requested by {username})")
    out = [{
        'id': r.id, 
        'filename': r.filename, 
        'mime_type': r.mime_type, 
        'uploaded_at': r.uploaded_at.isoformat(),
        'analysis_summary': r.analysis_summary,
        'processing_status': r.processing_status.value if r.processing_status else 'unknown',
        'report_date': r.report_date.isoformat() if r.report_date else None,
    } for r in reports]
    return ResponseSchema(code=200, status='success', message='Reports fetched', result={'reports': out})


@router.get('/debug/medical_reports/inspect')
def debug_inspect_reports(recipient_id: Optional[int] = None, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Temporary debug endpoint: returns DB url and counts. Requires authentication."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    try:
        import config
        from tables.medical_reports import MedicalReport
        total = db.query(MedicalReport).count()
        recip_count = None
        if recipient_id is not None:
            recip_count = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).count()
        return {
            'db_url': str(getattr(config, 'DATABASE_URL', 'unknown')),
            'engine_url': str(getattr(getattr(config, 'engine', ''), 'url', 'unknown')),
            'total_reports': total,
            'recipient_reports': recip_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug inspect failed: {e}")


@router.post('/recipients/{recipient_id}/summarize', response_model=ResponseSchema)
def summarize_recipient_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Re-run summarization for an existing recipient's uploaded reports and save the summary."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        _recalculate_aggregate_summary(db, recipient_id)
        # return the updated summary
        db.refresh(recipient)
        return ResponseSchema(code=200, status='success', message='Longitudinal history synthesized', result={'summary': recipient.report_summary or ''})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to summarize reports: {e}")


@router.delete('/recipients/{recipient_id}/alerts/{alert_id}', response_model=ResponseSchema)
def remove_alert(recipient_id: int, alert_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete a medical alert."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    from tables.medical_conditions import MedicalAlert
    alert = db.query(MedicalAlert).filter_by(id=alert_id, care_recipient_id=recipient_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    db.delete(alert)
    db.commit()
    return ResponseSchema(code=200, status='success', message='Alert deleted successfully')


@router.get('/recipients/{recipient_id}/reports/{report_id}/download')
def download_report(recipient_id: int, report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Stream an individual report back to the authenticated caretaker if they own the recipient."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(__import__('tables.medical_reports', fromlist=['MedicalReport']).MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    data = report.data or b''
    stream = BytesIO(data)
    headers = {
        'Content-Disposition': f'attachment; filename="{report.filename}"'
    }
    return StreamingResponse(stream, media_type=report.mime_type or 'application/octet-stream', headers=headers)


@router.get('/recipients/{recipient_id}/reports/{report_id}/extract_preview', response_model=ResponseSchema)
def extract_preview(recipient_id: int, report_id: int, force_refresh: bool = False, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Return a short preview of extracted text for a given report (debug endpoint)."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    try:
        # Check cache if not forcing refresh
        if not force_refresh and report.analysis_summary:
            print(f"[recipients] Returning cached snapshot for report_id={report_id}")
            return ResponseSchema(code=200, status='success', message='Cached preview', result={'extracted_preview': report.analysis_summary})

        extracted = extract_text_from_bytes(report.data or b'', report.mime_type)
        if not extracted:
             return ResponseSchema(code=200, status='success', message='No text found', result={'extracted_preview': 'No textual content could be extracted from this file.'})
             
        # Instead of raw preview, return a clinical snapshot
        print(f"[recipients] Generating clinical snapshot for report_id={report_id} (force_refresh={force_refresh})")
        summary = summarize_text_via_gemini(extracted, target_words=150)
        
        # Save to cache
        report.analysis_summary = summary
        db.add(report)
        db.commit()
        
        return ResponseSchema(code=200, status='success', message='Extract preview', result={'extracted_preview': summary})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to generate report preview: {e}")


@router.post('/recipients/{recipient_id}/reports/base64', response_model=ResponseSchema)
def upload_report_base64(recipient_id: int, payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Accept a JSON payload with base64-encoded file to simplify client uploads.
    Payload: { filename: str, mime_type: str, b64: str }
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        print(f"[recipients] Received base64 upload request for recipient_id={recipient_id} filename={payload.get('filename')}")
        filename = payload.get('filename')
        mime = payload.get('mime_type') or 'application/octet-stream'
        b64 = payload.get('b64')
        if not filename or not b64:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Missing filename or b64 payload')
        import base64
        data = base64.b64decode(b64)
        report = create_medical_report(db, recipient_id, filename, mime, data)

        # Auto-update summary
        _recalculate_aggregate_summary(db, recipient_id)

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")


@router.post('/recipients', response_model=ResponseSchema)
def add_care_recipient(payload: CareRecipientCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Add a new care recipient for the current caretaker."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_recipient = CareRecipient(
        caretaker_id=caretaker.id,
        full_name=payload.full_name,
        email=payload.email,
        phone_number=payload.phone_number,
        age=payload.age,
        gender=payload.gender,
        city=payload.city,
        height=payload.height,
        weight=payload.weight,
        blood_group=payload.blood_group,
        emergency_contact=payload.emergency_contact,
        respiratory_condition_status=payload.respiratory_condition_status
    )
    db.add(new_recipient)
    db.commit()
    db.refresh(new_recipient)

    return ResponseSchema(code=200, status="success", message="Care recipient added", result={"id": new_recipient.id})


@router.patch('/recipients/{recipient_id}', response_model=ResponseSchema)
def update_care_recipient(recipient_id: int, payload: RecipientUpdate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Update care recipient details."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=404, detail="Caretaker not found")
        
    print(f"[recipients] PATCH update attempt: recipient_id={recipient_id}, user={username}, caretaker_id={caretaker.id}")
    
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        print(f"[recipients] PATCH failed: Recipient {recipient_id} not found for caretaker {caretaker.id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    if payload.full_name is not None: recipient.full_name = payload.full_name
    if payload.email is not None: recipient.email = payload.email
    if payload.phone_number is not None: recipient.phone_number = payload.phone_number
    if payload.age is not None: recipient.age = payload.age
    if payload.gender is not None: recipient.gender = payload.gender
    if payload.city is not None: recipient.city = payload.city
    if payload.height is not None: recipient.height = payload.height
    if payload.weight is not None: recipient.weight = payload.weight
    if payload.blood_group is not None: recipient.blood_group = payload.blood_group
    if payload.emergency_contact is not None: recipient.emergency_contact = payload.emergency_contact
    if payload.respiratory_condition_status is not None: recipient.respiratory_condition_status = payload.respiratory_condition_status

    db.commit()
    return ResponseSchema(code=200, status="success", message="Recipient updated")


@router.delete('/recipients/{recipient_id}', response_model=ResponseSchema, name="remove_recipient_account")
def remove_care_recipient_account(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete a care recipient and all their data (recordings, reports, analysis)."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    db.delete(recipient)
    db.commit()
    return ResponseSchema(code=200, status="success", message="Recipient removed successfully")


# ═══════════════════════════════════════════════════════════════════════
#  SENSOR FUSION INSIGHTS ENGINE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@router.get("/recipients/{recipient_id}/insights")
async def get_patient_insights(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Full Sensor Fusion Insights — aggregates ALL data sources (vitals, audio,
    video, medical reports, environment) and generates cross-domain health
    conclusions using deterministic rules + Gemini AI interpretation.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    try:
        from services.insights_engine import generate_full_insights
        result = generate_full_insights(recipient_id, db)
        return result
    except Exception as e:
        print(f"[insights] Error generating insights: {e}")
        import traceback
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


@router.get("/recipients")
@router.get("/care-recipients")
async def list_care_recipients(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=404, detail="User not found")
    recipients = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).all()
    return [
        {
            "id": r.id,
            "full_name": r.full_name,
            "name": r.full_name,
            "age": r.age,
            "gender": r.gender.value if r.gender else None,
            "risk_score": r.risk_score
        } for r in recipients
    ]

@router.get("/care-recipients/{recipient_id}/profile")
async def get_care_recipient_profile(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Returns the complete longitudinal health record for a care recipient.
    Powers the unified Care-Recipient Profile Dashboard.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    from tables.medical_conditions import PatientCondition, ConditionStatus, LabValue, MedicalAlert
    from tables.medical_reports import MedicalReport
    from tables.allergies import Allergy
    from tables.medications import Medication, MedicationHistory
    from tables.vital_signs import VitalSign
    from tables.environment import EnvironmentSensor

    # 1. General Info
    general_info = {
        "id": recipient.id,
        "name": recipient.full_name,
        "preferred_name": recipient.preferred_name,
        "age": recipient.age,
        "gender": recipient.gender.value if recipient.gender else None,
        "height": recipient.height,
        "weight": recipient.weight,
        "blood_group": recipient.blood_group,
        "phone_number": recipient.phone_number,
        "email": recipient.email,
        "city": recipient.city,
        "emergency_contact": recipient.emergency_contact,
        "respiratory_condition_status": recipient.respiratory_condition_status,
        "registration_date": str(recipient.registration_date) if recipient.registration_date else None,
        "risk_score": recipient.risk_score,
        "risk_factors": recipient.risk_factors_breakdown.get("factors", []) if isinstance(recipient.risk_factors_breakdown, dict) else [],
        "doctor_remarks": recipient.doctor_remarks
    }

    # 2. Active Conditions
    active_conditions_raw = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()
    active_conditions = [
        {
            "id": c.id,
            "disease_name": c.disease_name,
            "status": c.status.value,
            "severity": c.severity.value if c.severity else None,
            "start_date": str(c.first_detected) if c.first_detected else None
        } for c in active_conditions_raw
    ]

    # 3. Medical History (Resolved)
    medical_history_raw = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status == ConditionStatus.resolved
    ).all()
    medical_history = [
        {
            "id": c.id,
            "disease_name": c.disease_name,
            "resolved_date": str(c.resolved_date) if c.resolved_date else None,
            "start_date": str(c.first_detected) if c.first_detected else None
        } for c in medical_history_raw
    ]

    # 4. Allergies
    allergies_raw = db.query(Allergy).filter(Allergy.care_recipient_id == recipient_id).all()
    allergies = [
        {
            "id": a.allergy_id,
            "allergen": a.allergen,
            "type": a.allergy_type.value if a.allergy_type else None,
            "reaction": a.reaction,
            "severity": a.severity,
            "status": a.status.value if a.status else None
        } for a in allergies_raw
    ]

    # 5. Medications
    meds_raw = db.query(Medication).filter(Medication.care_recipient_id == recipient_id).all()
    active_medications = [
        {
            "id": m.medication_id,
            "medicine_name": m.medicine_name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "schedule_time": m.schedule_time,
            "start_date": str(m.start_date) if m.start_date else None,
            "end_date": str(m.end_date) if m.end_date else None,
            "duration_days": (m.end_date - m.start_date).days if m.end_date and m.start_date else None,
            "status": m.status.value if m.status else None,
            "current_stock": m.current_stock or 0,
            "doses_per_day": m.doses_per_day or 1,
            "predicted_finish_date": str(datetime.date.today() + datetime.timedelta(days=(m.current_stock or 0) // (m.doses_per_day or 1))) if m.current_stock and m.doses_per_day and m.doses_per_day > 0 else None,
            "auto_order_enabled": getattr(m, 'auto_order_enabled', True),
            "last_auto_order_date": str(m.last_auto_order_date) if getattr(m, 'last_auto_order_date', None) else None
        } for m in meds_raw
    ]

    # 6. Medication History
    med_history_raw = db.query(MedicationHistory).filter(MedicationHistory.care_recipient_id == recipient_id).all()
    medication_history = [
        {
            "id": h.history_id,
            "medicine_name": h.medicine_name,
            "dosage": h.dosage,
            "end_date": str(h.end_date) if h.end_date else None,
            "termination_reason": h.termination_reason
        } for h in med_history_raw
    ]

    # 7. Medical Reports
    reports_raw = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).all()
    medical_reports = [
        {
            "id": r.id,
            "filename": r.filename,
            "uploaded_at": str(r.uploaded_at) if r.uploaded_at else None,
            "report_date": str(r.report_date) if r.report_date else None,
            "status": r.processing_status.value if r.processing_status else None
        } for r in reports_raw
    ]

    # 8. Test Trends (Lab Results) — Joined with Reports for authoritative chronological ordering
    from tables.medical_reports import MedicalReport
    labs_raw = db.query(LabValue).outerjoin(
        MedicalReport, LabValue.report_id == MedicalReport.id
    ).filter(
        LabValue.care_recipient_id == recipient_id
    ).order_by(
        MedicalReport.report_date.asc(),
        LabValue.recorded_date.asc(),
        LabValue.id.asc()
    ).all()

    test_trends = [
        {
            "id": l.id,
            "test_name": l.metric_name,
            "test_value": l.normalized_value,
            "test_unit": l.normalized_unit,
            "test_date": str(l.recorded_date or (l.report.report_date if l.report else None)),
            "is_abnormal": l.is_abnormal,
            "report_id": l.report_id,
            "reference_low": l.reference_range_low,
            "reference_high": l.reference_range_high
        } for l in labs_raw
    ]

    # 9. Vitals (Recent 30)
    vitals_raw = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id).order_by(VitalSign.recorded_at.desc()).limit(30).all()
    vitals = [
        {
            "recorded_at": str(v.recorded_at) if v.recorded_at else None,
            "heart_rate": v.heart_rate,
            "systolic_bp": v.systolic_bp,
            "diastolic_bp": v.diastolic_bp,
            "oxygen_saturation": v.oxygen_saturation,
            "temperature": v.temperature,
            "sleep_score": v.sleep_score,
            "bmi": v.bmi
        } for v in vitals_raw
    ]
    vitals.reverse() # Chronological

    # 10. Environment (Recent 24 hours or latest 20)
    env_raw = db.query(EnvironmentSensor).filter(EnvironmentSensor.care_recipient_id == recipient_id).order_by(EnvironmentSensor.timestamp.desc()).limit(20).all()
    environment = [
        {
            "timestamp": str(e.timestamp) if e.timestamp else None,
            "temperature_c": e.temperature_c,
            "humidity_percent": e.humidity_percent,
            "aqi": e.aqi
        } for e in env_raw
    ]
    environment.reverse()

    # 11. Alerts (Unread & Recent)
    alerts_raw = db.query(MedicalAlert).filter(MedicalAlert.care_recipient_id == recipient_id).order_by(MedicalAlert.created_at.desc()).limit(15).all()
    alerts = [
        {
            "id": a.id,
            "severity": a.severity.value if a.severity else None,
            "message": a.message,
            "is_read": a.is_read,
            "created_at": str(a.created_at) if a.created_at else None
        } for a in alerts_raw
    ]

    # Caretaker Account Info
    caretaker_info = {
        "id": caretaker.id,
        "full_name": caretaker.full_name or caretaker.username,
        "username": caretaker.username,
        "email": caretaker.email or "",
        "phone_number": caretaker.phone_number or ""
    }

    # All assigned recipients for switching
    all_recipients_raw = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).all()
    all_recipients = [
        {
            "id": r.id,
            "name": r.full_name,
            "age": r.age,
            "gender": r.gender.value if r.gender else None
        } for r in all_recipients_raw
    ]

    return {
        "status": "success",
        "caretaker": caretaker_info,
        "recipients": all_recipients,
        "general_info": general_info,
        "conditions": active_conditions,
        "medical_history": medical_history,
        "allergies": allergies,
        "medications": active_medications,
        "medication_history": medication_history,
        "medical_reports": medical_reports,
        "lab_values": test_trends,
        "vitals": vitals,
        "environment": environment,
        "alerts": alerts
    }


# --- Medical Record Management ---

# Conditions
@router.post("/care-recipients/{recipient_id}/conditions", response_model=ResponseSchema)
async def add_condition(recipient_id: int, data: ConditionInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    # Simple check for recipient
    from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity
    import datetime
    
    cond = PatientCondition(
        care_recipient_id=recipient_id,
        disease_code=data.disease_code or "CUSTOM",
        disease_name=data.disease_name,
        status=ConditionStatus(data.status),
        severity=ConditionSeverity(data.severity) if data.severity else None,
        first_detected=datetime.date.fromisoformat(data.first_detected) if data.first_detected else datetime.date.today(),
        last_updated=datetime.date.today()
    )
    db.add(cond)
    db.commit()
    return {"code": 200, "status": "success", "message": "Condition added", "result": {"id": cond.id}}

@router.patch("/care-recipients/{recipient_id}/conditions/{condition_id}", response_model=ResponseSchema)
async def update_condition(recipient_id: int, condition_id: int, data: ConditionInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity
    cond = db.query(PatientCondition).filter(PatientCondition.id == condition_id, PatientCondition.care_recipient_id == recipient_id).first()
    if not cond: raise HTTPException(404, "Condition not found")
    
    cond.disease_name = data.disease_name
    cond.status = ConditionStatus(data.status)
    if data.severity: cond.severity = ConditionSeverity(data.severity)
    db.commit()
    return {"code": 200, "status": "success", "message": "Condition updated"}


# Medications
@router.post("/care-recipients/{recipient_id}/medications", response_model=ResponseSchema)
async def add_medication(recipient_id: int, data: MedicationInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medications import Medication, MedicationStatus
    import datetime
    # Parse start_date from request or default to today
    start = datetime.date.today()
    if data.start_date:
        try:
            start = datetime.date.fromisoformat(data.start_date)
        except ValueError:
            pass
    # Compute end_date only if duration_days is set (None = lifetime)
    end = None
    if data.duration_days and data.duration_days > 0:
        end = start + datetime.timedelta(days=data.duration_days)

    med = Medication(
        care_recipient_id=recipient_id,
        medicine_name=data.medicine_name,
        dosage=data.dosage,
        frequency=data.frequency,
        schedule_time=data.schedule_time,
        start_date=start,
        end_date=end,
        status=MedicationStatus(data.status),
        current_stock=data.current_stock or 0,
        doses_per_day=data.doses_per_day or 1,
        auto_order_enabled=data.auto_order_enabled if data.auto_order_enabled is not None else True
    )
    db.add(med)
    db.commit()
    return {"code": 200, "status": "success", "message": "Medication added", "result": {"id": med.medication_id}}

@router.patch("/care-recipients/{recipient_id}/medications/{med_id}", response_model=ResponseSchema)
async def update_medication(recipient_id: int, med_id: int, data: MedicationInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medications import Medication, MedicationStatus
    import datetime
    med = db.query(Medication).filter(Medication.medication_id == med_id, Medication.care_recipient_id == recipient_id).first()
    if not med:
        raise HTTPException(404, "Medication not found")

    med.medicine_name = data.medicine_name
    med.dosage = data.dosage
    med.frequency = data.frequency
    med.schedule_time = data.schedule_time
    med.status = MedicationStatus(data.status)
    med.current_stock = data.current_stock or 0
    med.doses_per_day = data.doses_per_day or 1
    if data.auto_order_enabled is not None:
        med.auto_order_enabled = data.auto_order_enabled

    # Update start_date if provided
    if data.start_date:
        try:
            med.start_date = datetime.date.fromisoformat(data.start_date)
        except ValueError:
            pass
    if not med.start_date:
        med.start_date = datetime.date.today()

    # Recompute end_date: None = lifetime (never auto-complete)
    if data.duration_days and data.duration_days > 0 and med.start_date:
        med.end_date = med.start_date + datetime.timedelta(days=data.duration_days)
    else:
        med.end_date = None  # lifetime — clear any previous end_date

    db.commit()
    return {"code": 200, "status": "success", "message": "Medication updated"}


@router.patch("/care-recipients/{recipient_id}/medications/{med_id}/auto-order", response_model=ResponseSchema)
async def toggle_auto_order(recipient_id: int, med_id: int, payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Toggle auto-reorder via Tata 1mg for a specific medication."""
    from tables.medications import Medication
    med = db.query(Medication).filter(
        Medication.medication_id == med_id,
        Medication.care_recipient_id == recipient_id
    ).first()
    if not med:
        raise HTTPException(404, "Medication not found")
    
    enabled = payload.get("auto_order_enabled")
    if enabled is None:
        raise HTTPException(400, "Missing auto_order_enabled in payload")
    
    med.auto_order_enabled = bool(enabled)
    db.commit()
    return {"code": 200, "status": "success", "message": f"Auto-order {'enabled' if enabled else 'disabled'} for {med.medicine_name}"}



# Allergies
@router.post("/care-recipients/{recipient_id}/allergies", response_model=ResponseSchema)
async def add_allergy(recipient_id: int, data: AllergyInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.allergies import Allergy, AllergyType, AllergyStatus
    allg = Allergy(
        care_recipient_id=recipient_id,
        allergen=data.allergen,
        allergy_type=AllergyType(data.allergy_type),
        reaction=data.reaction,
        severity=data.severity,
        status=AllergyStatus(data.status)
    )
    db.add(allg)
    db.commit()
    return {"code": 200, "status": "success", "message": "Allergy added", "result": {"id": allg.allergy_id}}

@router.patch("/care-recipients/{recipient_id}/allergies/{all_id}", response_model=ResponseSchema)
async def update_allergy(recipient_id: int, all_id: int, data: AllergyInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.allergies import Allergy, AllergyType, AllergyStatus
    allg = db.query(Allergy).filter(Allergy.allergy_id == all_id, Allergy.care_recipient_id == recipient_id).first()
    if not allg: raise HTTPException(404, "Allergy not found")
    
    allg.allergen = data.allergen
    allg.allergy_type = AllergyType(data.allergy_type)
    allg.reaction = data.reaction
    allg.severity = data.severity
    allg.status = AllergyStatus(data.status)
    db.commit()
    return {"code": 200, "status": "success", "message": "Allergy updated"}

# Recommendations
@router.get("/care-recipients/{recipient_id}/recommendations", response_model=ResponseSchema)
@router.get("/recipients/{recipient_id}/recommendations", response_model=ResponseSchema)
async def get_recommendations(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from services.insights_engine import get_active_recommendations
    
    # Use the unified query function from insights_engine
    recs = get_active_recommendations(str(recipient_id), db, hours=48)
    
    results = []
    for r in recs:
        results.append({
            "id": r.id,
            "metric": r.metric,
            "severity": r.severity,
            "message": r.caregiver_message or r.message, 
            "do_this_now": r.do_this_now,
            
            # Enhanced Fields
            "why_this_matters": r.why_this_matters,
            "time_window": r.time_window,
            "snapshot_summary": r.snapshot_summary,
            "next_check": {
                "when": r.next_check_when,
                "look_for": r.next_check_look_for,
                "if_worse": r.next_check_if_worse
            },
            "root_cause": r.root_cause,

            "trigger_value": r.trigger_value,
            "reference_range": r.reference_range,
            "source": r.source,
            "model_used": r.model_used,
            "confidence_score": r.confidence_score,
            "actions": r.actions,
            "call_doctor": r.call_doctor == "true",
            "call_ambulance": r.call_ambulance == "true",
            "health_state": r.health_state,
            "title": r.title,
            "condition_group": r.condition_group,
            "today_actions": r.today_actions or [],
            "resolved_at": r.resolved_at.isoformat() + "Z" if r.resolved_at else None,
            "archived": r.archived,
            "created_at": (r.created_at.isoformat() + "Z") if r.created_at else None
        })

    return {"code": 200, "status": "success", "message": "Recommendations retrieved", "result": results}


@router.post("/care-recipients/{recipient_id}/recommendations/refresh", response_model=ResponseSchema)
@router.post("/recipients/{recipient_id}/recommendations/refresh", response_model=ResponseSchema)
async def refresh_recommendations(recipient_id: int, db: Session = Depends(get_db)):
    """Triggers an on-demand regeneration of clinical recommendations using the dual-model pipeline."""
    from services.insights_engine import run_recommendation_pipeline
    try:
        # Run the full end-to-end pipeline
        await run_recommendation_pipeline(str(recipient_id), db)
        
        # Keep legacy deterministic action processing for rules
        _process_recommendation_actions(recipient_id, db)
        
        return {"code": 200, "status": "success", "message": "Dual-model recommendations refreshed successfully"}
    except Exception as e:
        print(f"[rec_refresh] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/care-recipients/{recipient_id}/health-status", response_model=ResponseSchema)
async def get_health_status(recipient_id: int, db: Session = Depends(get_db)):
    """Backend endpoint for Architect Rule #11 - State of Health categorical score."""
    from services.recommendation_engine import get_state_of_health
    status = get_state_of_health(recipient_id, db)
    return {"code": 200, "status": "success", "message": "Health status retrieved", "result": status}


def _process_recommendation_actions(recipient_id: int, db: Session):
    """Detects structured payloads in recent recommendations and routes to stubs."""
    from tables.medical_recommendations import MedicalRecommendation
    import datetime
    
    # Only look at recommendations from the last 10 minutes (current run)
    since = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id,
        MedicalRecommendation.created_at >= since,
        MedicalRecommendation.action_payload != None
    ).all()
    
    for rec in recs:
        payload = rec.action_payload
        if not payload:
            continue
        action_type = payload.get("type")
        params = payload.get("suggested_params", {})
        
        print(f"[rec_actions] Processing ACTION: {action_type} for {rec.metric}")
        
        # REQUIRES_CONFIG: Add real API endpoints for Lab/Pharmacy integrations
        if action_type == "lab_order":
            # TODO: POST /api/external/diagnostics/order
            print(f"[STUB] Created Lab Order for {rec.metric}")
        elif action_type == "prescription":
            # TODO: POST /api/external/pharmacy/prescription
            print(f"[STUB] Created Prescription draft for {params.get('drug', 'medication')}")
        elif action_type == "monitor":
            # TODO: Trigger higher frequency IoT vitals polling
            print(f"[STUB] Escalated monitoring frequency for Recipient {recipient_id}")
@router.patch("/care-recipients/{recipient_id}/recommendations/{rec_id}/resolve", response_model=ResponseSchema)
async def resolve_recommendation(recipient_id: int, rec_id: int, db: Session = Depends(get_db)):
    """Mark a recommendation as resolved (caregiver completed the action)."""
    from tables.medical_recommendations import MedicalRecommendation
    import datetime
    
    rec = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.id == rec_id,
        MedicalRecommendation.care_recipient_id == recipient_id
    ).first()
    
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    rec.resolved_at = datetime.datetime.utcnow()
    db.commit()
    return {"code": 200, "status": "success", "message": "Recommendation marked as done"}


# ═══════════════════════════════════════════════════════════════════════
#  WEEKLY HEALTH REPORT — On-Demand Generate & Download
# ═══════════════════════════════════════════════════════════════════════

@router.post("/recipients/{recipient_id}/weekly-reports/generate")
async def generate_weekly_report_on_demand(
    recipient_id: int,
    force: bool = False,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Generate (or retrieve existing) weekly health report for a care recipient
    and return it as a downloadable PDF file.
    Pass ?force=true to regenerate even if a report for this week already exists.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    try:
        from services.weekly_report_generator import generate_weekly_report, _week_bounds
        from tables.weekly_reports import WeeklyReport
        import json as _json, re as _re

        if force:
            week_start, _ = _week_bounds()
            existing = db.query(WeeklyReport).filter(
                WeeklyReport.care_recipient_id == recipient_id,
                WeeklyReport.week_start == week_start
            ).first()
            if existing:
                db.delete(existing)
                db.commit()

        result = generate_weekly_report(recipient_id, db)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # ── Normalise keys: 'already_exists' uses different names ─────────────
        if result.get("status") == "already_exists":
            result["sections"]     = result.get("sections") or result.get("report_data") or {}
            result["ai_narrative"] = result.get("ai_narrative") or result.get("narrative") or {}
            result["overall_score"]= result.get("overall_score") or result.get("score") or 0

        # ── Strip non-ASCII (↑↓→—) so Helvetica doesn't crash ─────────────────
        def _sanitize(obj):
            if isinstance(obj, str):
                return obj.encode("ascii", "ignore").decode("ascii")
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(i) for i in obj]
            return obj
        result = _sanitize(result)

        pdf_bytes = _build_weekly_report_pdf(result, recipient, caretaker)
        safe_name = recipient.full_name.replace(" ", "_")
        filename = f"weekly_report_{safe_name}_{result.get('week_start', 'latest')}.pdf"

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Weekly report generation failed: {str(e)}")


def _build_weekly_report_pdf(result: dict, recipient, caretaker) -> bytes:
    """Build a clean, readable PDF weekly health report using fpdf2 (pure Python)."""
    from fpdf import FPDF

    # ── Colours (R, G, B) ──────────────────────────────────────────────────────
    C_BRAND   = (30,  58, 138)   # dark navy
    C_ACCENT  = (59, 130, 246)   # blue
    C_LIGHT   = (239, 246, 255)  # very light blue
    C_SUCCESS = (16, 185, 129)   # green
    C_WARN    = (245, 159, 11)   # amber
    C_DANGER  = (239, 68,  68)   # red
    C_MUTED   = (100, 116, 139)  # slate
    C_ROW     = (248, 250, 252)  # near-white
    C_WHITE   = (255, 255, 255)
    C_BLACK   = (15,  23,  42)

    class WeeklyReportPDF(FPDF):
        def header(self):
            pass  # custom header drawn in body
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*C_MUTED)
            self.cell(0, 5, "CareTaker.ai  |  AI-generated - consult a licensed medical professional for clinical decisions", align="C")

    pdf = WeeklyReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(20, 15, 20)

    w = pdf.w - 40   # usable width

    week_start = result.get("week_start", "")
    week_end   = result.get("week_end", "")
    now_str    = datetime.datetime.utcnow().strftime("%B %d, %Y  %H:%M UTC")

    # ── HEADER BANNER ──────────────────────────────────────────────────────────
    pdf.set_fill_color(*C_BRAND)
    pdf.rect(20, 15, w, 20, "F")
    pdf.set_xy(20, 15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(w * 0.4, 10, "CareTaker.ai", align="L", border=0)
    pdf.cell(w * 0.6, 10, "Weekly Health Report", align="R", border=0)
    pdf.ln(10)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(w * 0.4, 8, f"Patient: {recipient.full_name}", align="L", border=0)
    pdf.cell(w * 0.35, 8, f"Week: {week_start}  to  {week_end}", align="L", border=0)
    pdf.cell(w * 0.25, 8, now_str, align="R", border=0)
    pdf.ln(14)

    # ── WELLNESS SCORE ─────────────────────────────────────────────────────────
    score      = result.get("overall_score") or 0
    risk_trend = (result.get("risk_trend") or "stable").lower()
    trend_txt  = "Improving" if risk_trend == "improving" else ("Declining" if risk_trend == "declining" else "Stable")
    trend_col  = C_SUCCESS if risk_trend == "improving" else (C_DANGER if risk_trend == "declining" else C_WARN)

    pdf.set_text_color(*C_BRAND)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Wellness Score", ln=True)

    pdf.set_fill_color(*C_LIGHT)
    pdf.set_draw_color(*C_ACCENT)
    box_y = pdf.get_y()
    pdf.rect(20, box_y, w, 16, "FD")
    pdf.set_xy(20, box_y + 2)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*C_BRAND)
    pdf.cell(w * 0.25, 12, f"{score:.0f}", align="C", border=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(w * 0.15, 12, "/ 100", align="L", border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*trend_col)
    arrow = "▲" if risk_trend == "improving" else ("▼" if risk_trend == "declining" else "●")
    # fpdf2 safe: avoid non-latin chars, use text only
    trend_label = f"Trend: {trend_txt}"
    pdf.cell(w * 0.6, 12, trend_label, align="C", border=0)
    pdf.ln(20)

    def section_title(title):
        pdf.set_text_color(*C_BRAND)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(*C_WHITE)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_draw_color(*C_ACCENT)
        pdf.set_line_width(0.4)
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.line(20, y, 20 + w, y)
        pdf.set_line_width(0.2)
        pdf.ln(2)

    def sub_title(title):
        pdf.set_text_color(*C_BRAND)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, title, ln=True)

    def body_text(txt):
        pdf.set_text_color(*C_BLACK)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(w, 5, str(txt) if txt else "")
        pdf.ln(1)

    def bullet(txt, num=None):
        pdf.set_text_color(*C_BLACK)
        pdf.set_font("Helvetica", "", 9)
        prefix = f"{num}." if num else "-"
        pdf.set_x(24)
        pdf.multi_cell(w - 4, 5, f"{prefix}  {txt}")

    def data_table(section_label, rows):
        """Render a two-column data table for a health section."""
        # Table header
        pdf.set_fill_color(*C_LIGHT)
        pdf.set_text_color(*C_BRAND)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(w, 6, f"  {section_label}", fill=True, ln=True)
        # Rows
        for i, (label, value, unit) in enumerate(rows):
            val_str = f"{value} {unit}".strip() if value is not None else "N/A"
            fill_col = C_ROW if i % 2 == 0 else C_WHITE
            pdf.set_fill_color(*fill_col)
            pdf.set_text_color(*C_MUTED)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(w * 0.55, 5.5, f"  {label}", fill=True, border=0)
            pdf.set_text_color(*C_BLACK)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(w * 0.45, 5.5, str(val_str), fill=True, border=0, ln=True)
        pdf.ln(3)

    # ── AI NARRATIVE ───────────────────────────────────────────────────────────
    narrative = result.get("ai_narrative") or {}
    if narrative:
        section_title("AI Health Summary")

        exec_sum = narrative.get("executive_summary", "")
        if exec_sum:
            body_text(exec_sum)

        highlights = narrative.get("highlights") or []
        if highlights:
            sub_title("Highlights This Week")
            for i, h in enumerate(highlights, 1):
                bullet(h, i)
            pdf.ln(2)

        concerns = narrative.get("concerns") or []
        if concerns:
            sub_title("Areas of Concern")
            for i, c in enumerate(concerns, 1):
                bullet(c, i)
            pdf.ln(2)

        actions = narrative.get("caregiver_actions") or []
        if actions:
            sub_title("Recommended Actions for Caregiver")
            for i, a in enumerate(actions, 1):
                bullet(a, i)
            pdf.ln(2)

        mood_note = narrative.get("emotional_wellbeing", "")
        if mood_note:
            sub_title("Emotional Wellbeing")
            body_text(mood_note)

        doc_note = narrative.get("doctor_note", "")
        if doc_note:
            sub_title("Note for Doctor")
            pdf.set_fill_color(254, 252, 232)   # warm yellow bg
            pdf.set_draw_color(*C_WARN)
            pdf.set_line_width(0.5)
            y0 = pdf.get_y()
            pdf.set_x(20)
            pdf.multi_cell(w, 5, doc_note, border=1, fill=True)
            pdf.set_line_width(0.2)
            pdf.ln(2)

    # ── HEALTH DATA BREAKDOWN ─────────────────────────────────────────────────
    section_title("Weekly Health Data Breakdown")
    sections = result.get("sections") or {}

    # ── Active Conditions (shown first as most important) ────────────────────
    cond_data = sections.get("condition_updates", {})
    active_names = cond_data.get("active_condition_names") or []
    active_count_val = cond_data.get("active_condition_count", 0)
    data_table("Active Conditions (Disease Burden)", [
        ("Total active conditions", active_count_val, ""),
        ("Conditions", ", ".join(active_names) if active_names else "None on record", ""),
        ("New this week", len(cond_data.get("new_diagnoses") or []), ""),
    ])

    data_table("Vitals", [
        ("Heart Rate (avg)",     sections.get("vitals_summary",{}).get("avg_heart_rate"),    "bpm"),
        ("Blood Pressure (avg)", sections.get("vitals_summary",{}).get("avg_systolic_bp"),   "mmHg"),
        ("SpO2 (avg)",           sections.get("vitals_summary",{}).get("avg_spo2"),          "%"),
        ("Sleep Score (avg)",    sections.get("vitals_summary",{}).get("avg_sleep_score"),   "/ 10"),
        ("Temperature (avg)",    sections.get("vitals_summary",{}).get("avg_temperature"),   "C"),
        ("Readings logged this week", sections.get("vitals_summary",{}).get("readings"),     ""),
    ])
    adh_pct = sections.get("medication_adherence",{}).get("adherence_pct")
    adh_display = f"{adh_pct}" if adh_pct is not None else "Not tracked"
    data_table("Medication Adherence", [
        ("Total doses scheduled", sections.get("medication_adherence",{}).get("total_doses"),  ""),
        ("Doses taken",           sections.get("medication_adherence",{}).get("taken"),         ""),
        ("Doses missed",          sections.get("medication_adherence",{}).get("missed"),        ""),
        ("Adherence rate",        adh_display,                                                   "%" if adh_pct is not None else ""),
    ])
    data_table("Lab Results", [
        ("New lab results this week", sections.get("lab_summary",{}).get("new_labs_this_week"), ""),
        ("Abnormal values",           sections.get("lab_summary",{}).get("abnormal_count"),     ""),
    ])
    data_table("Respiratory Activity", [
        ("Cough count (7 days)",  sections.get("respiratory_activity",{}).get("cough_count_7d"),  ""),
        ("Sneeze count (7 days)", sections.get("respiratory_activity",{}).get("sneeze_count_7d"), ""),
        ("Daily cough average",   sections.get("respiratory_activity",{}).get("cough_daily_avg"), "/ day"),
    ])
    data_table("Physical Activity", [
        ("Fall incidents",  sections.get("physical_activity",{}).get("fall_count"),          ""),
        ("Activity score",  sections.get("physical_activity",{}).get("avg_activity_score"),  "/ 10"),
        ("Mobility score",  sections.get("physical_activity",{}).get("avg_mobility_score"),  "/ 10"),
    ])
    data_table("Environment", [
        ("Avg room temperature", sections.get("environment",{}).get("avg_room_temp_c"), "C"),
        ("Avg humidity",         sections.get("environment",{}).get("avg_humidity"),    "%"),
        ("Extreme readings",     sections.get("environment",{}).get("extreme_readings"), ""),
    ])
    data_table("Mood & Wellbeing", [
        ("Voice sessions", sections.get("mood_wellbeing",{}).get("conversations"), ""),
        ("Dominant mood",  sections.get("mood_wellbeing",{}).get("dominant_mood"),  ""),
    ])
    data_table("Alerts Summary", [
        ("Critical alerts", sections.get("alerts_recap",{}).get("critical_count"), ""),
        ("High alerts",     sections.get("alerts_recap",{}).get("high_count"),     ""),
        ("Total alerts",    sections.get("alerts_recap",{}).get("total_alerts"),   ""),
    ])

    # ── EXTRA DETAIL BULLETS ───────────────────────────────────────────────────
    vitals_alerts = sections.get("vitals_summary",{}).get("alerts") or []
    if vitals_alerts:
        sub_title("Vitals Alerts")
        for va in vitals_alerts:
            bullet(va)
        pdf.ln(2)

    key_changes = sections.get("lab_summary",{}).get("key_changes") or []
    if key_changes:
        sub_title("Notable Lab Changes")
        for kc in key_changes:
            bullet(kc)
        pdf.ln(2)

    problem_meds = sections.get("medication_adherence",{}).get("problem_meds") or []
    if problem_meds:
        sub_title("Missed Medication Details")
        for pm in problem_meds:
            bullet(pm)
        pdf.ln(2)

    # ── FOOTER META ────────────────────────────────────────────────────────────
    caretaker_name = getattr(caretaker, "username", getattr(caretaker, "full_name", ""))
    pdf.set_text_color(*C_MUTED)
    pdf.set_font("Helvetica", "I", 7)
    pdf.cell(0, 5,
        f"Report ID: {result.get('report_id', '')}  |  Caretaker: {caretaker_name}  "
        f"|  Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        ln=True, align="C"
    )

    return bytes(pdf.output())


@router.get("/recipients/{recipient_id}/weekly-reports/latest")
async def get_latest_weekly_report(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Return metadata of the most recently generated weekly report for a recipient."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    from tables.weekly_reports import WeeklyReport
    report = db.query(WeeklyReport).filter(
        WeeklyReport.care_recipient_id == recipient_id
    ).order_by(WeeklyReport.created_at.desc()).first()

    if not report:
        raise HTTPException(status_code=404, detail="No weekly reports found yet")

    return {
        "report_id": report.id,
        "week_start": str(report.week_start),
        "week_end": str(report.week_end),
        "overall_score": report.overall_score,
        "risk_trend": report.risk_trend,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }
