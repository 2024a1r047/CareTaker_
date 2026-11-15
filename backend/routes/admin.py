from fastapi import APIRouter, Depends, HTTPException, status, Header, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from datetime import timedelta, datetime
from typing import Optional, List

from config import get_db, ACCESS_TOKEN_EXPIRE_MINUTES
from tables.admin import Admin, AuditLog
from tables.users import CareTaker, Doctor, CareRecipient
from tables.video_analysis import VideoAnalysis
from repository.users import JWTRepo

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ─────────────────────────────────────────────
#  Auth Helpers
# ─────────────────────────────────────────────

def _get_admin_from_token(authorization: Optional[str], db: Session) -> Admin:
    """Decode JWT and return the Admin object, or raise 401."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    decoded = JWTRepo.decode_token(parts[1])
    if not decoded or not decoded.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Look up in admins table specifically
    role = decoded.get("role")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access only")

    admin = db.query(Admin).filter(Admin.username == decoded["sub"]).first()
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found")
    return admin


def _log_action(db: Session, admin: Admin, action: str,
                target_type: str = None, target_id: int = None,
                target_name: str = None, detail: str = None):
    """Write an audit log entry."""
    entry = AuditLog(
        admin_id=admin.id,
        admin_username=admin.username,
        action=action,
        target_type=target_type,
        target_id=target_id,
        target_name=target_name,
        detail=detail
    )
    db.add(entry)
    db.commit()


# ─────────────────────────────────────────────
#  Admin Login
# ─────────────────────────────────────────────

@router.post("/login")
async def admin_login(payload: dict, db: Session = Depends(get_db)):
    """Admin-specific login. Returns a JWT with role=admin encoded."""
    username = payload.get("username", "").strip()
    password = payload.get("password", "")

    admin = db.query(Admin).filter(Admin.username == username).first()
    if not admin or admin.password != password:
        raise HTTPException(status_code=401, detail="Incorrect admin credentials")

    token = JWTRepo.generate_token(
        {"sub": admin.username, "role": "admin"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES * 8)  # 4h session
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "admin": {
            "id": admin.id,
            "username": admin.username,
            "full_name": admin.full_name,
            "is_super_admin": admin.is_super_admin
        }
    }


# ─────────────────────────────────────────────
#  Analytics  
# ─────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Platform-wide stats for the admin overview cards."""
    admin = _get_admin_from_token(authorization, db)

    total_caretakers = db.query(func.count(CareTaker.id)).scalar()
    total_doctors = db.query(func.count(Doctor.id)).scalar()

    # Use raw SQL because is_verified was added via ALTER TABLE, not declared in ORM
    try:
        pending_result = db.execute(text("SELECT COUNT(*) FROM doctors WHERE is_verified = FALSE OR is_verified IS NULL")).scalar()
        verified_result = db.execute(text("SELECT COUNT(*) FROM doctors WHERE is_verified = TRUE")).scalar()
        pending_doctors = int(pending_result or 0)
        verified_doctors = int(verified_result or 0)
    except Exception as e:
        pending_doctors = 0
        verified_doctors = 0

    total_recipients = db.query(func.count(CareRecipient.id)).scalar()
    total_fall_events = 0
    try:
        total_fall_events = db.query(func.sum(VideoAnalysis.fall_count)).scalar() or 0
    except Exception:
        pass

    total_video_uploads = db.query(func.count(VideoAnalysis.id)).scalar()

    return {
        "total_caretakers": total_caretakers,
        "total_doctors": total_doctors,
        "verified_doctors": verified_doctors,
        "pending_doctors": pending_doctors,
        "total_recipients": total_recipients,
        "total_fall_events": int(total_fall_events),
        "total_video_uploads": total_video_uploads,
    }


# ─────────────────────────────────────────────
#  Doctor Verification
# ─────────────────────────────────────────────

@router.get("/doctors")
async def list_all_doctors(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """List all doctors with their verification status."""
    _get_admin_from_token(authorization, db)
    doctors = db.query(Doctor).order_by(desc(Doctor.created_at)).all()
    result = []
    for d in doctors:
        result.append({
            "id": d.id,
            "username": d.username,
            "full_name": d.full_name,
            "email": d.email,
            "phone_number": d.phone_number,
            "specialization": d.specialization,
            "is_verified": getattr(d, "is_verified", False),
            "verified_at": getattr(d, "verified_at", None),
            "rejection_reason": getattr(d, "rejection_reason", None),
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })
    return result


@router.get("/doctors/pending")
async def list_pending_doctors(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """List only unverified doctors."""
    _get_admin_from_token(authorization, db)
    try:
        # Use raw SQL because is_verified was added via ALTER TABLE, not declared in ORM
        rows = db.execute(
            text("SELECT id, username, full_name, email, specialization, created_at FROM doctors WHERE is_verified = FALSE OR is_verified IS NULL ORDER BY created_at DESC")
        ).fetchall()
        return [
            {
                "id": r[0],
                "username": r[1],
                "full_name": r[2],
                "email": r[3],
                "specialization": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
    except Exception:
        # Fallback: return all doctors if column doesn't exist
        doctors = db.query(Doctor).order_by(desc(Doctor.created_at)).all()
        return [
            {
                "id": d.id,
                "username": d.username,
                "full_name": d.full_name,
                "email": d.email,
                "specialization": d.specialization,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in doctors
        ]


@router.patch("/doctors/{doctor_id}/verify")
async def verify_doctor(
    doctor_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Approve a doctor — marks them as verified."""
    admin = _get_admin_from_token(authorization, db)
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    try:
        doctor.is_verified = True
        doctor.verified_at = datetime.utcnow()
        doctor.verified_by_admin_id = admin.id
        doctor.rejection_reason = None
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")

    _log_action(db, admin, "VERIFY_DOCTOR", "doctor", doctor.id,
                doctor.full_name, f"Verified by {admin.username}")
    return {"message": f"Dr. {doctor.full_name} has been verified successfully."}


@router.patch("/doctors/{doctor_id}/reject")
async def reject_doctor(
    doctor_id: int,
    payload: dict,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Reject a doctor and record the reason."""
    admin = _get_admin_from_token(authorization, db)
    reason = payload.get("reason", "No reason provided")
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    try:
        doctor.is_verified = False
        doctor.rejection_reason = reason
        doctor.verified_at = None
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Rejection failed: {e}")

    _log_action(db, admin, "REJECT_DOCTOR", "doctor", doctor.id,
                doctor.full_name, f"Reason: {reason}")
    return {"message": f"Dr. {doctor.full_name} has been rejected.", "reason": reason}


# ─────────────────────────────────────────────
#  Caretaker Management
# ─────────────────────────────────────────────

@router.get("/caretakers")
async def list_caretakers(
    authorization: Optional[str] = Header(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List all caretakers, optionally filtered by name or email."""
    _get_admin_from_token(authorization, db)
    query = db.query(CareTaker)
    if search:
        term = f"%{search}%"
        query = query.filter(
            (CareTaker.full_name.ilike(term)) | (CareTaker.email.ilike(term)) | (CareTaker.username.ilike(term))
        )
    caretakers = query.order_by(desc(CareTaker.created_at)).all()
    return [
        {
            "id": c.id,
            "username": c.username,
            "full_name": c.full_name,
            "email": c.email,
            "phone_number": c.phone_number,
            "recipient_count": len(c.care_recipients),
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in caretakers
    ]


@router.get("/caretakers/{caretaker_id}")
async def get_caretaker_detail(
    caretaker_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Get detail view of a specific caretaker and their recipients."""
    _get_admin_from_token(authorization, db)
    c = db.query(CareTaker).filter(CareTaker.id == caretaker_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Caretaker not found")
    return {
        "id": c.id,
        "username": c.username,
        "full_name": c.full_name,
        "email": c.email,
        "phone_number": c.phone_number,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "care_recipients": [
            {
                "id": r.id,
                "full_name": r.full_name,
                "age": r.age,
                "gender": r.gender.value if r.gender else None,
                "city": r.city,
            }
            for r in c.care_recipients
        ]
    }


@router.delete("/caretakers/{caretaker_id}")
async def delete_caretaker(
    caretaker_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Permanently delete a caretaker account and all associated data."""
    admin = _get_admin_from_token(authorization, db)
    c = db.query(CareTaker).filter(CareTaker.id == caretaker_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Caretaker not found")
    name = c.full_name
    try:
        # Feature tables added later (camera falls, medicine tracking, video analyses, sensors) point to
        # the caretaker / their patients WITHOUT "on delete cascade", so the database refused to delete
        # any caretaker who had used them. Remove that data first (children before parents), then the
        # caretaker; everything else already cascades.
        rids = [r[0] for r in db.execute(text("SELECT id FROM care_recipients WHERE caretaker_id = :c"),
                                         {"c": caretaker_id}).fetchall()]
        params = {"c": caretaker_id, "r": rids or [-1]}
        for table, cols in [
            ("medicine_sightings",  ("caretaker_id", "recipient_id")),
            ("medicine_refs",       ("caretaker_id", "recipient_id")),
            ("table_watch_events",  ("caretaker_id", "recipient_id")),
            ("fall_events",         ("caretaker_id", "recipient_id")),
            ("video_analysis",      ("caretaker_id", "recipient_id")),
            ("environment_sensors", (None, "care_recipient_id")),
        ]:
            care_col, rec_col = cols
            where = [f"{rec_col} = ANY(:r)"] + ([f"{care_col} = :c"] if care_col else [])
            db.execute(text(f"DELETE FROM {table} WHERE " + " OR ".join(where)), params)
        # Delete in the database itself: its ON DELETE CASCADE rules remove every patient record
        # (vitals, reports, weekly reports, recommendations, ...). Deleting through the ORM instead
        # tried to blank the patient id on some of them (e.g. weekly_reports) and failed.
        db.execute(text("DELETE FROM care_recipients WHERE caretaker_id = :c"), {"c": caretaker_id})
        db.execute(text("DELETE FROM caretakers WHERE id = :c"), {"c": caretaker_id})
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not remove caretaker: {e}")
    _log_action(db, admin, "DELETE_CARETAKER", "caretaker", caretaker_id, name, "Deleted by admin")
    return {"message": f"Caretaker '{name}' and all associated data has been removed."}


# ─────────────────────────────────────────────
#  Audit Log
# ─────────────────────────────────────────────

@router.get("/audit-log")
async def get_audit_log(
    authorization: Optional[str] = Header(None),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Paginated admin audit log."""
    _get_admin_from_token(authorization, db)
    logs = db.query(AuditLog).order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()
    return [
        {
            "id": l.id,
            "admin_username": l.admin_username,
            "action": l.action,
            "target_type": l.target_type,
            "target_name": l.target_name,
            "detail": l.detail,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None,
        }
        for l in logs
    ]
