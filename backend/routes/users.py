from fastapi import APIRouter, Depends, HTTPException, status, Header, Form
from sqlalchemy.orm import Session
from datetime import timedelta
import json
from typing import Optional, List, Dict

from models.users import ResponseSchema, Register, Login, CaretakerUpdate
from tables.users import CareTaker, CareRecipient, Doctor
from tables.vital_signs import VitalSign
from tables.medications import Medication, MedicationStatus
from sqlalchemy import desc
from config import get_db, ACCESS_TOKEN_EXPIRE_MINUTES
from repository.users import UsersRepo, JWTRepo
from utils.email import send_registration_email

router = APIRouter(tags=['Authentication'])


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        print("No Authorization header provided")
        return None
    
    try:
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            print(f"Invalid Authorization header format: {auth_header[:50]}...")
            return None
            
        token = parts[1]
        if not token:
            print("Empty token")
            return None
            
        print(f"Attempting to decode token: {token[:10]}...")
        decoded = JWTRepo.decode_token(token)
        
        if not decoded or not isinstance(decoded, dict):
            print("Invalid token format after decoding")
            return None
            
        username = decoded.get('sub')
        print(f"Successfully authenticated user: {username}")
        return username
        
    except Exception as e:
        print(f"Token validation error: {str(e)}")
        return None


@router.get('/profile')
async def profile(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    try:
        username = _get_username_from_auth(authorization)
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

        user = UsersRepo.find_by_username(db, CareTaker, username)
        role = 'caretaker'
        if not user:
            user = UsersRepo.find_by_username(db, Doctor, username)
            role = 'doctor'
            
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        if role == 'caretaker':
            recipients = []
            for r in user.care_recipients:
                # Query vitals directly from DB (same approach as insights_engine)
                latest_vital = db.query(VitalSign).filter(
                    VitalSign.care_recipient_id == r.id
                ).order_by(desc(VitalSign.recorded_at)).first()

                vitals_data = None
                if latest_vital:
                    vitals_data = {
                        'heart_rate': latest_vital.heart_rate,
                        'systolic_bp': latest_vital.systolic_bp,
                        'diastolic_bp': latest_vital.diastolic_bp,
                        'oxygen_saturation': latest_vital.oxygen_saturation,
                        'sleep_score': latest_vital.sleep_score,
                        'temperature': latest_vital.temperature,
                        'bmi': latest_vital.bmi,
                        'height': latest_vital.height,
                        'weight': latest_vital.weight,
                        'recorded_at': latest_vital.recorded_at.isoformat() if latest_vital.recorded_at else None
                    }

                # Fetch active medications for voice bot medicine reminders
                from tables.medications import Medication, MedicationStatus
                active_meds = db.query(Medication).filter(
                    Medication.care_recipient_id == r.id,
                    Medication.status == MedicationStatus.active
                ).all()
                meds_data = [
                    {
                        'medicine_name': m.medicine_name,
                        'dosage': m.dosage,
                        'frequency': m.frequency,
                        'schedule_time': m.schedule_time
                    }
                    for m in active_meds
                ]

                # Fetch recent unread alerts for the notification center
                from tables.medical_conditions import MedicalAlert
                recent_alerts = db.query(MedicalAlert).filter(
                    MedicalAlert.care_recipient_id == r.id,
                    MedicalAlert.is_read == False
                ).order_by(desc(MedicalAlert.created_at)).limit(5).all()
                alerts_data = [
                    {
                        'id': a.id,
                        'message': a.message,
                        'severity': a.severity.value if a.severity else None,
                        'created_at': a.created_at.isoformat() if a.created_at else None,
                        'is_read': a.is_read
                    }
                    for a in recent_alerts
                ]

                recipients.append({
                    'id': r.id,
                    'full_name': r.full_name,
                    'preferred_name': r.preferred_name,
                    'email': r.email,
                    'phone_number': r.phone_number,
                    'age': r.age,
                    'gender': r.gender.value if r.gender else None,
                    'respiratory_condition_status': bool(r.respiratory_condition_status),
                    'report_summary': r.report_summary,
                    'vitals': vitals_data,
                    'height': r.height,
                    'weight': r.weight,
                    'blood_group': r.blood_group,
                    'emergency_contact': r.emergency_contact,
                    'medications': meds_data,
                    'alerts': alerts_data,
                    'risk_score': r.risk_score
                })

            return {
                'status': 'success',
                'role': 'caretaker',
                'caretaker': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'phone_number': user.phone_number,
                    'face_registered': user.face_descriptor is not None
                },
                'care_recipients': recipients
            }
        else:
            return {
                'status': 'success',
                'role': 'doctor',
                'doctor': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'phone_number': user.phone_number,
                    'specialization': user.specialization,
                    'bio': user.bio,
                    'experience_years': user.experience_years,
                    'certifications': user.certifications or [],
                    'clinic_address': user.clinic_address,
                    'vpa_address': user.vpa_address,
                    'is_verified': bool(user.is_verified),
                    'verified_at': user.verified_at.isoformat() if user.verified_at else None,
                    'member_since': user.created_at.isoformat() if user.created_at else None,
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")


@router.patch('/me', response_model=ResponseSchema)
async def update_caretaker_profile(payload: CaretakerUpdate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Update caretaker's own details."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.email is not None:
        user.email = payload.email
    if payload.phone_number is not None:
        user.phone_number = payload.phone_number
    if payload.password is not None and payload.password:
        user.password = payload.password

    db.commit()
    db.refresh(user)

    return ResponseSchema(code=200, status="success", message="Profile updated successfully", result={
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name
    })


@router.patch('/doctor/me')
async def update_doctor_profile(payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Doctor edits their own profile (Settings in the doctor portal)."""
    import re as _re
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    doc = UsersRepo.find_by_username(db, Doctor, username)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doctor account not found")

    def clean(k):
        v = payload.get(k)
        return v.strip() if isinstance(v, str) else v

    if clean('full_name'):
        doc.full_name = clean('full_name')
    if 'specialization' in payload:
        doc.specialization = clean('specialization') or None
    if clean('email') and clean('email') != doc.email:
        email = clean('email')
        if not _re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise HTTPException(status_code=400, detail="Please enter a valid email address")
        if db.query(Doctor).filter(Doctor.email == email, Doctor.id != doc.id).first():
            raise HTTPException(status_code=400, detail="This email is already used by another doctor")
        doc.email = email
    if clean('phone_number') and clean('phone_number') != doc.phone_number:
        phone = _re.sub(r"\D", "", clean('phone_number'))
        if len(phone) != 10:
            raise HTTPException(status_code=400, detail="Phone number must have 10 digits")
        if db.query(Doctor).filter(Doctor.phone_number == phone, Doctor.id != doc.id).first():
            raise HTTPException(status_code=400, detail="This phone number is already used by another doctor")
        doc.phone_number = phone
    if 'bio' in payload:
        doc.bio = clean('bio') or None
    if 'clinic_address' in payload:
        doc.clinic_address = clean('clinic_address') or None
    if 'vpa_address' in payload:
        doc.vpa_address = clean('vpa_address') or None
    if 'experience_years' in payload:
        try:
            yrs = int(payload.get('experience_years') or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Years of experience must be a number")
        if not 0 <= yrs <= 70:
            raise HTTPException(status_code=400, detail="Years of experience must be between 0 and 70")
        doc.experience_years = yrs
    if 'certifications' in payload:
        certs = payload.get('certifications')
        if isinstance(certs, str):
            certs = [c.strip() for c in _re.split(r"[,\n]", certs) if c.strip()]
        doc.certifications = certs or []
    if payload.get('new_password'):
        if (payload.get('current_password') or '') != doc.password:
            raise HTTPException(status_code=400, detail="Current password is not correct")
        if len(payload['new_password']) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
        doc.password = payload['new_password']

    try:
        db.commit()
        db.refresh(doc)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not save profile: {e}")
    return {"status": "success", "message": "Profile updated", "doctor": {
        "full_name": doc.full_name, "email": doc.email, "phone_number": doc.phone_number,
        "specialization": doc.specialization, "bio": doc.bio, "experience_years": doc.experience_years,
        "certifications": doc.certifications or [], "clinic_address": doc.clinic_address,
        "vpa_address": doc.vpa_address, "is_verified": bool(doc.is_verified)}}


@router.delete('/me', response_model=ResponseSchema)
async def delete_caretaker_account(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete caretaker account and all associated data."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(user)
    db.commit()

    return ResponseSchema(code=200, status="success", message="Account deleted successfully")

# ---------- SIGNUP ----------
@router.post('/signup', response_model=ResponseSchema)
async def signup(request: Register, db: Session = Depends(get_db)):
    try:
        # Check if user already exists in either table
        existing_caretaker = UsersRepo.find_by_username(db, CareTaker, request.username)
        existing_doctor = UsersRepo.find_by_username(db, Doctor, request.username)
        
        if existing_caretaker or existing_doctor:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        if request.role == 'doctor':
            # Create Doctor entry
            doctor = Doctor(
                email=request.email,
                username=request.username,
                phone_number=request.phone_number,
                password=request.password,
                full_name=request.full_name,
                specialization=request.specialization
            )
            db.add(doctor)
            db.commit()
            db.refresh(doctor)
            
            # Generate JWT token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            token = JWTRepo.generate_token(
                {"sub": doctor.username}, expires_delta=access_token_expires
            )

            return ResponseSchema(
                code=200,
                status="success",
                message="Doctor registered successfully!",
                result={"access_token": token, "token_type": "bearer", "role": "doctor"}
            )
        else:
            # Create CareTaker entry
            caretaker = CareTaker(
                email=request.email,
                username=request.username,
                phone_number=request.phone_number,
                password=request.password,
                full_name=request.full_name
            )
            db.add(caretaker)
            db.commit()
            db.refresh(caretaker)

            # Add care recipients
            created_recipients = []
            if request.care_recipients:
                for recipient in request.care_recipients:
                    new_recipient = CareRecipient(
                        caretaker_id=caretaker.id,
                        full_name=recipient.full_name,
                        email=recipient.email,
                        phone_number=recipient.phone_number,
                        age=recipient.age,
                        gender=recipient.gender,
                        respiratory_condition_status=recipient.respiratory_condition_status,
                        height=recipient.height,
                        weight=recipient.weight,
                        blood_group=recipient.blood_group,
                        emergency_contact=recipient.emergency_contact
                    )
                    db.add(new_recipient)
                    created_recipients.append(new_recipient)

                db.commit()
                for r in created_recipients:
                    db.refresh(r)

            # Send registration email
            await send_registration_email(request.email, request.username)

            # Generate JWT token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            token = JWTRepo.generate_token(
                {"sub": caretaker.username}, expires_delta=access_token_expires
            )

            recipients_out = [{
                'id': r.id,
                'full_name': r.full_name,
                'email': r.email
            } for r in created_recipients]

            return ResponseSchema(
                code=200,
                status="success",
                message="Caretaker registered successfully!",
                result={"access_token": token, "token_type": "bearer", "care_recipients": recipients_out, "role": "caretaker"}
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        # Catch unique constraint violations (duplicate phone/email/username)
        if "UniqueViolation" in error_msg or "unique constraint" in error_msg.lower():
            if "phone_number" in error_msg:
                # Check if it was specifically a care recipient phone number conflict
                if "ix_care_recipients_phone_number" in error_msg:
                    raise HTTPException(status_code=400, detail="A care recipient with this phone number already exists")
                elif "ix_caretakers_phone_number" in error_msg:
                    raise HTTPException(status_code=400, detail="An account with this phone number already exists")
            
            if "ix_caretakers_email" in error_msg or "ix_care_recipients_email" in error_msg:
                raise HTTPException(status_code=400, detail="A user with this email already exists")
            elif "ix_caretakers_username" in error_msg:
                raise HTTPException(status_code=400, detail="Username already exists")
            else:
                # Fallback to generic parsing if constraint names are missed
                if "email" in error_msg.lower() and "Key (email)=" in error_msg:
                    raise HTTPException(status_code=400, detail="A user with this email already exists")
                if "username" in error_msg.lower() and "Key (username)=" in error_msg:
                    raise HTTPException(status_code=400, detail="Username already exists")
                
                raise HTTPException(status_code=400, detail="An account with these details already exists")
        raise HTTPException(status_code=500, detail=f"Signup failed: {error_msg}")


# ---------- NEW ENDPOINT: REGISTER WITH FACE ----------
@router.post('/register-with-face')
async def register_with_face(
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    phone_number: str = Form(...),
    password: str = Form(...),
    face_descriptor: str = Form(None),
    recipient_name: List[str] = Form(...),
    recipient_email: List[str] = Form([]),
    recipient_phone: List[str] = Form([]),
    recipient_age: List[int] = Form(...),
    recipient_gender: List[str] = Form(...),
    recipient_condition: List[str] = Form(...),
    db: Session = Depends(get_db)
):
    """
    New endpoint for registration with face data from webcam
    """
    try:
        # Check if caretaker already exists
        existing_user = UsersRepo.find_by_username(db, CareTaker, username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Parse face descriptor if provided
        face_descriptor_data = None
        if face_descriptor and face_descriptor != 'null':
            try:
                face_descriptor_data = json.loads(face_descriptor)
                print(f"Face descriptor received with {len(face_descriptor_data)} elements")
            except json.JSONDecodeError:
                print("Warning: Invalid face descriptor format - continuing without face data")

        # Create CareTaker entry with face descriptor
        caretaker = CareTaker(
            email=email,
            username=username,
            phone_number=phone_number,
            password=password,
            full_name=full_name,
            face_descriptor=face_descriptor_data  # Store the face data
        )
        db.add(caretaker)
        db.commit()
        db.refresh(caretaker)

        # Add care recipients
        created_recipients = []
        for i in range(len(recipient_name)):
            new_recipient = CareRecipient(
                caretaker_id=caretaker.id,
                full_name=recipient_name[i],
                email=recipient_email[i] if i < len(recipient_email) else None,
                phone_number=recipient_phone[i] if i < len(recipient_phone) else None,
                age=recipient_age[i],
                gender=recipient_gender[i],
                respiratory_condition_status=recipient_condition[i].lower() == 'true'
            )
            db.add(new_recipient)
            created_recipients.append(new_recipient)

        db.commit()
        for r in created_recipients:
            db.refresh(r)

        # Send registration email
        await send_registration_email(email, username)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": caretaker.username}, expires_delta=access_token_expires
        )

        # Return response with recipients
        recipients_out = [{
            'id': r.id,
            'full_name': r.full_name,
            'email': r.email
        } for r in created_recipients]

        return {
            "code": 200,
            "status": "success",
            "message": "Caretaker registered successfully!" + (" Face data stored." if face_descriptor_data else ""),
            "result": {
                "access_token": token, 
                "token_type": "bearer",
                "face_registered": bool(face_descriptor_data),
                "care_recipients": recipients_out
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        # Catch unique constraint violations (duplicate phone/email/username)
        if "UniqueViolation" in error_msg or "unique constraint" in error_msg.lower():
            if "phone_number" in error_msg:
                # Check if it was specifically a care recipient phone number conflict
                if "ix_care_recipients_phone_number" in error_msg:
                    raise HTTPException(status_code=400, detail="A care recipient with this phone number already exists")
                elif "ix_caretakers_phone_number" in error_msg:
                    raise HTTPException(status_code=400, detail="An account with this phone number already exists")
            
            if "ix_caretakers_email" in error_msg or "ix_care_recipients_email" in error_msg:
                raise HTTPException(status_code=400, detail="A user with this email already exists")
            elif "ix_caretakers_username" in error_msg:
                raise HTTPException(status_code=400, detail="Username already exists")
            else:
                # Fallback to generic parsing if constraint names are missed
                if "email" in error_msg.lower() and "Key (email)=" in error_msg:
                    raise HTTPException(status_code=400, detail="A user with this email already exists")
                if "username" in error_msg.lower() and "Key (username)=" in error_msg:
                    raise HTTPException(status_code=400, detail="Username already exists")
                
                raise HTTPException(status_code=400, detail="An account with these details already exists")
        raise HTTPException(status_code=500, detail=f"Registration failed: {error_msg}")


# ---------- UPDATE FACE FOR EXISTING USER ----------
@router.post('/update-face')
async def update_face(
    face_descriptor: str = Form(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Update face descriptor for an existing user
    """
    try:
        username = _get_username_from_auth(authorization)
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

        user = UsersRepo.find_by_username(db, CareTaker, username)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Parse face descriptor
        try:
            face_descriptor_data = json.loads(face_descriptor)
            user.face_descriptor = face_descriptor_data
            db.commit()
            
            return {
                "code": 200,
                "status": "success", 
                "message": "Face descriptor updated successfully",
                "result": {"face_registered": True}
            }
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid face descriptor format")

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update face: {str(e)}")


# ---------- LOGIN ----------
@router.post('/login', response_model=ResponseSchema)
async def login(request: Login, db: Session = Depends(get_db)):
    try:
        print(f"\n=== Login Request ===")
        print(f"Request data: {request.dict()}")
        print(f"Database URL: {db.bind.url if hasattr(db, 'bind') else 'No DB connection'}")
        
        # Check if user exists in Doctor table
        user = UsersRepo.find_by_username(db, Doctor, request.username)
        role = 'doctor'
        
        if not user:
            # Check if user exists in CareTaker table
            user = UsersRepo.find_by_username(db, CareTaker, request.username)
            role = 'caretaker'

        if not user:
            # Not found exactly: also accept the username in any letter case, or the e-mail address
            # ("dheemant" for "Dheemant", or "dheemantsharma123@gmail.com")
            from sqlalchemy import func, or_
            typed = (request.username or "").strip().lower()
            for model, r in ((Doctor, 'doctor'), (CareTaker, 'caretaker')):
                user = db.query(model).filter(or_(func.lower(model.username) == typed,
                                                  func.lower(model.email) == typed)).first()
                if user:
                    role = r
                    break

        print(f"User found: {user is not None}, Role: {role if user else 'N/A'}")
        
        if not user or user.password != request.password.strip():
            print(f"Login failed: Incorrect username or password for user '{request.username}'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = JWTRepo.generate_token(
            {"sub": user.username},
            expires_delta=access_token_expires
        )
        
        print(f"Login successful for {role}: {user.username}")
        
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": role
        }
        
        if role == 'caretaker':
            user_data["face_registered"] = user.face_descriptor is not None
        else:
            user_data["specialization"] = user.specialization
            # Soft verification flag — doctor can still log in but frontend can show a banner
            user_data["is_verified"] = getattr(user, "is_verified", False)
            user_data["verification_pending"] = not getattr(user, "is_verified", False)

        login_message = "Login successful"
        if role == 'doctor' and not getattr(user, "is_verified", False):
            login_message = "Login successful — your account is pending admin verification"

        return ResponseSchema(
            code=200,
            status="success",
            message=login_message,
            result={
                "access_token": token, 
                "token_type": "bearer",
                "user": user_data
            }
        )

    except HTTPException as e:
        print("Login failed with HTTPException")
        print("=================\n")
        raise
    except Exception as e:
        print(f"Unexpected error during login: {str(e)}")
        print("=================\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/refresh-token")
async def refresh_token(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Swap the login token for a fresh one - also when it has already expired.

    Monitoring pages (room camera, voice assistant) stay open for hours or overnight; they call this
    every ~20 minutes and when the laptop wakes up, so the caretaker is never logged out and falls /
    medicine sightings keep being saved. The token must still carry our signature and belong to an
    account that exists (a forged or deleted-account token is refused); pressing Logout ends it.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    from jose import jwt as _jwt, JWTError as _JWTError
    from config import SECRET_KEY, ALGORITHM
    try:
        payload = _jwt.decode(authorization.split(" ", 1)[1].strip(), SECRET_KEY, algorithms=[ALGORITHM],
                              options={"verify_exp": False})
    except _JWTError:
        payload = None
    sub = payload.get("sub") if payload else None
    if not sub or not (db.query(CareTaker.id).filter(CareTaker.username == sub).first()
                       or db.query(Doctor.id).filter(Doctor.username == sub).first()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    data = {k: v for k, v in payload.items() if k != "exp"}
    token = JWTRepo.generate_token(data, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer", "expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES}
