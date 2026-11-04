from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Enum, Text, JSON
from sqlalchemy.orm import relationship
from config import Base
import datetime
import enum


# ---------- ENUM for Gender ----------
class GenderEnum(str, enum.Enum):
    male = "Male"
    female = "Female"
    other = "Other"


# ---------- CareTaker Table ----------
class CareTaker(Base):
    __tablename__ = "caretakers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String(10), unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    face_descriptor = Column(JSON, nullable=True)  # stores face data as JSON array
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationship: one caretaker → many recipients
    care_recipients = relationship("CareRecipient", back_populates="caretaker", cascade="all, delete-orphan")

    # Relationship: caretaker -> video analyses (history)
    video_analyses = relationship("VideoAnalysis", back_populates="caretaker")


# ---------- Doctor Table ----------
class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String(10), unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    specialization = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


# ---------- CareRecipient Table ----------
class CareRecipient(Base):
    __tablename__ = "care_recipients"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id", ondelete="CASCADE"), nullable=False)

    full_name = Column(String, nullable=False)
    preferred_name = Column(String(100), nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String(10), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(Enum(GenderEnum), nullable=True)
    city = Column(String, nullable=True)
    respiratory_condition_status = Column(Boolean, default=False)
    
    # New Profile Fields
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    blood_group = Column(String, nullable=True)
    emergency_contact = Column(String, nullable=True)
    registration_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    # aggregated summary of uploaded medical reports for this recipient
    report_summary = Column(Text, nullable=True)

    # Medical History System (v3)
    risk_score = Column(Float, nullable=True)                       # 0-100 deterministic risk score
    risk_factors_breakdown = Column(JSON, nullable=True)            # Transparent factor-by-factor explanation
    last_analysis_date = Column(DateTime, nullable=True)
    last_report_date = Column(DateTime, nullable=True)              # For monitoring gap detection

    # Relationship back to caretaker
    caretaker = relationship("CareTaker", back_populates="care_recipients")

    # medical reports relation
    medical_reports = relationship('MedicalReport', back_populates='care_recipient', cascade='all, delete-orphan')

    # Doctor Remarks
    doctor_remarks = Column(Text, nullable=True)

    # Video analysis relation
    video_analyses = relationship('VideoAnalysis', back_populates='recipient', cascade='all, delete-orphan')

    # Vital Signs relation
    vital_signs = relationship('VitalSign', back_populates='recipient', cascade='all, delete-orphan')

    # Medical History System relations
    patient_conditions = relationship('PatientCondition', back_populates='care_recipient', cascade='all, delete-orphan')
    lab_values = relationship('LabValue', back_populates='care_recipient', cascade='all, delete-orphan')
    medical_alerts = relationship('MedicalAlert', back_populates='care_recipient', cascade='all, delete-orphan')

    # Voice Bot relations
    conversation_messages = relationship('ConversationMessage', back_populates='care_recipient', cascade='all, delete-orphan')
    proactive_reminders = relationship('ProactiveReminder', back_populates='care_recipient', cascade='all, delete-orphan')

    # Environment relations
    environment_readings = relationship('EnvironmentSensor', back_populates='care_recipient', cascade='all, delete-orphan')

    # New Profile relations
    active_medications = relationship('Medication', back_populates='care_recipient', cascade='all, delete-orphan')
    medication_history = relationship('MedicationHistory', back_populates='care_recipient', cascade='all, delete-orphan')
    dose_logs = relationship('MedicationDoseLog', back_populates='care_recipient', cascade='all, delete-orphan')
    allergies = relationship('Allergy', back_populates='care_recipient', cascade='all, delete-orphan')
    lab_orders = relationship('LabOrderDetail', back_populates='care_recipient', cascade='all, delete-orphan')