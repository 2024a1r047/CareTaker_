import os
import sys
from sqlalchemy.orm import Session
from datetime import datetime

sys.path.append(os.path.join(os.getcwd()))

from config import SessionLocal
# Import ALL tables to satisfy relationships
from tables.medical_recommendations import MedicalRecommendation
from tables.users import CareRecipient, CareTaker
from tables.video_analysis import VideoAnalysis
from tables.medical_conditions import PatientCondition, LabValue
from tables.vital_signs import VitalSign
from tables.medications import Medication
from tables.medication_dose_logs import MedicationDoseLog
from tables.allergies import Allergy
from tables.conversation_history import ConversationMessage
from tables.medical_reports import MedicalReport
from tables.audio_events import AudioEvent
from tables.environment import EnvironmentSensor

def check_recs():
    db = SessionLocal()
    recipient_id = 45 # From the screenshot URL
    recs = db.query(MedicalRecommendation).filter(MedicalRecommendation.care_recipient_id == recipient_id).order_by(MedicalRecommendation.created_at.desc()).all()
    
    print(f"--- Recommendations for Recipient {recipient_id} ---")
    for r in recs:
        print(f"ID: {r.id}")
        print(f"Metric: {r.metric}")
        print(f"Title: {r.title}")
        print(f"Condition Group: {r.condition_group}")
        print(f"Message: {r.message[:100]}...")
        print(f"Why Matters: {r.why_this_matters}")
        print(f"Created At: {r.created_at}")
        print("-" * 30)
    db.close()

if __name__ == "__main__":
    check_recs()
