import sys
import os

sys.path.append('c:/CareTaker/backend')

from routes.recipients import _build_weekly_report_pdf

class MockRecipient:
    full_name = "Test User"

class MockCaretaker:
    pass

result = {
    "week_start": "2023-10-01",
    "week_end": "2023-10-07",
    "overall_score": 85,
    "risk_trend": "stable",
    "sections": {
        "condition_updates": {"active_condition_names": ["Hypertension"], "active_condition_count": 1, "new_diagnoses": []},
        "vitals_summary": {"avg_heart_rate": 72, "avg_systolic_bp": 120, "avg_spo2": 98, "avg_sleep_score": 8, "avg_temperature": 36.5, "readings": 5},
        "medication_adherence": {"total_doses": 10, "taken": 10, "missed": 0, "adherence_pct": 100},
        "lab_summary": {"new_labs_this_week": 0},
        "physical_activity": {"avg_activity_score": 5, "fall_count": 0},
        "environment": {"avg_room_temp_c": 22},
        "respiratory_activity": {"cough_count_7d": 2}
    },
    "ai_narrative": {
        "executive_summary": "All good",
        "highlights": ["Great"],
        "concerns": ["None"],
        "caregiver_actions": ["Keep it up"],
        "emotional_wellbeing": "Happy",
        "doctor_note": "No notes"
    }
}

try:
    _build_weekly_report_pdf(result, MockRecipient(), MockCaretaker())
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
