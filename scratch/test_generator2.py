import sys
import os

sys.path.append('c:/CareTaker/backend')

from config import SessionLocal
import tables.video_analysis
from services.weekly_report_generator import generate_weekly_report
from routes.recipients import _build_weekly_report_pdf
from tables.users import CareRecipient, CareTaker

def main():
    db = SessionLocal()
    try:
        recipient = db.query(CareRecipient).filter(CareRecipient.id == 1).first()
        caretaker = db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
        
        print("Generating data...")
        result = generate_weekly_report(1, db)
        print("Result:", result.keys())
        
        # Test PDF gen
        print("Generating PDF...")
        # ensure keys are handled properly as done in route
        if result.get("status") == "already_exists":
            result["sections"]     = result.get("sections") or result.get("report_data") or {}
            result["ai_narrative"] = result.get("ai_narrative") or result.get("narrative") or {}
            result["overall_score"]= result.get("overall_score") or result.get("score") or 0

        # Sanitize
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
        print("PDF Size:", len(pdf_bytes))
        print("SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()
