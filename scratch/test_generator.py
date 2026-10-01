import sys
import os

sys.path.append('c:/CareTaker/backend')

from config import SessionLocal
from services.weekly_report_generator import generate_weekly_report

def main():
    db = SessionLocal()
    try:
        # recipient_id=1
        result = generate_weekly_report(1, db)
        print("Result keys:", result.keys())
        if "error" in result:
            print("Error:", result["error"])
        else:
            print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()
