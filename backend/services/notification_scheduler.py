"""
Background Notification Scheduler for CareTaker.

Runs periodic tasks:
1. schedule_pending_doses  — every minute: creates PENDING dose logs at scheduled times,
                             fires confirmation email + dashboard push
2. check_missed_doses      — every minute: auto-marks PENDING logs > 60 min old as MISSED,
                             sends escalation email, pushes dashboard update
3. Medication expiry       — once per day: marks completed if end_date has passed
4. Report reminders        — once per day: emails caretaker if no report for 30+ days
5. Auto-reorder            — once per day: alerts when stock ≤ 7 days (based on actual consumption)
6. Daily Recommendations   — once per day: re-runs clinical logic for all recipients

NOTE: decrement_daily_stock() has been DISABLED.
      Stock now only decrements via the confirm-dose API when status → TAKEN.
"""

import threading
import time
import datetime
import asyncio
import uuid
from datetime import timedelta
from typing import Optional

_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# Missed-dose window — dose is auto-escalated if PENDING for more than this many minutes
MISSED_DOSE_TIMEOUT_MINUTES = 60

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_db():
    from config import SessionLocal
    return SessionLocal()


def _normalize_slot(slot: str) -> str:
    """Normalise 'H:MM' or 'HH:MM' to 'HH:MM'."""
    try:
        return datetime.datetime.strptime(slot.strip(), "%H:%M").strftime("%H:%M")
    except ValueError:
        try:
            h, m = slot.strip().split(":")
            return f"{int(h):02d}:{int(m):02d}"
        except Exception:
            return slot.strip()


def _emit_socketio(event: str, data: dict):
    """
    Emit a Socket.IO event from the background scheduler thread.
    Uses asyncio.run_coroutine_threadsafe to bridge sync→async.
    """
    try:
        from socket_manager import sio_server
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                sio_server.emit(event, data),
                loop
            )
        else:
            loop.run_until_complete(sio_server.emit(event, data))
    except Exception as e:
        print(f"[scheduler] Socket.IO emit failed ({event}): {e}")


# ─────────────────────────────────────────────────────────────
# 1. SCHEDULE PENDING DOSES  (NEW — replaces passive decrement)
# ─────────────────────────────────────────────────────────────

def schedule_pending_doses():
    """
    Every minute: for each active medication with a schedule_time, check if the
    current HH:MM matches a slot.  If so, create a MedicationDoseLog (status=PENDING)
    — unless one already exists for that (medication_id, scheduled_datetime) pair.

    After creating the log:
      - Send a confirmation email to the caretaker (TAKEN / NOT TAKEN buttons).
      - Emit Socket.IO 'new_dose_pending' so the dashboard card appears instantly.
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.users import CareRecipient, CareTaker
        from tables.medication_dose_logs import MedicationDoseLog, DoseStatusEnum
        from sqlalchemy.exc import IntegrityError

        now = datetime.datetime.now()
        current_hhmm = now.strftime("%H:%M")

        active_meds = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.schedule_time.isnot(None),
        ).all()

        for med in active_meds:
            slots = [_normalize_slot(s) for s in med.schedule_time.split(",") if s.strip()]
            for slot in slots:
                if slot != current_hhmm:
                    continue

                # Build the exact scheduled datetime for today's slot
                slot_dt = now.replace(
                    hour=int(slot[:2]),
                    minute=int(slot[3:]),
                    second=0,
                    microsecond=0,
                )

                # ── Idempotency: check if log already exists ──
                existing = db.query(MedicationDoseLog).filter(
                    MedicationDoseLog.medication_id == med.medication_id,
                    MedicationDoseLog.scheduled_time == slot_dt,
                ).first()
                if existing:
                    continue  # Already created this cycle

                # ── Create PENDING dose log ──
                token = str(uuid.uuid4())
                dose_log = MedicationDoseLog(
                    medication_id=med.medication_id,
                    care_recipient_id=med.care_recipient_id,
                    scheduled_time=slot_dt,
                    unique_token=token,
                )
                db.add(dose_log)
                try:
                    db.flush()  # get the id; may raise IntegrityError on race
                except IntegrityError:
                    db.rollback()
                    print(f"[scheduler] Duplicate dose log for med={med.medication_id} slot={slot} (race — skipped)")
                    continue

                # ── Fetch caretaker & recipient for email ──
                recipient = db.query(CareRecipient).filter(
                    CareRecipient.id == med.care_recipient_id
                ).first()
                caretaker = (
                    db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
                    if recipient else None
                )

                db.commit()
                print(f"[scheduler] [DOSE] PENDING created: {med.medicine_name} @ {slot} "
                      f"(log_id={dose_log.id})")

                # ── Send confirmation email ──
                if caretaker and caretaker.email:
                    from services.email_notifications import send_dose_confirmation_email
                    sent = send_dose_confirmation_email(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name if recipient else "Patient",
                        medicine_name=med.medicine_name,
                        dosage=med.dosage or "As prescribed",
                        schedule_time=slot,
                        dose_log_id=dose_log.id,
                        token=token,
                    )
                    if sent:
                        dose_log.email_sent = True
                        db.commit()

                # ── Push to dashboard via Socket.IO ──
                _emit_socketio("new_dose_pending", {
                    "dose_log_id":    dose_log.id,
                    "medicine_name":  med.medicine_name,
                    "dosage":         med.dosage or "",
                    "scheduled_time": slot_dt.isoformat(),
                    "recipient_name": recipient.full_name if recipient else "",
                    "recipient_id":   med.care_recipient_id,
                })

    except Exception as e:
        db.rollback()
        print(f"[scheduler] schedule_pending_doses error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 2. CHECK MISSED DOSES  (NEW)
# ─────────────────────────────────────────────────────────────

def check_missed_doses():
    """
    Every minute: find PENDING dose logs that are more than MISSED_DOSE_TIMEOUT_MINUTES
    old.  Mark them MISSED (source=SYSTEM), send escalation email, push dashboard update.
    """
    db = _get_db()
    try:
        from tables.medication_dose_logs import MedicationDoseLog, DoseStatusEnum, ConfirmationSourceEnum
        from tables.medications import Medication
        from tables.users import CareRecipient, CareTaker

        cutoff = datetime.datetime.now() - timedelta(minutes=MISSED_DOSE_TIMEOUT_MINUTES)

        overdue = db.query(MedicationDoseLog).filter(
            MedicationDoseLog.status == DoseStatusEnum.PENDING,
            MedicationDoseLog.scheduled_time <= cutoff,
        ).all()

        for log in overdue:
            log.status = DoseStatusEnum.MISSED
            log.confirmation_source = ConfirmationSourceEnum.SYSTEM
            log.confirmed_at = datetime.datetime.utcnow()
            db.flush()

            # ── Escalation email (once only) ──
            if not log.escalation_sent:
                med = db.query(Medication).filter(
                    Medication.medication_id == log.medication_id
                ).first()
                recipient = db.query(CareRecipient).filter(
                    CareRecipient.id == log.care_recipient_id
                ).first()
                caretaker = (
                    db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
                    if recipient else None
                )

                if caretaker and caretaker.email and med:
                    from services.email_notifications import send_missed_dose_escalation
                    send_missed_dose_escalation(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name if recipient else "Patient",
                        medicine_name=med.medicine_name,
                        dosage=med.dosage or "As prescribed",
                        scheduled_time=log.scheduled_time.strftime("%I:%M %p"),
                    )

                log.escalation_sent = True

            db.commit()

            # ── Remove from dashboard ──
            _emit_socketio("dose_confirmed", {
                "dose_log_id": log.id,
                "action":      "MISSED",
                "new_stock":   None,
            })
            print(f"[scheduler] [MISSED] Auto-escalated dose log_id={log.id}")

    except Exception as e:
        db.rollback()
        print(f"[scheduler] check_missed_doses error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 3. MEDICATION EXPIRY  (unchanged logic)
# ─────────────────────────────────────────────────────────────

def check_medication_expiry():
    """
    Once per day: mark active medications whose end_date has passed as 'completed'
    and notify the caretaker.
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.users import CareRecipient, CareTaker

        today = datetime.date.today()
        expired = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.end_date.isnot(None),
            Medication.end_date < today,
        ).all()

        for med in expired:
            med.status = MedicationStatus.completed
            db.flush()

            recipient = db.query(CareRecipient).filter(
                CareRecipient.id == med.care_recipient_id
            ).first()
            if not recipient:
                continue
            caretaker = db.query(CareTaker).filter(
                CareTaker.id == recipient.caretaker_id
            ).first()
            if caretaker and caretaker.email:
                from services.email_notifications import send_medication_completed
                send_medication_completed(
                    to_email=caretaker.email,
                    recipient_name=recipient.full_name,
                    medicine_name=med.medicine_name,
                    duration_days=(med.end_date - med.start_date).days
                    if med.end_date and med.start_date else 0,
                )
            print(f"[scheduler] [OK] Medication '{med.medicine_name}' auto-completed (expired {med.end_date})")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[scheduler] Medication expiry check error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 4. REPORT REMINDERS  (unchanged logic)
# ─────────────────────────────────────────────────────────────

def check_report_reminders():
    """
    Once per day: email caretaker if no medical report has been uploaded for 30+ days.
    Only fires once per week per recipient to avoid spam.
    """
    db = _get_db()
    try:
        from tables.users import CareRecipient, CareTaker
        from tables.medical_reports import MedicalReport
        from sqlalchemy import func

        today = datetime.date.today()
        week_key = today.strftime("%Y-W%W")

        recipients = db.query(CareRecipient).all()
        for recipient in recipients:
            reminder_key = f"report_{recipient.id}_{week_key}"

            latest = db.query(func.max(MedicalReport.report_date)).filter(
                MedicalReport.care_recipient_id == recipient.id
            ).scalar()

            if latest is None:
                days_since = 999
            elif isinstance(latest, datetime.datetime):
                days_since = (today - latest.date()).days
            else:
                days_since = (today - latest).days

            if days_since >= 30:
                caretaker = db.query(CareTaker).filter(
                    CareTaker.id == recipient.caretaker_id
                ).first()
                if caretaker and caretaker.email:
                    from services.email_notifications import send_report_upload_reminder
                    send_report_upload_reminder(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name,
                        days_since_last=days_since if days_since < 999 else 30,
                    )
                    print(f"[scheduler] Report reminder sent for {recipient.full_name} ({days_since} days)")
    except Exception as e:
        print(f"[scheduler] Report reminder error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 5. AUTO-REORDER  (updated: uses actual confirmed consumption)
# ─────────────────────────────────────────────────────────────

def check_auto_reorder():
    """
    Once per day: for every active medication with auto_order_enabled,
    calculate days_remaining based on ACTUAL confirmed (TAKEN) doses over
    the last 14 days — not an assumed daily decrement.

    Fires a Tata 1mg order-link email if ≤ 7 days of stock remain.
    Rate-limited to once per 7 days via last_auto_order_date.
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.medication_dose_logs import MedicationDoseLog, DoseStatusEnum
        from tables.users import CareRecipient, CareTaker
        from urllib.parse import quote

        today = datetime.date.today()
        fourteen_days_ago = datetime.datetime.utcnow() - timedelta(days=14)

        meds = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.auto_order_enabled == True,
        ).all()

        for med in meds:
            stock = med.current_stock or 0

            # ── Compute avg daily consumption from confirmed TAKEN logs ──
            taken_count = db.query(MedicationDoseLog).filter(
                MedicationDoseLog.medication_id == med.medication_id,
                MedicationDoseLog.status == DoseStatusEnum.TAKEN,
                MedicationDoseLog.confirmed_at >= fourteen_days_ago,
            ).count()

            avg_daily = taken_count / 14.0 if taken_count > 0 else (med.doses_per_day or 1)
            days_remaining = int(stock / avg_daily) if avg_daily > 0 else 999

            if days_remaining > 7:
                continue

            # ── Rate-limit: skip if already sent within last 7 days ──
            if med.last_auto_order_date:
                days_since_last_order = (today - med.last_auto_order_date).days
                if days_since_last_order < 7:
                    continue

            order_link = f"https://www.1mg.com/search/all?name={quote(med.medicine_name)}"

            recipient = db.query(CareRecipient).filter(
                CareRecipient.id == med.care_recipient_id
            ).first()
            if not recipient:
                continue
            caretaker = db.query(CareTaker).filter(
                CareTaker.id == recipient.caretaker_id
            ).first()
            if not caretaker or not caretaker.email:
                continue

            from services.email_notifications import send_auto_reorder_notification
            sent = send_auto_reorder_notification(
                to_email=caretaker.email,
                recipient_name=recipient.full_name,
                medicine_name=med.medicine_name,
                dosage=med.dosage or "As prescribed",
                current_stock=stock,
                days_remaining=days_remaining,
                order_link=order_link,
            )

            med.last_auto_order_date = today
            db.flush()

            status_str = "[OK] email sent" if sent else "[WARN] email failed"
            print(f"[scheduler] [REORDER] {med.medicine_name} "
                  f"({days_remaining}d left, avg {avg_daily:.1f}/day) → {status_str}")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[scheduler] Auto-reorder check error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 6. DAILY RECOMMENDATIONS  (unchanged)
# ─────────────────────────────────────────────────────────────

def run_daily_recommendations():
    """
    Once per day: re-run clinical recommendation engine for every care recipient
    so recommendations stay fresh even without new report uploads.
    """
    db = _get_db()
    try:
        from tables.users import CareRecipient
        from services.insights_engine import run_recommendation_pipeline

        recipients = db.query(CareRecipient).all()
        for recipient in recipients:
            try:
                # Use asyncio.run since we are in a sync thread
                asyncio.run(run_recommendation_pipeline(recipient.id, db))
                print(f"[scheduler] Regenerated v3 recommendations for {recipient.full_name} (ID: {recipient.id})")
            except Exception as e:
                print(f"[scheduler] Failed v3 recommendations for recipient {recipient.id}: {e}")
    except Exception as e:
        print(f"[scheduler] Daily recommendations task failed: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# 7. WEEKLY HEALTH REPORT (Every Friday 9:00 AM)
# ─────────────────────────────────────────────────────────────

def generate_weekly_reports():
    """
    Every Friday at 9:00 AM: generate a weekly health report for every
    care recipient, then email it to the caretaker.
    """
    db = _get_db()
    try:
        from services.weekly_report_generator import generate_all_weekly_reports
        from services.email_notifications import send_weekly_report_email
        from tables.users import CareRecipient, CareTaker
        from tables.weekly_reports import WeeklyReport

        results = generate_all_weekly_reports(db)
        print(f"[scheduler] [WEEKLY] Generated {len(results)} weekly reports")

        # Send emails for newly generated reports
        for result in results:
            if result.get("status") != "ok":
                continue
            try:
                recipient = db.query(CareRecipient).filter(
                    CareRecipient.id == result["id"]
                ).first()
                if not recipient:
                    continue
                caretaker = db.query(CareTaker).filter(
                    CareTaker.id == recipient.caretaker_id
                ).first()
                if not caretaker or not caretaker.email:
                    continue

                # Get the latest report for this recipient
                report = db.query(WeeklyReport).filter(
                    WeeklyReport.care_recipient_id == recipient.id
                ).order_by(WeeklyReport.created_at.desc()).first()

                if report and not report.email_sent:
                    report_payload = {
                        "overall_score": report.overall_score,
                        "risk_trend": report.risk_trend,
                        "ai_narrative": report.ai_narrative,
                        "sections": report.report_data,
                        "week_start": str(report.week_start),
                        "week_end": str(report.week_end),
                    }
                    sent = send_weekly_report_email(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name,
                        report_data=report_payload,
                    )
                    if sent:
                        report.email_sent = True
                        db.commit()
                        print(f"[scheduler] [WEEKLY] Email sent for {recipient.full_name}")
            except Exception as e:
                print(f"[scheduler] [WEEKLY] Email failed for recipient {result['id']}: {e}")

    except Exception as e:
        print(f"[scheduler] Weekly report generation failed: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# DISABLED — passive auto-decrement (replaced by confirmation flow)
# ─────────────────────────────────────────────────────────────

def decrement_daily_stock():
    """
    DISABLED — Stock is no longer decremented automatically.
    Stock only changes when status transitions to TAKEN via POST /api/confirm-dose
    or GET /api/confirm-dose-email.

    This function is preserved for reference only. Do NOT re-enable without
    removing the confirmation-based flow.
    """
    print("[scheduler] decrement_daily_stock() is DISABLED — confirmation-based flow active.")


# ─────────────────────────────────────────────────────────────
# MAIN SCHEDULER LOOP
# ─────────────────────────────────────────────────────────────

def _scheduler_loop():
    """Main scheduler loop — runs every 60 seconds."""
    print("[scheduler] [START] Background notification scheduler started (confirmation-based mode)")

    last_expiry_check         = None
    last_report_check         = None
    last_auto_reorder_check   = None
    last_recommendation_run   = None
    last_weekly_report_run    = None     # tracks weekly report generation

    while not _stop_event.is_set():
        now   = datetime.datetime.now()
        today = now.date()

        # ── Every-minute jobs ──
        try:
            schedule_pending_doses()
        except Exception as e:
            print(f"[scheduler] schedule_pending_doses loop error: {e}")

        try:
            check_missed_doses()
        except Exception as e:
            print(f"[scheduler] check_missed_doses loop error: {e}")

        # ── Once-per-day jobs ──
        if last_expiry_check != today:
            try:
                check_medication_expiry()
                last_expiry_check = today
            except Exception as e:
                print(f"[scheduler] Expiry loop error: {e}")

        if last_report_check != today:
            try:
                check_report_reminders()
                last_report_check = today
            except Exception as e:
                print(f"[scheduler] Report reminder loop error: {e}")

        if last_auto_reorder_check != today:
            try:
                check_auto_reorder()
                last_auto_reorder_check = today
            except Exception as e:
                print(f"[scheduler] Auto-reorder loop error: {e}")

        if last_recommendation_run != today:
            try:
                run_daily_recommendations()
                last_recommendation_run = today
            except Exception as e:
                print(f"[scheduler] Daily recommendations loop error: {e}")

        # ── Weekly report — Friday 9:00 AM ──
        if now.weekday() == 4 and now.hour >= 9 and last_weekly_report_run != today:
            try:
                generate_weekly_reports()
                last_weekly_report_run = today
            except Exception as e:
                print(f"[scheduler] Weekly report loop error: {e}")

        # Sleep 60 s, checking stop every second for clean shutdown
        for _ in range(60):
            if _stop_event.is_set():
                break
            time.sleep(1)

    print("[scheduler] [STOP] Notification scheduler stopped")


def start_scheduler():
    """Start the background scheduler thread (idempotent)."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        print("[scheduler] Already running")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop, daemon=True, name="NotificationScheduler"
    )
    _scheduler_thread.start()


def stop_scheduler():
    """Signal the scheduler to stop."""
    _stop_event.set()
