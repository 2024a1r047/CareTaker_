"""Email Notification Service for CareTaker.

Sends:
- Medicine reminder emails at scheduled times
- Report upload reminders (no report for 30+ days)
- Medication completion notifications
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(override=True)


def _get_smtp_config():
    """Get SMTP configuration from environment."""
    return {
        "server": os.getenv("MAIL_SERVER", "smtp.gmail.com"),
        "port": int(os.getenv("MAIL_PORT", 587)),
        "username": os.getenv("MAIL_USERNAME"),
        "password": os.getenv("MAIL_PASSWORD"),
        "from_email": os.getenv("MAIL_FROM"),
        "starttls": os.getenv("MAIL_STARTTLS", "True").lower() == "true",
    }


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    config = _get_smtp_config()
    if not config["username"] or not config["password"]:
        print("[email] SMTP credentials not configured, skipping email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = config["from_email"]
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(config["server"], config["port"], timeout=30) as server:
            if config["starttls"]:
                server.starttls()
            server.login(config["username"], config["password"])
            server.sendmail(config["from_email"], to_email, msg.as_string())

        print(f"[email] ✅ Sent to {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"[email] ❌ Failed to send to {to_email}: {e}")
        return False


def send_medicine_reminder(to_email: str, recipient_name: str, medicine_name: str, dosage: str, schedule_time: str):
    """Send a medication reminder email."""
    subject = f"💊 Medicine Reminder: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">💊</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medicine Reminder</h2>
        </div>
        <div style="background: linear-gradient(135deg, #ebf8ff, #e6fffa); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Dosage:</strong> {dosage or 'As prescribed'}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Scheduled Time:</strong> {schedule_time}</p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            This is an automated reminder from CareTaker AI.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_medication_completed(to_email: str, recipient_name: str, medicine_name: str, duration_days: int):
    """Send a notification that a medication course is complete."""
    subject = f"✅ Medication Completed: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">✅</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medication Course Completed</h2>
        </div>
        <div style="background: linear-gradient(135deg, #f0fff4, #e6fffa); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Duration:</strong> {duration_days} days</p>
            <p style="margin: 8px 0; color: #38a169; font-weight: 600;">
                This medication has been automatically marked as completed.
            </p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            Consult your doctor before making any changes to the medication plan.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_report_upload_reminder(to_email: str, recipient_name: str, days_since_last: int):
    """Send a reminder that no medical report has been uploaded recently."""
    subject = f"📋 Report Reminder: No new reports for {recipient_name} in {days_since_last} days"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">📋</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medical Report Reminder</h2>
        </div>
        <div style="background: linear-gradient(135deg, #fffaf0, #fefcbf); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 8px 0; color: #c05621; font-weight: 600;">
                ⚠️ It has been {days_since_last} days since the last medical report was uploaded.
            </p>
            <p style="margin: 4px 0; color: #4a5568;">
                Regular medical reports help track health trends and detect issues early.
                Please upload a recent report to keep the health profile up to date.
            </p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            This is an automated reminder from CareTaker AI.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_auto_reorder_notification(to_email: str, recipient_name: str, medicine_name: str, dosage: str, current_stock: int, days_remaining: int, order_link: str):
    """Send a notification that an auto-reorder has been triggered via Tata 1mg."""
    subject = f"🛒 Auto-Reorder Triggered: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">🛒</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Auto-Reorder Alert</h2>
        </div>
        <div style="background: linear-gradient(135deg, #fff5f5, #fed7d7); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Dosage:</strong> {dosage or 'As prescribed'}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Current Stock:</strong> {current_stock} units</p>
            <p style="margin: 8px 0; color: #c53030; font-weight: 600;">
                ⚠️ Only {days_remaining} day(s) of stock remaining!
            </p>
        </div>
        <div style="text-align: center; margin: 20px 0;">
            <a href="{order_link}" 
               style="display: inline-block; background: linear-gradient(135deg, #FF6B35, #F72C25); color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 1rem;">
                🛍️ Order Now on Tata 1mg
            </a>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            CareTaker AI has detected low stock and generated this order link automatically.<br>
            Click the button above to review and complete the purchase on Tata 1mg.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Dose Confirmation Email  (actionable — caretaker clicks TAKEN / MISSED)
# ─────────────────────────────────────────────────────────────────────────────

def send_dose_confirmation_email(
    to_email: str,
    recipient_name: str,
    medicine_name: str,
    dosage: str,
    schedule_time: str,
    dose_log_id: int,
    token: str,
) -> bool:
    """
    Send a medicine confirmation email with two actionable buttons:
      - ✔ Mark as TAKEN  → GET /api/confirm-dose-email?token=TOKEN&action=TAKEN
      - ✖ Mark as NOT TAKEN → GET /api/confirm-dose-email?token=TOKEN&action=MISSED

    The token is a UUID4 tied to this specific dose log.  Both links are safe
    to embed in email clients because they use plain GET requests.
    """
    import os
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    taken_url  = f"{base_url}/api/confirm-dose-email?token={token}&action=TAKEN"
    missed_url = f"{base_url}/api/confirm-dose-email?token={token}&action=MISSED"

    subject = f"💊 Confirm Medicine: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;padding:28px;border-radius:16px;border:1px solid #e2e8f0;background:#fff;">

        <!-- Header -->
        <div style="text-align:center;margin-bottom:24px;">
            <span style="font-size:52px;">💊</span>
            <h2 style="color:#1e3a8a;margin:10px 0 4px;font-size:1.4rem;">Medicine Confirmation Required</h2>
            <p style="color:#64748b;margin:0;font-size:0.9rem;">Please confirm whether the dose was taken</p>
        </div>

        <!-- Info Card -->
        <div style="background:linear-gradient(135deg,#eff6ff,#e0f2fe);padding:20px;border-radius:12px;margin-bottom:24px;">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;width:120px;"><strong>Patient</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{recipient_name}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Medicine</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:700;">{medicine_name}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Dosage</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{dosage or 'As prescribed'}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Scheduled</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{schedule_time}</td></tr>
            </table>
        </div>

        <!-- Action Buttons -->
        <div style="display:flex;gap:12px;justify-content:center;margin-bottom:20px;">
            <a href="{taken_url}"
               style="display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:white;
                      padding:14px 28px;border-radius:10px;text-decoration:none;font-weight:700;
                      font-size:1rem;text-align:center;min-width:160px;">
                ✔&nbsp; Mark as TAKEN
            </a>
            <a href="{missed_url}"
               style="display:inline-block;background:#fff;color:#ef4444;border:2px solid #ef4444;
                      padding:12px 28px;border-radius:10px;text-decoration:none;font-weight:700;
                      font-size:1rem;text-align:center;min-width:160px;">
                ✖&nbsp; Not Taken
            </a>
        </div>

        <!-- Footer -->
        <p style="color:#94a3b8;font-size:0.78rem;text-align:center;margin:0;">
            This link expires after 24 hours.<br>
            Do not forward this email — the confirmation link is unique to this dose.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Missed Dose Escalation Email  (urgent alert to caretaker)
# ─────────────────────────────────────────────────────────────────────────────

def send_missed_dose_escalation(
    to_email: str,
    recipient_name: str,
    medicine_name: str,
    dosage: str,
    scheduled_time: str,
) -> bool:
    """
    Send an urgent escalation email when the system auto-marks a dose as MISSED
    (no confirmation received within 60 minutes of the scheduled time).
    """
    subject = f"🚨 MISSED DOSE ALERT: {medicine_name} — {recipient_name}"
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;padding:28px;border-radius:16px;border:2px solid #ef4444;background:#fff;">

        <!-- Urgent Header -->
        <div style="background:linear-gradient(135deg,#ef4444,#dc2626);border-radius:12px;padding:20px;text-align:center;margin-bottom:24px;">
            <span style="font-size:48px;">🚨</span>
            <h2 style="color:white;margin:8px 0 4px;font-size:1.4rem;">Missed Dose Alert</h2>
            <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.9rem;">Immediate attention required</p>
        </div>

        <!-- Info Card -->
        <div style="background:#fff5f5;padding:20px;border-radius:12px;border:1px solid #fecaca;margin-bottom:20px;">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;width:120px;"><strong>Patient</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:600;">{recipient_name}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Medicine</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:700;">{medicine_name}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Dosage</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{dosage or 'As prescribed'}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Scheduled</strong></td>
                    <td style="color:#c53030;font-size:0.9rem;font-weight:700;">{scheduled_time}</td></tr>
            </table>
        </div>

        <!-- Body Message -->
        <p style="color:#1e293b;font-size:1rem;font-weight:600;margin-bottom:8px;">
            ⚠️ {recipient_name} has <span style="color:#ef4444;">not confirmed</span> taking
            <strong>{medicine_name}</strong> scheduled at <strong>{scheduled_time}</strong>.
        </p>
        <p style="color:#475569;font-size:0.9rem;margin-bottom:20px;">
            This dose has been automatically marked as <strong>MISSED</strong>.
            Please contact the patient immediately and verify their status.
        </p>

        <!-- Footer -->
        <p style="color:#94a3b8;font-size:0.78rem;text-align:center;margin:0;">
            CareTaker AI — Automated Escalation Alert
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Weekly Health Report Email
# ─────────────────────────────────────────────────────────────────────────────

def send_weekly_report_email(
    to_email: str,
    recipient_name: str,
    report_data: dict,
) -> bool:
    """Send a beautifully formatted weekly health report email to the caretaker."""
    score = report_data.get("overall_score", 0)
    trend = report_data.get("risk_trend", "stable")
    narrative = report_data.get("ai_narrative") or {}
    sections = report_data.get("sections") or report_data.get("report_data") or {}
    week_start = report_data.get("week_start", "")
    week_end = report_data.get("week_end", "")

    # Score color
    if score >= 80:
        score_color, score_bg = "#059669", "#ecfdf5"
    elif score >= 60:
        score_color, score_bg = "#d97706", "#fffbeb"
    elif score >= 40:
        score_color, score_bg = "#ea580c", "#fff7ed"
    else:
        score_color, score_bg = "#dc2626", "#fef2f2"

    trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "➡️")

    # Build section summaries
    meds = sections.get("medication_adherence", {})
    vitals = sections.get("vitals_summary", {})
    labs = sections.get("lab_summary", {})
    mood = sections.get("mood_wellbeing", {})
    activity = sections.get("physical_activity", {})
    respiratory = sections.get("respiratory_activity", {})

    highlights = narrative.get("highlights", [])
    concerns = narrative.get("concerns", [])
    actions = narrative.get("caregiver_actions", [])
    emotional = narrative.get("emotional_wellbeing", "")

    highlights_html = "".join(f'<li style="color:#065f46;margin:4px 0;">✅ {h}</li>' for h in highlights)
    concerns_html = "".join(f'<li style="color:#991b1b;margin:4px 0;">⚠️ {c}</li>' for c in concerns)
    actions_html = "".join(f'<li style="color:#1e40af;margin:4px 0;">👉 {a}</li>' for a in actions)

    subject = f"📊 Weekly Health Report: {recipient_name} ({week_start} – {week_end})"
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:600px;margin:0 auto;padding:0;border-radius:16px;overflow:hidden;border:1px solid #e2e8f0;">
        <!-- Header -->
        <div style="background:linear-gradient(135deg,#1e3a8a,#3b82f6);padding:28px;text-align:center;">
            <h1 style="color:white;margin:0 0 4px;font-size:1.5rem;">📊 Weekly Health Report</h1>
            <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.95rem;">{recipient_name} — {week_start} to {week_end}</p>
        </div>

        <!-- Wellness Score -->
        <div style="background:{score_bg};padding:24px;text-align:center;border-bottom:1px solid #e2e8f0;">
            <p style="color:#64748b;margin:0 0 8px;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;">Wellness Score</p>
            <span style="font-size:3rem;font-weight:800;color:{score_color};">{score}</span>
            <span style="font-size:1.2rem;color:#94a3b8;">/100</span>
            <p style="margin:8px 0 0;color:#64748b;">{trend_icon} Trend: <strong>{trend.title()}</strong></p>
        </div>

        <div style="padding:24px;">
            <!-- AI Summary -->
            <div style="background:#f8fafc;padding:16px;border-radius:10px;margin-bottom:20px;border-left:4px solid #3b82f6;">
                <p style="color:#1e293b;font-size:0.95rem;margin:0;">{narrative.get("executive_summary", "Report generated.")}</p>
            </div>

            <!-- Highlights -->
            {"<div style='margin-bottom:16px;'><h3 style=\"color:#065f46;margin:0 0 8px;\">Highlights</h3><ul style='margin:0;padding-left:20px;'>" + highlights_html + "</ul></div>" if highlights else ""}

            <!-- Concerns -->
            {"<div style='margin-bottom:16px;'><h3 style=\"color:#991b1b;margin:0 0 8px;\">Concerns</h3><ul style='margin:0;padding-left:20px;'>" + concerns_html + "</ul></div>" if concerns else ""}

            <!-- Quick Stats -->
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">
                <div style="flex:1;min-width:120px;background:#eff6ff;padding:12px;border-radius:8px;text-align:center;">
                    <p style="margin:0;font-size:0.75rem;color:#64748b;">Med Adherence</p>
                    <p style="margin:4px 0 0;font-size:1.3rem;font-weight:700;color:#1e3a8a;">{meds.get('adherence_pct', '—')}%</p>
                </div>
                <div style="flex:1;min-width:120px;background:#f0fdf4;padding:12px;border-radius:8px;text-align:center;">
                    <p style="margin:0;font-size:0.75rem;color:#64748b;">Falls</p>
                    <p style="margin:4px 0 0;font-size:1.3rem;font-weight:700;color:#065f46;">{activity.get('fall_count', 0)}</p>
                </div>
                <div style="flex:1;min-width:120px;background:#fef3c7;padding:12px;border-radius:8px;text-align:center;">
                    <p style="margin:0;font-size:0.75rem;color:#64748b;">Coughs</p>
                    <p style="margin:4px 0 0;font-size:1.3rem;font-weight:700;color:#92400e;">{respiratory.get('cough_count_7d', 0)}</p>
                </div>
            </div>

            <!-- Emotional Wellbeing -->
            {"<div style='background:#faf5ff;padding:14px;border-radius:8px;margin-bottom:16px;border-left:4px solid #7c3aed;'><p style=\"margin:0 0 4px;font-weight:600;color:#6b21a8;\">🧠 Emotional Wellbeing</p><p style=\"margin:0;color:#4a5568;font-size:0.9rem;\">" + emotional + "</p></div>" if emotional else ""}

            <!-- Actions -->
            {"<div style='background:#eff6ff;padding:14px;border-radius:8px;margin-bottom:16px;'><h3 style=\"color:#1e40af;margin:0 0 8px;\">Caregiver Actions This Week</h3><ul style='margin:0;padding-left:20px;'>" + actions_html + "</ul></div>" if actions else ""}
        </div>

        <!-- Footer -->
        <div style="background:#f1f5f9;padding:16px;text-align:center;border-top:1px solid #e2e8f0;">
            <p style="color:#94a3b8;font-size:0.78rem;margin:0;">
                CareTaker AI — Automated Weekly Health Report<br>
                View the full report on your dashboard for more details.
            </p>
        </div>
    </div>
    """
    return send_email(to_email, subject, html)
