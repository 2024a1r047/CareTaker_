"""
Conversation Director - Deterministic Proactive Context Builder
"""

from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session
from sqlalchemy import desc
import random

from tables.conversation_history import ConversationMessage, SenderEnum, MoodEnum
from tables.medications import Medication
from tables.users import CareRecipient


ANNUAL_OCCASIONS = [
    (1, 14,  "मकर संक्रांति", "Makar Sankranti", "🪁"),
    (1, 26,  "गणतंत्र दिवस",  "Republic Day",    "🇮🇳"),
    (3, 8,   "होली",           "Holi",            "🎨"),
    (4, 14,  "बैसाखी",         "Baisakhi",        "🌾"),
    (8, 15,  "स्वतंत्रता दिवस", "Independence Day","🇮🇳"),
    (10, 24, "दशहरा",          "Dussehra",        "🎆"),
    (11, 12, "दीवाली",         "Diwali",          "🪔"),
    (12, 25, "क्रिसमस",        "Christmas",       "🎄"),
    (1, 1,   "नया साल मुबारक", "Happy New Year",  "🎊"),
]


def _get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _time_period(hour):
    if 5 <= hour < 12:   return "morning"
    if 12 <= hour < 17:  return "afternoon"
    if 17 <= hour < 21:  return "evening"
    return "night"


def _check_today_occasion(today):
    for month, day, name_hi, name_en, emoji in ANNUAL_OCCASIONS:
        try:
            festival_date = date(today.year, month, day)
            if abs((today - festival_date).days) <= 1:
                return {"name_hi": name_hi, "name_en": name_en, "emoji": emoji, "is_today": today == festival_date}
        except Exception:
            continue
    return None


def _last_interaction_gap(recipient_id, db):
    ist_offset = timedelta(hours=5, minutes=30)
    now_ist = datetime.utcnow() + ist_offset
    today_ist = datetime(now_ist.year, now_ist.month, now_ist.day)

    last = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at < (today_ist - ist_offset)
    ).order_by(desc(ConversationMessage.created_at)).first()

    if not last:
        return {"days_gap": 0, "last_snippet": "", "skipped": False, "is_new": True}

    last_ist = last.created_at + ist_offset
    last_date = datetime(last_ist.year, last_ist.month, last_ist.day).date()
    gap = (today_ist.date() - last_date).days

    recent = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at >= (last.created_at - timedelta(minutes=45))
    ).order_by(ConversationMessage.created_at).limit(4).all()

    snippet = " | ".join(m.message_text[:60] for m in recent)
    return {"days_gap": gap, "last_snippet": snippet[:200], "skipped": gap >= 1, "is_new": False}


def _due_medicine(recipient_id, db, hour):
    """Return medicine name if any active med is due within ±1 hour of current time."""
    meds = db.query(Medication).filter(
        Medication.care_recipient_id == recipient_id,
        Medication.status == "active"
    ).all()
    for med in meds:
        # 1. Use schedule_time field if available (e.g. "08:00", "20:00", "morning")
        sched = (med.schedule_time or "").strip().lower()
        if sched:
            # Try HH:MM format
            try:
                parts = sched.split(":")
                if len(parts) == 2:
                    med_hour = int(parts[0])
                    if abs(hour - med_hour) <= 1:
                        return med.medicine_name
            except ValueError:
                pass
            # Try word-based
            if sched in ("morning", "subah") and 5 <= hour <= 11:
                return med.medicine_name
            if sched in ("afternoon", "dopahar") and 12 <= hour <= 15:
                return med.medicine_name
            if sched in ("evening", "shaam") and 17 <= hour <= 20:
                return med.medicine_name
            if sched in ("night", "raat") and 20 <= hour <= 23:
                return med.medicine_name

        # 2. Fall back to frequency text matching
        freq = (med.frequency or "").lower()
        if "morning" in freq and 5 <= hour <= 11:
            return med.medicine_name
        if any(w in freq for w in ["evening", "night", "pm"]) and 19 <= hour <= 22:
            return med.medicine_name
        if ("twice" in freq or "bd" in freq or "2 times" in freq) and (5 <= hour <= 11 or 19 <= hour <= 22):
            return med.medicine_name
    return None


def _mood_trend(recipient_id, db):
    msgs = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at >= datetime.utcnow() - timedelta(days=7),
        ConversationMessage.mood_detected != None
    ).order_by(ConversationMessage.created_at).all()
    if len(msgs) < 3:
        return "stable"
    neg = {MoodEnum.sad, MoodEnum.lonely, MoodEnum.anxious, MoodEnum.distressed, MoodEnum.angry}
    mid = len(msgs) // 2
    f = sum(1 for m in msgs[:mid] if m.mood_detected in neg)
    s = sum(1 for m in msgs[mid:] if m.mood_detected in neg)
    if s > f + 1: return "worsening"
    if f > s + 1: return "improving"
    return "stable"


def build_proactive_context(recipient_id, db, language="hi"):
    now = _get_ist_now()
    today = now.date()
    hour = now.hour
    period = _time_period(hour)

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    first_name = (recipient.full_name or "").split()[0] if recipient else "ji"
    age = recipient.age if recipient else None

    gap_info = _last_interaction_gap(recipient_id, db)
    occasion = _check_today_occasion(today)
    med_due = _due_medicine(recipient_id, db, hour)
    mood_trend = _mood_trend(recipient_id, db)

    meal_context = None
    if 7 <= hour <= 9:     meal_context = "breakfast"
    elif 12 <= hour <= 14: meal_context = "lunch"
    elif 19 <= hour <= 21: meal_context = "dinner"

    return {
        "time_period": period,
        "hour": hour,
        "current_time_str": now.strftime("%I:%M %p"),
        "day_name": now.strftime("%A"),
        "first_name": first_name,
        "days_gap": gap_info["days_gap"],
        "last_snippet": gap_info["last_snippet"],
        "skipped_yesterday": gap_info["skipped"],
        "is_new": gap_info.get("is_new", False),
        "occasion": occasion,
        "medicine_due": med_due,
        "meal_context": meal_context,
        "mood_trend": mood_trend,
        "age": age,
        "language": language,
    }


def format_proactive_instruction(ctx):
    lines = []
    name = ctx["first_name"]
    period = ctx["time_period"]
    gap = ctx["days_gap"]

    lines.append("=== PROACTIVE OPENING CONTEXT ===")
    lines.append(f"Time: {ctx['current_time_str']}, {ctx['day_name']} ({period})")

    if ctx.get("occasion"):
        occ = ctx["occasion"]
        lines.append(f"TODAY'S OCCASION: {occ['emoji']} {occ['name_hi']} / {occ['name_en']} - acknowledge warmly.")

    if ctx.get("is_new"):
        lines.append("Gap: First conversation with the user. Give a warm time-appropriate greeting.")
    elif gap == 0:
        lines.append("Gap: User spoke earlier today. Continue the conversation warmly.")
    else:
        lines.append("Gap: Over 24 hours of silence. Start by naturally checking on them.")
        lines.append("Instruction: Do NOT say 'mujhe aapki yaad aayi' (AI shouldn't claim personal feelings).")
        lines.append("Say something like: 'Kal aap kahan the? Sab theek tha na? Kisi ki yaad aa rahi thi kya?'")
        if ctx.get("last_snippet"):
            lines.append(f"Last topic to follow up: \"{ctx['last_snippet'][:120]}\"")

    if ctx.get("medicine_due"):
        lines.append(f"Medicine due: {ctx['medicine_due']} - weave in a gentle reminder naturally.")

    if ctx.get("meal_context"):
        meal_words = {"breakfast": "naashta/breakfast", "lunch": "lunch/khana", "dinner": "dinner/raat ka khana"}
        lines.append(f"Meal time: {meal_words[ctx['meal_context']]} time - can ask if they have eaten.")

    if ctx["mood_trend"] == "worsening":
        lines.append("Mood trend (7 days): WORSENING - be extra gentle and nurturing today.")
    elif ctx["mood_trend"] == "improving":
        lines.append("Mood trend (7 days): IMPROVING - celebrate this warmly.")

    lines.append("")
    lines.append("RULE: YOU speak first. One warm opening sentence. Do NOT list all topics. Let conversation flow naturally.")

    return "\n".join(lines)
