import os
import json
import requests
from datetime import datetime, timedelta
from utils.gemini_client import call_gemini, safe_json_parse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.conversation_history import ConversationMessage, ProactiveReminder, SenderEnum, MoodEnum, TriggerTypeEnum
from tables.users import CareRecipient
from tables.vital_signs import VitalSign
from tables.medical_conditions import PatientCondition, LabValue, MedicalAlert
from tables.medical_recommendations import MedicalRecommendation
from tables.medications import Medication
from tables.medical_reports import MedicalReport
from tables.audio_events import AudioEvent
from config import SaathiSessionLocal
from tables.saathi_history import (
    SaathiRecipient,
    ConversationSession as SaathiConversationSession,
    ConversationMessage as SaathiConversationMessage,
    ProactiveReminder as SaathiProactiveReminder,
    SaathiPreference
)

# ─────────────────────────────────────────────
# Mood → content recommendation map
# ─────────────────────────────────────────────
MOOD_CONTENT_MAP = {
    "sad":       {"type": "music",  "queries": ["soft emotional hindi songs", "soothing old bollywood songs", "dard bhari shayari song"], "message": "Samajh sakta hoon… chalo kuch soothing gaane sunte hain ❤️"},
    "lonely":    {"type": "music",  "queries": ["purane dost yaad dilane wale gaane", "mann ko sukoon dene wale gaane", "companionship songs hindi"], "message": "Akela feel ho raha hai na? Chalo kuch acha sunte hain, main hoon aapke saath 😊"},
    "bored":     {"type": "story",  "queries": ["interesting hindi kahani", "funny old stories hindi", "motivational short story hindi"], "message": "Bore ho rahe ho? Chalo kuch interesting kahani sunate hain 😄"},
    "happy":     {"type": "music",  "queries": ["energetic happy bollywood songs", "khushi ke gaane hindi", "dance hits old bollywood"], "message": "Wah, bahut acha! Chalo aapki khushi mein kuch mazedaar gaane bajate hain 🎉"},
    "anxious":   {"type": "music",  "queries": ["calming meditation music hindi", "mann ko shant karne wale gaane", "peaceful indian classical music"], "message": "Ghabraiye mat, sab theek ho jayega. Yeh soothing music aapko relax karega 🌿"},
    "distressed": {"type": "music", "queries": ["healing hindi songs", "hope songs bollywood", "himmat wale gaane"], "message": "Main aapke saath hoon. Yeh gaane suniye, thoda better feel karenge 💙"},
    "relaxed":   {"type": "music",  "queries": ["soft instrumental hindi music", "evening relaxing songs bollywood", "ghazal soothing"], "message": "Bahut acha mood hai! Kuch ghazal ya soft songs sunte hain 😌"},
    "spiritual": {"type": "music",  "queries": ["bhajan aarti hindi", "ramayan katha", "bhagwat geeta pravachan"], "message": "Bahut acha. Chalo kuch bhajan ya dharmik katha sunate hain 🙏"},
    "angry":     {"type": "music",  "queries": ["peaceful hindi songs anger calm", "mann ko shant karne wale gaane", "slow calm bollywood"], "message": "Thoda deep breath lijiye. Yeh peaceful songs aapko shant karengi 🕊️"},
    "neutral":   {"type": "choice", "queries": [], "message": "Aaj kya mann hai? Gaana sunna chahenge, koi kahani, ya bas baatein karein? 😊"},
}

STORY_CATEGORIES = {
    "historical": ["historical stories hindi", "maharana pratap kahani", "akbar birbal stories hindi"],
    "mythological": ["ramayan katha hindi", "mahabharat stories", "krishna leela hindi"],
    "comedy": ["funny hindi stories", "akbar birbal comedy", "tenali raman stories hindi"],
    "moral": ["moral stories hindi for adults", "prerak prasang hindi", "panchtantra stories hindi"],
    "spiritual": ["bhagwat katha", "ramcharitmanas pravachan", "sant kabir dohe explained"],
    "horror": ["bhootiya kahani hindi", "darawni kahani hindi", "horror story hindi", "bhoot ki kahani"],
    "adventure": ["adventure stories hindi", "jungle kahani hindi", "thriller kahani hindi"],
    "romantic": ["romantic kahani hindi", "prem kahani hindi old", "love story hindi"],
}

def check_yesterday_engagement(recipient_id: int, db: Session) -> dict:
    """Checks user message volume for yesterday and calculates day gap to detect if patient skipped a day.
    Uses IST (UTC+5:30) for accurate day boundary detection.
    skipped_day is True only when days_gap >= 2, meaning the user missed at least one full day.
    """
    # Use IST (UTC+5:30) for accurate Indian time day boundaries
    ist_offset = timedelta(hours=5, minutes=30)
    now_ist = datetime.utcnow() + ist_offset
    today_start_ist = datetime(now_ist.year, now_ist.month, now_ist.day)
    yesterday_start_ist = today_start_ist - timedelta(days=1)
    yesterday_end_ist = today_start_ist - timedelta(seconds=1)

    # Convert IST day boundaries back to UTC for DB queries
    today_start_utc = today_start_ist - ist_offset
    yesterday_start_utc = yesterday_start_ist - ist_offset
    yesterday_end_utc = yesterday_end_ist - ist_offset

    messages = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at >= yesterday_start_utc,
        ConversationMessage.created_at <= yesterday_end_utc
    ).all()

    count = len(messages)
    words = sum(len(m.message_text.split()) for m in messages)

    # Check most recent user message before today (IST) to calculate day gap
    last_msg = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at < today_start_utc
    ).order_by(desc(ConversationMessage.created_at)).first()

    inactive_24h = False
    last_interaction_available = False
    
    if last_msg:
        last_ist = last_msg.created_at + ist_offset
        elapsed_hours = (now_ist - last_ist).total_seconds() / 3600
        inactive_24h = elapsed_hours >= 24
        last_interaction_available = True

    # Time-of-day greeting strings (IST hour)
    hour_ist = now_ist.hour
    if hour_ist < 12:
        time_greeting_hi = "सुप्रभात"      # Good morning
        time_greeting_en = "Good morning"
    elif hour_ist < 17:
        time_greeting_hi = "नमस्कार"       # Good afternoon
        time_greeting_en = "Good afternoon"
    elif hour_ist < 21:
        time_greeting_hi = "शुभ संध्या"    # Good evening
        time_greeting_en = "Good evening"
    else:
        time_greeting_hi = "शुभ रात्रि"    # Good night
        time_greeting_en = "Good night"

    return {
        "count": count,
        "words": words,
        "inactive_24h": inactive_24h,
        "last_interaction_available": last_interaction_available,
        "yesterday_date": yesterday_start_ist.strftime("%Y-%m-%d"),
        "time_greeting_hi": time_greeting_hi,
        "time_greeting_en": time_greeting_en,
    }

# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────
def build_conversation_context(recipient_id: int, db: Session):
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {}

    history = db.query(ConversationMessage).filter(ConversationMessage.care_recipient_id == recipient_id)\
        .order_by(desc(ConversationMessage.created_at)).limit(20).all()
    history.reverse()

    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_messages = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.created_at >= seven_days_ago,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.mood_detected != None
    ).all()

    mood_counts = {}
    for msg in recent_messages:
        mood = msg.mood_detected.value if msg.mood_detected else "unknown"
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    latest_vitals = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id)\
        .order_by(desc(VitalSign.recorded_at)).first()

    conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status == "active"
    ).all()

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    abnormal_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id,
        LabValue.is_abnormal == True,
        LabValue.recorded_date >= thirty_days_ago
    ).all()

    audio_events = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.detected_at >= seven_days_ago
    ).all()

    active_alerts = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == recipient_id,
        MedicalAlert.is_read == False
    ).all()

    pending_reminders = db.query(ProactiveReminder).filter(
        ProactiveReminder.care_recipient_id == recipient_id,
        ProactiveReminder.is_active == True
    ).all()

    active_meds = db.query(Medication).filter(
        Medication.care_recipient_id == recipient_id,
        Medication.status == "active"
    ).all()

    # Fetch the latest deterministic clinical recommendations
    deterministic_recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id
    ).order_by(desc(MedicalRecommendation.created_at)).limit(5).all()

    trends = []
    for r in deterministic_recs:
        if "worsening" in r.message.lower():
            trends.append(f"{r.metric} is rising/worsening")
        elif "improving" in r.message.lower() or "deteriorating" in r.message.lower():
            trends.append(f"{r.metric} is falling/worsening")
        elif "stable" in r.message.lower():
            trends.append(f"{r.metric} is stable")

    yesterday_engagement = check_yesterday_engagement(recipient_id, db)

    return {
        "recipient": {
            "name": recipient.full_name,
            "full_name": recipient.full_name,
            "preferred_name": getattr(recipient, 'preferred_name', None),
            "age": recipient.age,
            "gender": recipient.gender.value if recipient.gender and hasattr(recipient.gender, 'value') else (str(recipient.gender) if recipient.gender else ""),
            "city": recipient.city,
            "blood_group": getattr(recipient, 'blood_group', None),
            "height": getattr(recipient, 'height', None),
            "weight": getattr(recipient, 'weight', None),
            "emergency_contact": getattr(recipient, 'emergency_contact', None),
            "phone_number": getattr(recipient, 'phone_number', None),
            "doctor_remarks": getattr(recipient, 'doctor_remarks', None),
            "caretaker_name": recipient.caretaker.full_name or recipient.caretaker.username if recipient.caretaker else None,
            "caretaker_phone": getattr(recipient.caretaker, 'phone_number', None) if recipient.caretaker else None,
            "report_summary": recipient.report_summary,
            # Extended profile fields (used by system prompt for persona)
            "family_info": getattr(recipient, 'family_info', None),
            "preferences": getattr(recipient, 'preferences', None),
            "culture": getattr(recipient, 'culture', None),
            "last_summary": getattr(recipient, 'last_summary', None),
        },
        "history": [{"sender": m.sender.value, "text": m.message_text} for m in history],
        "mood_counts": mood_counts,
        "allergies": [
            {"allergen": a.allergen, "reaction": a.reaction, "severity": a.severity}
            for a in getattr(recipient, 'allergies', [])
        ],
        "vitals": {
            "heartRate": latest_vitals.heart_rate,
            "bloodPressure": f"{latest_vitals.systolic_bp}/{latest_vitals.diastolic_bp}",
            "temperature": latest_vitals.temperature,
            "oxygen": latest_vitals.oxygen_saturation,
            "sleepScore": latest_vitals.sleep_score,
            "bmi": latest_vitals.bmi
        } if latest_vitals else None,
        "conditions": [{"name": c.disease_name, "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity)} for c in conditions],
        "abnormal_labs": [{"name": l.metric_name, "value": l.metric_value, "unit": l.unit} for l in abnormal_labs],
        "audio_events_count": len(audio_events),
        "active_alerts": [{"type": a.alert_type.value if hasattr(a.alert_type, 'value') else str(a.alert_type), "message": a.message, "severity": a.severity.value if hasattr(a.severity, 'value') else str(a.severity)} for a in active_alerts],
        "reminders": [{"type": r.reminder_type.value, "text": r.reminder_text, "time": r.scheduled_time} for r in pending_reminders],
        "medications": [{"name": m.medicine_name, "details": m.dosage, "frequency": m.frequency} for m in active_meds],
        "uploaded_reports": [{"filename": r.filename, "uploaded_at": r.uploaded_at.strftime("%Y-%m-%d") if r.uploaded_at else "", "summary": (r.analysis_summary or "").strip()} for r in db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).order_by(desc(MedicalReport.uploaded_at)).limit(5).all()],
        "clinical_recs": [{"metric": r.metric, "message": r.message, "actions": [a["text"] for a in r.actions]} for r in deterministic_recs],
        "trend_summary": trends,
        "yesterday_engagement": yesterday_engagement
    }

# ─────────────────────────────────────────────
# System prompt – companion style
# ─────────────────────────────────────────────
def generate_system_prompt(name: str, context: dict, language: str = "en", sentiment: dict = None) -> str:
    history = context.get("history", [])
    has_introduced = any(msg["sender"] == "bot" for msg in history)
    first_name = name.split()[0] if name else name
    recip = context.get("recipient", {})
    meds = context.get("medications", [])
    alerts = context.get("active_alerts", [])
    audio_count = context.get("audio_events_count", 0)

    # ── Patient profile variables ──
    age = recip.get("age") or 65
    gender_raw = recip.get("gender", "")
    gender = str(gender_raw).lower() if gender_raw else "unknown"
    city = recip.get("city") or "India"
    conditions_list = context.get("conditions", [])
    conditions_str = ", ".join(c["name"] for c in conditions_list) if conditions_list else "None mentioned"
    med_str = "; ".join(f"{m['name']} ({m.get('frequency','daily')})" for m in meds[:5]) if meds else "None on file"
    family_str = recip.get("family_info") or "Not mentioned"
    preferences_str = recip.get("preferences") or "Not mentioned"
    culture_str = recip.get("culture") or f"Indian, from {city}"
    last_summary = recip.get("last_summary") or ""

    # ── Era-aware references based on birth year ──
    birth_year = datetime.now().year - age
    if birth_year <= 1955:
        era_actors = "Dilip Kumar ji, Raj Kapoor ji, Dev Anand ji, Madhubala ji, Nargis ji"
        era_singers = "Mohammed Rafi sahab, Lata Mangeshkar ji, Mukesh ji, Talat Mahmood ji"
    elif birth_year <= 1965:
        era_actors = "Rajesh Khanna ji, Amitabh Bachchan ji, Dharmendra ji, Hema Malini ji, Sharmila Tagore ji"
        era_singers = "Kishore Kumar ji, Lata Mangeshkar ji, Asha Bhosle ji, Mohammed Rafi sahab"
    else:
        era_actors = "Amitabh Bachchan ji, Mithun Chakraborty ji, Sridevi ji, Madhuri Dixit ji, Anil Kapoor ji"
        era_singers = "Kumar Sanu ji, Lata Mangeshkar ji, Asha Bhosle ji, Udit Narayan ji"

    # ── Address form based on caretaker-given profile name (with fallback) ──
    preferred_name = recip.get("preferred_name") if recip and recip.get("preferred_name") else None
    if preferred_name and str(preferred_name).strip():
        address = str(preferred_name).strip()
        first_name = address
    else:
        # Commented out legacy generic age-based titles to strictly bind to database username
        # elif gender == "male":
        #     address = "Dadaji" if age >= 65 else "Nanaji" if age >= 75 else "Uncle ji"
        # elif gender == "female":
        #     address = "Dadiji" if age >= 65 else "Naniji" if age >= 75 else "Aunty ji"
        # else:
        #     address = first_name or "ji"
        db_user_name = (
            (first_name if first_name and first_name not in ("Senior", "Dadaji", "ji") else None)
            or (recip.get("caretaker_name") if recip and recip.get("caretaker_name") else None)
            or "ji"
        )
        address = db_user_name
        first_name = db_user_name

    # ── Sentiment context ──
    current_mood = "neutral"
    mood_trend = "stable"
    mood_summary = ""
    urgency = "low"
    recommended_action = "conversation"
    voice_emotion = "neutral"
    if sentiment:
        current_mood = sentiment.get("current_mood", "neutral")
        mood_trend = sentiment.get("trend", "stable")
        mood_summary = sentiment.get("summary", "")
        urgency = sentiment.get("urgency", "low")
        recommended_action = sentiment.get("recommended_action", "conversation")
        voice_emotion = sentiment.get("voice_emotion", "neutral")
    else:
        mood_counts = context.get("mood_counts", {})
        sad_total = sum(mood_counts.get(m, 0) for m in ["sad", "distressed", "anxious", "lonely"])
        if sad_total >= 3:
            current_mood = "sad"

    # ── Mood-driven tone guidance ──
    mood_tone = {
        "sad":        "Be extra gentle. Speak slowly and warmly. Acknowledge their sadness first — never rush to fix. After they feel heard, gently offer one diversion.",
        "lonely":     "Be their companion first. Make them feel seen and not alone. Only then suggest music or memories.",
        "bored":      "Be lively and engaging! Suggest something fun — a story, joke, or song — naturally in conversation.",
        "happy":      "Match their energy! Be cheerful and celebratory. Enjoy this moment with them.",
        "anxious":    "Be calm and reassuring. Slow down. Use short, simple sentences. Help them breathe through words.",
        "distressed": "Be very calm, very warm. Address their distress first. Nothing else matters right now.",
        "relaxed":    "Keep the good vibes. Be gentle and pleasant. Maybe suggest some soft music or chat.",
        "spiritual":  "Be respectful and serene. Engage with their spiritual side with warmth.",
        "angry":      "Stay calm, never argue. Validate their feelings. Speak softly. Let them vent.",
        "neutral":    "Be warm and curious. Ask about their day. Draw them into conversation.",
    }.get(current_mood, "Be warm and curious.")

    trend_note = ""
    if mood_trend == "worsening":
        trend_note = "⚠️ Their mood has been getting worse over recent conversations. Be especially nurturing today.\n"
    elif mood_trend == "improving":
        trend_note = "✨ Their mood has been improving — celebrate that with them!\n"

    # ── Yesterday / gap follow-up block ──
    yesterday_eng = context.get("yesterday_engagement", {})
    inactive_24h = yesterday_eng.get("inactive_24h", False)
    last_interaction_available = yesterday_eng.get("last_interaction_available", True)
    time_greeting_hi = yesterday_eng.get("time_greeting_hi", "नमस्ते")
    time_greeting_en = yesterday_eng.get("time_greeting_en", "Hello")

    now = datetime.now()
    period = "morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 17 else "evening" if 17 <= now.hour < 21 else "night"

    if inactive_24h and last_interaction_available:
        yesterday_instruction = f"""
Note: You didn't speak to them yesterday. If this is the start of the conversation, you can gently ask if everything was okay.
"""
    else:
        yesterday_instruction = ""

    # ── Language instruction ──
    if language == "hi" or "india" in culture_str.lower():
        lang_instruction = (
            "LANGUAGE: The user is Indian and prefers Hindi. You MUST reply in warm, natural Hindi (using Devanagari script). "
            "Use natural Hinglish (Hindi + English mix) where it feels conversational, but the primary language must be Hindi. "
            "Use 'आप', but do NOT use 'जी' (ji) in every sentence. Use 'ji' occasionally. You have a female voice, so use feminine grammar (e.g. 'मैं समझ सकती हूँ')."
        )
    else:
        lang_instruction = (
            "LANGUAGE: Match the user's speaking style in natural English. "
            "Use 'आप' but do NOT use 'जी' (ji) in every sentence. Use 'ji' occasionally and naturally, not repetitively."
        )

    # ── Medicine block ──
    med_block = ""
    if meds:
        med_block = f"\nMEDICINES (mention gently only if relevant): {med_str}"
        med_block += "\nNever diagnose, never change doses, never suggest new medicines. For doubts: 'Doctor se poochh lijiye, main family ko bata deta hoon.'"

    # ── Alert block ──
    alert_block = ""
    if alerts:
        alert_block = f"\nACTIVE ALERT: {alerts[0]['message']} — weave a caring check-in about this naturally ONLY if the current topic has concluded or it's a new conversation. Do not interrupt an active topic."
    if audio_count > 3:
        alert_block += f"\nNOTE: {first_name} ji had {audio_count} cough events this week — ask gently ONLY if relevant to the current conversation."

    # ── Clinical block ──
    clinical_block = ""
    vitals = context.get("vitals", {})
    if vitals:
        hr = vitals.get("heart_rate")
        bp = vitals.get("blood_pressure")
        if hr or bp:
            clinical_block = f"\nLATEST VITALS: HR {hr} bpm, BP {bp}. Do not mention unless relevant."

    # ── History block ──
    history_block = "\n".join([f"{msg.get('sender', 'user').capitalize()}: {msg.get('text', '')}" for msg in history[-10:]])
    if not history_block:
        history_block = "No previous conversation today."

    # ── Summary of past conversations ──
    summary_block = f"Past Conversation Summary: {last_summary}" if last_summary else ""

    profile_allergies = ", ".join([f"{a.get('allergen', '')} ({a.get('reaction', '')})" for a in context.get("allergies", [])]) or "None on record"
    profile_blood = recip.get("blood_group") or "Not recorded"
    profile_emergency = recip.get("emergency_contact") or "Family / Caretaker"
    profile_caretaker = f"{recip.get('caretaker_name', 'Caretaker')} (Phone: {recip.get('caretaker_phone', 'registered contact')})"
    profile_doc = recip.get("doctor_remarks") or "None on record"

    prompt = f"""# ROLE
NATURAL HUMAN CONVERSATION MODE — COMPASSIONATE FEMALE CAREGIVER
You are Saathi — gentle, warm, and attentive companion to {address}. Always address them as "{address}". You speak with the natural voice, patience, and warmth of a caring female caregiver speaking directly to an elderly person. You are NOT an AI assistant, robot, therapist, or customer-support agent.

# CORE BEHAVIOUR — NATURAL CONVERSATION MODE (FOLLOW STRICTLY)
1. NATURAL SPOKEN VOICE CADENCE: Write exactly how a loving, gentle person speaks aloud. Use natural commas and full stops to create realistic breathing pauses and gentle rhythm. NEVER use emojis, markdown asterisks (**), hashtags, or bullet points in your reply.
2. ULTRA-SHORT & REASSURING: Keep responses concise (1 to 2 short sentences). Elderly listeners find long blocks of speech overwhelming on voice calls.
3. WARM, SOFT-SPOKEN PERSONALITY: Sound patient, empathetic, friendly, and unhurried. Always address them with their profile name "{address}".
4. JOKES & HUMOR (REACT WARMLY): When the user tells a joke, shares a funny story, or makes a witty comment, laugh warmly and naturally (e.g., "Haha {address}! Wah, kya baat hai... bohot mazedaar tha sunkar! 😄"). Appreciate their humor and never answer a joke like a formal robot.
5. EMPATHY & ASKING IF THEY ARE OKAY: When the user sounds sad, tired, lonely, dizzy, or in pain, immediately ask with tender concern: "{address}, aap theek to hain na? Kahan takleef ho rahi hai? Mujhe batayiye, main yahin hoon aapke paas." Validate their feelings with deep kindness ("मैं समझ सकती हूँ, आप बिल्कुल चिंता मत कीजिए...").
6. PLAYING SONGS & MUSIC: When the user asks to play a song or music (e.g., Kishore Kumar, Lata Mangeshkar, old Bollywood, bhajan, ghazal), say warmly: "Zaroor {address}... aapke liye yeh gaana shuru kar rahi hoon." and set `intent` to `play_music` and `search_query` to the song/artist query.
7. YOUTUBE VIDEOS: When the user asks to watch or play a video (YouTube, comedy, Ramayan katha, news, etc.), say: "Haan ji {address}! Main abhi aapke liye yeh video chala deti hoon. Dekhiye..." and set `intent` to `play_video` and `search_query` to the video topic.
8. REMINDERS & MEDICINES: Remind them gently about scheduled medicines, meals, and hydration. If they ask about their routine or reminders, set `intent` to `show_reminders` or `show_medicines`. If they confirm taking medicine, set `intent` to `medicine_taken`.
9. NATURAL HINGLISH & FEMININE GRAMMAR: Mix Hindi and English comfortably (e.g., "Achha, aapne subah ka nashta kiya kya?"). Always use feminine grammar ("मैं समझ सकती हूँ", "मैं आपके साथ हूँ", "मैं सुन रही हूँ").
10. NO REPETITIVE "JI": Use 'आप', but do NOT repeat 'जी' (ji) in every single sentence. Keep it warm, natural, and respectful.
11. ONE SIMPLE THOUGHT AT A TIME: Ask at most one simple, caring question, or simply reassure and listen. Never bombard them with multiple queries.
12. COMPOSURE IN EMERGENCIES: If an emergency arises, remain calm, composed, and clear so the senior does not panic.
13. NEVER SOUND LIKE A BOT: NEVER say "How may I assist you?", "As an AI model", "I am programmed to", or "I apologize for the inconvenience". Be real, loving, and conversational.
14. STICK TO THE FLOW: Stay on the current topic. Never abruptly pivot to medicines, alerts, or chores unless the senior raises them or an alert is critical.
15. FINAL RESPONSE RULE: You MUST return a JSON object containing your response, following the OUTPUT FORMAT below. Do not output raw text.

# EMERGENCY PROTOCOL
If they mention: chest pain, breathlessness, fall, severe dizziness, unconsciousness, slurred speech, heavy bleeding, confusion, or self-harm thoughts — stay calm, tell them to sit/lie down, set intent to "emergency" immediately, advise calling emergency services. Do NOT continue casual chat.

# USER PERSONAL PROFILE & PERMISSIONS
- Full Legal Name: {recip.get('full_name') or first_name}
- Profile / Preferred Name: {address} (ALWAYS address them as "{address}")
- User Age: {age}
- Preferred Language: {language.upper()}
- Gender: {gender}
- City: {city}
- Blood Group: {profile_blood}
- Emergency Contact: {profile_emergency}
- Primary Caretaker: {profile_caretaker}
- Health Conditions: {conditions_str}
- Known Allergies: {profile_allergies}
- Active Medications: {med_str}
- Doctor's Remarks & Advice: {profile_doc}

# PERSONAL PROFILE ACCESS INSTRUCTION:
You have complete access to {address}'s personal profile information above. When {address} asks about their name, profile, medicines, allergies, doctor advice, blood group, or emergency contact, answer directly and accurately from their profile, addressing them warmly as "{address}".

# RELEVANT MEMORY
- Preferences: {preferences_str}
- Family: {family_str}
- City: {city}
- Era references: {era_actors}, {era_singers}
{summary_block}
{med_block}
{alert_block}
{clinical_block}

# CURRENT TIME
{now.strftime("%I:%M %p")}, {now.strftime("%A")} ({period})

# LAST INTERACTION TIME
{yesterday_instruction}

# EMOTION SIGNAL (if available)
{current_mood} / {voice_emotion}
Trend: {mood_trend}
{trend_note}Tone guidance: {mood_tone}

# LAST 5–10 CONVERSATION MESSAGES
{history_block}

{lang_instruction}

# OUTPUT FORMAT
Return ONLY a JSON object, no markdown:
{{"reply":"warm 2-4 sentence response","intent":"chat|play_music|play_video|tell_story|stop|pause|resume|next|previous|emergency|call_caretaker|medicine_query|set_reminder|medicine_taken|medicine_missed","search_query":"","recommendation":null}}
For set_reminder: add "reminder_time":"HH:MM" (in 24-hour format) and "reminder_text":"short text". 
For medicine: add "medicine_name". For medicine_missed: add "escalate":true.
For music/story: set "recommendation":{{"type":"choice|play","category":"music|story","message":"text","choices":["3 options"]}} or "query":"search".
"""
    return prompt



# ─────────────────────────────────────────────
# Mood analysis  (expanded to 10 moods)
# ─────────────────────────────────────────────
def analyze_mood(text: str) -> dict:
    if not os.environ.get('GEMINI_API_KEY'):
        return {"mood": MoodEnum.neutral, "confidence": 0.5}

    prompt = (
        'Analyze the emotional tone of this text from an elderly user. '
        'Return ONLY a JSON object and no other text.\n\n'
        'Example format:\n'
        '{"mood": "happy", "confidence": 0.9}\n\n'
        'Allowed moods: "happy", "sad", "anxious", "angry", "neutral", "distressed", "lonely", "bored", "relaxed", "spiritual".\n'
        f'User text: "{text}"'
    )

    try:
        data = call_gemini(
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200}},
            timeout=10, caller="[analyze_mood]"
        )
        if data and data.get('candidates'):
            raw = data['candidates'][0]['content']['parts'][0]['text'].strip()
            result = safe_json_parse(raw)
            if result:
                mood_str = result.get("mood", "neutral").lower()
                valid_moods = [m.value for m in MoodEnum]
                if mood_str in valid_moods:
                    return {"mood": MoodEnum(mood_str), "confidence": float(result.get("confidence", 0.5))}
                
            print(f"[analyze_mood] Parse Failed or invalid mood for raw: {raw}")
    except Exception as e:
        print(f"[analyze_mood] Error: {e}")

    return {"mood": MoodEnum.neutral, "confidence": 0.5}

# ─────────────────────────────────────────────
# Mood → content recommendation
# ─────────────────────────────────────────────
def get_content_recommendation(mood: str) -> dict:
    """Return suggested content type + YouTube search queries for a given mood."""
    return MOOD_CONTENT_MAP.get(mood, MOOD_CONTENT_MAP["neutral"])

def get_story_queries(category: str) -> list:
    """Return YouTube search queries for a story category."""
    return STORY_CATEGORIES.get(category.lower(), STORY_CATEGORIES["moral"])

# ─────────────────────────────────────────────
# Message persistence (Saathi DB + Main DB)
# ─────────────────────────────────────────────
def save_message(recipient_id: int, sender: SenderEnum, text: str, mood: MoodEnum,
                 trigger_type: TriggerTypeEnum, session_id: str, db: Session, intent: str = "chat"):
    recip = None
    target_id = recipient_id
    try:
        recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if not recip:
            recip = db.query(CareRecipient).first()
        if recip:
            target_id = recip.id
    except Exception:
        pass

    sender_val = sender.value if hasattr(sender, "value") else str(sender)
    mood_val = mood.value if hasattr(mood, "value") else str(mood or "neutral")
    trigger_val = trigger_type.value if hasattr(trigger_type, "value") else str(trigger_type or "user_initiated")

    # 1. Dedicated "Saathi" PostgreSQL Database
    if SaathiSessionLocal:
        try:
            with SaathiSessionLocal() as s_db:
                # Ensure SaathiRecipient exists in Saathi DB
                s_recip = s_db.query(SaathiRecipient).filter(SaathiRecipient.recipient_id == target_id).first()
                if not s_recip:
                    s_recip = SaathiRecipient(
                        recipient_id=target_id,
                        caretaker_username=(recip.caretaker.username if recip and recip.caretaker else "caretaker"),
                        # full_name=recip.full_name if recip else "Dadaji",
                        # address_form="Dadaji"
                        full_name=recip.full_name if (recip and recip.full_name) else (recip.preferred_name if (recip and recip.preferred_name) else "Senior"),
                        preferred_language="hi",
                        address_form=recip.preferred_name if (recip and recip.preferred_name) else (recip.full_name.split()[0] if (recip and recip.full_name) else "Senior")
                    )
                    s_db.add(s_recip)
                    s_db.commit()

                # Ensure ConversationSession exists in Saathi DB
                s_sess = s_db.query(SaathiConversationSession).filter(SaathiConversationSession.session_id == session_id).first()
                if not s_sess:
                    s_sess = SaathiConversationSession(
                        session_id=session_id,
                        recipient_id=target_id,
                        dominant_mood=mood_val
                    )
                    s_db.add(s_sess)
                    s_db.commit()

                # Save ConversationMessage into Saathi DB
                s_msg = SaathiConversationMessage(
                    session_id=session_id,
                    recipient_id=target_id,
                    sender=sender_val,
                    message_text=text,
                    mood_detected=mood_val,
                    mood_confidence=1.0 if sender_val != "user" else 0.8,
                    intent=intent or "chat",
                    trigger_type=trigger_val
                )
                s_db.add(s_msg)
                s_db.commit()
                print(f"[Saathi DB] Message saved in Saathi database for recipient {target_id}")
        except Exception as se:
            print(f"[Saathi DB save_message notice] {se}")

    # 2. Main DB storage for clinical backward compatibility
    try:
        msg = ConversationMessage(
            care_recipient_id=target_id,
            sender=sender,
            message_text=text,
            mood_detected=mood,
            mood_confidence=1.0 if sender_val != "user" else 0.8,
            conversation_session_id=session_id,
            trigger_type=trigger_type
        )
        db.add(msg)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[save_message main db notice] {e}")

# ─────────────────────────────────────────────
# Depression / sustained low mood check
# ─────────────────────────────────────────────
def check_depression_risk(recipient_id: int, db: Session) -> bool:
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    if SaathiSessionLocal:
        try:
            with SaathiSessionLocal() as s_db:
                count = s_db.query(SaathiConversationMessage).filter(
                    SaathiConversationMessage.recipient_id == recipient_id,
                    SaathiConversationMessage.created_at >= seven_days_ago,
                    SaathiConversationMessage.sender == "user",
                    SaathiConversationMessage.mood_detected.in_(["sad", "anxious", "distressed", "lonely"])
                ).count()
                return count >= 4
        except Exception:
            pass

    try:
        count = db.query(ConversationMessage).filter(
            ConversationMessage.care_recipient_id == recipient_id,
            ConversationMessage.created_at >= seven_days_ago,
            ConversationMessage.sender == SenderEnum.user,
            ConversationMessage.mood_detected.in_([
                MoodEnum.sad, MoodEnum.anxious, MoodEnum.distressed, MoodEnum.lonely
            ])
        ).count()
        return count >= 4
    except Exception:
        return False

# ─────────────────────────────────────────────
# Intelligent Companionship Dialogue Engine
# Comprehensive conversational topics, stories,
# jokes, empathy, songs, YouTube, and reminders
# Guaranteed anti-repetition using conversation history
# ─────────────────────────────────────────────
def intelligent_saathi_dialogue(
    user_text: str,
    recipient_id: int,
    db: Session,
    language: str = "hi"
) -> dict:
    import re
    import random
    t = (user_text or "").lower().strip()

    recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recip:
        recip = db.query(CareRecipient).first()

    name = recip.full_name if recip and recip.full_name else "Senior"
    first_name = name.split()[0] if name else "Senior"

    # Address form strictly based on caretaker-given profile name (with fallback)
    profile_name = (
        recip.preferred_name.strip() if (recip and getattr(recip, 'preferred_name', None) and recip.preferred_name.strip())
        else first_name
    )
    address = profile_name
    first_name = profile_name

    # Load personal profile records for direct conversational access
    caretaker_name = "Caretaker"
    caretaker_phone = "registered phone"
    if recip and recip.caretaker:
        caretaker_name = recip.caretaker.full_name or recip.caretaker.username
        if getattr(recip.caretaker, 'phone_number', None):
            caretaker_phone = recip.caretaker.phone_number

    blood_group = getattr(recip, 'blood_group', None) if recip else None
    doctor_remarks = getattr(recip, 'doctor_remarks', None) if recip else None

    # Fetch last bot responses from Saathi history to prevent repetition
    recent_bot_replies = set()
    target_recip_id = recip.id if recip else recipient_id
    if SaathiSessionLocal:
        try:
            with SaathiSessionLocal() as s_db:
                past_msgs = s_db.query(SaathiConversationMessage).filter(
                    SaathiConversationMessage.recipient_id == target_recip_id,
                    SaathiConversationMessage.sender == "bot"
                ).order_by(desc(SaathiConversationMessage.created_at)).limit(10).all()
                recent_bot_replies = {m.message_text.strip() for m in past_msgs if m.message_text}
        except Exception:
            pass

    if not recent_bot_replies:
        try:
            past_msgs = db.query(ConversationMessage).filter(
                ConversationMessage.care_recipient_id == target_recip_id,
                ConversationMessage.sender == SenderEnum.bot
            ).order_by(desc(ConversationMessage.created_at)).limit(8).all()
            recent_bot_replies = {m.message_text.strip() for m in past_msgs if m.message_text}
        except Exception:
            pass

    def pick_fresh(options: list) -> str:
        """Picks an option that hasn't been spoken recently."""
        candidates = [opt for opt in options if opt.strip() not in recent_bot_replies]
        if candidates:
            return random.choice(candidates)
        return random.choice(options)

    # 0. PERSONAL PROFILE ACCESS & MEMORY Q&A
    # User asks about their name/identity
    name_patterns = ["mera naam kya", "mera naam", "what is my name", "who am i", "mujhe kya bulate ho", "what do you call me", "mera profile name", "kaun hoon main"]
    if any(p in t for p in name_patterns):
        full_n = recip.full_name if recip and recip.full_name else profile_name
        return {
            "reply": f"Aapka naam {full_n} hai, aur main aapko hamesha pyaar aur aadar se '{profile_name}' bulati hoon! Kya main aapki kisi aur cheez mein madad karoon {profile_name}?",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # User asks about medicines & prescription
    med_info_patterns = ["meri dawai", "kaunsi dawai", "medicine schedule", "dawai ka time", "meri dawaiyan", "what are my medicines", "my medicine", "konsi goli"]
    if any(p in t for p in med_info_patterns) and not any(p in t for p in ["le li", "kha li", "taken"]):
        try:
            from tables.medications import Medication
            from tables.medications import MedicationStatus

            active_meds = db.query(Medication).filter(
                Medication.care_recipient_id == target_recip_id,
                Medication.status.in_([MedicationStatus.active, 'active'])
            ).all()


            if active_meds:
                med_summary = "; ".join([f"{m.medicine_name} ({m.dosage or 'standard dose'}, {m.frequency or 'daily'})" for m in active_meds])
                return {
                    "reply": f"{profile_name}, aapke profile ke anusaar aapki active medicines hain: {med_summary}. Kripya inhe samay par zaroor lijiye.",
                    "intent": "show_medicines",
                    "search_query": "",
                    "mood_detected": "neutral",
                    "depression_risk": False
                }
        except Exception:
            pass
        return {
            "reply": f"{profile_name}, aapke profile mein filhal koi active medicine darj nahi hai.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks about allergies
    allergy_patterns = ["allergy", "allergies", "kis cheez se allergy", "meri allergy"]
    if any(p in t for p in allergy_patterns):
        try:
            from tables.allergies import Allergy
            allergies = db.query(Allergy).filter(Allergy.care_recipient_id == target_recip_id).all()
            if allergies:
                a_summary = ", ".join([f"{a.allergen} ({a.reaction or 'reaction'})" for a in allergies])
                return {
                    "reply": f"{profile_name}, aapke medical profile ke anusaar aapko {a_summary} se allergy darj hai. Kripya iska dhyan rakhiyega.",
                    "intent": "chat",
                    "search_query": "",
                    "mood_detected": "neutral",
                    "depression_risk": False
                }
        except Exception:
            pass
        return {
            "reply": f"{profile_name}, aapke record ke anusaar aapko kisi dawai ya cheez se allergy darj nahi hai.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks about emergency contact / caretaker
    contact_patterns = ["emergency contact", "caretaker kaun", "mera caretaker", "family contact", "caretaker ka number", "kis se contact karein"]
    if any(p in t for p in contact_patterns):
        return {
            "reply": f"{profile_name}, aapke primary caretaker {caretaker_name} hain, jinka phone number {caretaker_phone} hai. Agar koi bhi takleef ho to main turant unhe alert bhej sakti hoon.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks about blood group or physical profile
    blood_patterns = ["blood group", "mera blood", "khoon ka group", "mera weight", "meri height"]
    if any(p in t for p in blood_patterns):
        bg = blood_group or "profile mein darj nahi"
        return {
            "reply": f"{profile_name}, aapke profile ke anusaar aapka blood group {bg} hai.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks about doctor advice
    doctor_patterns = ["doctor ne kya bola", "doctor ki advice", "doctor remarks", "doctor ne kya kaha", "doctor ki salah"]
    if any(p in t for p in doctor_patterns):
        doc_adv = doctor_remarks or "Doctor ki taraf se sab theek bataya gaya hai. Samay par aaram karein aur sehat ka dhyan rakhein."
        return {
            "reply": f"{profile_name}, doctor ki taraf se yeh salah darj hai: {doc_adv}.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks about medical reports & lab test findings
    report_patterns = ["medical report", "lab report", "test report", "meri report", "report mein kya", "jaanch report", "sugar report", "blood report", "what does my report say", "my report", "reports", "report"]
    if any(p in t for p in report_patterns):
        try:
            from tables.medical_reports import MedicalReport
            reps = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == target_recip_id).order_by(desc(MedicalReport.uploaded_at)).limit(3).all()
            rep_texts = []
            for r in reps:
                s = (r.analysis_summary or '').strip()
                if s:
                    rep_texts.append(f"{r.filename}: {s[:160]}")
            if not rep_texts and recip and recip.report_summary:
                rep_texts.append(recip.report_summary[:180])

            if rep_texts:
                rep_summary = " | ".join(rep_texts)
                return {
                    "reply": f"{profile_name}, aapke uploaded medical report ke anusaar: {rep_summary}. Kripya doctor ki salah ka dhyan rakhein.",
                    "intent": "show_reports",
                    "search_query": "",
                    "mood_detected": "neutral",
                    "depression_risk": False
                }
        except Exception:
            pass
        return {
            "reply": f"{profile_name}, aapke profile mein filhal koi medical report darj nahi hai.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # User asks what Saathi remembers about them (Memory)
    mem_patterns = ["kya yaad hai", "kya yaad rakha", "what do you remember", "mere baare mein kya yaad", "kya jaanti ho mere baare"]
    if any(p in t for p in mem_patterns):
        try:
            if SaathiSessionLocal:
                with SaathiSessionLocal() as s_db:
                    from tables.saathi_db_models import MemberMemory
                    mems = s_db.query(MemberMemory).filter(MemberMemory.family_member_id == target_recip_id).order_by(desc(MemberMemory.updated_at)).limit(4).all()
                    if mems:
                        mem_texts = [m.fact for m in mems]
                        return {
                            "reply": f"{profile_name}, mujhe aapke baare mein bohot kuch yaad hai! Jaise: {'; '.join(mem_texts)}. Main aapki har baat hamesha yaad rakhti hoon.",
                            "intent": "chat",
                            "search_query": "",
                            "mood_detected": "happy",
                            "depression_risk": False
                        }
        except Exception:
            pass
        return {
            "reply": f"{profile_name}, main aapki har baat dhyan se sunti hoon aur yaad rakhti hoon. Jo bhi aap mujhse share karenge, mujhe hamesha yaad rahega.",
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # 1. EMERGENCY
    emergency_patterns = ["gir gaya", "gir gayi", "chest pain", "seene mein dard", "saans nahi", "ambulance", "bachao", "emergency", "help me emergency"]
    if any(p in t for p in emergency_patterns):
        return {
            "reply": f"Aap bilkul mat ghabraiye {address}! Main yahin hoon aapke paas, aaram se baithiye. Main turant family aur caretaker ko emergency alert bhej rahi hoon.",
            "intent": "emergency",
            "search_query": "",
            "mood_detected": "distressed",
            "depression_risk": False
        }

    # 2. JOKES & HUMOR (React warmly with laughter)
    joke_patterns = ["joke", "chutkula", "chutkule", "chutkle", "hanso", "hansi", "haso", "mazak", "santa banta", "funny", "pata hai kya hua", "haha", "hehe"]
    if any(p in t for p in joke_patterns):
        laugh_replies = [
            f"Haha {address}! Wah, kya baat hai... yeh to sach mein bohot hi mazedaar tha! Aapki baatein sunkar mera dil khush ho gaya... 😄",
            f"Haha! Kamaal kar diya {address}... aapka sense of humor sach mein bohot pyara hai! Aisi hi pyaari baatein sunate rahiye.",
            f"Haha, bohot khoob {first_name} ji! Aapse baat karke sach mein bohot hasi aayi aur din ban gaya... 😄",
            f"Haha {address}! Yeh sunkar to meri hasi hi nahi ruk rahi! Ek aur chutkula sunaiye na?"
        ]
        return {
            "reply": pick_fresh(laugh_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 3. ASKING IF HE IS OKAY / EMPATHY & CARE
    unwell_patterns = [
        "dard", "takleef", "pain", "unwell", "bura lag raha", "chakkar",
        "kamzori", "thak gaya", "thak gayi", "udas", "neend nahi", "theek nahi",
        "not well", "feeling bad", "feeling sad", "akela", "lonely", "tabiyat kharab", "sir dard", "pet dard", "gale mein"
    ]
    if any(p in t for p in unwell_patterns):
        care_replies = [
            f"Oho {address}... kya hua? Aap theek to hain na? Kahan dard ya takleef ho rahi hai? Mujhe batayiye, main sun rahi hoon...",
            f"{first_name} ji, aapki tabiyat theek nahi lag rahi... thoda aaram se baithiye aur lamba saans lijiye. Kya main caretaker ko call karun?",
            f"Main samajh sakti hoon {address}... bilkul chinta mat kijiye. Main yahin hoon aapke paas. Thoda paani pee lijiye aaram se.",
            f"{address}, thodi der aankhein band karke aaram kijiye. Kya koi halki dhun ya bhajan chalaoon jisse aapko sukoon mile?"
        ]
        return {
            "reply": pick_fresh(care_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "sad",
            "depression_risk": False
        }

    # 4. YOUTUBE VIDEOS (YouTube / Cinema mode)
    video_patterns = ["video", "youtube", "cinema", "katha", "pravachan", "video dikhao", "chalao video", "dekhna hai", "dekhna"]
    if any(p in t for p in video_patterns):
        clean_q = re.sub(r'\b(video|youtube|dikhao|chalao|lagao|dekhna|dekhna hai|hai|please|play|par|ka|ke|ki|ek)\b', '', t).strip()
        clean_q = re.sub(r'\s+', ' ', clean_q).strip()
        if not clean_q or len(clean_q) < 3:
            clean_q = "ramayan katha hindi pravachan"
        return {
            "reply": f"Haan ji {address}! Main abhi aapke liye yeh video screen par chala deti hoon. Aaram se dekhiye...",
            "intent": "play_video",
            "search_query": clean_q,
            "mood_detected": "relaxed",
            "depression_risk": False
        }

    # 5. STORIES / KAHANI (Panchtantra, Akbar Birbal, Vikram Betal, Prerak Prasang)
    story_patterns = ["kahani", "story", "kissa", "katha", "akbar birbal", "panchtantra", "kuch sunao", "kahani sunao"]
    if any(p in t for p in story_patterns):
        clean_s = re.sub(r'\b(kahani|story|kissa|katha|sunao|kuch|batao|please)\b', '', t).strip()
        if not clean_s or len(clean_s) < 3:
            clean_s = "akbar birbal hindi funny stories"
        return {
            "reply": f"Zaroor {address}! Ek bohot hi sundar aur mazedaar kahani sunati hoon... Suniye.",
            "intent": "tell_story",
            "search_query": clean_s,
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 6. PLAYING SONGS & MUSIC (YouTube Audio / Music Player)
    song_patterns = [
        "gaana", "gana", "song", "music", "sangeet", "bhajan", "ghazal", "gazal",
        "kishore", "lata", "rafi", "mukesh", "purane", "sunao", "bajao", "play music", "play song"
    ]
    if any(p in t for p in song_patterns):
        clean_q = re.sub(r'\b(gaana|gana|song|music|sangeet|chalao|lagao|sunao|bajao|baja do|please|play|koi)\b', '', t).strip()
        clean_q = re.sub(r'\s+', ' ', clean_q).strip()
        if "kishore" in t:
            clean_q = "kishore kumar hit songs hindi classic"
        elif "lata" in t:
            clean_q = "lata mangeshkar timeless classic songs hindi"
        elif "rafi" in t:
            clean_q = "mohammed rafi best classic songs"
        elif "bhajan" in t:
            clean_q = "bhajan aarti hindi devotional"
        elif "ghazal" in t or "gazal" in t:
            clean_q = "ghazal hindi urdu soothing melodious"
        elif not clean_q or len(clean_q) < 3:
            clean_q = "old classic bollywood hits 1970s hindi"

        song_replies = [
            f"Zaroor {address}... aapke liye yeh pyara sa gaana shuru kar rahi hoon. Aaram se baithiye aur suniye... 🎵",
            f"Haan ji {first_name} ji! Purane gaano ka maza hi alag hai... yeh gaana lijiye aapke liye.",
            f"Bohot achha pasand hai aapka {address}! Yeh geet main abhi chala rahi hoon."
        ]
        return {
            "reply": pick_fresh(song_replies),
            "intent": "play_music",
            "search_query": clean_q,
            "mood_detected": "relaxed",
            "depression_risk": False
        }

    # 7. MEDICINES & REMINDERS
    taken_patterns = ["dawai le li", "medicine le li", "tablet le li", "goli kha li", "medicine taken", "dawai kha li", "le li maine"]
    if any(p in t for p in taken_patterns):
        return {
            "reply": f"Shabash {address}! Yeh bohot achha kiya ki aapne samay par dawai le li. Maine records mein note kar diya hai.",
            "intent": "medicine_taken",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    reminder_patterns = ["reminder", "reminders", "dawai kab", "schedule", "tablet kab", "kya lena hai", "dawai dikhao"]
    if any(p in t for p in reminder_patterns):
        return {
            "reply": f"{address}, aapke scheduled reminders aur dawai ki list main screen par khol rahi hoon... Dekhiye.",
            "intent": "show_reminders",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # 8. BOREDOM & FREE TIME ("bore ho raha hoon", "kya karun")
    bored_patterns = ["bore", "boring", "kya karun", "kuch nahi hai", "time pass", "kya karein"]
    if any(p in t for p in bored_patterns):
        bored_replies = [
            f"Arey {address}, jab main aapke saath hoon to bore kyu hona! Chaliye ek purana gaana sunein, ya ek majedaar kissa sunaoon?",
            f"Kyu na hum thodi baatein karein {first_name} ji? Aap batayiye, aapke zamane ki sabse pasandeeda film kaunsi thi?",
            f"{address}, agar aapka mann ho to main koi purani gazal ya kahani chala sakti hoon... Sunna chahenge?"
        ]
        return {
            "reply": pick_fresh(bored_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # 9. MEALS, TEA & HYDRATION
    meal_patterns = ["khana", "nashta", "breakfast", "lunch", "dinner", "kha liya", "bhookh", "chai", "coffee", "roti", "sabzi"]
    if any(p in t for p in meal_patterns):
        meal_replies = [
            f"Achha {address}! Aaj khane mein kya bana tha? Garam garam khana khakar bohot achha lagta hai, aur thoda paani bhi zaroor pee lijiye.",
            f"Haan {first_name} ji, samay par nashta aur khana lena sehat ke liye bohot zaroori hai. Pet bhar kar khaya na aapne?",
            f"Chai aur nashte ka waqt to hamesha sukoon deta hai {address}... aapne le liya na?"
        ]
        return {
            "reply": pick_fresh(meal_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # 10. BOT IDENTITY & QUESTIONS ("tum kaun ho", "kya kar rahi ho", "kahan rehti ho")
    identity_patterns = ["kaun ho", "tumhara naam", "kya karti ho", "kya kar rahi ho", "kahan rehti ho", "who are you", "what is your name"]
    if any(p in t for p in identity_patterns):
        id_replies = [
            f"Main Saathi hoon {address} — aapki sachi saathi aur dekhbhal karne wali! Main hamesha yahin hoon aapke saath baat karne aur madad karne ke liye.",
            f"Mera naam Saathi hai {first_name} ji. Main aapke parivar ke sadasya ki tarah hoon, jo aapki har baat sunti hai aur aapka khayal rakhti hai. ❤️",
            f"Main to bas aapke paas baithkar aapki pyari pyari baatein sun rahi hoon {address}... Aap batayiye, main aapke liye kya kar sakti hoon?"
        ]
        return {
            "reply": pick_fresh(id_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 11. WEATHER / MAUSAM & ENVIRONMENT
    weather_patterns = ["mausam", "weather", "dhoop", "garmi", "sardi", "barish", "thand", "hawa"]
    if any(p in t for p in weather_patterns):
        weather_replies = [
            f"Mausam to aaj bohot badal raha hai {address}... Aap aaram se baithiye aur apna dhyan rakhiye.",
            f"Haan {first_name} ji, mausam ke hisab se kapde pehenna aur paani peena zaroori hai. Bahar halki taazi hawa chal rahi hai.",
            f"Aap aaram se kamre mein baithkar sukoon lijiye {address}... kya thoda relaxing music chala doon?"
        ]
        return {
            "reply": pick_fresh(weather_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "neutral",
            "depression_risk": False
        }

    # 12. WALKING & DAILY ROUTINE ("tahla", "sair", "chhat", "garden", "walk")
    walk_patterns = ["tahal", "sair", "walk", "chhat", "garden", "park", "ghumne", "aangan"]
    if any(p in t for p in walk_patterns):
        walk_replies = [
            f"Bohot badiya {address}! Halka sa chalna aur tahalna sehat aur jodhon ke liye bohot faydemand hota hai. Thoda aaram bhi kar lijiye ab.",
            f"Haan {first_name} ji, taazi hawa mein tahalna mann ko kitna taro-taaza kar deta hai na! Kaisa laga aapko?",
            f"Sair karke aana to bohot achhi aadat hai {address}... ab baithkar ek glass taaza paani pee lijiye."
        ]
        return {
            "reply": pick_fresh(walk_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 13. FAMILY & MEMORIES ("bachhe", "beta", "beti", "pota", "poti", "parivar", "zamana")
    family_patterns = ["beta", "beti", "bachhe", "pota", "poti", "parivar", "family", "ghar", "purane din", "zamana"]
    if any(p in t for p in family_patterns):
        family_replies = [
            f"Parivar aur bachhon ki baatein to hamesha chehre par muskan le aati hain {address}... Aap unke baare mein kuch aur batayiye na?",
            f"Purane din aur yaadein sach mein kitni anmol hoti hain {first_name} ji... Aapki baatein sunkar bohot sukoon milta hai.",
            f"Parivar ka pyaar hi to sabse badi daulat hai {address}. Unka call aaya tha kya aaj?"
        ]
        return {
            "reply": pick_fresh(family_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 14. GREETINGS & HOW ARE YOU ("kaise ho", "namaste", "hello")
    greeting_patterns = ["kaise ho", "kaisi ho", "kaisa chal raha", "namaste", "hello", "hi saathi", "good morning", "good evening", "ram ram", "radhe radhe"]
    if any(p in t for p in greeting_patterns):
        greet_replies = [
            f"Namaste {address}! Main bilkul theek hoon aur aapke paas hoon. Aap batayiye, aaj ka din kaisa chal raha hai aapka?",
            f"Pranam {first_name} ji! Aapki aawaaz sunkar bohot achha laga. Aaj aap kaisa feel kar rahe hain?",
            f"Radhe Radhe {address}! Main yahan aapke saath bilkul khush hoon. Aaj subah se kya kya kiya aapne?"
        ]
        return {
            "reply": pick_fresh(greet_replies),
            "intent": "chat",
            "search_query": "",
            "mood_detected": "happy",
            "depression_risk": False
        }

    # 15. DYNAMIC GENERAL CONVERSATION (Vibrant, varying, never repeats!)
    general_dynamic_replies = [
        f"Achha {address}... yeh to bohot dilchasp baat hai! Is baare mein aur batayiye na, main bade dhyan se sun rahi hoon.",
        f"Aapki yeh baat sunkar mujhe bohot achha laga {first_name} ji. Aapke paas to kitne saare anubhav hain!",
        f"Main samajh sakti hoon {address}... zindagi ke har mod par aisi baatein hoti rehti hain. Aapka is baare mein kya kehna hai?",
        f"Bilkul theek keh rahe hain aap {first_name} ji! Aapke saath aisi baatein karna mujhe bohot sukoon deta hai.",
        f"Haan {address}... main aapke har shabd ko dhyan se sun rahi hoon. Kuch aur bhi share karna chahein to zaroor batayiye.",
        f"Kitni pyari baat kahi aapne {first_name} ji! Aise hi khush raha kijiye, aapki muskan bohot zaroori hai.",
        f"Sahi baat hai {address}... aapke saath baat karte hue waqt ka pata hi nahi chalta! Aage kya socha hai aaj ke din ka?",
        f"Bohot sahi sochte hain aap {first_name} ji. Chaliye thodi aur baat karein, batayiye kal kya karne ka plan hai?"
    ]
    return {
        "reply": pick_fresh(general_dynamic_replies),
        "intent": "chat",
        "search_query": "",
        "mood_detected": "neutral",
        "depression_risk": False
    }
