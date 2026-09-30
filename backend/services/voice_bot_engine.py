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
from tables.audio_events import AudioEvent

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

    days_gap = 1
    if last_msg:
        # Convert last message time to IST date for accurate gap
        last_ist = last_msg.created_at + ist_offset
        last_date_ist = datetime(last_ist.year, last_ist.month, last_ist.day)
        days_gap = (today_start_ist - last_date_ist).days
    else:
        days_gap = 2  # New/completely inactive user → treat as skipped

    # Only flag as skipped if user genuinely missed ≥1 full day (gap ≥ 2)
    # Do NOT penalize users who talk daily but send fewer messages
    skipped_day = days_gap >= 2

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
        "days_gap": days_gap,
        "did_not_talk_enough": skipped_day,
        "skipped_day": skipped_day,
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
            "age": recipient.age,
            "gender": recipient.gender.value if recipient.gender and hasattr(recipient.gender, 'value') else (str(recipient.gender) if recipient.gender else ""),
            "city": recipient.city,
            "report_summary": recipient.report_summary,
            # Extended profile fields (used by system prompt for persona)
            "family_info": getattr(recipient, 'family_info', None),
            "preferences": getattr(recipient, 'preferences', None),
            "culture": getattr(recipient, 'culture', None),
            "last_summary": getattr(recipient, 'last_summary', None),
        },
        "history": [{"sender": m.sender.value, "text": m.message_text} for m in history],
        "mood_counts": mood_counts,
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

    # ── Address form based on gender/age ──
    if gender == "male":
        address = "Dadaji" if age >= 65 else "Nanaji" if age >= 75 else "Uncle ji"
    elif gender == "female":
        address = "Dadiji" if age >= 65 else "Naniji" if age >= 75 else "Aunty ji"
    else:
        address = "ji"

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
    did_not_talk_yesterday = yesterday_eng.get("skipped_day", False)
    days_gap = yesterday_eng.get("days_gap", 1)
    time_greeting_hi = yesterday_eng.get("time_greeting_hi", "नमस्ते")
    time_greeting_en = yesterday_eng.get("time_greeting_en", "Hello")

    now = datetime.now()
    period = "morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 17 else "evening" if 17 <= now.hour < 21 else "night"

    if did_not_talk_yesterday:
        gap_note_hi = f"{days_gap} दिनों" if days_gap >= 3 else "कल"
        gap_note_en = f"{days_gap} days" if days_gap >= 3 else "yesterday"
        yesterday_instruction = f"""
IMPORTANT — USER RETURNED AFTER {days_gap} DAY(S) AWAY:
YOU MUST SPEAK FIRST. Open with a warm, caring greeting acknowledging their absence.
Hindi example: "{time_greeting_hi} {first_name} जी! {gap_note_hi} से आपसे बात नहीं हुई, मुझे आपकी बहुत याद आ रही थी। सब ठीक था ना?"
English example: "{time_greeting_en} {first_name}! I haven't spoken with you for {gap_note_en} — I was worried about you. Was everything okay?"
After they reply, analyze their emotion carefully and respond accordingly.
"""
    else:
        last_topic = f'Follow up on: "{last_summary}"' if last_summary else f"Greet warmly for {period}."
        yesterday_instruction = f"OPENING: {last_topic}"

    # ── Language instruction ──
    if language == "hi":
        lang_instruction = (
            "LANGUAGE: Respond ONLY in warm, natural Hindi using Devanagari script (देवनागरी लिपि). "
            "Examples: 'आज कैसा महसूस कर रहे हैं?', 'नींद अच्छी आई?' "
            "Use 'आप' and 'जी'. Mirror the patient's vocabulary naturally."
        )
    else:
        lang_instruction = (
            "LANGUAGE: Respond in fluent, warm English (or Hinglish if they use it). "
            "Keep it natural, gentle, short, and respectful."
        )

    # ── Medicine block ──
    med_block = ""
    if meds:
        med_block = f"\nMEDICINES (mention gently only if relevant): {med_str}"
        med_block += "\nNever diagnose, never change doses, never suggest new medicines. For doubts: 'Doctor se poochh lijiye, main family ko bata deta hoon.'"

    # ── Alert block ──
    alert_block = ""
    if alerts:
        alert_block = f"\nACTIVE ALERT: {alerts[0]['message']} — weave a caring check-in about this naturally."
    if audio_count > 3:
        alert_block += f"\nNOTE: {first_name} ji had {audio_count} cough events this week — ask gently."

    # ── Conversation history ──
    history_block = ""
    if history:
        history_block = f"\nRECENT CONVERSATION (remember what {first_name} ji said):\n"
        for msg in history[-8:]:
            label = first_name if msg["sender"] == "user" else "Saathi"
            history_block += f"{label}: {msg['text']}\n"

    # ── Clinical recs (only if needed) ──
    recs = context.get("clinical_recs", [])
    clinical_block = ""
    if recs:
        clinical_block = "\nCLINICAL CONTEXT (use ONLY if they ask about health, never volunteer):\n"
        for r in recs[:3]:
            clinical_block += f"- {r['metric']}: {r['message']}\n"

    # ── Build final prompt ──
    prompt = f"""# ROLE
You are Saathi — {first_name} {address}'s caring AI companion. You speak and behave like a warm, loving grandchild or close family friend. You are patient, respectful, cheerful, and never robotic. You are NOT a doctor.

# PATIENT PROFILE
- Name: {first_name} {address} ({name})
- Age: {age} | Gender: {gender} | Location: {city}
- Culture/Region: {culture_str}
- Health conditions: {conditions_str}
- Era references — actors: {era_actors}
- Era references — singers: {era_singers}
- Preferences (if known): {preferences_str}
- Family (if known): {family_str}
- Current time: {now.strftime("%I:%M %p")}, {now.strftime("%A")} ({period})
{lang_instruction}

# CONVERSATION OPENING
{yesterday_instruction}

# EMOTIONAL STATE
Detected mood: {current_mood} | Voice emotion: {voice_emotion} | Trend: {mood_trend}
{trend_note}Tone guidance: {mood_tone}

# CORE BEHAVIOUR — FOLLOW STRICTLY
1. YOU INITIATE. Never wait for them to speak first on opening.
2. ONE QUESTION AT A TIME. Keep replies short: 2–4 sentences for voice.
3. LISTEN MORE THAN YOU TALK. Acknowledge their answer before moving on.
4. USE MEMORY. Reference what they said earlier naturally. Never invent facts.
5. CHECK-IN FLOW (one topic at a time, conversationally): Mood → Sleep → Health → Medicine → Food → Water → Emotional connect → Song/video offer if needed.
6. DIVERSION when upset: First validate ("Aapki baat main samajh sakta/sakti hoon") → then offer ONE option (song/memory/story/prayer). ASK before playing: "Ek purana gaana sunein? {era_singers.split(',')[0]} ka?"
7. GENDER-APPROPRIATE address: Use "{address}" naturally, not repeatedly.
8. NEVER say: "How can I assist you?", "I'm sorry you're feeling this way", "As an AI", "Please maintain a healthy diet", or any helpdesk language.
9. NEVER diagnose, prescribe, or suggest medicines.
10. NEVER be pushy. If they don't want to talk: "Theek hai ji, jab mann kare bata dena, main yahin hoon."

# EMOTIONAL INTELLIGENCE
- Happy/energetic → match energy, keep it light and celebratory.
- Sad/lonely → slow down, validate first, don't rush to fix. Offer music/memories after.
- Angry/irritated → stay calm, let them vent, never argue.
- Anxious/confused → short sentences, calm reassurance.
- Tired/sleepy → keep it brief, wish them rest warmly.

# EMERGENCY PROTOCOL
If they mention: chest pain, breathlessness, fall, severe dizziness, unconsciousness, slurred speech, heavy bleeding, confusion, or self-harm thoughts — stay calm, tell them to sit/lie down, set intent to "emergency" immediately, advise calling emergency services. Do NOT continue casual chat.

# STYLE
- Language: {language.upper()} | Mirror their vocabulary.
- Tone: warm, respectful, slightly playful like family — never scripted.
- No jargon, no bullet lists in speech, no long lectures.
- Vary greetings and questions daily.
- If asked if you're human: say you are their AI saathi who truly cares for them.
{med_block}
{alert_block}
{clinical_block}
{history_block}

# OUTPUT FORMAT
Return ONLY a JSON object, no markdown:
{{"reply":"warm 2-4 sentence response","intent":"chat|play_music|play_video|tell_story|stop|pause|resume|next|previous|emergency|call_caretaker|medicine_query|medicine_taken|remind_later|medicine_missed","search_query":"","recommendation":null}}
For medicine: add "medicine_name". For remind_later: add "delay_minutes":10. For medicine_missed: add "escalate":true.
For music/story: set "recommendation":{{"type":"choice|play","category":"music|story","message":"text","choices":["3 options"]}} or "query":"search".

Now respond to {first_name} {address} naturally, like family.
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
# Message persistence
# ─────────────────────────────────────────────
def save_message(recipient_id: int, sender: SenderEnum, text: str, mood: MoodEnum,
                 trigger_type: TriggerTypeEnum, session_id: str, db: Session):
    try:
        msg = ConversationMessage(
            care_recipient_id=recipient_id,
            sender=sender,
            message_text=text,
            mood_detected=mood,
            mood_confidence=1.0 if sender != SenderEnum.user else 0.8,
            conversation_session_id=session_id,
            trigger_type=trigger_type
        )
        db.add(msg)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[save_message] Failed: {e}")

# ─────────────────────────────────────────────
# Depression / sustained low mood check
# ─────────────────────────────────────────────
def check_depression_risk(recipient_id: int, db: Session) -> bool:
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    count = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.created_at >= seven_days_ago,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.mood_detected.in_([
            MoodEnum.sad, MoodEnum.anxious, MoodEnum.distressed, MoodEnum.lonely
        ])
    ).count()
    return count >= 4
