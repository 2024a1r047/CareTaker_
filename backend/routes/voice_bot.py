from fastapi import APIRouter, Depends, Header, HTTPException, status
import json
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from pydantic import BaseModel
import os
import requests
from config import get_db
from utils.gemini_client import call_gemini, safe_json_parse
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo
from dotenv import load_dotenv
from datetime import datetime, timedelta

from tables.conversation_history import (
    ProactiveReminder, ConversationMessage, MoodEnum, SenderEnum,
    TriggerTypeEnum, ReminderTypeEnum, RecurrenceEnum, VoiceBotPreferences
)
from services.voice_bot_engine import (
    build_conversation_context, generate_system_prompt, analyze_mood,
    save_message, check_depression_risk, get_content_recommendation, get_story_queries,
    intelligent_saathi_dialogue
)
from services.sentiment_engine import analyze_sentiment_with_history
from services.proactive_triggers import get_pending_triggers, create_default_reminders
from utils.tts_handler import generate_speech_base64, generate_speech_base64_async, is_hindi

load_dotenv()

router = APIRouter(tags=["VoiceBot"])


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        parts = auth_header.split()
        if len(parts) != 2:
            return None
        token = parts[1]
        from repository.users import JWTRepo
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None


def verify_and_get_caretaker(authorization: str, db: Session):
    if authorization:
        username = _get_username_from_auth(authorization)
        if username:
            caretaker = UsersRepo.find_by_username(db, CareTaker, username)
            if caretaker:
                return caretaker
    # Fallback to first caretaker in DB for dev testing
    caretaker = db.query(CareTaker).first()
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return caretaker


def _get_or_create_preferences(recipient_id: int, db: Session) -> VoiceBotPreferences:
    recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recip:
        recip = db.query(CareRecipient).first()
    if recip:
        recipient_id = recip.id

    prefs = db.query(VoiceBotPreferences).filter(VoiceBotPreferences.care_recipient_id == recipient_id).first()
    if not prefs and recip:
        prefs = VoiceBotPreferences(
            care_recipient_id=recipient_id,
            favorite_songs=[],
            favorite_stories=[],
            mood_content_preferences={},
            preferred_language="hi"
        )
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return prefs


# ─────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    recipient_id: Optional[int] = 1
    text: str
    session_id: str = "default_session"
    trigger_type: str = "user_initiated"
    language: str = "en"
    use_tts: bool = True


@router.post('/voice-bot/chat', response_model=ResponseSchema)
async def voice_bot_chat(payload: ChatRequest, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)

    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == payload.recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")
    payload.recipient_id = recipient.id

    user_text = payload.text
    if not user_text:
        return ResponseSchema(code=400, status="error", message="No text provided")

    trigger_enum = (
        TriggerTypeEnum(payload.trigger_type)
        if payload.trigger_type in [t.value for t in TriggerTypeEnum]
        else TriggerTypeEnum.user_initiated
    )

    # Run sentiment analysis across history + current message
    sentiment = analyze_sentiment_with_history(user_text, payload.recipient_id, db)
    
    # Extract mood from sentiment analysis
    mood_str = sentiment.get("current_mood", "neutral")
    mood_enum = MoodEnum(mood_str) if mood_str in [m.value for m in MoodEnum] else MoodEnum.neutral
    
    save_message(payload.recipient_id, SenderEnum.user, user_text, mood_enum, trigger_enum, payload.session_id, db)

    context = build_conversation_context(payload.recipient_id, db)
    system_prompt = generate_system_prompt(recipient.full_name, context, payload.language, sentiment)
    is_at_risk = check_depression_risk(payload.recipient_id, db) or sentiment.get("urgency") == "high"

    if not os.environ.get('GEMINI_API_KEY'):
        first_name = recipient.full_name.split()[0] if recipient.full_name else "ji"
        yesterday_eng = context.get("yesterday_engagement", {})
        time_hi = yesterday_eng.get("time_greeting_hi", "नमस्ते")
        time_en = yesterday_eng.get("time_greeting_en", "Hello")
        days_gap = yesterday_eng.get("days_gap", 1)
        if payload.trigger_type == "proactive_checkin" and yesterday_eng.get("skipped_day"):
            if payload.language == "hi":
                reply_text = f"{time_hi} {first_name}! कल आप कहां थे? सब ठीक था ना, किसी की याद आ रही थी क्या?"
            else:
                reply_text = f"{time_en} {first_name}! Where were you yesterday? Was everything okay? Were you missing someone?"
        elif payload.trigger_type == "proactive_checkin":
            if payload.language == "hi":
                reply_text = f"{time_hi} {first_name} जी! आज आप कैसा महसूस कर रहे हैं?"
            else:
                reply_text = f"{time_en} {first_name}! How are you feeling today?"
        else:
            if payload.language == "hi":
                reply_text = "मुझे अभी समझ नहीं आ रहा है, कृपया थोड़ी देर बाद कोशिश करें।"
            else:
                reply_text = "I'm having trouble connecting to my brain right now, please try again later."

        save_message(payload.recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral, trigger_enum, payload.session_id, db, intent="greeting" if payload.trigger_type == "proactive_checkin" else "chat")

        audio_base64 = ""
        if payload.use_tts:
            lang_code = "hi-IN" if is_hindi(reply_text) else payload.language
            audio_base64 = await generate_speech_base64_async(reply_text, lang_code)

        return ResponseSchema(
            code=200, status="success", message="Fallback AI response generated",
            result={
                "reply": reply_text,
                "intent": "chat",
                "search_query": "",
                "action_param": None,
                "audio_base64": audio_base64,
                "mood_detected": mood_str,
                "depression_risk": is_at_risk,
                "recommendation": None,
                "sentiment": {
                    "trend": sentiment.get("trend", "stable"),
                    "summary": sentiment.get("summary", "")
                }
            }
        )

    gemini_payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser says: {user_text}"}]}],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.8
        }
    }

    try:
        data = call_gemini(gemini_payload, timeout=20, caller="[voice_bot/chat]")
        if data and data.get('candidates'):
                ai_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                import json
                try:
                    ai_json = json.loads(ai_text)
                    reply_text = ai_json.get("reply")
                    thought = ai_json.get("thought", "")
                    if thought:
                        print(f"[Saathi Thought] {thought}")
                        
                    if not reply_text:
                        reply_text = ai_json.get("message") or ai_text
                    
                    intent = ai_json.get("intent", "chat")
                    search_query = ai_json.get("search_query", "")
                    action_param = ai_json.get("action_param")
                    medicine_name_from_ai = ai_json.get("medicine_name", "")
                    escalate_flag = ai_json.get("escalate", False)

                    # ── DOSE LOG INTEGRATION (Voice Channel) ──────────────────────────
                    # If Saathi confirms or misses a dose, write to MedicationDoseLog
                    # via the same shared pipeline used by dashboard and email channels.
                    if intent in ("medicine_taken", "medicine_missed"):
                        _voice_action = "TAKEN" if intent == "medicine_taken" else "MISSED"
                        try:
                            from tables.medication_dose_logs import (
                                MedicationDoseLog, DoseStatusEnum, ConfirmationSourceEnum
                            )
                            from tables.medications import Medication, MedicationStatus

                            # Find the most recent PENDING dose log for this recipient.
                            # If medicine_name was extracted by AI, prefer matching that drug.
                            pending_q = db.query(MedicationDoseLog).filter(
                                MedicationDoseLog.care_recipient_id == payload.recipient_id,
                                MedicationDoseLog.status == DoseStatusEnum.PENDING,
                            )
                            if medicine_name_from_ai:
                                # Join medications table to filter by name (case-insensitive)
                                pending_q = pending_q.join(
                                    Medication,
                                    Medication.medication_id == MedicationDoseLog.medication_id
                                ).filter(
                                    Medication.medicine_name.ilike(f"%{medicine_name_from_ai}%")
                                )

                            # Pick the most overdue pending log (oldest scheduled_time first)
                            pending_log = pending_q.order_by(
                                MedicationDoseLog.scheduled_time.asc()
                            ).first()

                            if pending_log:
                                pending_log.status             = DoseStatusEnum(_voice_action)
                                pending_log.confirmation_source = ConfirmationSourceEnum.VOICE
                                pending_log.confirmed_at        = datetime.utcnow()

                                new_stock = None
                                if _voice_action == "TAKEN":
                                    med = db.query(Medication).filter(
                                        Medication.medication_id == pending_log.medication_id
                                    ).with_for_update().first()
                                    if med:
                                        old = med.current_stock or 0
                                        daily = med.doses_per_day or 1
                                        new_stock = max(0, old - daily)
                                        med.current_stock = new_stock

                                db.commit()
                                print(f"[voice_bot] Dose log {pending_log.id} marked "
                                      f"{_voice_action} via VOICE for recipient {payload.recipient_id}")

                                # Push dashboard sync
                                try:
                                    import asyncio
                                    from socket_manager import sio_server
                                    loop = asyncio.get_event_loop()
                                    _payload = {
                                        "dose_log_id": pending_log.id,
                                        "action":      _voice_action,
                                        "new_stock":   new_stock,
                                    }
                                    if loop.is_running():
                                        asyncio.run_coroutine_threadsafe(
                                            sio_server.emit("dose_confirmed", _payload), loop
                                        )
                                    else:
                                        loop.run_until_complete(
                                            sio_server.emit("dose_confirmed", _payload)
                                        )
                                except Exception as _se:
                                    print(f"[voice_bot] Socket.IO push failed: {_se}")
                            else:
                                print(f"[voice_bot] No PENDING dose log found for recipient "
                                      f"{payload.recipient_id} (medicine='{medicine_name_from_ai}')")

                        except Exception as _de:
                            print(f"[voice_bot] Dose log update failed: {_de}")
                            # Non-fatal — voice response still returns normally
                    # ── END DOSE LOG INTEGRATION ──────────────────────────────────────

                except Exception as ex:
                    print(f"Failed to parse Gemini JSON: {ex}")
                    reply_text = ai_text
                    intent = "chat"
                    search_query = ""
                    action_param = None
                    thought = ""

                save_message(payload.recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral, trigger_enum, payload.session_id, db, intent=intent)
 
                # Attach content recommendation (prioritize AI's merged recommendation field)
                recommendation = ai_json.get("recommendation")
                 
                # Fallback: if AI didn't provide one but mood is actionable, use default map
                if not recommendation and mood_str not in ("neutral",):
                    recommendation = get_content_recommendation(mood_str)
 
                # If we have a recommendation, ensure reply_text reflects its message if present
                if recommendation and recommendation.get("message"):
                    reply_text = recommendation.get("message", reply_text)

                # Secondary audio disabled: Only ElevenLabs Asha voice is permitted
                audio_base64 = ""

                return ResponseSchema(
                    code=200, status="success", message="AI response generated",
                    result={
                        "reply": reply_text,
                        "intent": intent,
                        "search_query": search_query,
                        "action_param": action_param,
                        "audio_base64": audio_base64,
                        "mood_detected": mood_str,
                        "depression_risk": is_at_risk,
                        "recommendation": recommendation,
                        "sentiment": {
                            "trend": sentiment.get("trend", "stable"),
                            "dominant_mood": sentiment.get("dominant_mood", mood_str),
                            "stability_score": sentiment.get("stability_score", 0.5),
                            "summary": sentiment.get("summary", ""),
                            "urgency": sentiment.get("urgency", "low"),
                            "mood_timeline": sentiment.get("mood_timeline", [])
                        }
                    }
                )
        # Intelligent companionship fallback — handles jokes, empathy, songs, YouTube videos, and reminders
        dialogue_res = intelligent_saathi_dialogue(user_text, payload.recipient_id, db, payload.language)
        fallback_reply = dialogue_res.get("reply")
        intent = dialogue_res.get("intent", "chat")
        search_query = dialogue_res.get("search_query", "")
        fallback_mood = dialogue_res.get("mood_detected", mood_str)

        lang_hi = "hi-IN" if is_hindi(fallback_reply) else payload.language
        fallback_audio = ""

        save_message(payload.recipient_id, SenderEnum.bot, fallback_reply, MoodEnum.neutral, trigger_enum, payload.session_id, db, intent=intent)

        return ResponseSchema(
            code=200, status="success", message="Intelligent caring response generated",
            result={
                "reply": fallback_reply,
                "intent": intent,
                "search_query": search_query,
                "action_param": None,
                "audio_base64": fallback_audio,
                "mood_detected": fallback_mood,
                "depression_risk": is_at_risk,
                "recommendation": None,
                "sentiment": {
                    "trend": sentiment.get("trend", "stable"),
                    "summary": sentiment.get("summary", "")
                }
            }
        )
    except Exception as e:
        print(f"[voice_bot/chat] Error: {e}")
        return ResponseSchema(code=500, status="error", message=f"Chat error: {str(e)}")


# ─────────────────────────────────────────────
# Detect mood from text + return content recommendation
# ─────────────────────────────────────────────
class MoodContentRequest(BaseModel):
    text: str                        # User's mood description in any language
    content_type: str = "music"      # "music" or "story"
    recipient_id: Optional[int] = 1

@router.post('/voice-bot/detect-and-recommend', response_model=ResponseSchema)
async def detect_and_recommend(
    payload: MoodContentRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Analyzes user's text to detect mood, then returns a warm
    Hinglish response + content recommendation (YouTube queries).
    Used when user says 'play music' or 'sunao kahani' without
    specifying mood — bot asks mood, user replies, this endpoint handles it.
    """
    verify_and_get_caretaker(authorization, db)

    # 1. Full sentiment analysis — history + current message
    sentiment = analyze_sentiment_with_history(payload.text, payload.recipient_id, db)
    mood_str = sentiment["current_mood"]
    confidence = sentiment["confidence"]
    dominant_mood = sentiment["dominant_mood"]
    trend = sentiment["trend"]

    # If dominant historical mood differs from current and trend is worsening,
    # weight the recommendation toward the dominant (persistent) mood
    effective_mood = mood_str
    if trend == "worsening" and dominant_mood != mood_str:
        effective_mood = dominant_mood  # use persistent mood for recommendation

    # 2. Get content recommendation for effective mood
    recommendation = get_content_recommendation(effective_mood)

    # 3. Generate a warm Hinglish confirmation message via Gemini
    warm_message = recommendation.get("message", "Chalo kuch acha sunate hain 😊")

    if os.environ.get('GEMINI_API_KEY') and confidence > 0.4:
        prompt = (
            f"An elderly user said: \"{payload.text}\"\n"
            f"Current mood: {mood_str} (confidence: {confidence:.0%})\n"
            f"Dominant mood recently: {dominant_mood}\n"
            f"Emotional trend: {trend} (improving/worsening/stable)\n"
            f"Emotional summary: {sentiment.get('summary', '')}\n"
            f"They want: {payload.content_type}\n\n"
            f"Write ONE short, warm, empathetic Hinglish sentence (max 15 words) acknowledging their emotional state "
            f"and telling them you're going to play {'music' if payload.content_type == 'music' else 'a story'} for them. "
            f"If trend is worsening, be extra nurturing. If improving, be celebratory. "
            f"Use emojis. Be like a caring friend. Example: 'Samajh gaya... chalo kuch soothing gaane sunte hain ❤️'"
        )
        try:
            data = call_gemini(
                {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 60, "temperature": 0.7}},
                timeout=8, caller="[detect-and-recommend]"
            )
            if data and data.get("candidates"):
                warm_message = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"[detect-and-recommend] Gemini warm message error: {e}")

    return ResponseSchema(
        code=200, status="success", message="Mood detected",
        result={
            "mood": mood_str,
            "effective_mood": effective_mood,
            "dominant_mood": dominant_mood,
            "trend": trend,
            "confidence": confidence,
            "warm_message": warm_message,
            "recommendation": recommendation,
            "content_type": payload.content_type,
            "sentiment_summary": sentiment.get("summary", ""),
            "urgency": sentiment.get("urgency", "low"),
            "mood_timeline": sentiment.get("mood_timeline", [])
        }
    )


# ─────────────────────────────────────────────
# Content recommendation by mood
# ─────────────────────────────────────────────
@router.get('/voice-bot/recommend/{recipient_id}', response_model=ResponseSchema)
async def get_recommendation(
    recipient_id: int, mood: str = "neutral",
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    recommendation = get_content_recommendation(mood)
    return ResponseSchema(code=200, status="success", message="Recommendation fetched", result=recommendation)


@router.get('/voice-bot/preferences/{recipient_id}', response_model=ResponseSchema)
async def get_preferences(
    recipient_id: int,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(recipient_id, db)
    return ResponseSchema(
        code=200, status="success", message="Preferences fetched",
        result={
            "preferred_language": prefs.preferred_language or "hi",
            "favorite_songs": prefs.favorite_songs or [],
            "favorite_stories": prefs.favorite_stories or [],
            "mood_content_preferences": prefs.mood_content_preferences or {}
        }
    )


# ─────────────────────────────────────────────
# Story category queries
# ─────────────────────────────────────────────
@router.get('/voice-bot/story-queries', response_model=ResponseSchema)
async def get_story_category_queries(
    category: str = "moral",
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    queries = get_story_queries(category)
    return ResponseSchema(code=200, status="success", message="Story queries fetched", result={"category": category, "queries": queries})


# ─────────────────────────────────────────────
# Daily greeting check
# ─────────────────────────────────────────────
@router.get('/voice-bot/daily-greeting/{recipient_id}', response_model=ResponseSchema)
async def check_daily_greeting(
    recipient_id: int,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    caretaker = verify_and_get_caretaker(authorization, db)
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")
    recipient_id = recipient.id

    prefs = _get_or_create_preferences(recipient_id, db)
    today_ist = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    already_greeted = prefs.last_greeted_date == today_ist

    if not already_greeted:
        prefs.last_greeted_date = today_ist
        db.commit()

    # Get yesterday engagement — includes IST-based time greetings
    from services.voice_bot_engine import check_yesterday_engagement
    yesterday_eng = check_yesterday_engagement(recipient_id, db)
    did_not_talk_yesterday = yesterday_eng.get("skipped_day", False)
    days_gap = yesterday_eng.get("days_gap", 1)
    time_hi = yesterday_eng.get("time_greeting_hi", "नमस्ते")
    time_en = yesterday_eng.get("time_greeting_en", "Hello")

    first_name = recipient.full_name.split()[0] if recipient.full_name else recipient.full_name

    # IST hour for standard (non-skip) greeting
    hour_ist = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
    if hour_ist < 12:
        time_greeting_std = "Good morning! सुप्रभात 🌅"
        hi_greeting_std = f"सुप्रभात {first_name} जी! आज का दिन बहुत अच्छा हो आपका ☀️"
    elif hour_ist < 17:
        time_greeting_std = "Good afternoon! 🌤️"
        hi_greeting_std = f"नमस्कार {first_name} जी! दोपहर की शुभकामनाएं 🌤️"
    elif hour_ist < 21:
        time_greeting_std = "Good evening! 🌙"
        hi_greeting_std = f"शुभ संध्या {first_name} जी! आइए थोड़ी बातें करें 🌙"
    else:
        time_greeting_std = "Good night! 🌚"
        hi_greeting_std = f"शुभ रात्रि {first_name} जी! 🌚"

    if did_not_talk_yesterday:
        greeting_en = f"{time_en} {first_name}! Where were you yesterday? Was everything okay? Were you missing someone?"
        greeting_hi = f"{time_hi} {first_name}! कल आप कहां थे? सब ठीक था ना? किसी की याद आ रही थी क्या?"
    else:
        greeting_en = f"{time_greeting_std} {first_name}! How are you feeling today?"
        greeting_hi = f"{hi_greeting_std} आज आप कैसा महसूस कर रहे हैं?"

    return ResponseSchema(
        code=200, status="success", message="Daily greeting",
        result={
            "already_greeted": already_greeted,
            "did_not_talk_yesterday": did_not_talk_yesterday,
            "days_gap": days_gap,
            "yesterday_msg_count": yesterday_eng.get("count", 0),
            "time_greeting_hi": time_hi,
            "time_greeting_en": time_en,
            "greeting_en": greeting_en,
            "greeting_hi": greeting_hi,
        }
    )


# ─────────────────────────────────────────────
# Favorites – songs
# ─────────────────────────────────────────────
class FavoriteSongRequest(BaseModel):
    recipient_id: int
    title: str
    query: str
    youtube_id: str = ""


@router.post('/voice-bot/favorites/song', response_model=ResponseSchema)
async def add_favorite_song(
    payload: FavoriteSongRequest,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(payload.recipient_id, db)

    songs = list(prefs.favorite_songs or [])
    # Avoid duplicates
    if not any(s.get("title") == payload.title for s in songs):
        songs.append({"title": payload.title, "query": payload.query, "youtube_id": payload.youtube_id})
        prefs.favorite_songs = songs
        db.commit()

    return ResponseSchema(code=200, status="success", message="Song saved to favorites", result={"favorites": songs})


@router.get('/voice-bot/favorites/songs/{recipient_id}', response_model=ResponseSchema)
async def get_favorite_songs(
    recipient_id: int,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Favorite songs", result={"favorites": prefs.favorite_songs or []})


@router.delete('/voice-bot/favorites/song', response_model=ResponseSchema)
async def remove_favorite_song(
    recipient_id: int, title: str,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(recipient_id, db)
    prefs.favorite_songs = [s for s in (prefs.favorite_songs or []) if s.get("title") != title]
    db.commit()
    return ResponseSchema(code=200, status="success", message="Song removed from favorites")


# ─────────────────────────────────────────────
# Favorites – stories
# ─────────────────────────────────────────────
class FavoriteStoryRequest(BaseModel):
    recipient_id: int
    title: str
    category: str
    youtube_id: str = ""


@router.post('/voice-bot/favorites/story', response_model=ResponseSchema)
async def add_favorite_story(
    payload: FavoriteStoryRequest,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(payload.recipient_id, db)

    stories = list(prefs.favorite_stories or [])
    if not any(s.get("title") == payload.title for s in stories):
        stories.append({"title": payload.title, "category": payload.category, "youtube_id": payload.youtube_id})
        prefs.favorite_stories = stories
        db.commit()

    return ResponseSchema(code=200, status="success", message="Story saved to favorites", result={"favorites": stories})


@router.get('/voice-bot/favorites/stories/{recipient_id}', response_model=ResponseSchema)
async def get_favorite_stories(
    recipient_id: int,
    authorization: Optional[str] = Header(None), db: Session = Depends(get_db)
):
    verify_and_get_caretaker(authorization, db)
    prefs = _get_or_create_preferences(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Favorite stories", result={"favorites": prefs.favorite_stories or []})


# ─────────────────────────────────────────────
# Triggers
# ─────────────────────────────────────────────
@router.get('/voice-bot/triggers/{recipient_id}', response_model=ResponseSchema)
async def fetch_triggers(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    # Verify or fallback recipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recipient:
            return ResponseSchema(code=404, status="error", message="Recipient not found")
        recipient_id = recipient.id

    triggers = get_pending_triggers(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Fetched triggers", result={"triggers": triggers})


# ─────────────────────────────────────────────
# Reminders
# ─────────────────────────────────────────────
class ReminderCreate(BaseModel):
    recipient_id: int
    reminder_type: str
    reminder_text: str
    scheduled_time: str
    recurrence: str = "daily"


@router.post('/voice-bot/reminders', response_model=ResponseSchema)
async def create_reminder(payload: ReminderCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    recipient = db.query(CareRecipient).filter(CareRecipient.id == payload.recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        # Fallback to the first recipient of this caretaker
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recipient:
            return ResponseSchema(code=404, status="error", message="Recipient not found")
        payload.recipient_id = recipient.id

    r_type = ReminderTypeEnum(payload.reminder_type) if payload.reminder_type in [t.value for t in ReminderTypeEnum] else ReminderTypeEnum.custom
    recur = RecurrenceEnum(payload.recurrence) if payload.recurrence in [t.value for t in RecurrenceEnum] else RecurrenceEnum.daily

    reminder = ProactiveReminder(
        care_recipient_id=payload.recipient_id,
        reminder_type=r_type,
        reminder_text=payload.reminder_text,
        scheduled_time=payload.scheduled_time,
        recurrence=recur
    )
    db.add(reminder)
    db.commit()
    return ResponseSchema(code=200, status="success", message="Reminder created successfully")


@router.get('/voice-bot/reminders/{recipient_id}', response_model=ResponseSchema)
async def get_reminders(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    # Verify or fallback recipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recipient:
            return ResponseSchema(code=200, status="success", message="Fetched reminders", result={"reminders": []})
        recipient_id = recipient.id

    reminders = db.query(ProactiveReminder).filter(ProactiveReminder.care_recipient_id == recipient_id).all()
    res = [{
        "id": r.id,
        "type": r.reminder_type.value,
        "text": r.reminder_text,
        "time": r.scheduled_time,
        "recurrence": r.recurrence.value,
        "is_active": r.is_active
    } for r in reminders]
    return ResponseSchema(code=200, status="success", message="Fetched reminders", result={"reminders": res})


@router.delete('/voice-bot/reminders/{reminder_id}', response_model=ResponseSchema)
async def delete_reminder(reminder_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_and_get_caretaker(authorization, db)
    reminder = db.query(ProactiveReminder).filter(ProactiveReminder.id == reminder_id).first()
    if not reminder:
        return ResponseSchema(code=404, status="error", message="Reminder not found")
    db.delete(reminder)
    db.commit()
    return ResponseSchema(code=200, status="success", message="Reminder deleted successfully")


# ─────────────────────────────────────────────
# Mood trend
# ─────────────────────────────────────────────
@router.get('/voice-bot/mood-trend/{recipient_id}', response_model=ResponseSchema)
async def get_mood_trend(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    # Verify or fallback recipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recipient:
            return ResponseSchema(code=200, status="success", message="No recipient found", result={"trend": {}, "depression_risk": False})
        recipient_id = recipient.id

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
    is_at_risk = check_depression_risk(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Fetched mood trend", result={"trend": mood_counts, "depression_risk": is_at_risk})


# ─────────────────────────────────────────────
# Init default reminders
# ─────────────────────────────────────────────
@router.post('/voice-bot/init-reminders/{recipient_id}', response_model=ResponseSchema)
async def init_reminders(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    # Verify or fallback recipient
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recipient:
            return ResponseSchema(code=400, status="error", message="Recipient not found")
        recipient_id = recipient.id

    existing = db.query(ProactiveReminder).filter(ProactiveReminder.care_recipient_id == recipient_id).count()
    if existing > 0:
        return ResponseSchema(code=400, status="error", message="Reminders already initialized")
    create_default_reminders(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Default reminders initialized")


# ─────────────────────────────────────────────
# Emergency alert
# ─────────────────────────────────────────────
class EmergencyRequest(BaseModel):
    recipient_id: int
    message: str = "Emergency alert triggered from voice bot"


@router.post('/voice-bot/emergency', response_model=ResponseSchema)
async def trigger_emergency(
    payload: EmergencyRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    caretaker = verify_and_get_caretaker(authorization, db)
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == payload.recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")

    save_message(
        payload.recipient_id, SenderEnum.user, payload.message,
        MoodEnum.distressed, TriggerTypeEnum.user_initiated, "emergency", db, intent="emergency"
    )

    alert_sent = False
    if caretaker.email:
        try:
            from utils.email import send_fall_alert_email
            from datetime import timezone as _tz
            await send_fall_alert_email(caretaker.email, {
                "timestamp": datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "fall_count": 1,
                "location": "Voice Bot — Emergency Button",
                "fall_details": []
            })
            alert_sent = True
        except Exception as e:
            print(f"[emergency] Email alert failed: {e}")

    print(f"[EMERGENCY] Triggered for recipient {payload.recipient_id} by {caretaker.username}")
    return ResponseSchema(
        code=200, status="success", message="Emergency alert triggered",
        result={"alert_sent": alert_sent, "recipient": recipient.full_name}
    )


# ── STANDALONE TTS ENDPOINT ──
class TTSRequest(BaseModel):
    text: str
    language_code: str = "hi-IN"

@router.post('/voice-bot/tts', response_model=ResponseSchema)
async def get_standalone_tts(payload: TTSRequest):
    """Generates audio for given text using soft-spoken neural female TTS."""
    if not payload.text:
        return ResponseSchema(code=400, status="error", message="No text provided")
    
    audio_b64 = await generate_speech_base64_async(payload.text, payload.language_code)
    if not audio_b64:
        return ResponseSchema(code=500, status="error", message="TTS generation failed.")
    
    return ResponseSchema(
        code=200, status="success", message="TTS generated",
        result={"audio_base64": audio_b64}
    )
