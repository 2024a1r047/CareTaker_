from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import json
from sqlalchemy.orm import Session
from config import get_db
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo
from utils.gemini_client import call_gemini, safe_json_parse
from utils.tts_handler import generate_speech_base64, generate_speech_base64_async, is_hindi
import os

from tables.conversation_history import MoodEnum, SenderEnum, TriggerTypeEnum
from services.voice_bot_engine import (
    build_conversation_context, generate_system_prompt,
    save_message, check_depression_risk, get_content_recommendation,
    intelligent_saathi_dialogue
)
from services.sentiment_engine import analyze_sentiment_with_history
from services.conversation_director import build_proactive_context, format_proactive_instruction

router = APIRouter(tags=["VoiceBot WS"])


def _get_username_from_token(token: str):
    try:
        from repository.users import JWTRepo
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None

@router.websocket("/ws/voice-chat")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    
    caretaker = None
    recipient_id = 1 # Default
    saathi_session_id = None
    
    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON payload"})
                continue
            
            # Auth step (usually sent on first connection or with every message)
            token = payload.get("token")
            if token and not caretaker:
                username = _get_username_from_token(token)
                if username:
                    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
                    if caretaker and not saathi_session_id:
                        try:
                            from services.saathi_memory_service import start_session
                            saathi_session_id = start_session(caretaker.id, recipient_id)
                        except Exception as _serr:
                            print(f"[WS] start_session notice: {_serr}")
                    
            if not caretaker:
                await websocket.send_json({"error": "Unauthorized. Please provide a valid token."})
                continue
                
            ActionType = payload.get("type", "chat")
            
            if ActionType == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if ActionType == "interrupt":
                print("[WS] Barge-in registered.")
                continue

            # ── PROACTIVE INIT — Saathi speaks first ──────────────────────────
            if ActionType == "proactive_init":
                try:
                    try:
                        recipient_id = int(payload.get("recipient_id") or 1)
                    except (TypeError, ValueError):
                        recipient_id = 1
                    lang = payload.get("language", "hi")
                    session_id = str(saathi_session_id) if saathi_session_id else payload.get("session_id", "ws_session")

                    # Resolve recipient name
                    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
                    # Commented out legacy fallback to strictly resolve DB user name or username
                    # recipient_name = recipient.full_name if recipient else ""
                    # name_part = f"{recipient_name} ji" if recipient_name else "ji"
                    recipient_name = (
                        recipient.preferred_name.strip() if (recipient and getattr(recipient, "preferred_name", None) and recipient.preferred_name.strip())
                        else (recipient.full_name.split()[0] if (recipient and recipient.full_name) else (caretaker.username if caretaker else "ji"))
                    )
                    name_part = recipient_name

                    # Build full context
                    proactive_ctx = build_proactive_context(recipient_id, db, lang)
                    
                    proactive_instruction = format_proactive_instruction(proactive_ctx)

                    context = build_conversation_context(recipient_id, db)
                    system_prompt = generate_system_prompt(recipient_name, context, lang, None)

                    # Inject proactive instruction into the prompt
                    full_prompt = f"{system_prompt}\n\n{proactive_instruction}\n\nNow open the conversation naturally. Speak first."

                    gemini_payload = {
                        "contents": [{"parts": [{"text": full_prompt}]}],
                        "generationConfig": {"maxOutputTokens": 400, "temperature": 0.85}
                    }
                    data = call_gemini(gemini_payload, timeout=20, caller="[proactive_init]")

                    from services.saathi_memory_service import get_member_context
                    mem_ctx = get_member_context(recipient_id)
                    pref_name = mem_ctx.get("preferred_name") or (recipient_name.split()[0] if recipient_name else "Maa")
                    first_name = pref_name
                    # Specification: Exact opening greeting with natural pauses
                    reply_text = f"Namaste {first_name} ji... aaj aap kaisa feel kar rahe hain? ... Jo bhi aapke mann mein hai, mujhe batayiye. Main sun rahi hoon."
                    intent = "greeting"

                    save_message(recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral,
                                 TriggerTypeEnum.proactive_checkin, session_id, db, intent=intent)

                    # audio_base64 = None  # Only ElevenLabs Asha voice is permitted
                    audio_base64 = None
                    try:
                        audio_base64 = await generate_speech_base64_async(reply_text, lang)
                    except Exception as _t_err:
                        print(f"[WS proactive TTS] {_t_err}")

                    await websocket.send_json({
                        "type": "bot_response",
                        "message_id": payload.get("message_id"),
                        "reply": reply_text,
                        "intent": intent,
                        "search_query": "",
                        "audio_base64": audio_base64,
                        "mood_detected": "neutral",
                        "depression_risk": False,
                        "recommendation": None,
                        "sentiment": {"trend": proactive_ctx["mood_trend"], "summary": ""},
                        "is_proactive": True
                    })
                    continue
                except Exception as e:
                    print(f"[proactive_init] Error: {e}")
                    continue

            # Regular chat processing — wrapped per-message so exceptions never break the loop
            try:
                user_text = payload.get("text")
                try:
                    recipient_id = int(payload.get("recipient_id") or 1)
                except (TypeError, ValueError):
                    recipient_id = 1
                session_id = str(saathi_session_id) if saathi_session_id else payload.get("session_id", "ws_session")
                lang = payload.get("language", "hi")
                trigger_type = TriggerTypeEnum.user_initiated

                if not user_text:
                    continue

                # Resolve recipient name from DB
                recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
                recipient_name = (
                    recipient.preferred_name.strip() if (recipient and getattr(recipient, 'preferred_name', None) and recipient.preferred_name.strip())
                    else (recipient.full_name if recipient else "ji")
                )

                # 1. Sentiment & DB save

                sentiment = analyze_sentiment_with_history(user_text, recipient_id, db)
                mood_str = sentiment.get("current_mood", "neutral")
                mood_enum = MoodEnum(mood_str) if mood_str in [m.value for m in MoodEnum] else MoodEnum.neutral
                save_message(recipient_id, SenderEnum.user, user_text, mood_enum, trigger_type, session_id, db)
                # Sync user turn to saathi_db.conversation_messages
                try:
                    from services.saathi_memory_service import save_message_sync
                    save_message_sync(session_id, recipient_id, 'user', user_text, language=lang)
                except Exception:
                    pass

                # 2. Context & Gemini
                context = build_conversation_context(recipient_id, db)
                system_prompt = generate_system_prompt(recipient_name, context, lang, sentiment)
                is_at_risk = check_depression_risk(recipient_id, db) or sentiment.get("urgency") == "high"

                gemini_payload = {
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser says: {user_text}"}]}],
                    "generationConfig": {
                        "maxOutputTokens": 200,
                        "temperature": 0.7
                    }
                }

                data = call_gemini(gemini_payload, timeout=25, caller="[voice_bot_ws]")
                if data and data.get('candidates'):
                    ai_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                    try:
                        ai_json = json.loads(ai_text.replace("```json", "").replace("```", "").strip())
                        thought = ai_json.get("thought", "")
                        if thought: print(f"[Saathi Thought WS] {thought}")
                        reply_text = ai_json.get("reply") or ai_json.get("message") or ai_text
                        intent = ai_json.get("intent", "chat")
                        search_query = ai_json.get("search_query", "")
                        action_param = ai_json.get("action_param")
                        recommendation = ai_json.get("recommendation")
                        
                        if intent == "set_reminder":
                            r_time = ai_json.get("reminder_time")
                            r_text = ai_json.get("reminder_text")
                            if r_time and r_text:
                                try:
                                    from tables.medications import ProactiveReminder, ReminderTypeEnum, RecurrenceEnum
                                    
                                    # Verify or fallback recipient
                                    actual_recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
                                    if not actual_recipient:
                                        actual_recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
                                        
                                    if actual_recipient:
                                        new_rem = ProactiveReminder(
                                            care_recipient_id=actual_recipient.id,
                                            reminder_type=ReminderTypeEnum.custom,
                                            reminder_text=r_text,
                                            scheduled_time=r_time,
                                            recurrence=RecurrenceEnum.daily
                                        )
                                        db.add(new_rem)
                                        db.commit()
                                        intent = "show_reminders" # Force frontend to refresh reminders UI
                                    else:
                                        print("Failed to save voice reminder: No valid recipient found for caretaker")
                                except Exception as e:
                                    print(f"Error saving voice reminder: {e}")
                                    db.rollback()

                    except Exception:
                        reply_text = ai_text
                        intent = "chat"
                        search_query = ""
                        action_param = None
                        recommendation = None

                    save_message(recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral, trigger_type, session_id, db, intent=intent)
                    # Sync bot turn to saathi_db.conversation_messages
                    try:
                        from services.saathi_memory_service import save_message_sync
                        save_message_sync(session_id, recipient_id, 'assistant', reply_text, language=lang)
                    except Exception:
                        pass

                    if not recommendation and mood_str not in ("neutral",):
                        recommendation = get_content_recommendation(mood_str)

                    # audio_base64 = None  # Only ElevenLabs Asha voice is permitted
                    audio_base64 = None
                    try:
                        audio_base64 = await generate_speech_base64_async(reply_text, lang)
                    except Exception as _t_err:
                        print(f"[WS chat TTS] {_t_err}")

                    await websocket.send_json({
                        "type": "bot_response",
                        "message_id": payload.get("message_id"),
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
                            "summary": sentiment.get("summary", "")
                        }
                    })
                else:
                    # Intelligent companionship fallback — handles jokes, empathy, songs, YouTube videos, and reminders
                    dialogue_res = intelligent_saathi_dialogue(user_text, recipient_id, db, lang)
                    reply_text = dialogue_res.get("reply")
                    intent = dialogue_res.get("intent", "chat")
                    search_query = dialogue_res.get("search_query", "")
                    fallback_mood = dialogue_res.get("mood_detected", mood_str)

                    save_message(recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral, trigger_type, session_id, db, intent=intent)
                    # Sync bot turn to saathi_db.conversation_messages
                    try:
                        from services.saathi_memory_service import save_message_sync
                        save_message_sync(session_id, recipient_id, 'assistant', reply_text, language=lang)
                    except Exception:
                        pass
                    # fallback_audio = None  # Only ElevenLabs Asha voice is permitted
                    fallback_audio = None
                    try:
                        fallback_audio = await generate_speech_base64_async(reply_text, lang)
                    except Exception as _t_err:
                        print(f"[WS fallback TTS] {_t_err}")
                    await websocket.send_json({
                        "type": "bot_response",
                        "message_id": payload.get("message_id"),
                        "reply": reply_text,
                        "intent": intent,
                        "search_query": search_query,
                        "action_param": None,
                        "audio_base64": fallback_audio,
                        "mood_detected": fallback_mood,
                        "depression_risk": False,
                        "recommendation": None,
                        "sentiment": {"trend": "stable", "summary": ""}
                    })

            except Exception as e:
                print(f"[voice_bot_ws] Message processing error: {e}")
                try:
                    await websocket.send_json({"error": str(e)})
                except Exception:
                    pass
                
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except ConnectionResetError:
        pass
    except Exception as e:
        print(f"[WS] Unhandled error: {e}")
    finally:
        if saathi_session_id:
            try:
                from services.saathi_memory_service import end_session
                end_session(saathi_session_id)
                print(f"[WS] Closed Saathi session {saathi_session_id}")
            except Exception as _end_err:
                print(f"[WS] end_session notice: {_end_err}")
