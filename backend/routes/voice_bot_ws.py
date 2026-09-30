from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import json
from sqlalchemy.orm import Session
from config import get_db
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo
from utils.gemini_client import call_gemini
from utils.tts_handler import generate_speech_base64, is_hindi
import os

from tables.conversation_history import MoodEnum, SenderEnum, TriggerTypeEnum
from services.voice_bot_engine import (
    build_conversation_context, generate_system_prompt,
    save_message, check_depression_risk, get_content_recommendation
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
                    session_id = payload.get("session_id", "ws_session")

                    # Resolve recipient name
                    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
                    recipient_name = recipient.full_name if recipient else "ji"

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

                    if data and data.get("candidates"):
                        ai_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        try:
                            ai_json = json.loads(ai_text.replace("```json", "").replace("```", "").strip())
                            reply_text = ai_json.get("reply") or ai_json.get("message") or ai_text
                            intent = ai_json.get("intent", "chat")
                        except Exception:
                            reply_text = ai_text
                            intent = "chat"

                        save_message(recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral,
                                     TriggerTypeEnum.proactive_checkin, session_id, db)

                        lang_hi = "hi-IN" if is_hindi(reply_text) else "en-US"
                        audio_base64 = generate_speech_base64(reply_text, lang_hi)

                        await websocket.send_json({
                            "type": "bot_response",
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
                session_id = payload.get("session_id", "ws_session")
                lang = payload.get("language", "hi")
                trigger_type = TriggerTypeEnum.user_initiated

                if not user_text:
                    continue

                # Resolve recipient name from DB
                recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
                recipient_name = recipient.full_name if recipient else "ji"

                # 1. Sentiment & DB save

                sentiment = analyze_sentiment_with_history(user_text, recipient_id, db)
                mood_str = sentiment.get("current_mood", "neutral")
                mood_enum = MoodEnum(mood_str) if mood_str in [m.value for m in MoodEnum] else MoodEnum.neutral
                save_message(recipient_id, SenderEnum.user, user_text, mood_enum, trigger_type, session_id, db)

                # 2. Context & Gemini
                context = build_conversation_context(recipient_id, db)
                system_prompt = generate_system_prompt(recipient_name, context, lang, sentiment)
                is_at_risk = check_depression_risk(recipient_id, db) or sentiment.get("urgency") == "high"

                gemini_payload = {
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser says: {user_text}"}]}],
                    "generationConfig": {
                        "maxOutputTokens": 1500,
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
                    except Exception:
                        reply_text = ai_text
                        intent = "chat"
                        search_query = ""
                        action_param = None
                        recommendation = None

                    save_message(recipient_id, SenderEnum.bot, reply_text, MoodEnum.neutral, trigger_type, session_id, db)

                    if not recommendation and mood_str not in ("neutral",):
                        recommendation = get_content_recommendation(mood_str)

                    lang_hi = "hi-IN" if is_hindi(reply_text) else "en-US"
                    audio_base64 = generate_speech_base64(reply_text, lang_hi)

                    await websocket.send_json({
                        "type": "bot_response",
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
                    # Graceful fallback — send a friendly message instead of a bare error
                    fallback = "माफ़ कीजिए, अभी मुझे कुछ दिक्कत आ रही है। थोड़ी देर बाद फिर से कोशिश करें।"
                    await websocket.send_json({
                        "type": "bot_response",
                        "reply": fallback,
                        "intent": "chat",
                        "search_query": "",
                        "audio_base64": "",
                        "mood_detected": mood_str,
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
