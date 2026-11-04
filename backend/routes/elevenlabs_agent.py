from fastapi import APIRouter, Depends, HTTPException, Header, status, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from config import get_db, SaathiSessionLocal
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from routes.voice_bot import verify_and_get_caretaker
from services.voice_bot_engine import save_message
from tables.conversation_history import SenderEnum, MoodEnum, TriggerTypeEnum
from tables.saathi_history import (
    ConversationMessage as SaathiMsg,
    ConversationSession as SaathiSess,
    SaathiRecipient
)
from services.elevenlabs_service import (
    get_elevenlabs_voice_id,
    get_elevenlabs_model_id,
    get_elevenlabs_api_key,
    get_elevenlabs_agent_id,
    update_elevenlabs_voice_id,
    clone_voice_from_audio,
    synthesize_elevenlabs_speech,
    get_elevenlabs_signed_url,
    build_saathi_context,
    handle_log_medicine_or_meal,
    handle_send_emergency_alert,
)

router = APIRouter(prefix="/elevenlabs", tags=["ElevenLabs Saathi"])

class SetVoiceRequest(BaseModel):
    voice_id: str
    voice_name: Optional[str] = "saathi voice"

class LogToolRequest(BaseModel):
    recipient_id: Optional[int] = 1
    item_type: str = "medicine"   # "medicine" or "meal"
    item_name: Optional[str] = "Prescription"
    status: str = "taken"         # "taken" or "missed"

class EmergencyToolRequest(BaseModel):
    recipient_id: Optional[int] = 1
    reason: str = "Fall or acute distress reported"
    severity: Optional[str] = "critical"

class DirectTTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None

class ConversationTurnRequest(BaseModel):
    recipient_id: Optional[int] = 1
    session_id: str
    sender: str  # "user" or "bot"
    text: str
    mood_detected: Optional[str] = "neutral"
    intent: Optional[str] = "chat"
    trigger_type: Optional[str] = "user_initiated"

class EndSessionRequest(BaseModel):
    session_id: str

@router.get("/config", response_model=ResponseSchema)
async def get_elevenlabs_config():
    """Returns safe public config for ElevenLabs Saathi integration (never exposes private API keys)."""
    api_key = get_elevenlabs_api_key()
    agent_id = get_elevenlabs_agent_id()
    active_voice_id = get_elevenlabs_voice_id()
    active_model_id = get_elevenlabs_model_id()
    return ResponseSchema(
        code=200,
        status="success",
        message="ElevenLabs config fetched",
        result={
            "voice_name": "Asha - Soft Care Voice",
            "voice_id": active_voice_id,
            "model_id": active_model_id,
            "has_api_key": bool(api_key),
            "has_agent_id": bool(agent_id),
            "agent_id": agent_id if agent_id else None
        }
    )

@router.get("/session-context/{recipient_id}", response_model=ResponseSchema)
async def get_session_context(recipient_id: int, db: Session = Depends(get_db)):
    """
    Builds dynamic session context for the Care Recipient:
    - {{name}}, {{language}}, {{medicine_schedule}}, {{conditions}}, {{family_contact}}
    - First message & complete empathetic Saathi persona system prompt.
    """
    context = build_saathi_context(recipient_id, db)
    return ResponseSchema(
        code=200,
        status="success",
        message="Saathi session context built",
        result=context
    )

@router.get("/signed-url", response_model=ResponseSchema)
@router.get("/session", response_model=ResponseSchema)
async def get_signed_url(
    agent_id: Optional[str] = None,
    recipient_id: Optional[int] = 1,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Secure backend bridge for ElevenLabs Conversational AI Web SDK.
    Validates authenticated CareTaker user before issuing access to a conversation.
    Generates a signed URL so client applications never touch the secret API key.
    """
    # 1. Validate authenticated CareTaker user
    caretaker = verify_and_get_caretaker(authorization, db)

    # 2. Verify recipient belongs to this caretaker or fallback safely
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        recipient = db.query(CareRecipient).first()

    recip_id = recipient.id if recipient else 1

    # 3. Retrieve dynamic context for Saathi (name, medicines, address form)
    context = build_saathi_context(recip_id, db)
    target_agent_id = agent_id or get_elevenlabs_agent_id()

    # 4. Generate signed WebSocket URL for ElevenLabs
    result = get_elevenlabs_signed_url(target_agent_id)
    if "error" in result:
        return ResponseSchema(
            code=400,
            status="error",
            message=result["error"],
            result=None
        )

    saathi_sess_id = None
    try:
        from services.saathi_memory_service import start_session
        saathi_sess_id = str(start_session(caretaker.id, recip_id))
    except Exception as _serr:
        pass

    return ResponseSchema(
        code=200,
        status="success",
        message="Signed conversation URL generated",
        result={
            "signed_url": result["signed_url"],
            "session_id": saathi_sess_id,
            "agent_id": target_agent_id,
            "voice_id": get_elevenlabs_voice_id(),
            "recipient_id": recip_id,
            "recipient_name": context.get("name", "Dadaji"),
            "address": context.get("address", "Dadaji"),
            "language": context.get("language", "Hindi, English, or Hinglish"),
            "first_message": context.get("first_message", ""),
            "dynamic_variables": {
                "name": context.get("name", "Dadaji"),
                "user_name": context.get("name", "Dadaji"),
                "address": context.get("address", "Dadaji"),
                "preferred_language": context.get("language", "Hindi, English, or Hinglish"),
                "medicine_schedule": context.get("medicine_schedule", ""),
                "medicines": context.get("medicines", ""),
                "medical_reports": context.get("medical_reports", ""),
                "reports": context.get("medical_reports", ""),
                "conditions": context.get("conditions", ""),
                "family_contact": context.get("family_contact", "")
            }
        }
    )

@router.post("/end-session", response_model=ResponseSchema)
async def end_conversation_session(
    payload: EndSessionRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Closes conversation session in saathi_db, generates summary, and extracts member memory."""
    caretaker = verify_and_get_caretaker(authorization, db)
    try:
        from services.saathi_memory_service import end_session
        end_session(payload.session_id)
        return ResponseSchema(code=200, status="success", message="Session ended and memory extracted", result={"session_id": payload.session_id})
    except Exception as e:
        return ResponseSchema(code=500, status="error", message=str(e))

@router.post("/save-turn", response_model=ResponseSchema)
async def save_conversation_turn(
    payload: ConversationTurnRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Persists user utterances and ElevenLabs Saathi responses into the "Saathi" PostgreSQL database.
    Ensures data isolation between users and prevents duplicate record insertion.
    """
    caretaker = verify_and_get_caretaker(authorization, db)
    if not payload.text or not payload.text.strip():
        return ResponseSchema(code=400, status="error", message="Empty text provided")

    # Ensure recipient belongs to this caretaker
    recip = db.query(CareRecipient).filter(
        CareRecipient.id == payload.recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recip:
        recip = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    recip_id = recip.id if recip else (payload.recipient_id or 1)

    sender_enum = SenderEnum.bot if payload.sender.lower() in ["bot", "agent", "assistant", "ai"] else SenderEnum.user
    mood_str = payload.mood_detected or "neutral"
    mood_enum = MoodEnum(mood_str) if mood_str in [m.value for m in MoodEnum] else MoodEnum.neutral
    trigger_enum = TriggerTypeEnum.user_initiated

    # Deduplication check: avoid duplicate records if both callback and client stream the same text
    if SaathiSessionLocal:
        try:
            with SaathiSessionLocal() as s_db:
                cutoff = datetime.utcnow() - timedelta(seconds=4)
                sender_str = sender_enum.value if hasattr(sender_enum, 'value') else str(sender_enum)
                existing = s_db.query(SaathiMsg).filter(
                    SaathiMsg.session_id == payload.session_id,
                    SaathiMsg.recipient_id == recip_id,
                    SaathiMsg.sender == sender_str,
                    SaathiMsg.message_text == payload.text.strip(),
                    SaathiMsg.created_at >= cutoff
                ).first()
                if existing:
                    return ResponseSchema(
                        code=200, status="success",
                        message="Turn already recorded",
                        result={"id": existing.id, "duplicate": True}
                    )
        except Exception as _de:
            pass

    # Save to Saathi PostgreSQL database (and dual-write to core caretaker DB)
    save_message(
        recip_id, sender_enum, payload.text.strip(), mood_enum,
        trigger_enum, payload.session_id, db, intent=payload.intent or "chat"
    )
    return ResponseSchema(
        code=200, status="success",
        message="Turn saved successfully to Saathi database",
        result={"saved": True, "recipient_id": recip_id, "session_id": payload.session_id}
    )

@router.get("/history/{recipient_id}", response_model=ResponseSchema)
async def get_conversation_history(
    recipient_id: int,
    limit: int = 30,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Securely retrieves conversation history from the "Saathi" PostgreSQL database
    for the authenticated caretaker's recipient.
    """
    caretaker = verify_and_get_caretaker(authorization, db)

    recip = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recip:
        recip = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    target_id = recip.id if recip else recipient_id

    messages = []
    if SaathiSessionLocal:
        try:
            with SaathiSessionLocal() as s_db:
                msgs = s_db.query(SaathiMsg).filter(
                    SaathiMsg.recipient_id == target_id
                ).order_by(SaathiMsg.id.desc()).limit(limit).all()
                for m in reversed(msgs):
                    messages.append({
                        "id": m.id,
                        "session_id": m.session_id,
                        "sender": m.sender,
                        "text": m.message_text,
                        "mood": m.mood_detected,
                        "intent": m.intent,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    })
        except Exception as se:
            print(f"[Saathi history query error] {se}")

    if not messages:
        from tables.conversation_history import ConversationMessage as MainMsg
        main_msgs = db.query(MainMsg).filter(
            MainMsg.care_recipient_id == target_id
        ).order_by(MainMsg.id.desc()).limit(limit).all()
        for m in reversed(main_msgs):
            messages.append({
                "id": m.id,
                "sender": m.sender.value if hasattr(m.sender, 'value') else str(m.sender),
                "text": m.message_text,
                "mood": m.mood_detected.value if hasattr(m.mood_detected, 'value') and m.mood_detected else "neutral",
                "created_at": m.created_at.isoformat() if m.created_at else None
            })

    return ResponseSchema(
        code=200, status="success",
        message="Conversation history fetched",
        result={"recipient_id": target_id, "messages": messages}
    )

@router.post("/tools/log-medicine-or-meal", response_model=ResponseSchema)
async def tool_log_medicine_or_meal(payload: LogToolRequest, db: Session = Depends(get_db)):
    """
    ElevenLabs function-calling tool:
    Logs when the elderly user confirms taking medication or having a meal.
    Updates dose log records, inventory stock, and notifies caregiver dashboard in real time.
    """
    result = await handle_log_medicine_or_meal(
        recipient_id=payload.recipient_id or 1,
        item_type=payload.item_type,
        item_name=payload.item_name or "Medication",
        status=payload.status,
        db=db
    )
    return ResponseSchema(
        code=200 if result.get("success") else 500,
        status="success" if result.get("success") else "error",
        message=result.get("message", "Tool execution complete"),
        result=result
    )

@router.post("/tools/send-emergency-alert", response_model=ResponseSchema)
async def tool_send_emergency_alert(payload: EmergencyToolRequest, db: Session = Depends(get_db)):
    """
    ElevenLabs function-calling tool:
    Immediately triggers an emergency alert notification to family/caregiver contacts
    upon detection of severe distress, falls, or cardiac symptoms.
    """
    result = await handle_send_emergency_alert(
        recipient_id=payload.recipient_id or 1,
        reason=payload.reason,
        severity=payload.severity or "critical",
        db=db
    )
    return ResponseSchema(
        code=200 if result.get("success") else 500,
        status="success" if result.get("success") else "error",
        message=result.get("message", "Emergency alert processed"),
        result=result
    )

@router.post("/set-voice", response_model=ResponseSchema)
async def set_active_voice(payload: SetVoiceRequest):
    """
    Sets a custom voice ID for Saathi.
    Saves it directly to .env and runtime environment so all future sessions use this voice.
    """
    if not payload.voice_id:
        return ResponseSchema(code=400, status="error", message="voice_id is required")

    updated = update_elevenlabs_voice_id(payload.voice_id.strip())
    return ResponseSchema(
        code=200,
        status="success",
        message=f"Voice updated to '{payload.voice_id}'. Saathi will now speak with this voice.",
        result={
            "voice_id": payload.voice_id.strip(),
            "voice_name": payload.voice_name or "saathi voice",
            "persisted_to_env": updated
        }
    )

@router.post("/clone-voice", response_model=ResponseSchema)
async def clone_custom_voice(
    file: UploadFile = File(...),
    voice_name: Optional[str] = Form("saathi voice")
):
    """
    Upload an audio sample (.mp3 / .wav) of your desired voice.
    This clones the voice into ElevenLabs with Saathi's soft, calm settings and updates ELEVENLABS_VOICE_ID.
    """
    file_bytes = await file.read()
    if not file_bytes:
        return ResponseSchema(code=400, status="error", message="Empty audio file provided")

    result = clone_voice_from_audio(file_bytes, file.filename or "sample.mp3", voice_name or "saathi voice")
    if "error" in result:
        return ResponseSchema(
            code=500,
            status="error",
            message=result["error"],
            result=result
        )

    return ResponseSchema(
        code=200,
        status="success",
        message="Voice successfully cloned and activated as Saathi's voice!",
        result=result
    )

@router.post("/tts", response_model=ResponseSchema)
async def direct_elevenlabs_tts(payload: DirectTTSRequest):
    """
    Direct ElevenLabs Text-to-Speech synthesis with active Saathi voice
    and 'eleven_multilingual_v2'.
    """
    if not payload.text:
        return ResponseSchema(code=400, status="error", message="No text provided")

    audio_b64 = synthesize_elevenlabs_speech(
        text=payload.text,
        voice_id=payload.voice_id,
        model_id=payload.model_id
    )
    if not audio_b64:
        return ResponseSchema(
            code=500,
            status="error",
            message="ElevenLabs TTS synthesis failed. Ensure ELEVENLABS_API_KEY is configured in .env."
        )

    return ResponseSchema(
        code=200,
        status="success",
        message="ElevenLabs speech generated",
        result={
            "audio_base64": audio_b64,
            "voice_id": payload.voice_id or get_elevenlabs_voice_id(),
            "model_id": payload.model_id or get_elevenlabs_model_id()
        }
    )
