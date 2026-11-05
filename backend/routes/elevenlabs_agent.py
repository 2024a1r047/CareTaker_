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
    check_elevenlabs_quota,
    update_elevenlabs_credentials,
    get_elevenlabs_voice_id,
    get_elevenlabs_model_id,
    get_elevenlabs_api_key,
    get_elevenlabs_agent_id,
    update_elevenlabs_voice_id,
    clone_voice_from_audio,
    synthesize_elevenlabs_speech,
    get_elevenlabs_signed_url,
    build_saathi_context,
    build_agent_dynamic_variables,
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

class CaretakerMessageRequest(BaseModel):
    recipient_id: Optional[int] = 1
    message: str = ""
    urgent: Optional[bool] = False

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


class CredentialsRequest(BaseModel):
    api_key: Optional[str] = None
    agent_id: Optional[str] = None


@router.post("/set-credentials", response_model=ResponseSchema)
async def set_credentials(payload: CredentialsRequest):
    """Updates ELEVENLABS_API_KEY and/or ELEVENLABS_AGENT_ID safely in environment."""
    key = (payload.api_key or "").strip()
    agent = (payload.agent_id or "").strip()
    updated = update_elevenlabs_credentials(api_key=key if key else None, agent_id=agent if agent else None)
    if updated:
        return ResponseSchema(
            code=200,
            status="success",
            message="ElevenLabs credentials updated successfully in environment",
            result={"has_api_key": bool(key), "agent_id": agent}
        )
    return ResponseSchema(code=400, status="error", message="Failed to update credentials in .env")

@router.get("/config", response_model=ResponseSchema)
async def get_elevenlabs_config():
    """Returns safe public config for ElevenLabs Saathi integration (never exposes private API keys)."""
    api_key = get_elevenlabs_api_key()
    agent_id = get_elevenlabs_agent_id()
    active_voice_id = get_elevenlabs_voice_id()
    active_model_id = get_elevenlabs_model_id()
    quota = check_elevenlabs_quota()
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
            "agent_id": agent_id if agent_id else None,
            "quota_exceeded": quota.get("quota_exceeded", False),
            "remaining_characters": quota.get("remaining_characters", 0),
            "character_count": quota.get("character_count", 0),
            "character_limit": quota.get("character_limit", 30000),
            "tier": quota.get("tier", "creator")
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

    # 2. Verify recipient belongs to this caretaker (never fall back to another account's patient)
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        recipient = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        return ResponseSchema(
            code=404, status="error",
            message="No family member profile found for this account. Add one in Settings first.",
            result=None
        )

    recip_id = recipient.id

    # 3. Retrieve profile + memory for Asha (name, medicines, past calls) from the DBs
    context = build_saathi_context(recip_id, db)
    dynamic_variables = build_agent_dynamic_variables(recip_id, db)
    target_agent_id = agent_id or get_elevenlabs_agent_id()

    # 3b. Medical record briefing: uploaded reports + analysis, labs with normal ranges,
    #     vitals, care plan, and BP / heart-rate / sugar guidance. The page sends it to the
    #     agent as a contextual_update. Never blocks the call if something fails.
    medical_context = ""
    try:
        from services.agent_medical_context import build_medical_context, format_medical_context
        medical_context = format_medical_context(build_medical_context(recip_id, db))
    except Exception as _mcerr:
        print(f"[ElevenLabs] Medical context skipped for recipient {recip_id}: {_mcerr}")

    # 3c. Contacts the agent can call (via the page's call_* client tools)
    contacts = {
        "caretaker_name": caretaker.full_name or caretaker.username or "",
        "caretaker_phone": (getattr(caretaker, "phone_number", None) or "").strip(),
        "emergency_contact": (getattr(recipient, "emergency_contact", None) or "").strip(),
        "emergency_services": "112",
    }

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
        print(f"[Saathi DB] Could not start session in saathi_db: {_serr}")

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
            "recipient_name": dynamic_variables["user_name"],
            "address": dynamic_variables["user_name"],
            "language": context.get("language", "Hindi, English, or Hinglish"),
            "first_message": context.get("first_message", ""),
            "dynamic_variables": dynamic_variables,
            "medical_context": medical_context,
            "contacts": contacts
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
    if not recip:
        return ResponseSchema(code=404, status="error", message="Family member not found for this account")
    recip_id = recip.id

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

    # Also store the turn in saathi_db memory tables (member_sessions / member_messages).
    # end-session reads these to write the call summary and extract personal memory,
    # which Asha receives at the next login.
    try:
        from services.saathi_memory_service import save_message_async
        save_message_async(
            payload.session_id, recip_id,
            "assistant" if sender_enum == SenderEnum.bot else "user",
            payload.text.strip(),
            account_id=caretaker.id,
        )
    except Exception as _me:
        print(f"[Saathi DB] memory-table save notice: {_me}")

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

@router.post("/tools/message-caretaker", response_model=ResponseSchema)
async def tool_message_caretaker(payload: CaretakerMessageRequest, db: Session = Depends(get_db)):
    """
    ElevenLabs client tool `message_caretaker`: delivers a message from the patient to their
    caretaker. Always saved as an alert on the caretaker's dashboard (and pushed live via
    Socket.IO), so the message is delivered even without email/SMS configured. Returns the
    caretaker's name and phone so the page can also offer WhatsApp / SMS.
    """
    from tables.medical_conditions import MedicalAlert, AlertType, AlertSeverity

    recipient = db.query(CareRecipient).filter(CareRecipient.id == payload.recipient_id).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Family member not found")
    caretaker = db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
    patient_name = (getattr(recipient, "preferred_name", None) or recipient.full_name or "Your family member").strip()
    text = (payload.message or "").strip() or "Please call me when you can."

    saved = False
    try:
        db.add(MedicalAlert(
            care_recipient_id=recipient.id,
            alert_type=AlertType.critical if payload.urgent else AlertType.monitoring_gap,
            severity=AlertSeverity.high if payload.urgent else AlertSeverity.medium,
            message=f"💬 Message from {patient_name} (via Saathi): {text}",
        ))
        db.commit()
        saved = True
    except Exception as e:
        db.rollback()
        print(f"[Saathi message] could not save dashboard alert: {e}")

    try:
        from socket_manager import sio_server
        await sio_server.emit("caretaker_message", {
            "recipient_id": recipient.id, "recipient_name": patient_name,
            "message": text, "urgent": bool(payload.urgent),
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[Saathi message] socket push failed: {e}")

    return ResponseSchema(
        code=200 if saved else 500,
        status="success" if saved else "error",
        message="Message delivered to the caretaker's dashboard" if saved else "Could not save the message",
        result={
            "delivered_to_dashboard": saved,
            "caretaker_name": (caretaker.full_name or caretaker.username) if caretaker else "",
            "caretaker_phone": (getattr(caretaker, "phone_number", None) or "") if caretaker else "",
            "patient_name": patient_name,
            "message": text,
        },
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
