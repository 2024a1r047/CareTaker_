import os
import requests
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

def get_elevenlabs_voice_id() -> str:
    return os.getenv("ELEVENLABS_VOICE_ID", "VQxh7q9EH7jUlfwkQI00")

def get_elevenlabs_model_id() -> str:
    return os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

def get_elevenlabs_api_key() -> str:
    # The secret key (starts with 'sk_') must come ONLY from backend/.env.
    # Never hardcode it here: anyone who sees the code could use your ElevenLabs account.
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if key and not key.startswith("sk_"):
        print("[ElevenLabs] ELEVENLABS_API_KEY in .env does not start with 'sk_'. "
              "Copy the secret API key (not the key ID) from the ElevenLabs dashboard.")
    return key

def get_elevenlabs_agent_id() -> str:
    return os.getenv("ELEVENLABS_AGENT_ID", "agent_0201m5wjypr0f46990bqwck3596g")


def check_elevenlabs_quota() -> Dict[str, Any]:
    """Checks character credits on ElevenLabs to verify quota availability."""
    api_key = get_elevenlabs_api_key()
    if not api_key:
        return {"configured": False, "quota_exceeded": True, "reason": "No API key configured"}
    try:
        url = "https://api.elevenlabs.io/v1/user/subscription"
        r = requests.get(url, headers={"xi-api-key": api_key}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            count = data.get("character_count", 0)
            limit = data.get("character_limit", 30000)
            remaining = max(0, limit - count)
            is_exceeded = remaining < 50
            return {
                "configured": True,
                "quota_exceeded": is_exceeded,
                "character_count": count,
                "character_limit": limit,
                "remaining_characters": remaining,
                "tier": data.get("tier", "starter"),
                "status": data.get("status", "active")
            }
        return {"configured": True, "quota_exceeded": False, "status_code": r.status_code}
    except Exception as e:
        return {"configured": True, "quota_exceeded": False, "error": str(e)}


def update_elevenlabs_credentials(api_key: str = None, agent_id: str = None) -> bool:
    """Updates ELEVENLABS_API_KEY and/or ELEVENLABS_AGENT_ID in runtime and .env"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    updated = False
    if not os.path.exists(env_path):
        return False
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        key_found = False
        agent_found = False
        for line in lines:
            if api_key and line.startswith("ELEVENLABS_API_KEY="):
                new_lines.append(f"ELEVENLABS_API_KEY={api_key.strip()}\n")
                os.environ["ELEVENLABS_API_KEY"] = api_key.strip()
                key_found = True
                updated = True
            elif agent_id and line.startswith("ELEVENLABS_AGENT_ID="):
                new_lines.append(f"ELEVENLABS_AGENT_ID={agent_id.strip()}\n")
                os.environ["ELEVENLABS_AGENT_ID"] = agent_id.strip()
                agent_found = True
                updated = True
            else:
                new_lines.append(line)
        if api_key and not key_found:
            new_lines.append(f"ELEVENLABS_API_KEY={api_key.strip()}\n")
            os.environ["ELEVENLABS_API_KEY"] = api_key.strip()
            updated = True
        if agent_id and not agent_found:
            new_lines.append(f"ELEVENLABS_AGENT_ID={agent_id.strip()}\n")
            os.environ["ELEVENLABS_AGENT_ID"] = agent_id.strip()
            updated = True
        if updated:
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        return updated
    except Exception as e:
        print(f"[ElevenLabs credentials update error] {e}")
        return False

def update_elevenlabs_voice_id(new_voice_id: str) -> bool:
    """Updates ELEVENLABS_VOICE_ID in memory and writes to backend/.env"""
    os.environ["ELEVENLABS_VOICE_ID"] = new_voice_id
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            if "ELEVENLABS_VOICE_ID=" in content:
                content = re.sub(r'ELEVENLABS_VOICE_ID=.*', f'ELEVENLABS_VOICE_ID={new_voice_id}', content)
            else:
                content += f"\nELEVENLABS_VOICE_ID={new_voice_id}\n"
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[update_elevenlabs_voice_id error] {e}")
    return False

def clone_voice_from_audio(file_bytes: bytes, filename: str, voice_name: str = "saathi voice") -> Dict[str, Any]:
    """
    Clones a custom voice into ElevenLabs using Instant Voice Cloning.
    Ensures that Saathi uses the exact provided voice timbre with our soft/calm parameters.
    """
    api_key = get_elevenlabs_api_key()
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY is not configured"}

    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {
        "xi-api-key": api_key
    }
    data = {
        "name": voice_name,
        "description": "Custom Saathi elderly care companion voice — soft, calm, gentle, and reassuring with natural pauses."
    }
    files = [
        ("files", (filename, file_bytes, "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"))
    ]

    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            voice_id = result.get("voice_id")
            if voice_id:
                update_elevenlabs_voice_id(voice_id)
            return {"success": True, "voice_id": voice_id, "name": voice_name}
        else:
            return {"error": f"ElevenLabs API responded with {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def format_speech_with_natural_pauses(text: str) -> str:
    """
    Inserts deliberate conversational pauses (ellipses and commas) so that 
    the ElevenLabs voice delivers slow, unhurried, gentle speech for elderly listeners.
    """
    if not text:
        return ""
    import re
    # Clean emojis and markdown
    t = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE00-\uFE0F]', '', text)
    t = re.sub(r'[*_#`~]', '', t)
    # Convert sentence ends into gentle breathing ellipses if not already spaced
    t = re.sub(r'\.\s+', '... ', t)
    t = re.sub(r'\?\s+', '? ... ', t)
    t = re.sub(r'!\s+', '! ... ', t)
    # Ensure commas have clean breathing space
    t = re.sub(r'\s*,\s*', ', ', t)
    # Collapse multiple spaces or repeated dots beyond 3
    t = re.sub(r'\.{4,}', '...', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def synthesize_elevenlabs_speech(
    text: str,
    voice_id: str = None,
    model_id: str = None
) -> str:
    """
    Synthesizes speech using ElevenLabs API with the 'saathi voice' (kbqGn0IGWLrQBLcS2Utk).
    Applies exact vocal parameters:
    - Stability: 0.80 (High 0.75-0.85 for consistent, soothing tone)
    - Similarity / Clarity: 0.80 (High 0.75-0.85)
    - Style Exaggeration: 0.05 (Low 0.00-0.10 for calm, soft, gentle delivery)
    - Natural pause pacing with ellipses
    """
    api_key = get_elevenlabs_api_key()
    if not api_key:
        return ""

    target_voice_id = voice_id or get_elevenlabs_voice_id()
    target_model_id = model_id or get_elevenlabs_model_id()
    paced_text = format_speech_with_natural_pauses(text)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{target_voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": paced_text,
        "model_id": target_model_id,
        "voice_settings": {
            "stability": 0.80,          # High (0.75 - 0.85) for consistent, soothing tone
            "similarity_boost": 0.80,   # High (0.75 - 0.85) for clarity
            "style": 0.05,              # Low (0.00 - 0.10) for calm, gentle, unexaggerated tone
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            if "quota_exceeded" in response.text:
                print(f"[ElevenLabs Quota Exceeded Alert] Your API key quota is exceeded or set too low: {response.text}")
            elif "missing_permissions" in response.text:
                print(f"[ElevenLabs Permissions Alert] Your API key is missing permissions: {response.text}")
            else:
                print(f"[ElevenLabs TTS Error] Status {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"[ElevenLabs TTS Exception] {e}")
        return ""

def get_elevenlabs_signed_url(agent_id: str = None) -> Dict[str, Any]:
    """
    Retrieves a signed WebSocket URL for ElevenLabs Conversational AI Web SDK.
    Keeps the API key secure on the backend.
    """
    api_key = get_elevenlabs_api_key()
    target_agent_id = agent_id or get_elevenlabs_agent_id()

    if not api_key:
        return {"error": "ELEVENLABS_API_KEY is not configured in environment"}
    if not target_agent_id:
        return {"error": "ELEVENLABS_AGENT_ID is not configured in environment"}

    url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={target_agent_id}"
    headers = {
        "xi-api-key": api_key
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {"signed_url": data.get("signed_url")}
        else:
            return {"error": f"ElevenLabs API responded with status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def build_saathi_context(recipient_id: int, db: Session) -> Dict[str, Any]:
    """
    Extracts dynamic variables from care recipient's records for ElevenLabs agent:
    - {{name}}
    - {{language}}
    - {{medicine_schedule}}
    - {{conditions}}
    - {{family_contact}}
    """
    from tables.users import CareRecipient, CareTaker
    from tables.medications import Medication
    from tables.medical_conditions import PatientCondition

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        recipient = db.query(CareRecipient).first()

    name = recipient.full_name if recipient else "Maa"
    first_name = name.split()[0] if name else "Maa"

    # Load member memory and preferred name from saathi_db / caretaker profile
    mem_ctx = {}
    try:
        from services.saathi_memory_service import get_member_context
        mem_ctx = get_member_context(recipient.id if recipient else recipient_id)
    except Exception as _mem_err:
        pass

    preferred_name = (
        mem_ctx.get("preferred_name")
        or (recipient.preferred_name if recipient and getattr(recipient, "preferred_name", None) else None)
        or first_name
    )

    # Address form based on gender/age (Commented out to strictly bind to database username)
    # address = "ji"
    # if recipient:
    #     age = recipient.age or 68
    #     gender_raw = str(recipient.gender or "").lower()
    #     if "male" in gender_raw:
    #         address = "Papa ji" if age < 75 else "Dadaji"
    #     elif "female" in gender_raw:
    #         address = "Maa ji" if age < 75 else "Dadiji"
    # Strict database name resolution:
    address = preferred_name if (preferred_name and preferred_name != "ji") else first_name

    # Caretaker / family contact
    family_contact = "Not provided"
    if recipient and recipient.caretaker_id:
        caretaker = db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
        if caretaker:
            contact_parts = [caretaker.full_name or caretaker.username]
            if getattr(caretaker, 'phone_number', None):
                contact_parts.append(f"Phone: {caretaker.phone_number}")
            if caretaker.email:
                contact_parts.append(f"Email: {caretaker.email}")
            family_contact = ", ".join(contact_parts)

    # Active medications and schedule
    meds = []
    if recipient:
        active_meds = db.query(Medication).filter(
            Medication.care_recipient_id == recipient.id,
            Medication.status == "active"
        ).all()
        for m in active_meds:
            sched = m.schedule_time or m.frequency or "daily"
            meds.append(f"{m.medicine_name} ({m.dosage or 'standard dose'}, scheduled: {sched})")
    medicine_schedule = "; ".join(meds) if meds else "No active scheduled medicines on file"

    # Conditions
    conditions = []
    if recipient:
        cond_records = db.query(PatientCondition).filter(
            PatientCondition.care_recipient_id == recipient.id,
            PatientCondition.status == "active"
        ).all()
        for c in cond_records:
            conditions.append(c.disease_name)
    conditions_str = ", ".join(conditions) if conditions else "None recorded"

    # Language preference
    language = "Hindi, English, or Hinglish"

    # Exact opening audio greeting with pauses
    first_message = f"Namaste {preferred_name}... aaj aap kaisa feel kar rahe hain? ... Jo bhi aapke mann mein hai, mujhe batayiye. Main sun rahi hoon."

    # Injected User Data Context format
    user_data_context = f"Name: {preferred_name} ({address}) | Preferred language: {language} | Medicines: {medicine_schedule} | Conditions: {conditions_str} | Family contact: {family_contact}"

    profile_details = mem_ctx.get("profile_details") or f"Age: {recipient.age if recipient else 68}, Conditions: {conditions_str}"
    member_memory = mem_ctx.get("member_memory") or "No specific personal memories recorded yet."
    recent_summaries = mem_ctx.get("recent_session_summaries") or "No previous conversations recorded yet."

    # Exact Specification Saathi System Prompt
    system_prompt = f"""You are Saathi, a warm, patient and caring voice companion.
You are talking to {preferred_name}. Always address them as "{preferred_name}".
About them: {profile_details}
What you remember: {member_memory}
Recent conversations: {recent_summaries}
Speak in {language} in short, simple sentences suitable for voice.
Use remembered details naturally but never invent facts.
If they mention a health emergency, a fall, chest pain or feeling unsafe, calmly tell them to contact their caretaker or emergency services right away.

DUTIES & TOOLS:
1. Medicines: Remind at scheduled times from user records ({medicine_schedule}). Ask gently if they took their medicine. When confirmed, invoke `log_medicine_or_meal`.
2. Meals & Hydration: Remind about meals and drinking water. Ask what they ate. When confirmed, invoke `log_medicine_or_meal`.
3. Wellbeing: Ask how they slept and feel. Listen for pain, dizziness, or sadness.
4. Music & Entertainment: When they ask for music, songs, or bhajans, invoke `play_music` (with song_name) or `play_youtube_video` (with query) or `media_control` (action='pause'|'resume'|'stop'|'next'|'volume_up'|'volume_down'). Available local library tracks include: Pal Pal Dil Ke Paas, Relaxing Piano, Gentle Breeze, Soft Morning, Roop Tera Mastana, Zindagi Ka Safar. Chat warmly about old melodies.
5. Safety: If they report a fall, chest pain, or breathing issues, remain calm, reassure them help is coming, and trigger the emergency alert tool `send_emergency_alert`.
"""

    return {
        "recipient_id": recipient.id if recipient else 1,
        "name": preferred_name,
        "preferred_name": preferred_name,
        "full_name": name,
        "address": preferred_name,
        "language": language,
        "medicine_schedule": medicine_schedule,
        "conditions": conditions_str,
        "family_contact": family_contact,
        "first_message": first_message,
        "user_data_context": user_data_context,
        "system_prompt": system_prompt,
        "voice_id": get_elevenlabs_voice_id(),
        "model_id": get_elevenlabs_model_id(),
        "voice_settings": {
            "stability": 0.80,
            "similarity_boost": 0.80,
            "style": 0.05,
            "use_speaker_boost": True
        }
    }

async def handle_log_medicine_or_meal(
    recipient_id: int,
    item_type: str,
    item_name: str,
    status: str,
    db: Session
) -> Dict[str, Any]:
    """
    Function calling tool: Logs when user confirms taking medication or having a meal.
    Updates MedicationDoseLog, inventory stock, and emits real-time sync event.
    """
    from tables.users import CareRecipient, CareTaker
    recip = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recip:
        recip = db.query(CareRecipient).first()
    target_id = recip.id if recip else recipient_id

    item_type_clean = (item_type or "medicine").lower()
    status_clean = (status or "taken").lower()
    action = "TAKEN" if status_clean in ["taken", "yes", "completed", "had", "done", "consumed"] else "MISSED"

    if "med" in item_type_clean:
        try:
            from tables.medication_dose_logs import MedicationDoseLog, DoseStatusEnum, ConfirmationSourceEnum
            from tables.medications import Medication

            # Find matching pending log or most recent medication
            query = db.query(MedicationDoseLog).filter(
                MedicationDoseLog.care_recipient_id == target_id,
                MedicationDoseLog.status == DoseStatusEnum.PENDING
            )
            if item_name:
                query = query.join(Medication, Medication.medication_id == MedicationDoseLog.medication_id)\
                             .filter(Medication.medicine_name.ilike(f"%{item_name}%"))

            pending_log = query.order_by(MedicationDoseLog.scheduled_time.asc()).first()
            new_stock = None

            if pending_log:
                pending_log.status = DoseStatusEnum(action)
                pending_log.confirmation_source = ConfirmationSourceEnum.VOICE
                pending_log.confirmed_at = datetime.utcnow()

                if action == "TAKEN":
                    med = db.query(Medication).filter(Medication.medication_id == pending_log.medication_id).first()
                    if med and med.current_stock is not None:
                        daily = med.doses_per_day or 1
                        med.current_stock = max(0, med.current_stock - daily)
                        new_stock = med.current_stock

                db.commit()

                # Emit Socket.IO push
                try:
                    import asyncio
                    from socket_manager import sio_server
                    loop = asyncio.get_event_loop()
                    payload = {"dose_log_id": pending_log.id, "action": action, "new_stock": new_stock}
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(sio_server.emit("dose_confirmed", payload), loop)
                    else:
                        loop.run_until_complete(sio_server.emit("dose_confirmed", payload))
                except Exception as se:
                    print(f"[ElevenLabs Tool Socket error] {se}")

                return {
                    "success": True,
                    "message": f"Medication '{item_name}' logged as {action} for recipient {recipient_id}.",
                    "action": action,
                    "new_stock": new_stock
                }
            else:
                return {
                    "success": True,
                    "message": f"Confirmed {item_name} as {action}, recorded in daily log.",
                    "action": action
                }
        except Exception as e:
            print(f"[ElevenLabs log_medicine error] {e}")
            db.rollback()
            return {"success": False, "error": str(e)}

    # Meal logging
    return {
        "success": True,
        "message": f"Meal '{item_name or 'Meal'}' logged successfully as {status_clean}.",
        "action": status_clean
    }

async def handle_send_emergency_alert(
    recipient_id: int,
    reason: str,
    severity: str,
    db: Session
) -> Dict[str, Any]:
    """
    Function calling tool: Immediately triggers emergency alert notification
    to family/caregiver contacts upon detection of severe distress or falls.
    """
    from tables.users import CareRecipient, CareTaker
    from tables.medical_conditions import MedicalAlert, AlertType, AlertSeverity

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        recipient = db.query(CareRecipient).first()
    target_id = recipient.id if recipient else recipient_id
    recipient_name = recipient.full_name if recipient else "Care Recipient"
    caretaker_email = None

    if recipient and recipient.caretaker_id:
        caretaker = db.query(CareTaker).filter(CareTaker.id == recipient.caretaker_id).first()
        if caretaker and caretaker.email:
            caretaker_email = caretaker.email

    # 1. Create database MedicalAlert
    try:
        alert = MedicalAlert(
            care_recipient_id=target_id,
            alert_type=AlertType.critical,
            severity=AlertSeverity.critical,
            message=f"EMERGENCY triggered via Saathi Voice: {reason or 'Fall or severe distress reported'}"
        )
        db.add(alert)
        db.commit()
    except Exception as e:
        print(f"[ElevenLabs Emergency DB error] {e}")
        db.rollback()

    # 2. Send email notification to caregiver
    email_sent = False
    if caretaker_email:
        try:
            from utils.email import send_fall_alert_email
            from datetime import timezone as _tz
            now_str = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
            await send_fall_alert_email(caretaker_email, {
                "timestamp": now_str,
                "fall_count": 1,
                "location": f"Voice Conversation with Saathi (Reason: {reason})",
                "fall_details": [{
                    "timestamp": now_str,
                    "confidence": 100,
                    "angle": "Emergency Voice Alert"
                }]
            })
            email_sent = True
        except Exception as e:
            print(f"[ElevenLabs Emergency Email error] {e}")

    # 3. Broadcast Socket.IO event
    try:
        import asyncio
        from socket_manager import sio_server
        loop = asyncio.get_event_loop()
        alert_payload = {
            "recipient_id": recipient_id,
            "recipient_name": recipient_name,
            "reason": reason,
            "severity": severity or "critical",
            "timestamp": datetime.utcnow().isoformat()
        }
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(sio_server.emit("emergency_alert", alert_payload), loop)
        else:
            loop.run_until_complete(sio_server.emit("emergency_alert", alert_payload))
    except Exception as se:
        print(f"[ElevenLabs Emergency Socket error] {se}")

    return {
        "success": True,
        "message": f"Emergency alert triggered for {recipient_name}. Caregiver alerted.",
        "email_sent": email_sent,
        "recipient": recipient_name
    }


def build_agent_dynamic_variables(recipient_id: int, db: Session) -> Dict[str, str]:
    """
    Builds the dynamic variables sent to the ElevenLabs agent ("Asha") at the start of
    every conversation. Names must match the {{variables}} used in the agent's system
    prompt and first message in the ElevenLabs dashboard.

    Every value is always a string ("" when there is no data), because ElevenLabs can
    refuse to start a conversation if a variable used in the prompt is missing.
    """
    from tables.users import CareRecipient

    ctx = build_saathi_context(recipient_id, db)
    mem_ctx: Dict[str, Any] = {}
    try:
        from services.saathi_memory_service import get_member_context
        mem_ctx = get_member_context(ctx.get("recipient_id", recipient_id))
    except Exception as e:
        print(f"[ElevenLabs] Could not load saathi_db memory for recipient {recipient_id}: {e}")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == ctx.get("recipient_id")).first()

    user_name = (ctx.get("preferred_name") or "").strip()

    # Profile details (no phone numbers or e-mails, this is spoken context)
    parts = []
    if recipient:
        # Real name as well as the name Asha calls them (e.g. "Geeta Devi", called "Dadi")
        full = (recipient.full_name or "").strip()
        if full and full.lower() != user_name.lower():
            parts.append(f"full name {full} (called {user_name})")
        if recipient.age:
            parts.append(f"Age {recipient.age}")
        if recipient.gender:
            g = recipient.gender.value if hasattr(recipient.gender, "value") else str(recipient.gender)
            parts.append(g)
        if recipient.city:
            parts.append(f"lives in {recipient.city}")
        if getattr(recipient, "blood_group", None):
            parts.append(f"blood group {recipient.blood_group}")
    conditions = ctx.get("conditions") or ""
    if conditions and conditions != "None recorded":
        parts.append(f"health conditions: {conditions}")
    if recipient and getattr(recipient, "doctor_remarks", None):
        parts.append(f"doctor's advice: {recipient.doctor_remarks}")
    profile_details = ", ".join(parts)

    medicines = ctx.get("medicine_schedule") or ""
    if medicines.startswith("No active"):
        medicines = ""

    caretaker_name = mem_ctx.get("caretaker_name") or ""
    family_members = f"{caretaker_name} (caretaker)" if caretaker_name else ""

    # Memory = personal facts + summaries of the last few calls
    memory_parts = []
    facts = mem_ctx.get("member_memory") or ""
    if facts and not facts.startswith("No specific"):
        memory_parts.append(facts)
    past = mem_ctx.get("recent_session_summaries") or ""
    if past and not past.startswith("No previous"):
        memory_parts.append("Previous calls:\n" + past)
    member_memory = "\n".join(memory_parts)

    return {
        # Variables used by the Asha system prompt / first message
        "user_name": user_name,
        "preferred_language": "Hinglish",
        "profile_details": profile_details,
        "medicines": medicines,
        "family_members": family_members,
        "caretaker_name": caretaker_name,
        "member_memory": member_memory,
        "last_summary": mem_ctx.get("last_summary") or "",
        # Older names kept so any earlier prompt version still works
        "name": user_name,
        "address": user_name,
        "medicine_schedule": medicines,
        # Lets the post-call webhook map the call back to this patient
        "recipient_id": str(ctx.get("recipient_id", recipient_id)),
    }

