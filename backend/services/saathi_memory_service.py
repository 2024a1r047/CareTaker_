"""
Saathi Memory & Conversation Service
Handles:
1. Conversation session lifecycle and async non-blocking message logging in saathi_db
2. Member memory extraction and upserting (health, medicine, family, likes, dislikes, routine, mood)
3. Name resolution (preferred_name with fallback) and standardized system prompt generation
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from config import SaathiSessionLocal, SessionLocal
from tables.saathi_db_models import (
    FamilyMember,
    ConversationSession,
    ConversationMessage,
    MemberMemory,
)
from tables.users import CareRecipient, CareTaker
from utils.gemini_client import call_gemini, safe_json_parse

logger = logging.getLogger(__name__)

# Valid categories per specification
VALID_CATEGORIES = {
    "health",
    "medicine",
    "family",
    "likes",
    "dislikes",
    "routine",
    "mood",
}

SYSTEM_PROMPT_TEMPLATE = """You are Saathi, a warm, patient and caring voice companion.
You are talking to {{preferred_name}}. Always address them as "{{preferred_name}}".
About them: {{profile_details}}
What you remember: {{member_memory}}
Recent conversations: {{recent_session_summaries}}
Speak in {{preferred_language}} in short, simple sentences suitable for voice.
Use remembered details naturally but never invent facts.
If they mention a health emergency, a fall, chest pain or feeling unsafe, calmly tell them to contact their caretaker or emergency services right away.

PERSONAL PROFILE ACCESS & INSTRUCTIONS:
- You have full access to {{preferred_name}}'s personal profile information above (full name, medicines and schedule, diagnosed conditions, known allergies, emergency contacts, vital signs, and doctor remarks).
- When {{preferred_name}} asks about their personal details, medicines, schedule, allergies, doctor's remarks, blood group, or emergency contact, answer directly, accurately, and reassuringly from their profile.
- Always address them as "{{preferred_name}}".
"""


def _to_valid_uuid(val: Union[str, uuid.UUID]) -> uuid.UUID:
    """Safely converts string or UUID to a valid UUID without throwing ValueError for custom session IDs."""
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, AttributeError):
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))


def _ensure_family_member_exists(s_db: Session, family_member_id: int, account_id: Optional[int] = None) -> FamilyMember:
    """Ensures family_member record exists in saathi_db, syncing from care_recipients if needed."""
    member = s_db.query(FamilyMember).filter(FamilyMember.id == family_member_id).first()
    if member:
        return member

    # Look up in main caretaker database
    with SessionLocal() as main_db:
        recip = main_db.query(CareRecipient).filter(CareRecipient.id == family_member_id).first()
        if recip:
            act_id = account_id or recip.caretaker_id or 1
            member = FamilyMember(
                id=recip.id,
                account_id=act_id,
                full_name=recip.full_name or "Family Member",
                preferred_name=recip.preferred_name,
                language="hi",
                profile_details={
                    "age": recip.age,
                    "gender": str(recip.gender) if recip.gender else None,
                    "city": recip.city,
                    "report_summary": recip.report_summary,
                }
            )
            s_db.add(member)
            s_db.commit()
            s_db.refresh(member)
            return member

    # Fallback placeholder if not found
    member = FamilyMember(
        id=family_member_id,
        account_id=account_id or 1,
        full_name="Family Member",
        preferred_name="Family Member",
        language="hi",
        profile_details={}
    )
    s_db.add(member)
    s_db.commit()
    s_db.refresh(member)
    return member


def start_session(account_id: int, family_member_id: int) -> uuid.UUID:
    """Creates a new conversation session in saathi_db."""
    with SaathiSessionLocal() as s_db:
        _ensure_family_member_exists(s_db, family_member_id, account_id)
        session = ConversationSession(
            account_id=account_id,
            family_member_id=family_member_id,
            started_at=datetime.now(timezone.utc),
        )
        s_db.add(session)
        s_db.commit()
        s_db.refresh(session)
        logger.info(f"[Saathi DB] Started session {session.id} for member {family_member_id}")
        return session.id


def save_message_sync(
    session_id: Union[str, uuid.UUID],
    family_member_id: int,
    role: str,
    content: str,
    language: str = "hi",
    audio_url: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[uuid.UUID]:
    """Saves a conversation message in saathi_db with retries."""
    sess_uuid = _to_valid_uuid(session_id)
    role_normalized = "assistant" if role.lower() in ("assistant", "bot", "saathi") else "user"

    for attempt in range(max_retries):
        try:
            with SaathiSessionLocal() as s_db:
                # Ensure session exists
                sess = s_db.query(ConversationSession).filter(ConversationSession.id == sess_uuid).first()
                if not sess:
                    _ensure_family_member_exists(s_db, family_member_id)
                    sess = ConversationSession(
                        id=sess_uuid,
                        account_id=1,
                        family_member_id=family_member_id,
                        started_at=datetime.now(timezone.utc),
                    )
                    s_db.add(sess)
                    s_db.commit()

                msg = ConversationMessage(
                    session_id=sess_uuid,
                    family_member_id=family_member_id,
                    role=role_normalized,
                    content=content,
                    language=language,
                    audio_url=audio_url,
                    created_at=datetime.now(timezone.utc),
                )
                s_db.add(msg)
                s_db.commit()
                s_db.refresh(msg)
                return msg.id
        except Exception as e:
            logger.warning(f"[Saathi DB] save_message retry {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (2 ** attempt))
            else:
                logger.error(f"[Saathi DB] Failed to save message after {max_retries} attempts: {e}")
                return None


def save_message_async(
    session_id: Union[str, uuid.UUID],
    family_member_id: int,
    role: str,
    content: str,
    language: str = "hi",
    audio_url: Optional[str] = None,
):
    """Fire-and-forget non-blocking message save so real-time voice latency is zero."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        loop.run_in_executor(
            None,
            save_message_sync,
            session_id,
            family_member_id,
            role,
            content,
            language,
            audio_url,
        )
    else:
        import threading
        t = threading.Thread(
            target=save_message_sync,
            args=(session_id, family_member_id, role, content, language, audio_url),
            daemon=True,
        )
        t.start()


def generate_session_summary(messages: List[Dict[str, str]]) -> str:
    """Generates a concise 2-3 sentence summary of the conversation using Gemini."""
    if not messages:
        return "Brief check-in with no messages."

    transcript_lines = []
    for m in messages:
        r = "User" if m["role"] == "user" else "Saathi"
        transcript_lines.append(f"{r}: {m['content']}")
    transcript = "\n".join(transcript_lines)

    prompt = f"""Summarize this elderly care conversation in 2-3 concise, caring sentences.
Focus on: what the member shared (feelings, health, activities) and any topics discussed.

Transcript:
{transcript}

Summary:"""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 150, "temperature": 0.3}
        }
        res = call_gemini(payload, timeout=15, caller="[session_summary]")
        if res and res.get("candidates"):
            text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text
    except Exception as e:
        logger.warning(f"Failed to generate session summary with Gemini: {e}")

    # Fallback summary
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    if user_msgs:
        return f"Member talked about: {', '.join(user_msgs[:3])}."
    return "Check-in conversation completed."


def extract_and_upsert_memory(session_id: Union[str, uuid.UUID], family_member_id: int):
    """
    Extracts new personal facts from the session and upserts into member_memory.
    Categories: health, medicine, family, likes, dislikes, routine, mood.
    Never overwrites caretaker-entered profile fields.
    """
    sess_uuid = _to_valid_uuid(session_id)

    with SaathiSessionLocal() as s_db:
        messages = (
            s_db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == sess_uuid)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        if not messages:
            return

        user_turns = [
            {"id": str(m.id), "content": m.content}
            for m in messages
            if m.role == "user" and len(m.content.strip()) > 3
        ]
        if not user_turns:
            return

        user_text_dump = "\n".join([f"[{t['id']}] {t['content']}" for t in user_turns])

        prompt = f"""Extract personal facts about this elderly person from their messages into JSON.
Categories MUST strictly be one of: "health", "medicine", "family", "likes", "dislikes", "routine", "mood".
Only extract genuine personal facts (e.g., likes ginger tea, takes blood pressure medicine in morning, daughter's name is Priya, walked in the garden).
Do NOT include trivial greetings like "namaste" or "theek hoon".

Return ONLY a JSON list of objects:
[
  {{"category": "category_name", "fact": "clear concise fact", "source_id": "message_id_from_brackets"}}
]

User messages:
{user_text_dump}
"""
        extracted = []
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 400, "temperature": 0.2}
            }
            res = call_gemini(payload, timeout=20, caller="[memory_extract]")
            if res and res.get("candidates"):
                raw = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                extracted = safe_json_parse(raw) or []
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")

        if not extracted or not isinstance(extracted, list):
            # Heuristic extractor fallback across all 7 categories
            import re
            extracted = []
            for t in user_turns:
                txt = t["content"].lower()
                tid = t["id"]
                if any(k in txt for k in ["dawa", "medicine", "goli", "bp", "sugar", "tablet", "capsule"]):
                    extracted.append({"category": "medicine", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["dard", "pain", "chhati", "sir", "bukhar", "fever", "knee", "ghutna", "chakkar", "dizzy"]):
                    extracted.append({"category": "health", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["beta", "beti", "pota", "poti", "bahu", "son", "daughter", "wife", "husband", "priya", "amit", "rahul"]):
                    extracted.append({"category": "family", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["pasand", "like", "accha lagta", "chai", "tea", "geet", "bhajan", "music", "song"]):
                    extracted.append({"category": "likes", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["nafrat", "dislike", "bura lagta", "khatta", "kadwa", "shor", "loud"]):
                    extracted.append({"category": "dislikes", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["subah", "shaam", "walk", "sair", "uthata", "sota", "routine", "garden"]):
                    extracted.append({"category": "routine", "fact": t["content"], "source_id": tid})
                elif any(k in txt for k in ["khush", "happy", "udas", "sad", "gham", "bored", "bore", "lonely", "alone"]):
                    extracted.append({"category": "mood", "fact": t["content"], "source_id": tid})

        # Upsert each fact
        for item in extracted:
            cat = str(item.get("category", "")).lower().strip()
            fact = str(item.get("fact", "")).strip()
            src_id_str = item.get("source_id")
            source_uuid = None
            if src_id_str:
                try:
                    source_uuid = uuid.UUID(str(src_id_str).strip("[] "))
                except Exception:
                    pass

            if cat not in VALID_CATEGORIES or not fact:
                continue

            # Look for existing fact in same category for smart upsert
            existing = (
                s_db.query(MemberMemory)
                .filter(
                    MemberMemory.family_member_id == family_member_id,
                    MemberMemory.category == cat,
                )
                .all()
            )

            # Check for keyword overlap
            matched = None
            fact_words = set(fact.lower().split())
            for ex in existing:
                ex_words = set(ex.fact.lower().split())
                overlap = len(fact_words.intersection(ex_words))
                if overlap >= 3 or (len(ex_words) > 0 and overlap / len(ex_words) > 0.6):
                    matched = ex
                    break

            if matched:
                matched.fact = fact
                matched.source_message_id = source_uuid or matched.source_message_id
                matched.updated_at = datetime.now(timezone.utc)
                logger.info(f"[Saathi Memory] Updated existing {cat} fact: {fact}")
            else:
                new_mem = MemberMemory(
                    family_member_id=family_member_id,
                    category=cat,
                    fact=fact,
                    source_message_id=source_uuid,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                s_db.add(new_mem)
                logger.info(f"[Saathi Memory] Added new {cat} fact: {fact}")

        s_db.commit()


def end_session_sync(session_id: Union[str, uuid.UUID]):
    """Closes the conversation session, generates summary, and extracts memory."""
    sess_uuid = _to_valid_uuid(session_id)

    with SaathiSessionLocal() as s_db:
        session = s_db.query(ConversationSession).filter(ConversationSession.id == sess_uuid).first()
        if not session:
            logger.warning(f"[Saathi DB] end_session: session {session_id} not found")
            return

        session.ended_at = datetime.now(timezone.utc)

        # Load messages for summary
        messages = (
            s_db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == sess_uuid)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        summary_text = generate_session_summary(msg_dicts)
        session.summary = summary_text
        s_db.commit()

        # Extract memories
        extract_and_upsert_memory(sess_uuid, session.family_member_id)
        logger.info(f"[Saathi DB] Closed session {session_id} with summary: {summary_text}")


def end_session(session_id: Union[str, uuid.UUID]):
    """Async wrapper to close session and extract memories in background."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        loop.run_in_executor(None, end_session_sync, session_id)
    else:
        import threading
        t = threading.Thread(target=end_session_sync, args=(session_id,), daemon=True)
        t.start()


def get_member_context(family_member_id: int) -> Dict[str, Any]:
    """
    Loads all member context required to build the standardized Saathi system prompt:
    - preferred_name (caretaker given name with fallback)
    - profile_details
    - member_memory (categorized)
    - recent_session_summaries
    """
    preferred_name = "ji"
    full_name = "Family Member"
    profile_details = "Elderly family member"
    language = "Hindi and English"

    # 1. Read caretaker-given name from care_recipients (or saathi_db.family_members)
    with SessionLocal() as main_db:
        recip = main_db.query(CareRecipient).filter(CareRecipient.id == family_member_id).first()
        if recip:
            full_name = recip.full_name or "Family Member"
            # Exact requirement: Read the caretaker-given name. Fallback: first name.
            if recip.preferred_name and recip.preferred_name.strip():
                preferred_name = recip.preferred_name.strip()
            else:
                preferred_name = full_name.split()[0] if full_name else "ji"

            details_parts = [f"Name: {full_name}"]
            if preferred_name and preferred_name != full_name:
                details_parts.append(f"Profile / Preferred Name: {preferred_name}")
            if recip.age:
                details_parts.append(f"Age: {recip.age}")
            if recip.gender:
                g_val = recip.gender.value if hasattr(recip.gender, 'value') else str(recip.gender)
                details_parts.append(f"Gender: {g_val}")
            if recip.city:
                details_parts.append(f"City: {recip.city}")
            if getattr(recip, 'blood_group', None):
                details_parts.append(f"Blood Group: {recip.blood_group}")
            if getattr(recip, 'height', None) and getattr(recip, 'weight', None):
                details_parts.append(f"Height/Weight: {recip.height} cm, {recip.weight} kg")
            if getattr(recip, 'emergency_contact', None):
                details_parts.append(f"Emergency Contact: {recip.emergency_contact}")
            if getattr(recip, 'phone_number', None):
                details_parts.append(f"Phone: {recip.phone_number}")

            if getattr(recip, 'caretaker', None):
                c_info = recip.caretaker.full_name or recip.caretaker.username
                if getattr(recip.caretaker, 'phone_number', None):
                    c_info += f" (Phone: {recip.caretaker.phone_number})"
                details_parts.append(f"Primary Caretaker: {c_info}")

            # Active medications from DB
            try:
                from tables.medications import Medication, MedicationStatus
                active_meds = main_db.query(Medication).filter(
                    Medication.care_recipient_id == recip.id,
                    Medication.status.in_([MedicationStatus.active, 'active'])
                ).all()
                if active_meds:
                    m_list = [f"{m.medicine_name} ({m.dosage or 'standard dose'}, {m.frequency or 'daily'}, time: {m.schedule_time or 'as prescribed'})" for m in active_meds]
                    details_parts.append(f"Active Medications: {'; '.join(m_list)}")
                else:
                    details_parts.append("Active Medications: None on record")
            except Exception:
                pass

            # Known Allergies from DB
            try:
                from tables.allergies import Allergy
                allergies = main_db.query(Allergy).filter(
                    Allergy.care_recipient_id == recip.id
                ).all()
                if allergies:
                    a_list = [f"{a.allergen} ({a.reaction or 'reaction'}, severity: {a.severity or 'moderate'})" for a in allergies]
                    details_parts.append(f"Known Allergies: {'; '.join(a_list)}")
                else:
                    details_parts.append("Known Allergies: None on record")
            except Exception:
                pass

            # Diagnosed Medical Conditions from DB
            try:
                from tables.medical_conditions import PatientCondition
                conds = main_db.query(PatientCondition).filter(
                    PatientCondition.care_recipient_id == recip.id,
                    PatientCondition.status == "active"
                ).all()
                if conds:
                    details_parts.append(f"Medical Conditions: {', '.join([c.disease_name for c in conds])}")
            except Exception:
                pass

            # Doctor Remarks
            if getattr(recip, 'doctor_remarks', None):
                details_parts.append(f"Doctor's Advice/Remarks: {recip.doctor_remarks}")

            # Latest Vitals
            try:
                from tables.vital_signs import VitalSign
                vitals = main_db.query(VitalSign).filter(
                    VitalSign.care_recipient_id == recip.id
                ).order_by(VitalSign.recorded_at.desc()).first()
                if vitals:
                    v_items = []
                    if vitals.systolic_bp and vitals.diastolic_bp:
                        v_items.append(f"BP {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg")
                    if vitals.heart_rate:
                        v_items.append(f"Pulse {vitals.heart_rate} bpm")
                    if vitals.oxygen_saturation:
                        v_items.append(f"SpO2 {vitals.oxygen_saturation}%")
                    if vitals.temperature:
                        v_items.append(f"Temp {vitals.temperature} F")
                    if v_items:
                        details_parts.append(f"Latest Vitals: {', '.join(v_items)}")
            except Exception:
                pass

            if recip.report_summary:
                details_parts.append(f"Health summary: {recip.report_summary}")
            profile_details = " | ".join(details_parts)

    # 2. Read memories & recent summaries from saathi_db
    memory_lines = []
    summaries = []

    with SaathiSessionLocal() as s_db:
        # Check and sync family_members preferred_name
        mem_rec = s_db.query(FamilyMember).filter(FamilyMember.id == family_member_id).first()
        if mem_rec:
            if preferred_name != "ji":
                mem_rec.preferred_name = preferred_name
                s_db.commit()
            elif mem_rec.preferred_name and mem_rec.preferred_name.strip():
                preferred_name = mem_rec.preferred_name.strip()

        # Load memories
        mems = (
            s_db.query(MemberMemory)
            .filter(MemberMemory.family_member_id == family_member_id)
            .order_by(MemberMemory.category.asc(), MemberMemory.updated_at.desc())
            .all()
        )
        for m in mems:
            memory_lines.append(f"[{m.category.capitalize()}] {m.fact}")

        # Load recent 3 session summaries
        recent_sessions = (
            s_db.query(ConversationSession)
            .filter(
                ConversationSession.family_member_id == family_member_id,
                ConversationSession.summary != None,
            )
            .order_by(ConversationSession.started_at.desc())
            .limit(3)
            .all()
        )
        for s in recent_sessions:
            date_str = s.started_at.strftime("%b %d") if s.started_at else "Recently"
            summaries.append(f"- {date_str}: {s.summary}")

    member_memory_str = "\n".join(memory_lines) if memory_lines else "No specific personal memories recorded yet."
    recent_summaries_str = "\n".join(summaries) if summaries else "No previous conversations recorded yet."

    return {
        "family_member_id": family_member_id,
        "preferred_name": preferred_name,
        "full_name": full_name,
        "profile_details": profile_details,
        "member_memory": member_memory_str,
        "recent_session_summaries": recent_summaries_str,
        "preferred_language": language,
    }


def build_system_prompt(family_member_id: int) -> str:
    """Builds the voice agent's system prompt using the exact user-specified template."""
    ctx = get_member_context(family_member_id)
    pref_name = ctx["preferred_name"]

    prompt = (
        SYSTEM_PROMPT_TEMPLATE
        .replace("{{preferred_name}}", pref_name)
        .replace("{{profile_details}}", ctx["profile_details"])
        .replace("{{member_memory}}", ctx["member_memory"])
        .replace("{{recent_session_summaries}}", ctx["recent_session_summaries"])
        .replace("{{preferred_language}}", ctx["preferred_language"])
    )
    return prompt
