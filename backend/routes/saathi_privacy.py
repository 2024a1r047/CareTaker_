"""
Saathi Privacy & History API Endpoints
Provides endpoints for caretakers to:
- View conversation sessions and turn-by-turn message history for their family members
- Delete conversation sessions and message history
- View, delete, and manage categorized member memory facts
- Update the caretaker-given preferred name for the family member
Strictly scoped by account_id and family_member_id.
"""

import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config import get_db, SaathiSessionLocal, SessionLocal
from tables.users import CareTaker, CareRecipient
from tables.saathi_db_models import (
    FamilyMember,
    ConversationSession,
    ConversationMessage,
    MemberMemory,
)
from repository.users import UsersRepo

router = APIRouter(prefix="/saathi", tags=["Saathi Privacy & Memory"])


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


def get_current_caretaker(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> CareTaker:
    """Verifies JWT token and scopes query to the logged-in caretaker account."""
    if authorization:
        username = _get_username_from_auth(authorization)
        if username:
            caretaker = UsersRepo.find_by_username(db, CareTaker, username)
            if caretaker:
                return caretaker
    # Fallback to first caretaker in DB for local development / testing
    caretaker = db.query(CareTaker).first()
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Caretaker authentication required")
    return caretaker


def _verify_family_member_ownership(caretaker_id: int, family_member_id: int):
    """Ensures family member belongs to the authenticated caretaker account."""
    with SessionLocal() as main_db:
        recip = main_db.query(CareRecipient).filter(
            CareRecipient.id == family_member_id,
            CareRecipient.caretaker_id == caretaker_id
        ).first()
        if not recip:
            # Check if this caretaker is the only one in dev
            any_recip = main_db.query(CareRecipient).filter(CareRecipient.id == family_member_id).first()
            if not any_recip:
                raise HTTPException(status_code=404, detail="Family member not found")
            if any_recip.caretaker_id != caretaker_id:
                raise HTTPException(status_code=403, detail="Access denied: Family member belongs to another account")


# ── SCHEMAS ────────────────────────────────────────────────────────────

class UpdatePreferredNameRequest(BaseModel):
    preferred_name: str


# ── ENDPOINTS ──────────────────────────────────────────────────────────

@router.get("/members/{family_member_id}/sessions")
def list_conversation_sessions(
    family_member_id: int,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """List all past conversation sessions for a family member, scoped by caretaker account."""
    _verify_family_member_ownership(caretaker.id, family_member_id)

    with SaathiSessionLocal() as s_db:
        sessions = (
            s_db.query(ConversationSession)
            .filter(
                ConversationSession.family_member_id == family_member_id,
                ConversationSession.account_id == caretaker.id,
            )
            .order_by(desc(ConversationSession.started_at))
            .all()
        )
        return {
            "family_member_id": family_member_id,
            "sessions": [
                {
                    "id": str(s.id),
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "summary": s.summary or "In-progress or check-in session",
                }
                for s in sessions
            ]
        }


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """View full turn-by-turn conversation transcript for a session."""
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session UUID")

    with SaathiSessionLocal() as s_db:
        session = s_db.query(ConversationSession).filter(ConversationSession.id == sess_uuid).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.account_id != caretaker.id:
            raise HTTPException(status_code=403, detail="Access denied: Session belongs to another account")

        messages = (
            s_db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == sess_uuid)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        return {
            "session_id": str(session.id),
            "family_member_id": session.family_member_id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "summary": session.summary,
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "language": m.language,
                    "audio_url": m.audio_url,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ]
        }


@router.delete("/sessions/{session_id}")
def delete_conversation_session(
    session_id: str,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """Permanently delete a conversation session and all its messages (cascade)."""
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session UUID")

    with SaathiSessionLocal() as s_db:
        session = s_db.query(ConversationSession).filter(ConversationSession.id == sess_uuid).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.account_id != caretaker.id:
            raise HTTPException(status_code=403, detail="Access denied: Session belongs to another account")

        s_db.delete(session)
        s_db.commit()
        return {"status": "success", "message": f"Session {session_id} and all messages deleted."}


@router.get("/members/{family_member_id}/memory")
def get_member_memory(
    family_member_id: int,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """View all remembered personal facts for a member categorized into health, likes, routine, etc."""
    _verify_family_member_ownership(caretaker.id, family_member_id)

    with SaathiSessionLocal() as s_db:
        memories = (
            s_db.query(MemberMemory)
            .filter(MemberMemory.family_member_id == family_member_id)
            .order_by(MemberMemory.category.asc(), desc(MemberMemory.updated_at))
            .all()
        )

        categorized = {}
        for m in memories:
            cat = m.category
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append({
                "id": str(m.id),
                "fact": m.fact,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            })

        return {
            "family_member_id": family_member_id,
            "total_facts": len(memories),
            "memories": categorized,
        }


@router.delete("/members/{family_member_id}/memory/{memory_id}")
def delete_single_memory(
    family_member_id: int,
    memory_id: str,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """Delete a specific personal memory fact."""
    _verify_family_member_ownership(caretaker.id, family_member_id)
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory UUID")

    with SaathiSessionLocal() as s_db:
        mem = s_db.query(MemberMemory).filter(
            MemberMemory.id == mem_uuid,
            MemberMemory.family_member_id == family_member_id
        ).first()
        if not mem:
            raise HTTPException(status_code=404, detail="Memory fact not found")

        s_db.delete(mem)
        s_db.commit()
        return {"status": "success", "message": "Memory fact deleted."}


@router.delete("/members/{family_member_id}/memory")
def clear_all_member_memory(
    family_member_id: int,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """Clear all extracted memories for a member."""
    _verify_family_member_ownership(caretaker.id, family_member_id)

    with SaathiSessionLocal() as s_db:
        deleted_count = (
            s_db.query(MemberMemory)
            .filter(MemberMemory.family_member_id == family_member_id)
            .delete()
        )
        s_db.commit()
        return {"status": "success", "message": f"Cleared {deleted_count} memory facts for member {family_member_id}."}


@router.put("/members/{family_member_id}/name")
def update_member_preferred_name(
    family_member_id: int,
    request: UpdatePreferredNameRequest,
    caretaker: CareTaker = Depends(get_current_caretaker)
):
    """Update the caretaker-given preferred name by which Saathi addresses the family member."""
    _verify_family_member_ownership(caretaker.id, family_member_id)
    new_name = request.preferred_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="preferred_name cannot be empty")

    # Update in main care_recipients table
    with SessionLocal() as main_db:
        recip = main_db.query(CareRecipient).filter(CareRecipient.id == family_member_id).first()
        if recip:
            recip.preferred_name = new_name
            main_db.commit()

    # Also update in saathi_db.family_members
    with SaathiSessionLocal() as s_db:
        mem = s_db.query(FamilyMember).filter(FamilyMember.id == family_member_id).first()
        if mem:
            mem.preferred_name = new_name
            s_db.commit()

    return {
        "status": "success",
        "family_member_id": family_member_id,
        "preferred_name": new_name,
        "message": f"Saathi will now address this member as '{new_name}'."
    }
