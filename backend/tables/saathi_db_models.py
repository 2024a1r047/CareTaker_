"""
SQLAlchemy models for saathi_db database:
- FamilyMember
- ConversationSession
- ConversationMessage
- MemberMemory
"""

import uuid
from sqlalchemy import (
    Column, String, Text, BigInteger, DateTime, ForeignKey, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship as orm_relationship

SaathiBase = declarative_base()


class FamilyMember(SaathiBase):
    __tablename__ = "family_members"

    id = Column(BigInteger, primary_key=True)
    account_id = Column(BigInteger, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    preferred_name = Column(String(255), nullable=True)
    relationship = Column(String(100), nullable=True)
    language = Column(String(50), default="hi")
    profile_details = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    sessions = orm_relationship("ConversationSession", back_populates="family_member", cascade="all, delete-orphan")
    messages = orm_relationship("ConversationMessage", back_populates="family_member", cascade="all, delete-orphan")
    memories = orm_relationship("MemberMemory", back_populates="family_member", cascade="all, delete-orphan")


class ConversationSession(SaathiBase):
    __tablename__ = "conversation_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(BigInteger, nullable=False, index=True)
    family_member_id = Column(BigInteger, ForeignKey("family_members.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(Text, nullable=True)

    family_member = orm_relationship("FamilyMember", back_populates="sessions")
    messages = orm_relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan", order_by="ConversationMessage.created_at")


class ConversationMessage(SaathiBase):
    __tablename__ = "conversation_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    family_member_id = Column(BigInteger, ForeignKey("family_members.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    language = Column(String(20), default="hi")
    audio_url = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = orm_relationship("ConversationSession", back_populates="messages")
    family_member = orm_relationship("FamilyMember", back_populates="messages")

    __table_args__ = (
        Index("idx_conv_messages_member_created", "family_member_id", "created_at"),
    )


class MemberMemory(SaathiBase):
    __tablename__ = "member_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_member_id = Column(BigInteger, ForeignKey("family_members.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(50), nullable=False)  # 'health' | 'medicine' | 'family' | 'likes' | 'dislikes' | 'routine' | 'mood'
    fact = Column(Text, nullable=False)
    source_message_id = Column(UUID(as_uuid=True), ForeignKey("conversation_messages.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    family_member = orm_relationship("FamilyMember", back_populates="memories")
    source_message = orm_relationship("ConversationMessage")

    __table_args__ = (
        Index("idx_member_memory_member_cat", "family_member_id", "category"),
    )
