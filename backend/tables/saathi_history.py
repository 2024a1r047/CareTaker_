from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
import datetime

# Independent Base for the Saathi PostgreSQL Database
SaathiBase = declarative_base()


class SaathiRecipient(SaathiBase):
    """
    Mirror/Companion record for the care recipient in the Saathi database.
    Associates conversations and preferences with the correct user across databases.
    """
    __tablename__ = "saathi_recipients"

    recipient_id = Column(Integer, primary_key=True, index=True)
    caretaker_username = Column(String, index=True, nullable=True)
    full_name = Column(String, nullable=False, default="Dadaji")
    preferred_language = Column(String, default="hi")
    address_form = Column(String, default="Dadaji")
    preferred_voice_id = Column(String, nullable=True, default="VQxh7q9EH7jUlfwkQI00")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships within the Saathi database
    sessions = relationship("ConversationSession", back_populates="recipient", cascade="all, delete-orphan")
    messages = relationship("ConversationMessage", back_populates="recipient", cascade="all, delete-orphan")
    reminders = relationship("ProactiveReminder", back_populates="recipient", cascade="all, delete-orphan")
    preference = relationship("SaathiPreference", back_populates="recipient", uselist=False, cascade="all, delete-orphan")


class ConversationSession(SaathiBase):
    """
    Represents an active or past conversational visit/session with Saathi.
    """
    __tablename__ = "conversation_sessions"

    session_id = Column(String, primary_key=True, index=True)
    recipient_id = Column(Integer, ForeignKey("saathi_recipients.recipient_id", ondelete="CASCADE"), nullable=False)
    
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    dominant_mood = Column(String, default="neutral")
    summary = Column(Text, nullable=True)

    recipient = relationship("SaathiRecipient", back_populates="sessions")
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")


class ConversationMessage(SaathiBase):
    """
    Individual turn in the conversation stored persistently in the Saathi database.
    """
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("conversation_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    recipient_id = Column(Integer, ForeignKey("saathi_recipients.recipient_id", ondelete="CASCADE"), nullable=False, index=True)
    
    sender = Column(String, nullable=False)  # "user", "bot", or "system"
    message_text = Column(Text, nullable=False)
    
    mood_detected = Column(String, default="neutral")
    mood_confidence = Column(Float, default=1.0)
    intent = Column(String, default="chat")
    trigger_type = Column(String, default="user_initiated")  # "user_initiated", "proactive_checkin", "reminder"
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    recipient = relationship("SaathiRecipient", back_populates="messages")
    session = relationship("ConversationSession", back_populates="messages")


class ProactiveReminder(SaathiBase):
    """
    Scheduled reminders created for or spoken by Saathi in the Saathi database.
    """
    __tablename__ = "proactive_reminders"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recipient_id = Column(Integer, ForeignKey("saathi_recipients.recipient_id", ondelete="CASCADE"), nullable=False, index=True)

    reminder_type = Column(String, nullable=False, default="custom")  # "medicine", "water", "food", "walk", "custom"
    reminder_text = Column(String, nullable=False)
    scheduled_time = Column(String, nullable=False)  # "HH:MM"
    recurrence = Column(String, default="daily")     # "daily", "once"
    
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    recipient = relationship("SaathiRecipient", back_populates="reminders")


class SaathiPreference(SaathiBase):
    """
    Stores content favorites, music, stories, and companion settings for the user.
    """
    __tablename__ = "saathi_preferences"

    recipient_id = Column(Integer, ForeignKey("saathi_recipients.recipient_id", ondelete="CASCADE"), primary_key=True, index=True)
    
    favorite_songs = Column(JSON, default=list)       # [{"title": str, "query": str, "youtube_id": str}]
    favorite_videos = Column(JSON, default=list)      # [{"title": str, "query": str, "category": str}]
    mood_music_map = Column(JSON, default=dict)       # {"sad": "music", "lonely": "music", ...}
    last_greeted_date = Column(String, nullable=True) # "YYYY-MM-DD"
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    recipient = relationship("SaathiRecipient", back_populates="preference")
