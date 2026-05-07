"""
WeeklyReport Table
------------------
Stores generated weekly health reports for each care recipient.
One report per recipient per week.
"""

import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    ForeignKey, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship
from config import Base


class WeeklyReport(Base):
    __tablename__ = "weekly_reports"

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(
        Integer,
        ForeignKey("care_recipients.id", ondelete="CASCADE"),
        nullable=False, index=True
    )

    # Week boundaries
    week_start = Column(Date, nullable=False)
    week_end = Column(Date, nullable=False)

    # Structured data — all 9 domain sections
    report_data = Column(JSON, nullable=False)

    # AI-generated narrative (executive summary, highlights, concerns, actions)
    ai_narrative = Column(JSON, nullable=True)

    # Week-over-week comparison data
    comparison = Column(JSON, nullable=True)

    # Which Gemini model wrote the narrative (None = rule-based only)
    model_used = Column(String, nullable=True)

    # Deterministic wellness score (0-100)
    overall_score = Column(Float, nullable=True)

    # Risk trend: improving / stable / declining
    risk_trend = Column(String, nullable=True)

    # Delivery tracking
    email_sent = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    care_recipient = relationship("CareRecipient", backref="weekly_reports")

    # One report per recipient per week
    __table_args__ = (
        UniqueConstraint(
            "care_recipient_id", "week_start",
            name="uq_weekly_report_recipient_week"
        ),
    )

    def __repr__(self):
        return (
            f"<WeeklyReport id={self.id} recipient={self.care_recipient_id} "
            f"week={self.week_start}..{self.week_end} score={self.overall_score}>"
        )
