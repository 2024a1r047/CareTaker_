"""
Table Watch: camera check of the water glass and the medicines left on the table.

One row per event seen by the in-browser Table Watch (frontend/table_watch.js):
  item   = "water" | "medicine"
  event  = "baseline"        zone marked: glass full / medicines placed
           "taken"           medicines picked up / glass picked up and put back changed (drank)
           "missing"         glass or medicines no longer in their place
           "returned"        glass put back
           "untouched_alert" water not touched for too long / medicines still there after the due time
"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text

from config import Base


class TableWatchEvent(Base):
    __tablename__ = "table_watch_events"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id"), nullable=True, index=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id"), nullable=True, index=True)

    item = Column(String(20), nullable=False)              # water | medicine
    event = Column(String(30), nullable=False)             # baseline | taken | missing | returned | untouched_alert
    detail = Column(Text, nullable=True)                   # readable text shown in the history
    idle_minutes = Column(Float, nullable=True)            # minutes untouched (for alerts)
    confidence = Column(Float, nullable=True)              # 0-100

    occurred_at = Column(DateTime, default=datetime.utcnow, index=True)
