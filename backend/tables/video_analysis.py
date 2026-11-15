from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from config import Base

class VideoAnalysis(Base):
    __tablename__ = "video_analysis"

    id = Column(Integer, primary_key=True, index=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id"), nullable=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    video_filename = Column(String, nullable=True)
    
    # Analysis Metrics
    has_fall = Column(Boolean, default=False)
    fall_count = Column(Integer, default=0)
    inactivity_duration_seconds = Column(Integer, default=0)
    activity_score = Column(Float, default=0.0) # 0-10 scale
    mobility_score = Column(Float, default=0.0) # 0-10 scale
    
    # Relationships
    recipient = relationship("CareRecipient", back_populates="video_analyses")
    caretaker = relationship("CareTaker", back_populates="video_analyses")


class FallEvent(Base):
    """
    One row per detected fall, kept permanently (uploaded videos, live camera streams and
    in-browser live monitoring). Holds the fall summary: when it happened, how confident the
    detector was, and whether / how quickly the person got up afterwards.
    """
    __tablename__ = "fall_events"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id"), nullable=True, index=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id"), nullable=True, index=True)
    video_analysis_id = Column(Integer, ForeignKey("video_analysis.id"), nullable=True, index=True)

    source = Column(String, default="upload")            # upload | live_stream | tfjs_live
    video_filename = Column(String, nullable=True)        # analysed video (uploads)
    camera_name = Column(String, nullable=True)           # live streams

    occurred_at = Column(DateTime, default=datetime.utcnow)   # wall-clock time (UTC)
    video_time_sec = Column(Float, nullable=True)         # position in the uploaded video

    confidence = Column(Float, nullable=True)             # 0-100 %
    detection_method = Column(String, nullable=True)      # pose_model | lstm | disappearance | tfjs

    got_up = Column(Boolean, default=False)               # person stood up again after the fall
    time_to_get_up_sec = Column(Float, nullable=True)     # seconds on the ground before getting up
    time_on_ground_sec = Column(Float, nullable=True)     # if not up: seconds still down when video/session ended
    got_up_at = Column(DateTime, nullable=True)

    summary = Column(Text, nullable=True)                 # human-readable fall summary
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
