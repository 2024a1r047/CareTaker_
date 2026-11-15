"""
Medicines the room camera should recognise (uploaded photo or taught in front of the camera),
and where each one was last seen.

The browser (frontend/table_watch.js) turns the photo into MobileNet fingerprints, stores them
here so every device and the AI assistant share the same list, and reports the location each
time the camera sees the medicine ("on the dining table, right side of the room").
"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text

from config import Base


class MedicineRef(Base):
    __tablename__ = "medicine_refs"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id"), nullable=True, index=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id"), nullable=True, index=True)

    name = Column(String(120), nullable=False)
    image = Column(Text, nullable=True)            # small JPEG data URL (thumbnail of the photo)
    source = Column(String(20), default="upload")  # upload | camera
    mode = Column(String(20), default="mobilenet") # fingerprint type: mobilenet | simple
    embeddings = Column(Text, nullable=True)       # JSON [[...], ...] positive fingerprints
    negatives = Column(Text, nullable=True)        # JSON [[...], ...] "not this medicine" fingerprints

    last_seen_at = Column(DateTime, nullable=True)
    last_location = Column(Text, nullable=True)    # English, e.g. "on the dining table, right side of the room"
    last_location_hi = Column(Text, nullable=True) # Hinglish for the voice assistant
    last_box = Column(Text, nullable=True)         # JSON {x,y,w,h} as fractions of the picture
    last_confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class MedicineSighting(Base):
    """
    History of where the room camera found each medicine.
    One row per "visit": repeated sightings at the same place within 10 minutes extend the row
    (last_seen_at, sightings) instead of adding a new one.
    """
    __tablename__ = "medicine_sightings"

    id = Column(Integer, primary_key=True, index=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id"), nullable=True, index=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id"), nullable=True, index=True)
    medicine_ref_id = Column(Integer, ForeignKey("medicine_refs.id", ondelete="SET NULL"), nullable=True, index=True)
    medicine_name = Column(String(120), nullable=False)

    location = Column(Text, nullable=True)          # English, e.g. "on the pillow, on the bed, left side of the room"
    location_hi = Column(Text, nullable=True)       # Hinglish
    confidence = Column(Float, nullable=True)       # best match % during this visit
    probable = Column(Integer, default=1)           # 1 = only a strong guess so far, 0 = confirmed
    sightings = Column(Integer, default=1)          # how many reports this visit combined

    first_seen_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow, index=True)
