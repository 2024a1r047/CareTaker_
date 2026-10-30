from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set in .env file")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print("DB session failed:", e)
        raise  # re-raise so FastAPI knows
    finally:
        db.close()

# ── Saathi Companion Database Configuration ──────────────────────
# Dedicated PostgreSQL database for voice assistant conversations & companion data
SAATHI_DATABASE_URL = os.getenv("SAATHI_DATABASE_URL")
if not SAATHI_DATABASE_URL and DATABASE_URL:
    if "/" in DATABASE_URL:
        SAATHI_DATABASE_URL = DATABASE_URL.rsplit("/", 1)[0] + "/Saathi"

saathi_engine = None
SaathiSessionLocal = None

if SAATHI_DATABASE_URL:
    try:
        saathi_engine = create_engine(SAATHI_DATABASE_URL, pool_pre_ping=True)
        SaathiSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=saathi_engine)
    except Exception as e:
        print(f"[Saathi DB] Engine initialization warning: {e}")

def get_saathi_db():
    """Dependency that yields a session for the Saathi companion database with graceful fallback."""
    if SaathiSessionLocal:
        db = SaathiSessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set in .env file")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Audio Pipeline Config
AUDIO_CHUNK_SIZE = 4096
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
AUDIO_FORMAT = "paInt16"
AUDIO_VAD_THRESHOLD = 800
AUDIO_MAX_SILENCE_CHUNKS = 30
AUDIO_TARGET_SR = 16000