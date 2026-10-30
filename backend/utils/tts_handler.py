import os
import re
import base64
import asyncio
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

# Global circuit breaker for Google Cloud TTS (if used as fallback)
_gcp_tts_unavailable = False

# High-quality neural female voices for empathetic elderly care
VOICE_MAP = {
    # Hindi / Hinglish: Soft, warm, compassionate caregiver voice
    "hi": "hi-IN-SwaraNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    # Indian English: Gentle, polite, natural conversational rhythm
    "en-IN": "en-IN-NeerjaNeural",
    # General / US English: Warm, friendly, soothing conversational voice
    "en": "en-US-JennyNeural",
    "en-US": "en-US-JennyNeural",
}

# Common Hindi/Hinglish marker words (case-insensitive) to detect Hinglish in Latin script
HINGLISH_KEYWORDS = {
    "namaste", "pranam", "karein", "kijiye", "theek", "kaise", "kaisa", "kaisi",
    "aap", "aapka", "aapki", "aapke", "main", "meri", "mera", "mere", "haan",
    "nahi", "achha", "acha", "suno", "sunte", "gaane", "kahani", "shant", "sukoon",
    "chinta", "khana", "dawa", "dawain", "subah", "shaam", "dophar", "raat",
    "saathi", "dadaji", "nanaji", "dadiji", "naniji", "chalo", "baatein", "samajh"
}

def is_hindi(text: str) -> bool:
    """Detects if text contains Devanagari characters or recognizable Hinglish phrasing."""
    if not text:
        return False
    # Check Devanagari range
    if any("\u0900" <= char <= "\u097f" for char in text):
        return True
    # Check for prevalent Hinglish words
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if any(w in HINGLISH_KEYWORDS for w in words):
        return True
    return False

def clean_text_for_speech(text: str) -> str:
    """
    Cleans and segments text for realistic, natural speech delivery:
    - Removes emojis, markdown symbols, and technical formatting
    - Expands common abbreviations so words are pronounced clearly
    - Normalizes punctuation to create gentle, human-like breathing pauses
    """
    if not text:
        return ""

    # 1. Remove emojis (prevent TTS reading emoji descriptions out loud)
    cleaned = re.sub(
        r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE00-\uFE0F]',
        '',
        text
    )

    # 2. Remove markdown formatting: bold, italic, headers, bullet points, code blocks
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'`[^`]*`', '', cleaned)
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
    cleaned = re.sub(r'[*_~]', '', cleaned)
    cleaned = re.sub(r'^\s*[-*•]\s+', '', cleaned, flags=re.MULTILINE)

    # 3. Expand common abbreviations for elderly understanding
    abbrev_map = [
        (r'\bBP\b', 'blood pressure'),
        (r'\bHR\b', 'heart rate'),
        (r'\bDr\.\s*', 'Doctor '),
        (r'\bmg\b', 'milligrams'),
        (r'\bmin\b', 'minutes'),
        (r'\bsec\b', 'seconds'),
        (r'\bapprox\.\s*', 'approximately '),
        (r'\bapprox\b', 'approximately'),
    ]
    for pattern, replacement in abbrev_map:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 4. Normalize conversational pauses and clean extra punctuation
    cleaned = re.sub(r'\s*\.{2,}\s*', '... ', cleaned)  # Normalize ellipsis
    cleaned = re.sub(r'\s*([,;])\s*', r'\1 ', cleaned)  # Ensure single space after commas
    cleaned = re.sub(r'\s*([.?!])\s*', r'\1 ', cleaned)  # Clean sentence boundaries
    cleaned = re.sub(r'[!]{2,}', '!', cleaned)
    cleaned = re.sub(r'[?]{2,}', '?', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def resolve_voice_name(text: str, language_code: str = "hi-IN", voice_override: str = None) -> str:
    """Resolves the most natural-sounding female voice based on language and text content."""
    if voice_override:
        return voice_override

    # If text is in Hindi/Devanagari or Hinglish, use the SwaraNeural voice
    if is_hindi(text) or (language_code and language_code.lower().startswith("hi")):
        return VOICE_MAP["hi-IN"]

    # Indian English preference
    if language_code and (language_code.lower() == "en-in" or "in" in language_code.lower()):
        return VOICE_MAP["en-IN"]

    return VOICE_MAP.get(language_code, VOICE_MAP["en-US"])

async def _synthesize_edge_tts(text: str, voice_name: str) -> str:
    """Synthesizes speech using Microsoft Edge Neural TTS with elderly-friendly prosody."""
    import edge_tts

    # Soft, relaxed speech rate (-4%) and natural pitch (+0Hz) for elderly comprehension
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_name,
        rate="-4%",
        pitch="+0Hz",
        volume="+0%"
    )

    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_bytes.extend(chunk.get("data", b""))

    if audio_bytes:
        return base64.b64encode(audio_bytes).decode("utf-8")
    return ""

def _synthesize_google_fallback(text: str, language_code: str) -> str:
    """Fallback using Google Cloud Text-to-Speech if edge-tts fails and credentials exist."""
    global _gcp_tts_unavailable
    if _gcp_tts_unavailable:
        return ""

    try:
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        if language_code.startswith("hi"):
            gcp_voice = "hi-IN-Neural2-A"
        else:
            gcp_voice = "en-US-Neural2-F"

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=gcp_voice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.96,
            pitch=0.0
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode("utf-8")
    except Exception as e:
        err = str(e)
        if "credentials" in err.lower() or "403" in err or "billing" in err.lower():
            _gcp_tts_unavailable = True
        print(f"[TTS GCP Fallback Note] {e}")
        return ""

async def generate_speech_base64_async(text: str, language_code: str = "hi-IN", voice_override: str = None) -> str:
    """
    Primary async speech generation function:
    1. Preprocesses text for smooth pronunciation and natural rhythm.
    2. Uses high-fidelity neural female voice with soft, relaxed caregiver delivery.
    3. Falls back gracefully to Google Cloud TTS if needed.
    """
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return ""

    voice_name = resolve_voice_name(clean_text, language_code, voice_override)

    # 1. Primary: ElevenLabs (Voice ID kbqGn0IGWLrQBLcS2Utk - "saathi voice", model eleven_multilingual_v2)
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    if eleven_key:
        try:
            from services.elevenlabs_service import synthesize_elevenlabs_speech, get_elevenlabs_voice_id, get_elevenlabs_model_id
            b64_audio = synthesize_elevenlabs_speech(clean_text, get_elevenlabs_voice_id(), get_elevenlabs_model_id())
            if b64_audio:
                return b64_audio
        except Exception as e:
            print(f"[ElevenLabs TTS Fallback Note] {e}, attempting Edge Neural fallback...")

    # 2. Secondary: Edge Neural TTS (Natural, soft, expressive female voice)
    try:
        b64_audio = await _synthesize_edge_tts(clean_text, voice_name)
        if b64_audio:
            return b64_audio
    except Exception as e:
        print(f"[TTS Edge Error] {e}, attempting Google Cloud fallback...")

    # 2. Secondary fallback: Google Cloud TTS
    try:
        lang = "hi-IN" if is_hindi(clean_text) else language_code
        gcp_audio = _synthesize_google_fallback(clean_text, lang)
        if gcp_audio:
            return gcp_audio
    except Exception as e:
        print(f"[TTS Fallback Error] {e}")

    return ""

def generate_speech_base64(text: str, language_code: str = "hi-IN", voice_override: str = None) -> str:
    """
    Synchronous thread-safe wrapper for generate_speech_base64_async.
    Safe to call from both synchronous code and within already-running event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an active event loop (e.g. FastAPI / Uvicorn thread)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(generate_speech_base64_async(text, language_code, voice_override))
            ).result(timeout=20)
    else:
        return asyncio.run(generate_speech_base64_async(text, language_code, voice_override))
