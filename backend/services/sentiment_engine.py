"""
Sentiment Engine — Context-Aware Mood Analysis for Caretaker AI

Instead of analyzing just the current message, this engine:
1. Pulls the last N user messages from conversation history
2. Sends the full conversation arc + current message to Gemini
3. Returns a rich SentimentContext:
   - current_mood       : mood of the current message
   - dominant_mood      : most frequent mood over recent history
   - trend              : "improving" | "worsening" | "stable"
   - stability_score    : 0.0 (very volatile) to 1.0 (very stable)
   - recommended_action : "music" | "story" | "conversation" | "reminder" | "alert"
   - urgency            : "low" | "medium" | "high"
   - summary            : one-line human-readable emotional summary
   - confidence         : 0.0 to 1.0

This context is fed into the system prompt and recommendation engine.
"""

import os
import re
import json
import requests
from datetime import datetime, timedelta
from utils.gemini_client import call_gemini, safe_json_parse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.conversation_history import ConversationMessage, SenderEnum, MoodEnum

# Circuit breaker: skip Gemini sentiment call after a quota failure
# (saves the remaining quota for the main voice bot response)
_sentiment_cb_open = False
_sentiment_cb_until: datetime | None = None
_SENTIMENT_CB_COOLDOWN = 300  # seconds

def _sentiment_gemini_allowed() -> bool:
    global _sentiment_cb_open, _sentiment_cb_until
    if _sentiment_cb_open:
        if _sentiment_cb_until and datetime.utcnow() >= _sentiment_cb_until:
            _sentiment_cb_open = False
            _sentiment_cb_until = None
        else:
            return False
    return True

def _sentiment_gemini_fail():
    global _sentiment_cb_open, _sentiment_cb_until
    _sentiment_cb_open = True
    _sentiment_cb_until = datetime.utcnow() + timedelta(seconds=_SENTIMENT_CB_COOLDOWN)
    print("[sentiment_engine] Circuit breaker OPEN — skipping Gemini for 5 min to preserve quota")


# ─────────────────────────────────────────────
# Trend helpers
# ─────────────────────────────────────────────
NEGATIVE_MOODS = {"sad", "anxious", "distressed", "lonely", "angry"}
POSITIVE_MOODS = {"happy", "relaxed", "spiritual"}
NEUTRAL_MOODS  = {"neutral", "bored"}

def _mood_score(mood: str) -> float:
    """Map mood to numeric score: positive=1, neutral=0, negative=-1."""
    if mood in POSITIVE_MOODS:
        return 1.0
    if mood in NEGATIVE_MOODS:
        return -1.0
    return 0.0

def _compute_trend(mood_timeline: list) -> str:
    """Compare first half vs second half of mood scores to determine trend."""
    if len(mood_timeline) < 3:
        return "stable"
    scores = [_mood_score(m) for m in mood_timeline]
    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / max(len(scores[:mid]), 1)
    second_half_avg = sum(scores[mid:]) / max(len(scores[mid:]), 1)
    delta = second_half_avg - first_half_avg
    if delta > 0.3:
        return "improving"
    if delta < -0.3:
        return "worsening"
    return "stable"

def _compute_stability(mood_timeline: list) -> float:
    """How consistent the moods are. 1.0 = all same mood, 0.0 = all different."""
    if not mood_timeline:
        return 1.0
    unique = len(set(mood_timeline))
    return round(1.0 - (unique - 1) / max(len(mood_timeline), 1), 2)

def _dominant_mood(mood_timeline: list) -> str:
    if not mood_timeline:
        return "neutral"
    counts = {}
    for m in mood_timeline:
        counts[m] = counts.get(m, 0) + 1
    return max(counts, key=counts.get)

def _partial_json_extract(raw: str) -> dict:
    """Field-by-field regex fallback for truncated Gemini JSON."""
    patterns = {
        "current_mood":        r'"current_mood"\s*:\s*"([^"]+)"',
        "dominant_mood":       r'"dominant_mood"\s*:\s*"([^"]+)"',
        "trend":               r'"trend"\s*:\s*"([^"]+)"',
        "stability_score":     r'"stability_score"\s*:\s*([0-9.]+)',
        "recommended_action":  r'"recommended_action"\s*:\s*"([^"]+)"',
        "urgency":             r'"urgency"\s*:\s*"([^"]+)"',
        "summary":             r'"summary"\s*:\s*"([^"]*)"',
        "confidence":          r'"confidence"\s*:\s*([0-9.]+)',
    }
    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, raw)
        if m:
            val = m.group(1)
            if key in ("stability_score", "confidence"):
                try:
                    val = float(val)
                except ValueError:
                    pass
            result[key] = val
    return result if result else None


def _recommend_action(current_mood: str, trend: str, urgency: str) -> str:
    """Rule-based recommendation before Gemini override."""
    if urgency == "high":
        return "alert"
    if current_mood in ("sad", "lonely", "distressed") and trend == "worsening":
        return "conversation"
    if current_mood in ("bored",):
        return "story"
    if current_mood in ("spiritual",):
        return "story"
    if current_mood in ("anxious", "angry"):
        return "music"
    if current_mood in ("happy", "relaxed"):
        return "music"
    return "music"


# ─────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────
def analyze_sentiment_with_history(
    current_text: str,
    recipient_id: int,
    db: Session,
    history_limit: int = 10
) -> dict:
    """
    Full sentiment analysis using conversation history + current message.
    Returns a SentimentContext dict.
    """

    # 1. Fetch last N user messages from DB
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    past_messages = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.created_at >= seven_days_ago
    ).order_by(desc(ConversationMessage.created_at)).limit(history_limit).all()
    past_messages.reverse()  # chronological

    # Build mood timeline from stored moods (fast — no extra API call)
    stored_moods = [
        m.mood_detected.value
        for m in past_messages
        if m.mood_detected is not None
    ]

    # Extract text snippets for Gemini context
    history_snippets = [
        {"text": m.message_text[:120], "mood": m.mood_detected.value if m.mood_detected else "unknown"}
        for m in past_messages[-6:]  # last 6 for brevity
    ]

    # 2. Quick local computation (no API needed)
    trend = _compute_trend(stored_moods)
    stability = _compute_stability(stored_moods)
    dominant = _dominant_mood(stored_moods)

    # 3. FAST PATH: Try local keyword-based mood detection first
    local_mood = "neutral"
    local_confidence = 0.3
    text_lower = current_text.lower()
    
    if any(kw in text_lower for kw in ["sad", "udaas", "dukhi", "cry", "rona", "dard", "hurt", "pain", "takleef", "pareshan"]):
        local_mood = "sad"
        local_confidence = 0.8
    elif any(kw in text_lower for kw in ["khush", "happy", "badhiya", "mast", "great", "mazza", "good mood"]):
        local_mood = "happy"
        local_confidence = 0.8
    elif any(kw in text_lower for kw in ["akela", "lonely", "yaad", "miss", "alone"]):
        local_mood = "lonely"
        local_confidence = 0.8
    elif any(kw in text_lower for kw in ["bore", "boring", "kya karun", "time pass"]):
        local_mood = "bored"
        local_confidence = 0.8
    elif any(kw in text_lower for kw in ["ghabra", "anxious", "nervous", "dar", "tension", "scared", "worry"]):
        local_mood = "anxious"
        local_confidence = 0.8
    elif any(kw in text_lower for kw in ["gussa", "angry", "naraz", "irritat", "annoyed", "frustrated"]):
        local_mood = "angry"
        local_confidence = 0.8

    gemini_result = None
    if local_confidence >= 0.7:
        gemini_result = {
            "current_mood": local_mood,
            "dominant_mood": dominant,
            "trend": trend,
            "stability_score": float(stability),
            "recommended_action": _recommend_action(local_mood, trend, "low"),
            "urgency": "low",
            "summary": "",
            "confidence": local_confidence
        }
    else:
        # 4. Gemini deep analysis — sends history + current message
        api_key = os.environ.get('GEMINI_API_KEY')

        if api_key and (history_snippets or current_text) and _sentiment_gemini_allowed():
            history_text = "\n".join(
                [f"  [{i+1}] (mood={h['mood']}) \"{h['text']}\"" for i, h in enumerate(history_snippets)]
            ) or "  (no prior messages)"

            prompt = f"""You are analyzing the emotional state of an elderly user for a care assistant.

Recent conversation history (chronological, user messages only):
{history_text}

Current message (just now): "{current_text}"

Respond with EXACTLY ONE JSON object and NO OTHER TEXT. No markdown, no code fences.

{{"current_mood":"happy|sad|anxious|angry|neutral|distressed|lonely|bored|relaxed|spiritual","dominant_mood":"same options","trend":"improving|worsening|stable","stability_score":0.0,"recommended_action":"music|story|conversation|reminder|alert","urgency":"low|medium|high","summary":"Short Hinglish summary max 10 words","confidence":0.0}}
"""

            try:
                data = call_gemini(
                    {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200}},
                    timeout=10, caller="[sentiment_engine]"
                )
                if data and data.get("candidates"):
                    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    gemini_result = safe_json_parse(raw)
                    if not gemini_result:
                        gemini_result = _partial_json_extract(raw)
                    if not gemini_result:
                        print(f"[sentiment_engine] JSON parse failed: {raw[:100]}")
                else:
                    _sentiment_gemini_fail()
            except Exception as e:
                print(f"[sentiment_engine] Gemini error: {e}")
                _sentiment_gemini_fail()

    # 4. Merge Gemini result with local computation
    if gemini_result:
        current_mood = gemini_result.get("current_mood", "neutral")
        dominant_mood = gemini_result.get("dominant_mood", dominant)
        trend = gemini_result.get("trend", trend)
        stability = gemini_result.get("stability_score", stability)
        recommended_action = gemini_result.get("recommended_action", "music")
        urgency = gemini_result.get("urgency", "low")
        summary = gemini_result.get("summary", "")
        confidence = gemini_result.get("confidence", 0.7)
    else:
        # Fallback to local computation
        current_mood = "neutral"
        dominant_mood = dominant
        recommended_action = _recommend_action(current_mood, trend, "low")
        urgency = "low"
        summary = ""
        confidence = 0.5

    # Validate mood values
    valid_moods = [m.value for m in MoodEnum]
    if current_mood not in valid_moods:
        current_mood = "neutral"
    if dominant_mood not in valid_moods:
        dominant_mood = dominant if dominant in valid_moods else "neutral"

    return {
        "current_mood": current_mood,
        "dominant_mood": dominant_mood,
        "trend": trend,
        "stability_score": round(float(stability), 2),
        "recommended_action": recommended_action,
        "urgency": urgency,
        "summary": summary,
        "confidence": round(float(confidence), 2),
        "history_length": len(stored_moods),
        "mood_timeline": stored_moods[-6:],  # last 6 for frontend display
    }


def build_sentiment_prompt_block(sentiment: dict) -> str:
    """Format sentiment context as a block to inject into the system prompt."""
    if not sentiment:
        return ""

    trend_emoji = {"improving": "📈", "worsening": "📉", "stable": "➡️"}.get(sentiment.get("trend", "stable"), "➡️")
    urgency_note = ""
    if sentiment.get("urgency") == "high":
        urgency_note = "\n⚠️ HIGH URGENCY: User may be in emotional distress. Prioritize emotional support over any content suggestions."
    elif sentiment.get("urgency") == "medium":
        urgency_note = "\nNote: User seems moderately distressed. Be extra warm and supportive."

    block = f"""
### 🧠 Sentiment Analysis (AI-computed from conversation history):
- Current mood: **{sentiment.get('current_mood', 'neutral')}**
- Dominant mood (recent history): **{sentiment.get('dominant_mood', 'neutral')}**
- Emotional trend: {trend_emoji} **{sentiment.get('trend', 'stable')}**
- Stability score: {sentiment.get('stability_score', 0.5)} (1.0 = very stable)
- Emotional summary: "{sentiment.get('summary', '')}"
- Recommended intervention: **{sentiment.get('recommended_action', 'music')}**{urgency_note}

Use this context to personalize your response. If trend is worsening, be more nurturing. If improving, celebrate with them.
"""
    return block
