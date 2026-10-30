"""
Central Gemini API client.

Configure via .env:
    GEMINI_API_KEY=<your key>
    GEMINI_MODEL=gemini-2.5-flash-lite   # switch model here anytime
    GEMINI_API_ENDPOINT=<optional full override URL>

Switch model:  just change GEMINI_MODEL in .env and restart the server.
Available models (free tier): gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash
"""

import os
import requests
import time
import json
import re
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Optimized model order: Stable models first, slower preview models last.
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash"
]

def safe_json_parse(text: str) -> dict | None:
    """
    Tries to parse text as JSON. If it fails, uses regex to extract 
    the first JSON block { ... } and parses that.
    """
    if not text: return None
    
    # 1. Direct parse
    try:
        # Clean up common LLM artifacts
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        pass
        
    # 2. Regex extraction (for when LLM adds markdown or chat text)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
        
    return None

def get_gemini_url(model: str, api_key: str) -> str:
    """Build the Gemini URL for a specific model."""
    endpoint = os.environ.get("GEMINI_API_ENDPOINT")
    if endpoint:
        return f"{endpoint}?key={api_key}"
    return f"{BASE_URL}/{model}:generateContent?key={api_key}"

def call_gemini(payload: dict, timeout: int = 30, caller: str = "", model_override: str = None) -> dict | None:
    """
    Make a Gemini API call with automatic model switching on failure.
    If model_override is provided, it attempts that model first.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(f"{caller} [Gemini] ERROR: GEMINI_API_KEY is not set in .env")
        return None

    if not api_key.startswith("AIza"):
        print(f"{caller} [Gemini] Notice: Key in .env does not start with Google's 'AIza' prefix. Using Saathi Intelligent Dialogue Engine.")
        return None

    # Get primary model from override or env
    primary_model = model_override or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    
    # Create the sequence of models to try
    models_to_try = [primary_model]
    for model in FALLBACK_MODELS:
        if model not in models_to_try:
            models_to_try.append(model)

    # Enhanced Internal retries with exponential backoff (Safety Pattern ✅ 1)
    for i, model in enumerate(models_to_try):
        url = get_gemini_url(model, api_key)
        
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)

                if resp.status_code == 200:
                    if i > 0:
                        print(f"{caller} [Gemini] Success using fallback model: {model}")
                    return resp.json()

                if resp.status_code == 400:
                    print(f"{caller} [Gemini] HTTP 400: Invalid API key. Using Saathi Intelligent Dialogue Engine.")
                    return None

                # Quota errors (429) or Server Busy (503) -> Backoff then retry ONCE for same model, then switch
                if resp.status_code in [429, 503]:
                    if attempt < 1:
                        wait_time = 2
                        print(f"{caller} [Gemini] Model '{model}' busy ({resp.status_code}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"{caller} [Gemini] Model '{model}' exhausted. Switching...")
                        break 

                # Deprecated or missing model (404) -> Try next
                if resp.status_code == 404:
                    print(f"{caller} [Gemini] Model '{model}' not found. Skipping...")
                    break

                # Internal Error (500) -> Wait 1s and retry ONCE before switching
                if resp.status_code == 500 and attempt == 0:
                    time.sleep(1)
                    continue

                print(f"{caller} [Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
                break

            except requests.exceptions.Timeout:
                print(f"{caller} [Gemini] Timeout for model '{model}'. Trying next...")
                break # Switch model on timeout
            except Exception as e:
                print(f"{caller} [Gemini] Connection error with model '{model}': {e}")
                time.sleep(1)
                continue # Try second attempt for the same model

    print(f"{caller} [Gemini] All models failed. Please check your API key and quotas.")
    return None

def get_embedding(text: str, model: str = "text-embedding-004") -> list[float] | None:
    """
    Get vector embedding for a piece of text.
    Uses text-embedding-004 (Gemini Embedding 2).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key: return None
    
    url = f"{BASE_URL}/{model}:embedContent?key={api_key}"
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]}
    }
    
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("embedding", {}).get("values")
        return None
    except Exception as e:
        print(f"[Embedding] Error: {e}")
        return None
