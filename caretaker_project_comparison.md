# Caretaker.ai — In-Depth Technical Guide & Contribution Breakdown

This document provides a highly technical, interview-ready breakdown of **Fall Detection**, **Conversational Bot**, and a clear mapping of **your contributions (Palak)** versus **what your teammates built upon**.

---

## 🔷 Part 1: How the Fall Detection System Works (In-Depth)

The fall detection pipeline is a state-of-the-art **hybrid deep learning model** designed to process spatial keypoints over a temporal sequence. Unlike simple posture-based classifiers (which mistake sleeping or sitting for a fall), this architecture models the *transition* of a fall over time.

### 1. Frame Capture & Keypoint Extraction (YOLOv8n-pose)
* **Model:** `yolov8n-pose.pt` (nano variant) — chosen for its sub-10ms inference speed on standard CPUs.
* **Extraction:** For every incoming video frame, YOLO detects the elderly individual and extracts **17 COCO keypoints** (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles). Each keypoint consists of $(x, y)$ coordinates and a confidence value.
* **Frame Optimization:** A frame skip of `2` (`FRAME_SKIP=2`) is utilized to reduce computational overhead without losing temporal integrity.

### 2. Spatial Normalization (`normalize_keypoints`)
Feeding raw coordinates into a neural network fails because the values change depending on where the person is standing or how close they are to the camera. We solve this using custom spatial normalization:
1. **Hip Centering:** Calculate the hip midpoint (average of left/right hips) and subtract it from all 17 keypoint coordinates. This centers the person at $(0, 0)$.
2. **Torso Scaling:** Calculate the torso length as the Euclidean distance between the shoulder midpoint and hip midpoint. Divide all coordinates by this torso length.
3. **Confidence Masking:** Keypoints with confidence values $< 0.2$ are zeroed out to prevent noisy jitter.
* **Output:** A **34-dimensional float32 vector** (17 keypoints $\times$ 2 coordinates) that is entirely scale-invariant and position-invariant.

### 3. Temporal Modeling (BiGRU + Self-Attention)
* **Sliding Sequence Buffer:** Normalized frames are loaded into a rolling `deque(maxlen=30)` representing a 30-frame sequence (~1.5 seconds of motion). It slides with a stride of 15 frames (50% overlap).
* **Bidirectional GRU (BiGRU):** The 30-frame sequence is fed into a 2-layer Bidirectional Gated Recurrent Unit (128 hidden units per direction). 
  * *Why Bidirectional?* It captures temporal patterns forward (the action of falling) and backward (the subsequent state of immobility), providing the model with full event context.
* **Additive Self-Attention:** Instead of using the last GRU state (which suffers from forgetting), a self-attention layer assigns weights to all 30 frames. The transition frames (where the rapid collapse happens) receive high attention weights.
* **Classifier Head:** A dense layer maps the attention context vector to logits representing `[Normal, Fall]`.

### 4. Real-Time Confirmation Engine
To prevent false alarms in a live setting (e.g., someone sitting down quickly or tying a shoelace):
* **Stricter Thresholds:** The model output threshold is elevated from 0.80 (in training) to **0.92** in live mode.
* **Consecutive Frame Count:** A fall is only confirmed and logged if the confidence exceeds 0.92 for **10 consecutive frames**.
* **Inactivity Centroid:** Once a fall is confirmed, a secondary `InactivityMonitor` tracks the bounding box centroid. If the centroid doesn't move for 30 seconds, a critical alert is triggered.

---

## 🔷 Part 2: How the Conversational Bot ("Saathi") Works (In-Depth)

**Saathi** is an emotionally aware conversational AI companion designed for the elderly. It lives in `voice_bot.html` and communicates with the backend via WebSockets.

```
[Elderly User] 🗣️ Speech/Text 
     ↓
[WebSocket Connection]
     ↓
[Gemini AI Classifier] ──(Detects Intent)──> Play music, tell story, health query, or chat
     ↓
[Sentiment Engine] ────(Analyzes last 10 messages)──> 10 Moods / Trends / Depression Risk
     ↓
[YouTube Engine] ──────(Shifted from Spotify)──> Quota-safe search, quality-ranks, plays embeds
     ↓
[Google Cloud TTS] ────(Natural Hinglish Voice)──> Response read aloud to user
```

### 1. Intent Dispatcher (Gemini LLM Classifier)
When a user says something, it goes to a central **Gemini-powered intent router** which parses the text and dispatches it:
* `music` / `story` (e.g., "play an old song" or "tell me a spiritual story")
* `health_query` / `medicine_reminder` (accesses the database context to answer clinical questions safely)
* `general_chitchat` (standard companion conversation)

### 2. The Sentiment Engine (`sentiment_engine.py`)
To make Saathi emotionally intelligent, a sentiment layer sits on top:
* **Context Fetching:** It extracts the last 10 messages from the database.
* **10-Mood Classification:** It prompts Gemini to classify the user's current mood into one of 10 emotions: `happy`, `sad`, `anxious`, `angry`, `neutral`, `distressed`, `lonely`, `bored`, `relaxed`, `spiritual`.
* **Trend Analysis:** Tracks if the mood is improving (📈), stable (➡️), or worsening (📉).
* **Depression Risk Alert:** If the database records 4 or more sad/lonely/anxious/distressed moods over the last 7 days, the system automatically flags a **Depression Risk** and alerts the caregiver.

### 3. YouTube Media Playback (The Spotify Shift)
* **Legacy Endpoint:** `/spotify/search` (retained for backward compatibility).
* **YouTube Data API v3:** The bot uses the YouTube API to query songs/stories based on the user's mood.
* **Quality Filtering:**
  * Excludes short clips / YouTube Shorts.
  * Enforces minimum duration (songs $\ge$ 3 mins, stories $\ge$ 5 mins).
  * Sorts candidates by real view count to fetch the highest-quality, official videos.
* **HTML Scrape Fallback:** If the YouTube API daily quota is exceeded, it automatically falls back to an HTML scraper using `urllib` and regex to extract watch IDs so media never stops playing.

---

## 🔷 Part 3: Contribution Breakdown (Palak vs. Teammates)

This breakdown gives you a flawless narrative to present in your interview: *"I built the foundational backend architecture, database schema, and the initial voice bot and media features. Once the core was stable, my teammates joined to harden the machine learning models and add deep emotional intelligence."*

### 🔨 1. What You Built (Palak's Contributions)
You established the bedrock of the entire Caretaker.ai application:
* **Core Backend Framework & DB Schema:** Set up the FastAPI structure, PostgreSQL integration, tables (users, caretakers, recipients, lab values, etc.), and JWT security.
* **Voice Bot V1:** Created the initial chatbot logic (`chatbot` and `voice bot working` commits) integrating Web Speech API and basic chat response handlers.
* **The Spotify Player Concept:** Wrote the original `/spotify/search` endpoint and the front-end JS player integrations to play music for the patient.
* **Doctor's Dashboard:** Built the initial Doctor remarks flow, patient vitals logging panels, and patient profile list.
* **Video Frame Capture:** Set up the initial WebRTC / camera streaming interface on the dashboard to allow frame transmission.

### 🚀 2. How Your Teammates Continued Your Work
Your teammates built advanced layers on top of your rock-solid foundation:
* **The YouTube Shift:** Because the **Spotify Developer API** had strict developer-account limits, token timeouts, and failed to play songs/videos directly in the web player without premium accounts, your teammate shifted the backend of your `/spotify/search` endpoint to use **YouTube Data API v3** combined with a custom **HTML scraper fallback**. They kept your endpoint name so the front-end wouldn't break, but added smart filters (dropping shorts, minimum duration checks, sorting by view count) so that music and stories now play flawlessly.
* **Adding the Sentiment & Emotion Engine:** Your teammate added the `sentiment_engine.py` on top of your voice bot. This introduced the 10-mood classification, trend tracking, and depression risk alerts to make the companion aware of the elder's emotional state.
* **Fall Detection Hardening:** Your teammates took the basic camera stream and connected it to the full **YOLOv8-pose + BiGRU + Self-Attention** deep learning pipeline. They wrote the hip-centered torso scaling normalizer, built the live 10-frame consecutive confirmation engine to kill false positives, and optimized the video processing thread to run at 3x speed.
* **Google Cloud TTS:** Upgraded the bot's speech output from standard web speech to a premium, natural-sounding Hindi voice using Google Cloud Text-to-Speech.
