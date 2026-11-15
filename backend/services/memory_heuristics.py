"""
Rule-based memory helpers for Saathi / Asha, used when Gemini is not available.

The old fallback stored every sentence the patient said as a "fact" — including questions
("मैंने कौन सी उम्र में join किया था?"), commands ("gaana lagao") and filler. These helpers
keep only real first-person statements, clean them up, categorise them, and build a short
topic-based session summary.
"""
import re
from typing import Dict, List, Optional, Tuple

_QUESTION = re.compile(
    r"\?|(^|\s)(kya|kyaa|kaun|kaunsi|kaun si|kab|kitna|kitni|kitne|kaise|kaisa|kaisi|kyun|kyon|kahan|kahaan|"
    r"what|when|why|how|who|which|where|"
    r"क्या|कौन|कब|कितन\S*|कैसे|कैसा|कैसी|क्यों|कहाँ|कहां)(\s|$)", re.I)
_COMMAND = re.compile(
    r"(^|\s)(batao|bataiye|bataao|batana|lagao|lagaiye|chalao|chalaiye|bajao|sunao|sunaiye|karo|kariye|kar do|"
    r"bhejo|bhej do|band karo|roko|ruko|play|stop|call|message|"
    r"बताओ|बताइए|बताइये|लगाओ|चलाओ|सुनाओ|बजाओ|करो|कीजिए|भेजो|रोको|बंद करो)(\s|$|[.,!।])", re.I)
_FIRST_PERSON = re.compile(
    r"(^|\s)(main|mai|mein|mujhe|mujhko|mera|meri|mere|maine|hum|hamara|hamari|hamare|i|i'm|i am|my|me|"
    r"मैं|मैने|मैंने|मुझे|मुझको|मेरा|मेरी|मेरे|हम|हमारा|हमारी|हमारे)(\s|$|[.,!।])", re.I)
_FILLER_START = re.compile(
    r"^(\s*(haan|haanji|han|ha|ji|achha|acha|accha|ok|okay|chalo|hmm|um|uh|toh|to|"
    r"हाँ|हां|जी|अच्छा|ठीक है|चलो|तो|अ)\s*[,.!।-]*\s*)+", re.I)
_TRANSIENT = re.compile(
    r"(le li|le liya|kha li|kha liya|ki dawai li|li thi|li thin|ली थी|ली थीं|ले ली|खा ली|"
    r"abhi|right now|आज सुबह|aaj subah)", re.I)

# category -> keywords (Roman Hinglish + Devanagari)
_CATEGORIES: List[Tuple[str, Tuple[str, ...]]] = [
    ("health", ("dard", "pain", "bukhar", "fever", "ghutna", "knee", "chakkar", "dizzy", "kamar", "sir dard",
                "ulti", "neend nahi", "saans", "khansi", "दर्द", "बुखार", "घुटन", "चक्कर", "उल्टी", "खांसी", "सांस", "नींद नहीं")),
    ("family", ("beta", "beti", "pota", "poti", "bahu", "damad", "pati", "patni", "son", "daughter", "grandson",
                "granddaughter", "wife", "husband", "bhai", "behen", "बेटा", "बेटी", "पोता", "पोती", "बहू", "पति", "पत्नी", "भाई", "बहन")),
    ("likes", ("pasand", "like", "love", "accha lagta", "achha lagta", "shauk", "पसंद", "अच्छा लगता", "शौक")),
    ("dislikes", ("pasand nahi", "nafrat", "dislike", "bura lagta", "पसंद नहीं", "बुरा लगता")),
    ("routine", ("subah", "shaam", "raat ko", "roz", "daily", "walk", "sair", "tahal", "pooja", "puja", "uthti", "uthta",
                 "soti", "sota", "सुबह", "शाम", "रोज़", "रोज", "टहल", "पूजा", "उठती", "सोती")),
    ("life", ("teacher", "teaching", "school", "padhaya", "padhati", "padhata", "padhaati", "naukri", "job", "retire",
              "kaam karti", "kaam karta", "gaon", "village", "shaadi", "shadi", "vivah", "guzar", "death", "nahi rahe",
              "covid", "corona", "janam", "paida", "bangalore", "bengaluru", "delhi", "mumbai", " it ", "company",
              "टीचर", "स्कूल", "पढ़ाया", "पढ़ाती", "पढ़ाता", "नौकरी", "रिटायर", "गांव", "गाँव", "शादी", "विवाह",
              "गुज़र", "गुजर", "देहांत", "कोविड", "कोरोना", "जन्म", "बैंगलोर", "बेंगलुरु")),
    ("mood", ("khush", "happy", "udas", "sad", "akela", "akeli", "lonely", "bore", "ghabrahat",
              "खुश", "उदास", "अकेल", "घबराहट")),
]


def _clean(s: str) -> str:
    s = re.sub(r"\s*(--|…|\.\.\.)\s*", " ", s or "")
    s = re.sub(r"(^|\s)(अ|um|uh|hmm)(?=\s|[,.])", " ", s, flags=re.I)
    s = _FILLER_START.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.-–!।")
    return s


def _sentences(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?।])\s+|\n+", text or "") if p.strip()]


def categorize(fact: str) -> Optional[str]:
    low = fact.lower()
    # dislikes must win over likes ("pasand nahi")
    order = {"dislikes": 0, "life": 1}
    for cat, words in sorted(_CATEGORIES, key=lambda c: order.get(c[0], 2)):
        if any(w in low for w in words):
            return cat
    return None


def is_clean_statement(fact: str) -> bool:
    """A statement (not a question / command / fragment); first person NOT required."""
    f = _clean(fact)
    return 8 <= len(f) <= 240 and not _QUESTION.search(f) and not _COMMAND.search(f)


def is_usable_fact(fact: str) -> bool:
    """True for a real statement about the person; False for questions, commands and fragments."""
    f = _clean(fact)
    if not (12 <= len(f) <= 240):
        return False
    if _QUESTION.search(f) or _COMMAND.search(f):
        return False
    return bool(_FIRST_PERSON.search(" " + f + " "))


def extract_facts(user_turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """[{id, content}] -> [{category, fact, source_id}] using only clean first-person statements."""
    out, seen = [], set()
    for turn in user_turns:
        for sent in _sentences(turn.get("content", "")):
            fact = _clean(sent)
            if not is_usable_fact(fact) or _TRANSIENT.search(fact):
                continue
            cat = categorize(fact)
            if not cat:
                continue
            key = re.sub(r"\W+", " ", fact.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append({"category": cat, "fact": fact[:240], "source_id": turn.get("id")})
    return out


_TOPICS = [
    ("health", ("dard", "pain", "bukhar", "chakkar", "ulti", "saans", "khansi", "bp", "sugar", "दर्द", "बुखार", "चक्कर", "उल्टी", "सांस")),
    ("medicines", ("dawai", "dawa", "goli", "medicine", "tablet", "inhaler", "दवा", "दवाई", "गोली")),
    ("reports", ("report", "test", "jaanch", "रिपोर्ट", "जांच")),
    ("family", ("beta", "beti", "pota", "poti", "bahu", "rahul", "बेटा", "बेटी", "पोता", "बहू")),
    ("music", ("gaana", "gana", "bhajan", "song", "music", "गाना", "भजन")),
    ("their past", ("teacher", "school", "naukri", "gaon", "टीचर", "स्कूल", "नौकरी", "गांव")),
    ("feelings", ("udas", "akela", "akeli", "khush", "ghabrahat", "उदास", "अकेल", "खुश", "घबराहट")),
]


def summarize(messages: List[Dict[str, str]]) -> str:
    """Short English summary of what the member talked about (no raw transcript dump)."""
    user = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user:
        return "Check-in conversation completed."
    blob = " ".join(user).lower()
    topics = [name for name, words in _TOPICS if any(w in blob for w in words)]
    facts = extract_facts([{"id": "", "content": u} for u in user])
    concern = next((f["fact"] for f in facts if f["category"] == "health"), None)
    if not concern:  # health complaints are often short ("उल्टी वाला मन हो रहा है")
        concern = next((_clean(u) for u in user
                        if categorize(_clean(u)) == "health" and not _QUESTION.search(u) and len(_clean(u)) <= 120), None)
    parts = [f"Talked about {', '.join(topics)}." if topics else "Had a general chat."]
    if concern:
        parts.append(f"Health concern mentioned: \"{concern}\".")
    shared = next((f["fact"] for f in facts if f["category"] in ("family", "life", "likes", "routine")), None)
    if shared:
        parts.append(f"Shared: \"{shared}\".")
    return " ".join(parts)
