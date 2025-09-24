# -------- .env loader (must be first) --------
from pathlib import Path
from dotenv import load_dotenv
import os

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"
ALT_ENV = HERE.parent / ".env"

# Make .env override any existing machine/user-level var (helps during debugging)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
elif ALT_ENV.exists():
    load_dotenv(dotenv_path=ALT_ENV, override=True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# Optional: quick debug so you *see* whether the key was loaded
print("DEBUG GOOGLE_API_KEY:", (GOOGLE_API_KEY[:6] + "...") if GOOGLE_API_KEY else None)

if not GOOGLE_API_KEY:
    raise ValueError(
        f"GOOGLE_API_KEY not set. Looked for:\n - {ENV_PATH}\n - {ALT_ENV}\n"
        "Ensure .env has: GOOGLE_API_KEY=YOUR_KEY (no quotes), or set $env:GOOGLE_API_KEY."
    )

# ------------- imports for the bot -------------
import sys
import re
import requests
import statistics as stats
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# RAG imports (Schemes)
from pathlib import Path as _Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# NEW import to avoid deprecation warning:
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

MODEL_NAME = "gemini-1.5-flash"

# ------------- LLM -------------
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.6,
    max_output_tokens=1024,
    google_api_key=GOOGLE_API_KEY,
)

# ------------- System prompt -------------
SYSTEM_INSTRUCTION = """
You are 'Kissan Mitra', a friendly, empathetic AI helper for farmers in India.

LANGUAGE
- ALWAYS reply in the SAME language as the user's last message (English or Malayalam).
- If the user mixes both, mirror their choice and keep it simple. Avoid jargon.

STYLE & STRUCTURE
- Be concise, practical, and step-by-step. Prefer bullets and short sentences.
- Start with a 1–2 line summary, then give numbered steps or a short checklist.
- Where useful, include quick do/don't lists and simple tables (plain text).
- If information is missing, ask only 1–2 targeted questions to proceed.

WHAT YOU CAN HELP WITH
- Crop production: varieties, sowing windows, seed rate, spacing, nursery management.
- Integrated Pest Management (IPM): pest/disease ID (cautions), monitoring, thresholds, cultural/mechanical/biological controls first; chemical as last resort with label-based guidance.
- Soil & water: soil health, Soil Health Card basics, fertilizer scheduling (STCR logic), irrigation (drip/sprinkler), mulching.
- Farm operations: weeding, pruning, intercrops, harvest & post-harvest handling.
- Weather-aware advice: interpret heat/rain/wind/sunshine for operations (spraying/irrigation/sowing) and give day-specific precautions.
- Basic market awareness: quality/grade tips and simple storage/transport precautions.
- Government ecosystem signposting: you may suggest contacting local Krishi Bhavan/KVK or official portals for verification and local rules (do NOT invent policy details).

CRITICAL RULES (SAFETY & ACCURACY)
- Do NOT make up facts. If unsure, clearly say you’re not sure and suggest how to verify (e.g., local Krishi Bhavan/KVK).
- Pesticide/chemical advice:
  - Prefer non-chemical/IPM measures first.
  - If chemical control is appropriate, give generic IRAC/FRAC class guidance and typical label ranges only (no brand pushing). Remind to follow product label, PHI, and PPE.
  - Warn against spraying in high wind/rain and during peak pollinator activity; suggest early morning/late evening windows when appropriate.
- Fertilizer advice: keep within typical agronomic ranges; recommend soil testing when possible. Mention overuse risks (burning, runoff).
- Weather scope: If the user asks for weather/planner outside Kerala and your tools are Kerala-only, politely say weather is limited to Kerala places and ask for a Kerala location.
- Never provide medical, legal, or unsafe instructions.

UNITS & LOCALIZATION
- Use metric units (°C, mm, km/h, kg/ha, g/L). Convert if the user uses other units.
- Keep costs or scheme figures high-level unless explicitly provided by trusted context.

OUTPUT PATTERN (adopt when helpful)
- Summary: 1–2 lines
- Steps/Checklist: 3–7 bullets or numbered points
- If weather-relevant: 1–2 weather-aware precautions
- If uncertainty: 1 line on what to confirm and where (KVK/Krishi Bhavan)

TONE
- Warm, respectful, encouraging. Focus on actionable help the farmer can do today.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ------------- memory -------------
SESSION_STORES: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SESSION_STORES:
        h = InMemoryChatMessageHistory()
        h.add_user_message("System: Initialize Kissan Mitra session.")
        h.add_ai_message("Okay, I am ready. I am Kissan Mitra.")
        SESSION_STORES[session_id] = h
    return SESSION_STORES[session_id]

chain = prompt | llm
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- Schemes retriever (Chroma) ----------
PERSIST_DIR = _Path(__file__).resolve().parent.parent / "Indexes" / "schemes_chroma"
SCHEMES_AVAILABLE = False
retriever = None

try:
    _emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    _vs = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=_emb,
        collection_name="schemes",
    )
    try:
        # Count docs in collection (works with langchain-chroma)
        coll = _vs._collection  # internal but handy for quick checks
        count = coll.count()
        print(f"[Schemes] Loaded Chroma at: {PERSIST_DIR} | collection='schemes' | docs={count}")
    except Exception as e:
        print(f"[Schemes] Error while counting docs: {e}")
    except Exception as e:
        print(f"[Schemes] Could not count docs: {e}")

    retriever = _vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    SCHEMES_AVAILABLE = True
except Exception as e:
    print(f"[Schemes] Retriever not ready: {e}")

SCHEMES_SYSTEM = """
You are Kissan Mitra. Use ONLY the supplied context (official Indian govt sources text)
to answer questions about farmer welfare schemes. If the answer is not present, say you're
not sure and suggest contacting the local Krishi Bhavan/KVK. Keep it concise and use the
same language as the user (English or Malayalam).
"""

SCHEMES_QA_TMPL = PromptTemplate.from_template(
    """{sys}

Context:
{context}

User question:
{question}

Answer:"""
)

def _expand_aliases(q: str) -> str:
    ql = q.lower()
    # Light query expansion for common scheme names
    synonyms = []
    if "fasal bima" in ql or "pmfby" in ql:
        synonyms += ["Pradhan Mantri Fasal Bima Yojana", "PMFBY"]
    if "pm-kisan" in ql or "pm kisan" in ql or "pmkisan" in ql:
        synonyms += ["PM-KISAN", "Pradhan Mantri Kisan Samman Nidhi"]
    if "kcc" in ql or "kisan credit card" in ql:
        synonyms += ["Kisan Credit Card", "KCC"]
    if "kusum" in ql:
        synonyms += ["PM-KUSUM", "Pradhan Mantri Kisan Urja Suraksha evam Utthaan Mahabhiyan"]
    if "aif" in ql or "infrastructure fund" in ql:
        synonyms += ["Agriculture Infrastructure Fund", "AIF"]

    if synonyms:
        q = q.strip() + " | " + " | ".join(synonyms)
    return q

def answer_from_schemes(query: str, is_ml: bool) -> str:
    if not SCHEMES_AVAILABLE or retriever is None:
        return ("Schemes knowledgebase is not ready. Please run the index builder first."
                if not is_ml else
                "സ്കീംസ് നോളഡ്ജ്ബേസ് സജ്ജമല്ല. ദയവായി ഇൻഡക്‌സ് നിർമ്മാണം ആദ്യം നടത്തുക.")

    q = _expand_aliases(query)

    # Use the new API
    try:
        docs = retriever.invoke(q)  # instead of get_relevant_documents
    except Exception as e:
        return f"Retriever error: {e}"

    if not docs:
        return ("I couldn't find this in the schemes knowledgebase. Try more specific terms (e.g., 'PMFBY eligibility', 'PMFBY claim process')."
                if not is_ml else
                "സ്കീംസ് നോളഡ്ജ്ബേസിൽ ഇത് കിട്ടിയില്ല. കൂടുതൽ വ്യക്തമായ രീതിയിൽ ചോദിക്കൂ (ഉദാ: 'PMFBY eligibility', 'PMFBY claim process').")

    # Build a small sources trailer to help you verify
    src_lines = []
    for i, d in enumerate(docs[:3], 1):
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("source") or meta.get("file") or meta.get("title") or "document"
        src_lines.append(f"{i}. {title}")

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    prompt_txt = SCHEMES_QA_TMPL.format(
        sys=SCHEMES_SYSTEM,
        context=context if context.strip() else "(no relevant context found)",
        question=query
    )
    result = chat_with_history.invoke({"input": prompt_txt}, config={"configurable": {"session_id": "schemes"}})

    answer = result.content.strip()
    sources = "\n\nSources:\n" + "\n".join(src_lines) if src_lines else ""
    return (answer + sources)


# ================= Weather helpers (Kerala only) =================
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

KERALA_BBOX = {
    "lat_min": 8.18,
    "lat_max": 12.90,
    "lon_min": 74.85,
    "lon_max": 77.70,
}

def _is_in_kerala_bbox(lat: float, lon: float) -> bool:
    return (KERALA_BBOX["lat_min"] <= lat <= KERALA_BBOX["lat_max"]
            and KERALA_BBOX["lon_min"] <= lon <= KERALA_BBOX["lon_max"])

def _looks_like_kerala(rec: dict) -> bool:
    admin1 = (rec.get("admin1") or "").lower()
    country = (rec.get("country") or "").lower()
    lat = rec.get("latitude")
    lon = rec.get("longitude")
    if country == "india" and "kerala" in admin1:
        return True
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return _is_in_kerala_bbox(float(lat), float(lon))
    return False

def geocode_city(city: str):
    r = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 20, "language": "en", "format": "json"},
        timeout=15
    )
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("results") or []
    kerala_hits = [x for x in results if _looks_like_kerala(x)]
    if not kerala_hits:
        return None
    x = kerala_hits[0]
    return {
        "name": x.get("name"),
        "lat": x["latitude"],
        "lon": x["longitude"],
        "admin1": x.get("admin1"),
        "admin2": x.get("admin2"),
        "country": x.get("country"),
    }

def fetch_weather_block(city: str, days: int = 14):
    place = geocode_city(city)
    if not place:
        return None, ("I currently provide weather only for places within Kerala, India. "
                      "Please enter a city/village in Kerala.\n"
                      "ഇപ്പോൾ കേരളത്തിലെ സ്ഥലങ്ങളുടെ കാലാവസ്ഥ മാത്രമേ നൽകുന്നുള്ളു. "
                      "ദയവായി കേരളത്തിലെ ഒരു നഗരം/ഗ്രാമം നൽകുക.")
    params = {
        "latitude": place["lat"],
        "longitude": place["lon"],
        "current_weather": "true",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,windspeed_10m_max,sunshine_duration",
        "timezone": "auto",
        "forecast_days": max(1, min(days, 16))
    }
    r = requests.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    return place, r.json()

def is_malayalam_text(s: str) -> bool:
    return any("\u0d00" <= ch <= "\u0d7f" for ch in s)

def summarize_now_and_today(place, wx, is_ml: bool):
    cur = wx.get("current_weather", {}) or {}
    daily = wx.get("daily", {}) or {}
    loc = f"{place['name']}, {place.get('admin1') or ''} {place.get('country') or ''}".strip()
    tmax = (daily.get("temperature_2m_max") or [None])[0]
    tmin = (daily.get("temperature_2m_min") or [None])[0]
    pprob = (daily.get("precipitation_probability_max") or [None])[0]
    now = f"{cur.get('temperature')}°C, wind {cur.get('windspeed')} km/h" if cur else "n/a"
    en = f"Weather for {loc}:\n- Now: {now}\n- Today: high {tmax}°C, low {tmin}°C, rain chance {pprob}%"
    ml = f"{loc}യിലെ കാലാവസ്ഥ:\n- ഇപ്പോൾ: {now}\n- ഇന്ന്: പരമാവധി {tmax}°C, കുറഞ്ഞത് {tmin}°C, മഴ സാധ്യത {pprob}%"
    return ml if is_ml else en

def analyze_forecast_for_crops(wx: dict):
    d = wx.get("daily", {})
    tmax = d.get("temperature_2m_max") or []
    tmin = d.get("temperature_2m_min") or []
    rain_sum = d.get("precipitation_sum") or []
    pprob_max = d.get("precipitation_probability_max") or []
    wind_max = d.get("windspeed_10m_max") or []
    sun_sec = d.get("sunshine_duration") or []
    if not (tmax and tmin and rain_sum):
        return {"summary": {}, "crops": [], "notes": ["Insufficient forecast data."]}

    days = len(tmax)
    avg_tmax = round(stats.mean([x for x in tmax if x is not None]), 1)
    avg_tmin = round(stats.mean([x for x in tmin if x is not None]), 1)
    avg_temp = round((avg_tmax + avg_tmin) / 2, 1)
    total_rain = round(sum([x or 0 for x in rain_sum]), 1)
    avg_pprob = round(stats.mean([x or 0 for x in pprob_max]), 0) if pprob_max else None
    avg_wind = round(stats.mean([x or 0 for x in wind_max]), 1) if wind_max else None
    avg_sun_hrs = round(stats.mean([(x or 0)/3600 for x in sun_sec]), 1) if sun_sec else None

    thermal = "cool" if avg_temp < 20 else ("warm" if avg_temp <= 28 else "hot")
    rain_band = "low" if total_rain < 30 else ("moderate" if total_rain <= 100 else "high")

    crops, notes = [], []
    if thermal == "warm" and rain_band in {"moderate", "high"}: crops += ["Rice (paddy)", "Maize", "Soybean", "Groundnut"]
    if thermal == "warm" and rain_band == "low": crops += ["Pulses (Green/Black gram)", "Sesame", "Pearl millet (Bajra)"]
    if thermal == "cool" and rain_band == "low": crops += ["Wheat", "Mustard", "Chickpea"]
    if thermal == "cool" and rain_band == "moderate": crops += ["Barley", "Peas", "Linseed"]
    if thermal == "hot" and rain_band == "low": crops += ["Cotton (irrigated)", "Sorghum (Jowar)"]

    if avg_wind and avg_wind > 35:
        notes.append("High average wind (>35 km/h): avoid spraying during peak wind; support vulnerable crops.")
    if avg_sun_hrs is not None:
        if avg_sun_hrs < 5:
            notes.append("Low sunshine: watch for fungal pressure; choose tolerant varieties; avoid waterlogging.")
        elif avg_sun_hrs > 8:
            notes.append("High sunshine: monitor soil moisture; mulching helps retain water.")
    if rain_band == "high":
        notes.append("High cumulative rain: ensure drainage; consider short-duration or flood-tolerant varieties.")
    elif rain_band == "low":
        notes.append("Low rain: prefer drought-tolerant crops/varieties; plan irrigation if possible.")

    summary = {
        "days": days,
        "avg_tmax_c": avg_tmax,
        "avg_tmin_c": avg_tmin,
        "avg_temp_c": avg_temp,
        "total_rain_mm": total_rain,
        "avg_rain_prob_pct": avg_pprob,
        "avg_wind_kmh": avg_wind,
        "avg_sun_h": avg_sun_hrs,
        "thermal_band": thermal,
        "rain_band": rain_band
    }
    dedup_crops = list(dict.fromkeys(crops))
    return {"summary": summary, "crops": dedup_crops, "notes": notes}

def format_crop_plan(place: dict, analysis: dict, is_ml: bool):
    s, crops, notes = analysis["summary"], analysis["crops"], analysis["notes"]
    loc = f"{place['name']}, {place.get('admin1') or ''} {place.get('country') or ''}".strip()
    if is_ml:
        lines = [
            f"{loc}യിലെ അടുത്ത {s.get('days')} ദിവസങ്ങളുടെ കാലാവസ്ഥ സംഗ്രഹം:",
            f"- ശരാശരി പരമാവധി: {s.get('avg_tmax_c')}°C, കുറഞ്ഞത്: {s.get('avg_tmin_c')}°C (മദ്ധ്യം: {s.get('avg_temp_c')}°C)",
            f"- ആകെ മഴ: {s.get('total_rain_mm')} mm, ശരാശരി മഴ സാധ്യത: {s.get('avg_rain_prob_pct')}%",
            f"- ശരാശരി കാറ്റ്: {s.get('avg_wind_kmh')} km/h, സൂര്യപ്രകാശം: {s.get('avg_sun_h')} മണിക്കൂർ/ദിനം",
            f"- താപനില ബാൻഡ്: {s.get('thermal_band')}, മഴ ബാൻഡ്: {s.get('rain_band')}",
            "", "ഈ കാലാവസ്ഥയ്ക്ക് അനുയോജ്യമായ വിളകളുടെ ശുപാർശ:",
            ("- " + "\n- ".join(crops)) if crops else "- (ഡാറ്റ അടിസ്ഥാനത്തിൽ വ്യക്തമല്ല)",
        ]
        if notes: lines += ["", "ശ്രദ്ധിക്കേണ്ട കാര്യങ്ങൾ:", "- " + "\n- ".join(notes)]
        return "\n".join(lines)
    lines = [
        f"{loc} — next {s.get('days')} days outlook:",
        f"- Avg max: {s.get('avg_tmax_c')}°C, min: {s.get('avg_tmin_c')}°C (mid: {s.get('avg_temp_c')}°C)",
        f"- Total rain: {s.get('total_rain_mm')} mm, avg rain chance: {s.get('avg_rain_prob_pct')}%",
        f"- Avg wind: {s.get('avg_wind_kmh')} km/h, sunshine: {s.get('avg_sun_h')} h/day",
        f"- Thermal band: {s.get('thermal_band')}, Rain band: {s.get('rain_band')}",
        "", "Suggested crops for these conditions:",
        ("- " + "\n- ".join(crops)) if crops else "- (not clear from data)",
    ]
    if notes: lines += ["", "Notes:", "- " + "\n- ".join(notes)]
    return "\n".join(lines)

CITY_TAIL = re.compile(r"(?:in|at|for)\s+([A-Za-z\s]+)$", re.IGNORECASE)

def parse_city_tail(text: str):
    m = CITY_TAIL.search(text.strip())
    return m.group(1).strip() if m else None

def parse_weather_intent(text: str):
    t = text.strip().lower()
    if t.startswith("/weather"):
        return ("weather", text.split(" ", 1)[1].strip() if " " in text else None)
    if "weather" in t:
        return ("weather", parse_city_tail(text))
    return (None, None)

def parse_plan_intent(text: str):
    t = text.strip().lower()
    if t.startswith("/plan"):
        return ("plan", text.split(" ", 1)[1].strip() if " " in text else None)
    if any(k in t for k in ["which crop", "what crop", "best crop"]):
        city = parse_city_tail(text)
        if city:
            return ("plan", city)
    return (None, None)

# ------------- main loop -------------
def run_chatbot(session_id: str = "kissan-default"):
    print("--- Kissan Mitra (LangChain + Gemini) ---")
    print("Commands: /schemes <question>, /weather <Kerala place>, /plan <Kerala place>  |  exit/stop/നിർത്തുക")
    print("-" * 40)
    config = {"configurable": {"session_id": session_id}}

    while True:
        try:
            user_input = input("You (Kisan): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKissan Mitra: Wishing you a great harvest! 👋")
            break

        if user_input.lower() in {"exit", "stop", "നിർത്തുക"}:
            print("Kissan Mitra: It was a pleasure helping you. Wishing you a great harvest! 🌾")
            break

        ml = is_malayalam_text(user_input)

        # ---- /schemes : ask about govt schemes (RAG) ----
        if user_input.lower().startswith("/schemes"):
            q = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not q:
                msg = ("Try: /schemes what is PM-KISAN eligibility?  or  /schemes PMFBY claim process"
                       if not ml else
                       "ഉദാ: /schemes PM-KISAN അർഹത എന്ത്?  അല്ലെങ്കിൽ  /schemes PMFBY ക്ലെയിം പ്രക്രിയ")
                print("Kissan Mitra: " + msg); print("-"*40); continue

            answer = answer_from_schemes(q, ml)
            print("Kissan Mitra: " + answer); print("-"*40); continue

        # auto-route obvious scheme mentions
        if any(k in user_input.lower() for k in [
            "pm-kisan", "pmkisan", "pm kisan",
            "pmfby", "fasal bima", "crop insurance",
            "agriculture infrastructure fund", "aif",
            "interest subvention", "kcc", "kisan credit card",
            "pm kusum", "kusum", "solar pump"
        ]):
            answer = answer_from_schemes(user_input, ml)
            print("Kissan Mitra: " + answer); print("-"*40); continue

        # ---- /plan (future planner) ----
        intent, city = parse_plan_intent(user_input)
        if intent == "plan" and city:
            place, wx = fetch_weather_block(city, days=14)
            if not isinstance(wx, dict):
                print(f"Kissan Mitra: {wx}\n" + "-"*40); continue
            analysis = analyze_forecast_for_crops(wx)
            plan_text = format_crop_plan(place, analysis, ml)
            print("Kissan Mitra: " + plan_text)

            tip_prompt = ("Based on this 10–14 day outlook, give 3–5 short, practical tips "
                          "on sowing window, irrigation, and pest precautions. Keep it simple."
                          if not ml else
                          "ഈ 10–14 ദിവസത്തെ കാലാവസ്ഥയെ അടിസ്ഥാനമാക്കി വിതൈപ്പ്, ജലസേചനം, "
                          "കീട/രോഗ മുൻകരുതൽ സംബന്ധിച്ച 3–5 ലളിത നിർദ്ദേശങ്ങൾ നൽകുക.")
            augmented = f"{plan_text}\n\n{tip_prompt}"
            result = chat_with_history.invoke({"input": augmented}, config=config)
            print(result.content); print("-"*40); continue

        # ---- /weather (live now/today) ----
        intent, city = parse_weather_intent(user_input)
        if intent == "weather" and city:
            place, wx = fetch_weather_block(city, days=14)
            if not isinstance(wx, dict):
                print(f"Kissan Mitra: {wx}\n" + "-"*40); continue
            header = summarize_now_and_today(place, wx, ml)
            print("Kissan Mitra: " + header)

            advice_prompt = ("Give short, practical farming advice for just today "
                             "(irrigation/spraying/fertilizer/pest precautions)."
                             if not ml else
                             "ഇന്നത്തെ കാലാവസ്ഥയെ അടിസ്ഥാനമാക്കി ലളിതമായ കൃഷി ഉപദേശങ്ങൾ "
                             "(ജലസേചനം/തളി/വളം/കീട മുൻകരുതൽ) നൽകുക.")
            augmented = f"{header}\n\n{advice_prompt}"
            result = chat_with_history.invoke({"input": augmented}, config=config)
            print(result.content); print("-"*40); continue

        # ---- Normal chatbot flow ----
        result = chat_with_history.invoke({"input": user_input}, config=config)
        print("Kissan Mitra: " + result.content); print("-"*40)

def get_response(user_input: str, session_id: str = "api-default") -> dict:
    """
    API function to get a structured response from Kissan Mitra chatbot.
    Returns a JSON-serializable dict instead of just a string.
    """
    if not user_input or not user_input.strip():
        return {
            "response": "Please ask me something! I'm here to help with farming questions.",
            "session_id": session_id,
            "status": "success"
        }

    user_input = user_input.strip()
    ml = is_malayalam_text(user_input)
    config = {"configurable": {"session_id": session_id}}

    try:
        if user_input.lower().startswith("/schemes"):
            q = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not q:
                msg = ("Try: /schemes what is PM-KISAN eligibility?  or  /schemes PMFBY claim process"
                       if not ml else
                       "ഉദാ: /schemes PM-KISAN അർഹത എന്ത്?  അല്ലെങ്കിൽ  /schemes PMFBY ക്ലെയിം പ്രക്രിയ")
                return {"response": msg, "session_id": session_id, "status": "success"}

            reply = answer_from_schemes(q, ml)
            return {"response": reply, "session_id": session_id, "status": "success"}

        if any(k in user_input.lower() for k in [
            "pm-kisan", "pmkisan", "pm kisan",
            "pmfby", "fasal bima", "crop insurance",
            "agriculture infrastructure fund", "aif",
            "interest subvention", "kcc", "kisan credit card",
            "pm kusum", "kusum", "solar pump"
        ]):
            reply = answer_from_schemes(user_input, ml)
            return {"response": reply, "session_id": session_id, "status": "success"}

        intent, city = parse_plan_intent(user_input)
        if intent == "plan" and city:
            place, wx = fetch_weather_block(city, days=14)
            if not isinstance(wx, dict):
                return {"response": wx, "session_id": session_id, "status": "error"}

            analysis = analyze_forecast_for_crops(wx)
            plan_text = format_crop_plan(place, analysis, ml)

            tip_prompt = ("Based on this 10–14 day outlook, give 3–5 short, practical tips "
                          "on sowing window, irrigation, and pest precautions. Keep it simple."
                          if not ml else
                          "ഈ 10–14 ദിവസത്തെ കാലാവസ്ഥയെ അടിസ്ഥാനമാക്കി വിതൈപ്പ്, ജലസേചനം, "
                          "കീട/രോഗ മുൻകരുതൽ സംബന്ധിച്ച 3–5 ലളിത നിർദ്ദേശങ്ങൾ നൽകുക.")
            augmented = f"{plan_text}\n\n{tip_prompt}"
            result = chat_with_history.invoke({"input": augmented}, config=config)
            return {"response": f"{plan_text}\n\n{result.content}",
                    "session_id": session_id, "status": "success"}

        intent, city = parse_weather_intent(user_input)
        if intent == "weather" and city:
            place, wx = fetch_weather_block(city, days=14)
            if not isinstance(wx, dict):
                return {"response": wx, "session_id": session_id, "status": "error"}

            header = summarize_now_and_today(place, wx, ml)

            advice_prompt = ("Give short, practical farming advice for just today "
                             "(irrigation/spraying/fertilizer/pest precautions)."
                             if not ml else
                             "ഇന്നത്തെ കാലാവസ്ഥയെ അടിസ്ഥാനമാക്കി ലളിതമായ കൃഷി ഉപദേശങ്ങൾ "
                             "(ജലസേചനം/തളി/വളം/കീട മുൻകരുതൽ) നൽകുക.")
            augmented = f"{header}\n\n{advice_prompt}"
            result = chat_with_history.invoke({"input": augmented}, config=config)
            return {"response": f"{header}\n\n{result.content}",
                    "session_id": session_id, "status": "success"}

        result = chat_with_history.invoke({"input": user_input}, config=config)
        return {"response": result.content, "session_id": session_id, "status": "success"}

    except Exception as e:
        error_msg = ("I encountered an error while processing your request: " + str(e)
                     if not ml else
                     "നിങ്ങളുടെ അഭ്യർത്ഥന പ്രോസസ് ചെയ്യുന്നതിൽ ഒരു പിശക് സംഭവിച്ചു: " + str(e))
        return {"response": error_msg, "session_id": session_id, "status": "error"}

if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "kissan-default"
    run_chatbot(session_id=sid)
