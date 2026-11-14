"""
AI assistant backend for the Lat/Long Wait dashboard.

Serves index.html and a /chat endpoint that answers questions with Gemini,
grounded in the neighborhood statistics from map_data.geojson.

Setup:  pip install -r requirements.txt
        put GEMINI_API_KEY=... in a .env file next to this script (never commit it)
Run:    uvicorn app:app --reload      then open http://127.0.0.1:8000
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# Tried in order; if one is overloaded (503/429) or times out, the next one answers.
MODELS = os.environ.get("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash,gemini-flash-latest").split(",")
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1)),
)

app = FastAPI()

# Allow the GitHub Pages site (or another host) to call this backend.
# Example: ALLOWED_ORIGINS=https://lleger0.github.io
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST"], allow_headers=["Content-Type"])


def load_neighborhoods():
    """Neighborhood statistics only (no geometry), so the prompt stays small."""
    with open(ROOT / "map_data.geojson") as f:
        geojson = json.load(f)
    rows = []
    for feature in geojson["features"]:
        p = feature["properties"]
        rows.append({
            "neighborhood": p["nta_name"],
            "borough": p["borough_name"],
            "unique_requests": p["n_requests"],
            "typical_actual_wait_days": p["median_actual_days"],
            "extra_wait_days_all_types": p["median_gap_days"],
            "extra_wait_days_heat": p["heat_gap"],
            "heat_requests": p["heat_n"],
            "extra_wait_days_plumbing": p["plumb_gap"],
            "plumbing_requests": p["plumb_n"],
            "pct_requests_slower_than_citywide_norm": p["share_waited_longer"],
            "pct_closed_no_access": p["no_access_share"],
            "pct_requests_from_10_busiest_buildings": p["top10_building_share"],
            "hotspot_simple_benchmark": p["hotspot"],
        })
    return rows


DATA_JSON = json.dumps(load_neighborhoods(), separators=(",", ":"))

SYSTEM_INSTRUCTION = f"""
You are the AI assistant on "Lat/Long Wait", a dashboard showing where NYC 311 service
falls behind. Answer questions using ONLY the neighborhood data below.

About the data (NYC 311 requests, 2024: Heat/Hot Water, Plumbing, Rodent):
- Each row is a 2020 Neighborhood Tabulation Area (NTA).
- "Extra wait" = actual wait minus the citywide median wait for the same complaint type,
  agency and month (median per neighborhood, in days). Positive = slower than similar
  requests citywide. It is NOT the total resolution time.
- Rows with fewer than 30 unique requests are too small to compare; say so if asked.
- "Closed as no access" means the request was closed without an inspector getting in;
  closed does not mean fixed.
- Hot spots come from Getis-Ord Gi* with false discovery rate correction. After
  correction, upper Manhattan (Harlem South and Harlem North) is the only significant
  slow cluster.
- Rodent requests mostly close the same day, so their waits are not meaningful.

Rules:
- Do not invent statistics. If the data cannot answer a question, say so.
- Do not confuse request volume with delays.
- Do not claim a neighborhood is discriminated against; the patterns are not causal.
- Keep answers short (under 150 words), in plain language, for city staff and residents.
- Write plain text only: no Markdown, no asterisks, no tables. Use simple dashes for lists.
- Round days to 2 decimals and percentages to whole numbers.

NEIGHBORHOOD DATA (JSON):
{DATA_JSON}
"""


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    view: dict = {}
    history: list[Turn] = Field(default=[], max_length=10)


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    contents = [
        types.Content(role="model" if t.role == "assistant" else "user", parts=[types.Part(text=t.content)])
        for t in request.history
    ]
    view = {k: str(v)[:100] for k, v in request.view.items()}
    contents.append(types.Content(role="user", parts=[types.Part(
        text=f"(The map currently shows: {json.dumps(view)})\n\n{request.question}")]))

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model.strip(),
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
            return {"answer": (response.text or "Sorry, I could not produce an answer.").strip()}
        except Exception as e:
            print(f"Gemini error ({model}): {e}")

    raise HTTPException(status_code=502, detail="The AI service is unavailable right now.")
