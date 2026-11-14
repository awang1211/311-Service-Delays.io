
import os
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# Gemini API client
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

class ChatRequest(BaseModel):
    question: str
    context: dict
    history: list = []

@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/chat")
def chat(request: ChatRequest):

    system_instructions = """
    You are the NYC 311 Service Equity Explorer AI Assistant.

    Answer questions using ONLY the map data provided.

    The data represents NYC 311 service requests from 2024.

    Explain statistics in simple language.

    Important:
    - Extra wait is the difference from similar citywide requests.
    - It is not the same as total resolution time.
    - Do not confuse request volume with service delays.
    - Do not claim that a neighborhood is disadvantaged
      or discriminated against without supporting evidence.
    - Do not invent statistics.
    - If the data does not answer a question, say so.
    - Explain that correlation does not establish causation.
    """

    prompt = f"""
    {system_instructions}

    MAP DATA:
    {json.dumps(request.context)}

    CONVERSATION HISTORY:
    {json.dumps(request.history)}

    USER QUESTION:
    {request.question}

    Answer clearly and concisely.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return {"answer": response.text}
