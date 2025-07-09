from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import os
from utils.prompt import SYSTEM_PROMPT
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI()

# Allow this frontend origin
origins = [
    "https://gangayaan.vercel.app"
]


# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(content={"status": "ok", "message": "API is healthy"})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")

    payload = {
        "model": "gemma2-9b-it",  # or "gemma-7b-it"
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        # ✅ This is where the response is created
        response = await client.post(GROQ_URL, headers=headers, json=payload)

        # ✅ Print Groq response inside this block
        print("Groq Response:", response.status_code, response.text)

        if response.status_code != 200:
            return {
                "error": "Groq API error",
                "status": response.status_code,
                "details": response.text
            }

        result = response.json()

        if "choices" not in result:
            return {
                "error": "Invalid Groq response format",
                "raw": result
            }

        return {
            "response": result["choices"][0]["message"]["content"]
        }
