import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def ask_claude(query: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": query}],
        },
        timeout=60,
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"Groq error: {data}")
    content = data["choices"][0]["message"].get("content")
    if not content:
        raise ValueError(f"Groq returned an empty message: {data}")
    return content