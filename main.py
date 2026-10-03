from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from embedder import embed
from cache import search, add, stats, save, load
from llm import ask_claude

load_dotenv()

THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.92))

@asynccontextmanager
async def lifespan(app):
    load()
    yield
    save()

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    text: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/query")
def query(q: Query):
    vec = embed(q.text)
    cached = search(vec, THRESHOLD)
    if cached:
        return {"response": cached, "source": "cache"}
    response = ask_claude(q.text)
    add(vec, response)
    return {"response": response, "source": "llm"}

@app.get("/stats")
def get_stats():
    return stats()