import time
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_first_call_is_llm():
    response = client.post("/query", json={"text": "what is the speed of light"})
    assert response.status_code == 200
    assert response.json()["source"] == "llm"

def test_same_query_hits_cache():
    client.post("/query", json={"text": "what is the boiling point of water"})
    response = client.post("/query", json={"text": "what is the boiling point of water"})
    assert response.json()["source"] == "cache"

def test_similar_query_hits_cache():
    client.post("/query", json={"text": "what is machine learning"})
    response = client.post("/query", json={"text": "what is machine learning"})
    assert response.json()["source"] == "cache"

def test_different_query_is_llm():
    client.post("/query", json={"text": "what is DNA"})
    response = client.post("/query", json={"text": "what is the capital of France"})
    assert response.json()["source"] == "llm"

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "hit_rate" in data
    assert "hits" in data
    assert "misses" in data

def test_cache_is_faster():
    query = {"text": "what is photosynthesis in plants"}

    start = time.time()
    client.post("/query", json=query)
    first_duration = time.time() - start

    start = time.time()
    client.post("/query", json=query)
    second_duration = time.time() - start

    print(f"\nLLM call: {first_duration:.2f}s")
    print(f"Cache hit: {second_duration:.2f}s")
    print(f"Speedup: {first_duration / second_duration:.1f}x faster")

    assert second_duration < first_duration