import faiss
import numpy as np
import time
import os
from dotenv import load_dotenv

load_dotenv()

DIM = 384
TTL = float(os.getenv("TTL_SECONDS", 86400))

index = faiss.IndexFlatIP(DIM)
store = []

hits = 0
misses = 0

def search(vec: np.ndarray, threshold: float):
    global hits, misses
    if index.ntotal == 0:
        misses += 1
        return None
    D, I = index.search(vec.reshape(1, -1), k=1)
    score = D[0][0]
    idx = I[0][0]
    if score >= threshold:
        age = time.time() - store[idx]["timestamp"]
        if age <= TTL:
            hits += 1
            return store[idx]["response"]
    misses += 1
    return None

def add(vec: np.ndarray, response: str):
    index.add(vec.reshape(1, -1))
    store.append({"response": response, "timestamp": time.time()})

def stats():
    total = hits + misses
    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total > 0 else 0,
        "cached_entries": index.ntotal
    }

def save():
    import pickle
    faiss.write_index(index, "cache_index.faiss")
    with open("cache_store.pkl", "wb") as f:
        pickle.dump(store, f)

def load():
    global index
    import pickle
    if os.path.exists("cache_index.faiss") and os.path.exists("cache_store.pkl"):
        index = faiss.read_index("cache_index.faiss")
        with open("cache_store.pkl", "rb") as f:
            store.extend(pickle.load(f))