# Semantic Cache for LLM APIs

A small FastAPI service that sits in front of an LLM and **reuses answers for questions that mean the same thing**, not only questions that match word-for-word.

Exact-match caches fail on “reset campus Wi‑Fi” vs “how do I reset university wifi login?”. This layer embeds each query, searches previous queries with FAISS, and returns the stored answer when cosine similarity is high enough. That skip is the whole product: **no Groq tokens, much lower latency**.

A demo UI is included: **Northline Helpdesk**, a campus IT support desk that talks to this cache.

---

## Why this exists

LLM calls are slow and billed per token. Support, FAQ, and internal-search traffic is full of repeats and paraphrases. Caching those calls:

- Cuts API spend on every **cache hit** (zero tokens)
- Turns a multi-second Groq round-trip into tens of milliseconds
- Still calls the model on genuinely new questions

Typical place to use it: a chatbot or helpdesk API. The UI in this repo is that pattern in a browser.

---

## How it works

```
User question
    → embed with all-MiniLM-L6-v2 (384-d, L2-normalized)
    → FAISS inner-product search (cosine similarity)
    → if score ≥ threshold and entry is within TTL
          return stored answer   (source: cache)
    → else
          call Groq              (source: llm)
          store vector + answer
```


| Piece               | Role                                                   |
| ------------------- | ------------------------------------------------------ |
| `embedder.py`       | Sentence-Transformers `all-MiniLM-L6-v2`               |
| `cache.py`          | FAISS `IndexFlatIP`, hit/miss counters, TTL, save/load |
| `llm.py`            | Groq chat completions                                  |
| `main.py`           | FastAPI: UI, `/query`, `/stats`                        |
| `static/index.html` | Helpdesk UI                                            |
| `test_cache.py`     | pytest against the API                                 |


Default similarity threshold is **0.92**. Default TTL is **24 hours**. Both are overridable via env vars.

On shutdown the index is written to `cache_index.faiss` and answers to `cache_store.pkl`. Those files reload on the next start. **Hit/miss counters do not persist**; they reset when the process restarts. `cached_entries` can therefore be larger than `total` for the current run.

---

## Demo application (web UI)

With the server running, open:

**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

You should see **Ask campus support**. This is not Swagger. Interactive API docs stay at `/docs`.

1. Click **Wi‑Fi reset** — first reply is a Groq call (`Groq · new answer`).
2. Click **Wi‑Fi reset** again — same answer from cache (`Cache hit · $0 tokens`).
3. Use **Similar Wi‑Fi** to try a paraphrase (may miss if similarity is under 0.92).
4. The **Cache this shift** panel is live `/stats`.

You can also type any question in the box.

---

## Local setup

**Requirements:** Python 3.11+ (3.13 works), a [Groq API key](https://console.groq.com/).

```bash
cd semantic-cache-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root (never commit this file):

```
GROQ_API_KEY=your_groq_key_here
```

Optional:

```
GROQ_MODEL=openai/gpt-oss-20b
SIMILARITY_THRESHOLD=0.92
TTL_SECONDS=86400
```

The default Groq model is `openai/gpt-oss-20b`. Groq retired `llama-3.1-8b-instant` for free/developer keys in August 2026.

Start the API:

```bash
source .venv/bin/activate
uvicorn main:app --reload
```

First boot downloads the embedding model from Hugging Face and can take a minute. A “set `HF_TOKEN`” warning is optional and safe to ignore locally.

Always run `uvicorn` **from the venv**. System Python will raise `No module named 'sentence_transformers'`.

---

## API


| Method | Path     | Purpose           |
| ------ | -------- | ----------------- |
| `GET`  | `/`      | Helpdesk UI       |
| `POST` | `/query` | Ask a question    |
| `GET`  | `/stats` | Cache metrics     |
| `GET`  | `/docs`  | OpenAPI / Swagger |


**POST `/query`**

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "what is the speed of light"}'
```

```json
{
  "response": "The speed of light in a vacuum is …",
  "source": "llm"
}
```

`source` is `"llm"` or `"cache"`. Repeat the same body to get `"cache"`.

**GET `/stats`**

```json
{
  "total": 2,
  "hits": 1,
  "misses": 1,
  "hit_rate": 0.5,
  "cached_entries": 1
}
```

---

## Tests

Tests load the real embedder and call Groq. They need the venv and `GROQ_API_KEY`.

```bash
source .venv/bin/activate
pytest test_cache.py -v
```


| Test                            | Expectation                                |
| ------------------------------- | ------------------------------------------ |
| `test_first_call_is_llm`        | New question → `source: llm`               |
| `test_same_query_hits_cache`    | Identical repeat → `cache`                 |
| `test_similar_query_hits_cache` | Same text again → `cache`                  |
| `test_different_query_is_llm`   | Unrelated question → `llm`                 |
| `test_stats_endpoint`           | Stats payload has hits / misses / hit_rate |
| `test_cache_is_faster`          | Second call is faster than the first       |


---

## Stack

- FastAPI + Uvicorn
- sentence-transformers (`all-MiniLM-L6-v2`)
- FAISS (`faiss-cpu`)
- Groq (`openai/gpt-oss-20b` by default)

---

## Project layout

```
semantic-cache-main/
  main.py              # app, routes
  embedder.py          # query embeddings
  cache.py             # FAISS cache + stats
  llm.py               # Groq client
  test_cache.py
  static/index.html    # Northline Helpdesk UI
  requirements.txt
  .env                 # local secrets (gitignored)
```
## Result 

<img width="962" height="795" alt="Screenshot 2026-10-03 at 12 34 47 PM" src="https://github.com/user-attachments/assets/5eef3b10-405f-4b53-93f0-a0b1e7f61dbc" />
<img width="549" height="667" alt="Screenshot 2026-10-03 at 12 35 04 PM" src="https://github.com/user-attachments/assets/f8cb3b5a-eac5-4593-a410-677270870fea" />
<img width="634" height="703" alt="Screenshot 2026-10-03 at 12 35 20 PM" src="https://github.com/user-attachments/assets/60738c43-45ee-4be6-8843-d75b2b0e293a" />


