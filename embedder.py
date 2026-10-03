from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0].astype("float32")