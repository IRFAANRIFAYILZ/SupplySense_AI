"""
retrieval.py
------------
RAG Step 5: Retrieve the most relevant document chunks for a user's question.

Works with whichever embedding backend ingestion.py used (sentence-transformers
or the TF-IDF fallback) -- this module reads rag/index/meta.json to know which
one, and re-embeds the query the same way the documents were embedded, then
ranks chunks by cosine similarity.
"""

import json
import pickle
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).parent / "index"


class Retriever:
    def __init__(self):
        with open(INDEX_DIR / "meta.json") as f:
            self.meta = json.load(f)
        with open(INDEX_DIR / "chunks.json") as f:
            self.chunks = json.load(f)
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")

        self.backend = self.meta["backend"]
        self.model = None
        self.vectorizer = None

        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            with open(INDEX_DIR / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "sentence-transformers":
            vec = self.model.encode([query], normalize_embeddings=True)[0]
        else:
            vec = self.vectorizer.transform([query]).toarray()[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.20):
        """Returns top_k chunks with cosine-similarity scores.
        Chunks below min_score are dropped -- this is what lets the system
        honestly say 'not enough information' instead of forcing an answer."""
        q_vec = self._embed_query(query)
        scores = self.embeddings @ q_vec  # cosine similarity (vectors are normalized)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "doc": chunk["doc"],
                "section": chunk["section"],
                "score": round(score, 4),
            })
        return results


if __name__ == "__main__":
    r = Retriever()
    test_queries = [
        "What is the penalty for late delivery?",
        "What is the minimum inventory level?",
        "What is the capital of France?",  # should retrieve nothing relevant
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for res in r.retrieve(q, top_k=3):
            print(f"  [{res['score']}] {res['doc']} — {res['section']}")
        if not r.retrieve(q, top_k=3):
            print("  (no sufficiently relevant chunks found)")
