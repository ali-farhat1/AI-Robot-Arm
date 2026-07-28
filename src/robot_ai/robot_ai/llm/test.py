"""
Hybrid memory = structured fields (exact match) + semantic embeddings (fuzzy match).

Requires: pip install sentence-transformers
Run on your own machine (needs internet on first run to download the model).
"""

import time
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class HybridMemory:
    def __init__(self, save_path: str = "memory.pkl"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # fixed pretrained weights, nothing to save here
        self.save_path = save_path
        self.records = []      # list of dicts: {event, object, location, text, time}
        self.embeddings = None
        self.load()  # restore memory from disk if it exists

    def save(self):
        """Save records + embeddings to disk so memory survives a restart."""
        with open(self.save_path, "wb") as f:
            pickle.dump({"records": self.records, "embeddings": self.embeddings}, f)

    def load(self):
        """Load previously saved memory, if any."""
        try:
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
            self.records = data["records"]
            self.embeddings = data["embeddings"]
        except FileNotFoundError:
            pass  # no saved memory yet, start fresh

    def add(self, event: str, obj: str = None, location: str = None, text: str = None):
        """
        event/object/location = structured fields, used for exact keyword matching.
        text = free description, used for semantic fallback search.
        """
        text = text or f"{event} {obj or ''} {location or ''}".strip()
        record = {
            "event": event,
            "object": obj,
            "location": location,
            "text": text,
            "time": time.time(),
        }
        self.records.append(record)

        vec = self.model.encode([text])[0]
        self.embeddings = vec.reshape(1, -1) if self.embeddings is None else np.vstack([self.embeddings, vec])
        self.save()

    def _keyword_score(self, query: str, record: dict) -> float:
        """1.0 if any query word matches a structured field, else 0.0."""
        words = set(re.findall(r"[a-z]+", query.lower()))
        fields = " ".join(str(v) for v in [record["event"], record["object"], record["location"]] if v).lower()
        return 1.0 if words & set(re.findall(r"[a-z]+", fields)) else 0.0

    def _semantic_scores(self, query: str) -> np.ndarray:
        """Cosine similarity between query and every stored memory (0 to 1 range)."""
        q_vec = self.model.encode([query])
        return cosine_similarity(q_vec, self.embeddings)[0]

    def recall(self, query: str, top_k: int = 3, keyword_weight: float = 0.6):
        """
        Combined ranking: every memory gets ONE score = a weighted mix of
        keyword match (exact, binary) and semantic similarity (fuzzy, 0-1).
        keyword_weight controls how much to trust exact field matches vs.
        meaning-based similarity. Higher = trust keywords more.
        """
        semantic = self._semantic_scores(query)

        combined = []
        for i, r in enumerate(self.records):
            kw = self._keyword_score(query, r)
            score = keyword_weight * kw + (1 - keyword_weight) * semantic[i]
            combined.append((i, score, kw, semantic[i]))

        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score, kw, sem in combined[:top_k]:
            tag = f"combined={score:.2f} (kw={kw:.0f}, sem={sem:.2f})"
            results.append((self.records[i]["text"], tag))
        return results


if __name__ == "__main__":
    mem = HybridMemory()

    mem.add(event="picked_up", obj="mug", location="kitchen counter", text="Picked up a small red mug from the kitchen counter.")
    mem.add(event="navigated", location="hallway", text="Navigated down the hallway, avoided a chair.")
    mem.add(event="placed", obj="mug", location="sink", text="Placed the small red mug in the sink.")
    mem.add(event="detected_person", location="front door", text="Detected a person waving near the front door.")
    mem.add(event="charging", obj="battery", location="dock", text="Charged battery overnight.")
    mem.add(event="opened_door", location="front door", text="Opened the front door for the person who waved.")

    for q in ["Battery Overnight?", "was the mug big or small?", "did anything happen near the entrance?"]:
        print(f"\nQuery: {q}")
        for text, source in mem.recall(q):
            print(f"  ({source}) {text}")