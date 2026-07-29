"""
Skye's long-term memory: multiple categories, each with fields that suit
what they represent, unified under one hybrid (keyword + semantic) search.

Requires: pip install sentence-transformers
"""

import time
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Which structured fields belong to each category. `text` and `time` are
# added automatically to every record regardless of category.
CATEGORY_FIELDS = {
    "event":       ["what", "outcome"],
    "people":      ["name", "relation", "facts"],
    "curiosities": ["question", "context"],
    "procedures":  ["name", "steps"],
    "reflections": ["insight", "related_to"],
}


class SkyeMemory:
    def __init__(self, save_path: str = "skye_memory.pkl"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.save_path = save_path
        self.records = []
        self.embeddings = None
        self.load()

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump({"records": self.records, "embeddings": self.embeddings}, f)

    def load(self):
        try:
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
            self.records = data["records"]
            self.embeddings = data["embeddings"]
        except FileNotFoundError:
            pass

    def add(self, category: str, text: str = None, **fields):
        """
        category: one of "event", "people", "curiosities", "procedures", "reflections"
        fields: whichever structured fields belong to that category (see CATEGORY_FIELDS)
        text: free description for semantic search. Auto-built from fields if not given.
        """
        if category not in CATEGORY_FIELDS:
            raise ValueError(f"Unknown category '{category}'. Choose from {list(CATEGORY_FIELDS)}")

        allowed = set(CATEGORY_FIELDS[category])
        unexpected = set(fields) - allowed
        if unexpected:
            raise ValueError(f"'{category}' doesn't use fields {unexpected}. Allowed: {allowed}")

        if text is None:
            text = " ".join(str(v) for v in fields.values() if v)

        record = {"category": category, "text": text, "time": time.time(), **fields}
        self.records.append(record)

        vec = self.model.encode([text])[0]
        self.embeddings = vec.reshape(1, -1) if self.embeddings is None else np.vstack([self.embeddings, vec])
        self.save()

    def _keyword_score(self, query: str, record: dict) -> float:
        words = set(re.findall(r"[a-z]+", query.lower()))
        field_text = " ".join(str(v) for k, v in record.items() if k not in ("time", "text") and v).lower()
        return 1.0 if words & set(re.findall(r"[a-z]+", field_text)) else 0.0

    def _semantic_scores(self, query: str) -> np.ndarray:
        q_vec = self.model.encode([query])
        return cosine_similarity(q_vec, self.embeddings)[0]

    def recall(self, query: str, top_k: int = 3, category: str = None, keyword_weight: float = 0.6):
        """
        category: optionally restrict search to one category (e.g. only "people").
        Leave as None to search everything.
        """
        semantic = self._semantic_scores(query)

        combined = []
        for i, r in enumerate(self.records):
            if category and r["category"] != category:
                continue
            kw = self._keyword_score(query, r)
            score = keyword_weight * kw + (1 - keyword_weight) * semantic[i]
            combined.append((i, score, kw, semantic[i]))

        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score, kw, sem in combined[:top_k]:
            r = self.records[i]
            tag = f"[{r['category']}] combined={score:.2f} (kw={kw:.0f}, sem={sem:.2f})"
            results.append((r["text"], tag))
        return results


if __name__ == "__main__":
    mem = SkyeMemory()

    mem.add("event", what="Grasped the mug on the first try", outcome="success")
    mem.add("people", name="Ali", relation="creator", facts="Built me, works on ROS2 arm projects")
    mem.add("curiosities", question="Why do soft objects need less grip force?", context="noticed during grasp attempts")
    mem.add("procedures", name="pick_and_place_mug", steps="approach, grip at 40% force, lift, move to target, release")
    mem.add("reflections", insight="I tend to drop objects when moving too fast after grasping", related_to="grasp failures")

    print("All memories:")
    for r in mem.records:
        print(" -", r["category"], "|", r["text"])

    print("\nQuery: 'who made me?'")
    for text, tag in mem.recall("who made me?"):
        print(f"  {tag} {text}")

    print("\nQuery: 'anything about dropping things?' (procedures + reflections only searched)")
    for text, tag in mem.recall("anything about dropping things?", category="reflections"):
        print(f"  {tag} {text}")