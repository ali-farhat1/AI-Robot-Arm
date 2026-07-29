import json
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = str(Path(__file__).resolve().parent / "memory")

SHORT_MEMORY = os.path.join(BASE, "chat_memory.json")
CURIOSITIES  = os.path.join(BASE, "curiosities.json")
EVENTS       = os.path.join(BASE, "events.json")
PEOPLE       = os.path.join(BASE, "people.json")
PROCEDURES   = os.path.join(BASE, "procedures.json")
REFLECTIONS  = os.path.join(BASE, "reflections.json")
WORLD_MODEL  = os.path.join(BASE, "world_model.json")

LONG_TERM_MAP = {
    "curiosities": CURIOSITIES,
    "events":      EVENTS,
    "people":      PEOPLE,
    "procedures":  PROCEDURES,
    "reflections": REFLECTIONS,
    "world_model": WORLD_MODEL,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return []

def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def _clear_short_term():
    _save(SHORT_MEMORY, [])

# ─── Short Term ──────────────────────────────────────────────────────────────

def short_term_memory(user, ai):
    data = _load(SHORT_MEMORY)
    skye_text = ai.get("text", "") if isinstance(ai, dict) else ai
    data.append({"User": user, "Skye": skye_text})
    _save(SHORT_MEMORY, data)

# ─── Long Term ───────────────────────────────────────────────────────────────

def long_term_memory(categorized: dict):
    """
    Receives the memory archivist's JSON output and saves each section.
    People are merged by name. All other sections are appended, skipping duplicates.
    Importance decay is handled by the archivist prompt — not here.
    """
    for section, path in LONG_TERM_MAP.items():
        incoming = categorized.get(section)
        if not incoming:
            continue

        existing = _load(path)

        if section == "people":
            existing_by_name = {p["name"]: p for p in existing if isinstance(p, dict) and "name" in p}

            for person in incoming:
                if not isinstance(person, dict) or "name" not in person:
                    continue

                name = person["name"]

                if name in existing_by_name:
                    ep = existing_by_name[name]

                    if person.get("description"):
                        ep["description"] = person["description"]
                    if person.get("relationship"):
                        ep["relationship"] = person["relationship"]

                    existing_obs = ep.get("observations", [])
                    for obs in person.get("observations", []):
                        if obs not in existing_obs:
                            existing_obs.append(obs)
                    ep["observations"] = existing_obs

                    ep["importance"]    = person.get("importance", ep.get("importance", 5))
                    ep["access_count"]  = person.get("access_count", ep.get("access_count", 0))
                    ep["last_accessed"] = person.get("last_accessed", ep.get("last_accessed", None))
                else:
                    existing_by_name[name] = person

            _save(path, list(existing_by_name.values()))

        else:
            existing_memories = [e.get("memory") for e in existing if isinstance(e, dict)]

            for entry in incoming:
                if not isinstance(entry, dict) or "memory" not in entry:
                    continue
                if entry["memory"] not in existing_memories:
                    existing.append(entry)
                    existing_memories.append(entry["memory"])

            _save(path, existing)

# ─── Recall ──────────────────────────────────────────────────────────────────

def refer_memory(max_short=5, top_n=5):
    """
    Builds the memory context string passed to Skye before each response.
    Returns recent conversation + top long-term memories by importance.
    """
    parts = []

    # Recent conversation
    short = _load(SHORT_MEMORY)
    if short:
        parts.append("=== Recent Conversation ===")
        for entry in short[-max_short:]:
            parts.append(f"User: {entry.get('User', '')}")
            parts.append(f"Skye: {entry.get('Skye', '')}")

    # Long-term sections
    section_labels = {
        "reflections": "Reflections",
        "procedures":  "Procedures",
        "people":      "People",
        "events":      "Events",
        "curiosities": "Curiosities",
        "world_model": "World Model",
    }

    for section, path in LONG_TERM_MAP.items():
        entries = _load(path)
        if not entries:
            continue

        label = section_labels.get(section, section.capitalize())
        parts.append(f"\n=== {label} ===")

        if section == "people":
            for person in entries:
                if not isinstance(person, dict):
                    continue
                parts.append(f"- {person.get('name', 'Unknown')}: {person.get('description', '')}")
                if person.get("relationship"):
                    parts.append(f"  Relationship: {person['relationship']}")
                for obs in person.get("observations", []):
                    parts.append(f"  • {obs}")
        else:
            # Sort by importance descending, take top_n
            sorted_entries = sorted(
                [e for e in entries if isinstance(e, dict)],
                key=lambda e: e.get("importance", 0),
                reverse=True
            )
            for entry in sorted_entries[:top_n]:
                parts.append(f"- {entry.get('memory', '')}")

    return "\n".join(parts)

# ─── Build Current Memories JSON (for archivist) ─────────────────────────────

def build_current_memories() -> dict:
    """
    Assembles the full current memory state as a dict.
    Passed to the memory archivist alongside new_memory.
    """
    current = {}
    for section, path in LONG_TERM_MAP.items():
        current[section] = _load(path)
    return current