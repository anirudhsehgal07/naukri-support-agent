

import json
from pathlib import Path
from datetime import datetime, timezone

HISTORY_FILE = Path(__file__).parent / "conversation_history.json"


def load_history():
    if not HISTORY_FILE.exists():
        return {}

    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def add_message(conversation_id, role, content):
    history = load_history()

    if conversation_id not in history:
        history[conversation_id] = []

    history[conversation_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    save_history(history)


def get_conversation(conversation_id):
    history = load_history()
    return history.get(conversation_id, [])


def start_new_conversation(conversation_id):
    history = load_history()
    history[conversation_id] = []
    save_history(history)
    return conversation_id
