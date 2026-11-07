
import json
import uuid
import os
from datetime import datetime, timezone


# Create logs folder
os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/agent_logs.jsonl"


# Generate unique trace ID
def generate_trace_id():
    return str(uuid.uuid4())


# Save structured log
def log_event(
    trace_id,
    question,
    intent,
    status,
    duration_ms
):

    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "question": question,
        "intent": intent,
        "status": status,
        "duration_ms": round(duration_ms, 2)
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_data) + "\n")
