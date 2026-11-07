
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from logging_utils import (
    generate_trace_id,
    log_event
)

from agent_workflow import app as agent_app
from application_lookup import check_job_application_status

from conversation_manager import (
    add_message,
    get_conversation
)

from input_guardrails import process_user_input


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

api = FastAPI(
    title="Naukri Recruitment Support API",
    description="API for recruitment support and application lookup",
    version="1.0.0"
)


# ==========================================
# 2. REQUEST AND RESPONSE MODELS
# ==========================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str = "user_001"


class ChatResponse(BaseModel):
    response: str
    intent: str
    validation_status: bool


class ApplicationResponse(BaseModel):
    record_id: str
    status: Optional[str] = None
    expected_salary_inr: Optional[int] = None
    escalation_score: Optional[float] = None
    escalation_required: Optional[bool] = None
    error: Optional[str] = None


# ==========================================
# 3. ENDPOINT 1: CHAT WITH AGENT
# ==========================================

@api.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):

    # Generate trace ID and start timer
    trace_id = generate_trace_id()
    start_time = time.perf_counter()

    # Apply input guardrails
    guardrail_result = process_user_input(
        request.question
    )

    masked_question = guardrail_result["masked_text"]

    # Block suspicious input
    if not guardrail_result["safe"]:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_event(
            trace_id,
            masked_question,
            "blocked",
            "blocked",
            duration_ms
        )

        raise HTTPException(
            status_code=400,
            detail=guardrail_result["message"]
        )

    # Load previous conversation
    history = get_conversation(
        request.conversation_id
    )

    # Run LangGraph agent
    result = agent_app.invoke({
        "question": masked_question,
        "history": history
    })

    response = result["response"]

    # Save user message
    add_message(
        request.conversation_id,
        "user",
        masked_question
    )

    # Save assistant response
    add_message(
        request.conversation_id,
        "assistant",
        response
    )

    # Calculate response time
    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Save structured log
    log_event(
        trace_id,
        masked_question,
        result["intent"],
        "success" if result["validation_status"] else "validation_failed",
        duration_ms
    )

    # Return API response
    return {
        "response": response,
        "intent": result["intent"],
        "validation_status": result["validation_status"]
    }


# ==========================================
# 4. ENDPOINT 2: APPLICATION STATUS
# ==========================================

@api.get(
    "/application/{record_id}",
    response_model=ApplicationResponse
)
def get_application_status(record_id: str):

    trace_id = generate_trace_id()
    start_time = time.perf_counter()

    result = check_job_application_status(
        record_id.upper()
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    status = "not_found" if result.get("error") else "success"

    log_event(
        trace_id,
        f"Application lookup: {record_id.upper()}",
        "lookup",
        status,
        duration_ms
    )

    return result


# ==========================================
# 5. ROOT ENDPOINT
# ==========================================

@api.get("/")
def home():

    trace_id = generate_trace_id()
    start_time = time.perf_counter()

    result = {
        "message": "Welcome to Naukri Recruitment Support API",
        "status": "Running"
    }

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_event(
        trace_id,
        "Home endpoint request",
        "home",
        "success",
        duration_ms
    )

    return result
