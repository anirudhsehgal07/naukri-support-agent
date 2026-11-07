
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_engine import ask_question
from application_lookup import check_job_application_status

from conversation_manager import (
    add_message,
    get_conversation,
    start_new_conversation
)

from response_validator import validate_response


# ==========================================
# 1. DEFINE THE WORKFLOW STATE
# ==========================================

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    record_id: str
    history: list
    contextual_question: str
    result: dict
    response: str
    validation_status: bool


# ==========================================
# 2. NODE 1: CLASSIFY USER INTENT
# ==========================================

def classify_intent(state: AgentState) -> dict:

    print("\nExecuting node: classify_intent")

    question = state["question"].strip()
    question_lower = question.lower()

    application_id = re.search(
        r"\bAPP\d{4}\b",
        question,
        re.IGNORECASE
    )

    lookup_keywords = [
        "application status",
        "status of my application",
        "check my application",
        "track my application",
        "application details"
    ]

    if application_id or any(
        keyword in question_lower
        for keyword in lookup_keywords
    ):
        return {
            "intent": "lookup",
            "record_id": (
                application_id.group(0).upper()
                if application_id else ""
            )
        }

    return {"intent": "rag"}


# ==========================================
# 3. NODE 2: RAG NODE
# ==========================================

def rag_node(state: AgentState) -> dict:

    print("\nExecuting node: rag_node")

    question = state["question"]
    history = state.get("history", [])

    contextual_question = question

    if history:

        previous_user_questions = [
            message["content"]
            for message in history
            if message["role"] == "user"
        ]

        if previous_user_questions:

            last_question = previous_user_questions[-1]

            follow_up_words = [
                "it", "that", "this", "there", "them"
            ]

            is_follow_up = any(
                re.search(
                    r"\b" + re.escape(word) + r"\b",
                    question.lower()
                )
                for word in follow_up_words
            )

            if is_follow_up:
                contextual_question = (
                    last_question + " " + question
                )

    print(
        "\nContextual question sent to RAG:",
        contextual_question
    )

    result = ask_question(contextual_question)

    return {
        "contextual_question": contextual_question,
        "result": result
    }


# ==========================================
# 4. NODE 3: APPLICATION LOOKUP NODE
# ==========================================

def lookup_node(state: AgentState) -> dict:

    print("\nExecuting node: lookup_node")

    record_id = state.get("record_id", "")

    if not record_id:
        return {
            "result": {
                "error": (
                    "Please provide your application ID, "
                    "such as APP0001."
                )
            }
        }

    result = check_job_application_status(record_id)

    return {"result": result}


# ==========================================
# 5. NODE 4: FINAL RESPONSE NODE
# ==========================================

def response_node(state: AgentState) -> dict:

    print("\nExecuting node: response_node")

    result = state.get("result", {})

    if "answer" in result:
        response_type = "rag"
        response_data = result

    elif "error" in result:
        response_type = "error"
        response_data = {"error": result["error"]}

    elif "record_id" in result:
        response_type = "application"
        response_data = result

    else:
        return {
            "response": "Unable to generate a valid response.",
            "validation_status": False
        }

    validation_result = validate_response(
        response_data,
        response_type
    )

    if not validation_result["valid"]:

        print("\nResponse validation failed!")
        print(validation_result["errors"])

        return {
            "response": (
                "Sorry, I could not generate a valid response."
            ),
            "validation_status": False
        }

    data = validation_result["data"]

    if response_type == "application":

        response = (
            f"Application ID: {data['record_id']}\n"
            f"Status: {data['status']}\n"
            f"Expected Salary: ₹{data['expected_salary_inr']:,}\n"
            f"Escalation Score: {data['escalation_score']}\n"
            f"Escalation Required: {data['escalation_required']}"
        )

    elif response_type == "rag":
        response = data["answer"]

    else:
        response = data["error"]

    print("\nResponse validation: PASSED")

    return {
        "response": response,
        "validation_status": True
    }


# ==========================================
# 6. CONDITIONAL ROUTING
# ==========================================

def route_by_intent(state: AgentState):

    if state["intent"] == "lookup":
        return "lookup"

    return "rag"


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("rag_node", rag_node)
workflow.add_node("lookup_node", lookup_node)
workflow.add_node("response_node", response_node)

workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "rag": "rag_node",
        "lookup": "lookup_node"
    }
)

workflow.add_edge("rag_node", "response_node")
workflow.add_edge("lookup_node", "response_node")

workflow.add_edge("response_node", END)


# Normal app for existing API imports
app = workflow.compile()


# ==========================================
# 8. RUN AGENT WITH SQLITE CHECKPOINTING
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print(" NAUKRI RECRUITMENT SUPPORT AGENT")
    print("======================================")

    conversation_id = "user_001"

    # SQLite saves workflow progress
    with SqliteSaver.from_conn_string(
        "checkpoints.sqlite"
    ) as checkpointer:

        # Compile graph with SQLite checkpointer
        checkpointed_app = workflow.compile(
            checkpointer=checkpointer
        )

        # Same conversation ID = same thread
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }

        while True:

            question = input(
                "\nEnter your question (or type /new or exit): "
            )

            if question.lower().strip() == "exit":
                print("Agent stopped.")
                break

            if question.lower().strip() == "/new":

                start_new_conversation(conversation_id)

                conversation_id = "user_001_new"

                config = {
                    "configurable": {
                        "thread_id": conversation_id
                    }
                }

                print("\nNew conversation started!")
                continue

            previous_messages = get_conversation(
                conversation_id
            )

            print(
                "\nPrevious messages:",
                len(previous_messages)
            )

            result = checkpointed_app.invoke(
                {
                    "question": question,
                    "history": previous_messages
                },
                config=config
            )

            response = result["response"]

            add_message(
                conversation_id,
                "user",
                question
            )

            add_message(
                conversation_id,
                "assistant",
                response
            )

            print("\nDetected intent:", result["intent"])

            print(
                "\nValidation status:",
                result["validation_status"]
            )

            print("\nAgent response:")
            print(response)
