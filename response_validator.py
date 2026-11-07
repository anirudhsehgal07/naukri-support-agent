
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


# ==========================================
# 1. APPLICATION LOOKUP RESPONSE SCHEMA
# ==========================================

class ApplicationResponse(BaseModel):

    record_id: str
    status: str
    expected_salary_inr: int = Field(ge=0)
    escalation_score: float = Field(ge=0, le=1)
    escalation_required: bool


# ==========================================
# 2. RAG RESPONSE SCHEMA
# ==========================================

class RAGResponse(BaseModel):

    question: str
    status: str
    answer: str
    best_distance: Optional[float] = None
    sources: list[str] = Field(default_factory=list)


# ==========================================
# 3. ERROR RESPONSE SCHEMA
# ==========================================

class ErrorResponse(BaseModel):

    error: str


# ==========================================
# 4. VALIDATION FUNCTION
# ==========================================

def validate_response(response: dict, response_type: str):

    try:

        if response_type == "application":
            validated = ApplicationResponse(**response)

        elif response_type == "rag":
            validated = RAGResponse(**response)

        elif response_type == "error":
            validated = ErrorResponse(**response)

        else:
            raise ValueError("Unknown response type")

        return {
            "valid": True,
            "data": validated.model_dump()
        }

    except ValidationError as error:

        return {
            "valid": False,
            "errors": error.errors()
        }


# ==========================================
# 5. TEST THE VALIDATOR
# ==========================================

if __name__ == "__main__":

    # Valid application response
    valid_application = {
        "record_id": "APP0001",
        "status": "Applied",
        "expected_salary_inr": 1114000,
        "escalation_score": 0.28,
        "escalation_required": False
    }

    print("\nVALID RESPONSE TEST:")
    print(
        validate_response(
            valid_application,
            "application"
        )
    )

    # Invalid application response
    invalid_application = {
        "record_id": "APP0001",
        "status": "Applied",
        "expected_salary_inr": -500,
        "escalation_score": 1.5,
        "escalation_required": "yes"
    }

    print("\nINVALID RESPONSE TEST:")
    print(
        validate_response(
            invalid_application,
            "application"
        )
    )
