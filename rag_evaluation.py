
from rag_engine import ask_question
import json


# ==========================================
# 1. MOCK LLM CONFIGURATION
# ==========================================

MOCK_LLM = True


# ==========================================
# 2. LLM-AS-A-JUDGE PROMPT
# ==========================================

JUDGE_PROMPT = """
You are an evaluator for a recruitment support RAG system.

Evaluate the system using three scores between 0 and 1.

1. CONTEXT RELEVANCE:
Did the system retrieve the correct knowledge-base
document for the user's question?

2. GROUNDEDNESS:
Is the generated answer supported by the retrieved
knowledge-base context?

3. ANSWER RELEVANCE:
Does the answer address the user's actual question?

For out-of-scope questions:
A correct refusal with no irrelevant sources receives
a score of 1 for all three criteria.

An incorrect answer to an out-of-scope question receives
a score of 0 for all three criteria.

Return scores between 0 and 1.

The evaluation runs in MOCK_LLM mode.
The deterministic mock judge follows this prompt
using predefined rules without an external API.
"""


# ==========================================
# 3. CREATE 15 TEST QUESTIONS
# ==========================================

test_cases = [

    {
        "question": "What are the eligibility criteria for applying for a job?",
        "expected_source": "eligibility.txt",
        "expected_keywords": ["eligibility", "criteria"]
    },
    {
        "question": "How can I schedule or reschedule an interview?",
        "expected_source": "interview_scheduling.txt",
        "expected_keywords": ["interview", "schedule"]
    },
    {
        "question": "Can candidates negotiate their offered salary?",
        "expected_source": "offer_negotiation.txt",
        "expected_keywords": ["offer", "salary"]
    },
    {
        "question": "What happens during background verification?",
        "expected_source": "background_verification.txt",
        "expected_keywords": ["background", "verification"]
    },
    {
        "question": "What is the notice period policy?",
        "expected_source": "notice_period.txt",
        "expected_keywords": ["notice", "period"]
    },
    {
        "question": "How does the employee referral bonus work?",
        "expected_source": "referral_bonus.txt",
        "expected_keywords": ["referral", "bonus"]
    },
    {
        "question": "Can an employee transfer to another department?",
        "expected_source": "internal_transfer.txt",
        "expected_keywords": ["internal", "transfer"]
    },
    {
        "question": "How long is the probation period?",
        "expected_source": "probation.txt",
        "expected_keywords": ["probation"]
    },
    {
        "question": "Are employees eligible to work remotely?",
        "expected_source": "remote_work.txt",
        "expected_keywords": ["remote", "work"]
    },
    {
        "question": "What are the diversity hiring guidelines?",
        "expected_source": "diversity_hiring.txt",
        "expected_keywords": ["diversity", "hiring"]
    },
    {
        "question": "What happens during an exit interview?",
        "expected_source": "exit_interview.txt",
        "expected_keywords": ["exit", "interview"]
    },
    {
        "question": "How long is applicant data retained?",
        "expected_source": "data_retention.txt",
        "expected_keywords": ["data", "retention"]
    },

    # Out-of-scope questions

    {
        "question": "What is the price of Bitcoin today?",
        "expected_source": None,
        "expected_keywords": []
    },
    {
        "question": "Who won the cricket match yesterday?",
        "expected_source": None,
        "expected_keywords": []
    },
    {
        "question": "How do I repair a car engine?",
        "expected_source": None,
        "expected_keywords": []
    }
]


# ==========================================
# 4. DETERMINISTIC MOCK JUDGE
# ==========================================

def mock_llm_judge(
    prompt,
    question,
    result,
    expected_source,
    expected_keywords
):
    """
    Simulate an LLM-as-a-judge under MOCK_LLM.

    The written prompt defines the evaluation criteria.
    Deterministic rules implement those criteria locally.
    No external AI API is called.
    """

    answer = result.get("answer", "").lower()

    sources = [
        str(source).lower()
        for source in result.get("sources", [])
    ]

    context_chunks = result.get(
        "retrieved_context", []
    )

    status = result.get("status", "").lower()

    # --------------------------------------
    # OUT-OF-SCOPE EVALUATION
    # --------------------------------------

    if expected_source is None:

        correct_refusal = (
            status == "out_of_scope"
            and len(sources) == 0
        )

        score = int(correct_refusal)

        return {
            "context_relevance": score,
            "groundedness": score,
            "answer_relevance": score,
            "judge_reason": (
                "Correct refusal of out-of-scope question."
                if correct_refusal
                else "Incorrect handling of out-of-scope question."
            )
        }

    # --------------------------------------
    # 1. CONTEXT RELEVANCE
    # --------------------------------------

    context_relevance = int(
        any(
            expected_source.lower() in source
            for source in sources
        )
    )

    # --------------------------------------
    # 2. GROUNDEDNESS
    # --------------------------------------

    answer_statements = [
        line.strip()[2:].strip()
        for line in result.get("answer", "").splitlines()
        if line.strip().startswith("- ")
    ]

    normalized_context = [
        " ".join(chunk.lower().split())
        for chunk in context_chunks
    ]

    supported_statements = 0

    for statement in answer_statements:

        normalized_statement = " ".join(
            statement.lower().split()
        )

        if any(
            normalized_statement in context
            for context in normalized_context
        ):
            supported_statements += 1

    if answer_statements:

        groundedness = (
            supported_statements / len(answer_statements)
        )

    else:
        groundedness = 0

    # --------------------------------------
    # 3. ANSWER RELEVANCE
    # --------------------------------------

    matching_keywords = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer
    )

    if expected_keywords:

        answer_relevance = (
            matching_keywords / len(expected_keywords)
        )

    else:
        answer_relevance = 0

    return {
        "context_relevance": round(context_relevance, 2),
        "groundedness": round(groundedness, 2),
        "answer_relevance": round(answer_relevance, 2),
        "judge_reason": (
            "Scores calculated using the deterministic "
            "MOCK_LLM evaluation rubric."
        )
    }


# ==========================================
# 5. RUN EVALUATION
# ==========================================

results = []

if not MOCK_LLM:
    raise RuntimeError(
        "This evaluation script is configured for MOCK_LLM only."
    )

for index, test in enumerate(test_cases, start=1):

    question = test["question"]

    print(f"\nTest {index}: {question}")

    # Get response from RAG engine
    rag_result = ask_question(question)

    # Evaluate using deterministic mock judge
    scores = mock_llm_judge(
        JUDGE_PROMPT,
        question,
        rag_result,
        test["expected_source"],
        test["expected_keywords"]
    )

    evaluation = {
        "query": question,
        "status": rag_result.get("status"),
        "sources": rag_result.get("sources", []),
        "context_relevance": scores["context_relevance"],
        "groundedness": scores["groundedness"],
        "answer_relevance": scores["answer_relevance"],
        "judge_reason": scores["judge_reason"]
    }

    results.append(evaluation)

    print("Status:", evaluation["status"])
    print("Sources:", evaluation["sources"])

    print(
        "Context relevance:",
        scores["context_relevance"]
    )

    print(
        "Groundedness:",
        scores["groundedness"]
    )

    print(
        "Answer relevance:",
        scores["answer_relevance"]
    )

    print("Judge reason:", scores["judge_reason"])


# ==========================================
# 6. CALCULATE AVERAGE SCORES
# ==========================================

total = len(results)

average_context = sum(
    item["context_relevance"] for item in results
) / total

average_groundedness = sum(
    item["groundedness"] for item in results
) / total

average_relevance = sum(
    item["answer_relevance"] for item in results
) / total


print("\n======================================")
print("       RAG EVALUATION SUMMARY")
print("======================================")

print("Total queries:", total)

print(
    "Average context relevance:",
    round(average_context, 2)
)

print(
    "Average groundedness:",
    round(average_groundedness, 2)
)

print(
    "Average answer relevance:",
    round(average_relevance, 2)
)


# ==========================================
# 7. SAVE RESULTS TO JSON
# ==========================================

report = {
    "evaluation_mode": "MOCK_LLM",
    "judge_type": "Deterministic prompt-guided mock judge",
    "judge_prompt": JUDGE_PROMPT,
    "total_queries": total,

    "average_context_relevance": round(
        average_context, 2
    ),

    "average_groundedness": round(
        average_groundedness, 2
    ),

    "average_answer_relevance": round(
        average_relevance, 2
    ),

    "individual_results": results
}

with open(
    "rag_evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )

print("\nResults saved to rag_evaluation_results.json")
