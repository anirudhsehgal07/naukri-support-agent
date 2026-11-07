
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sentence_based"

COSINE_DISTANCE_THRESHOLD = 0.63

MOCK_LLM = True

FALLBACK_MESSAGE = (
    "I'm sorry, but I can only assist with recruitment "
    "and HR-related queries. I can help with interview "
    "scheduling, eligibility, application information, "
    "and other recruitment topics."
)


# ==========================================
# 2. LOAD MODEL AND VECTOR DATABASE
# ==========================================

print("Loading recruitment support agent...")

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Recruitment support agent ready!")


# ==========================================
# 3. RETRIEVE RELEVANT KNOWLEDGE
# ==========================================

def retrieve_context(question, top_k=3):
    """
    Convert the user question into an embedding
    and retrieve the top matching knowledge chunks.
    """

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        retrieved_chunks.append({
            "text": document,
            "source": metadata["source"],
            "distance": float(distance)
        })

    return retrieved_chunks


# ==========================================
# 4. GENERATE MOCK LLM ANSWER
# ==========================================

def generate_mock_answer(context_chunks):
    """
    Generate a deterministic answer using only
    the retrieved recruitment knowledge.
    """

    if not context_chunks:
        return FALLBACK_MESSAGE

    answer_parts = []

    for chunk in context_chunks:

        text = chunk["text"].strip()

        if text and text not in answer_parts:
            answer_parts.append(text)

    if not answer_parts:
        return FALLBACK_MESSAGE

    answer = (
        "According to the recruitment knowledge base:\n"
        + "\n".join(
            f"- {part}" for part in answer_parts
        )
    )

    return answer


# ==========================================
# 5. MAIN RAG FUNCTION
# ==========================================

def ask_question(question):
    """
    Main entry point for the recruitment agent.
    """

    if not question or not question.strip():

        return {
            "question": question,
            "status": "invalid",
            "answer": "Please enter a valid question.",
            "best_distance": None,
            "sources": [],
            "retrieved_context": []
        }

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_context(question.strip())

    if not chunks:

        return {
            "question": question,
            "status": "out_of_scope",
            "answer": FALLBACK_MESSAGE,
            "best_distance": None,
            "sources": [],
            "retrieved_context": []
        }

    # Step 2: Find the best cosine distance
    best_distance = chunks[0]["distance"]

    # Step 3: Identify the best-matching document
    best_source = chunks[0]["source"]

    # Step 4: Keep chunks from the best document
    # that also pass the calibrated threshold
    relevant_chunks = [
        chunk for chunk in chunks
        if (
            chunk["distance"] <= COSINE_DISTANCE_THRESHOLD
            and chunk["source"] == best_source
        )
    ]

    # Step 5: Reject if the best result is too distant
    if best_distance > COSINE_DISTANCE_THRESHOLD:

        return {
            "question": question,
            "status": "out_of_scope",
            "answer": FALLBACK_MESSAGE,
            "best_distance": round(best_distance, 4),
            "sources": [],
            "retrieved_context": []
        }

    # Step 6: Generate deterministic mock answer
    if MOCK_LLM:
        answer = generate_mock_answer(relevant_chunks)
    else:
        answer = FALLBACK_MESSAGE

    # Step 7: Prepare retrieved context
    retrieved_context = [
        chunk["text"] for chunk in relevant_chunks
    ]

    # Step 8: Return structured response
    return {
        "question": question,
        "status": "answered",
        "answer": answer,
        "best_distance": round(best_distance, 4),
        "sources": list(
            dict.fromkeys(
                chunk["source"]
                for chunk in relevant_chunks
            )
        ),
        "retrieved_context": retrieved_context
    }


# ==========================================
# 6. TERMINAL INTERFACE
# ==========================================

if __name__ == "__main__":

    print("\n========== NAUKRI SUPPORT AGENT ==========")

    while True:

        question = input(
            "\nAsk a recruitment question "
            "(type 'exit' to stop): "
        )

        if question.lower().strip() == "exit":

            print("Agent stopped.")
            break

        result = ask_question(question)

        print("\nStatus:", result["status"])

        print(
            "Best cosine distance:",
            result["best_distance"]
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:", result["sources"])
