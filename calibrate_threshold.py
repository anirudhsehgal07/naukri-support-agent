
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project configuration
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTIONS = [
    "fixed_size_overlap",
    "sentence_based"
]


# Questions that should be answered using our recruitment knowledge base
IN_SCOPE_QUERIES = [
    "How are interviews scheduled and how can a candidate request rescheduling?",
    "What happens during background verification before employment is finalized?",
    "What should a candidate know about the probation period?",
    "How does the employee referral bonus work?",
    "How long is applicant information retained?"
]


# Questions unrelated to our recruitment knowledge base
OUT_OF_SCOPE_QUERIES = [
    "What is the weather forecast for Delhi tomorrow?",
    "How do I bake a chocolate cake at home?",
    "Who won the latest FIFA World Cup?",
    "What is the price of this CPU?",
    "Which laptop should I buy for gaming?",
    "How do I repair a broken smartphone screen?",
    "What are the best tourist places in Goa?",
    "How can I learn stock market trading?"
]


def get_top_result(collection, model, question):
    """
    Find the most similar knowledge-base chunk for a question.
    """

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    distance = results["distances"][0][0]

    similarity = 1 - distance

    return {
        "question": question,
        "cosine_distance": round(distance, 4),
        "similarity": round(similarity, 4),
        "source": results["metadatas"][0][0]["source"],
        "matched_text": results["documents"][0][0]
    }


def main():

    print("\n========== THRESHOLD CALIBRATION ==========")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Connect to existing ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))

    all_results = {}

    # Test both chunking strategies
    for collection_name in COLLECTIONS:

        collection = client.get_collection(collection_name)

        print(f"\n\n========== {collection_name} ==========")

        in_scope_results = []
        out_scope_results = []

        # Test in-scope questions
        print("\n--- IN-SCOPE QUESTIONS ---")

        for question in IN_SCOPE_QUERIES:

            result = get_top_result(
                collection,
                model,
                question
            )

            in_scope_results.append(result)

            print("\nQuestion:", result["question"])
            print("Matched document:", result["source"])
            print("Cosine distance:", result["cosine_distance"])
            print("Similarity:", result["similarity"])

        # Test out-of-scope questions
        print("\n--- OUT-OF-SCOPE QUESTIONS ---")

        for question in OUT_OF_SCOPE_QUERIES:

            result = get_top_result(
                collection,
                model,
                question
            )

            out_scope_results.append(result)

            print("\nQuestion:", result["question"])
            print("Matched document:", result["source"])
            print("Cosine distance:", result["cosine_distance"])
            print("Similarity:", result["similarity"])

        all_results[collection_name] = {
            "in_scope": in_scope_results,
            "out_of_scope": out_scope_results
        }

        # Display score ranges
        in_scores = [
            item["similarity"]
            for item in in_scope_results
        ]

        out_scores = [
            item["similarity"]
            for item in out_scope_results
        ]

        print("\n--- SCORE SUMMARY ---")

        print(
            "In-scope similarity range:",
            min(in_scores),
            "to",
            max(in_scores)
        )

        print(
            "Out-of-scope similarity range:",
            min(out_scores),
            "to",
            max(out_scores)
        )

    # Save measured results for README documentation
    output_path = BASE_DIR / "calibration_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\n==========================================")
    print("Calibration measurements saved to:")
    print(output_path)
    print("\nThreshold measurement completed!")


if __name__ == "__main__":
    main()
