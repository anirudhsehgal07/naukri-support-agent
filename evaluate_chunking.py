
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
TOP_K_CHUNKS = 12
TOP_K_DOCUMENTS = 3

# Five test questions and their expected source documents
TEST_QUERIES = [
    {
        "question": "How does the employee referral bonus work?",
        "relevant_docs": ["referral_bonus.txt"]
    },
    {
        "question": "Can employees work remotely?",
        "relevant_docs": ["remote_work.txt"]
    },
    {
        "question": "What happens during an exit interview?",
        "relevant_docs": ["exit_interview.txt"]
    },
    {
        "question": "What qualifications must applicants have?",
        "relevant_docs": ["eligibility.txt"]
    },
    {
        "question": "How can an employee transfer to another department?",
        "relevant_docs": ["internal_transfer.txt"]
    }
]

# Load embedding model and vector database
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=DB_PATH)

fixed_collection = client.get_collection("fixed_size_overlap")
sentence_collection = client.get_collection("sentence_based")


def retrieve_top_documents(collection, question):
    """Retrieve chunks and return the top 3 unique source documents."""

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K_CHUNKS, collection.count())
    )

    documents = []
    seen_sources = set()

    sources = results["metadatas"][0]

    for metadata in sources:
        source = metadata["source"]

        # Keep only the first occurrence of each document
        if source not in seen_sources:
            seen_sources.add(source)
            documents.append(source)

        if len(documents) == TOP_K_DOCUMENTS:
            break

    return documents


def evaluate_collection(collection, strategy_name):
    """Calculate Precision@3 and Recall@3 for each question."""

    total_precision = 0
    total_recall = 0

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {strategy_name}")
    print("=" * 60)

    for test in TEST_QUERIES:
        question = test["question"]
        relevant_docs = set(test["relevant_docs"])

        retrieved_docs = retrieve_top_documents(
            collection,
            question
        )

        relevant_found = set(retrieved_docs) & relevant_docs

        precision = len(relevant_found) / TOP_K_DOCUMENTS
        recall = len(relevant_found) / len(relevant_docs)

        total_precision += precision
        total_recall += recall

        print(f"\nQuestion: {question}")
        print(f"Retrieved documents: {retrieved_docs}")
        print(f"Relevant documents: {list(relevant_docs)}")
        print(f"Precision@3: {precision:.3f}")
        print(f"Recall@3: {recall:.3f}")

    count = len(TEST_QUERIES)

    average_precision = total_precision / count
    average_recall = total_recall / count

    print("\n--- AVERAGE RESULTS ---")
    print(f"Average Precision@3: {average_precision:.3f}")
    print(f"Average Recall@3: {average_recall:.3f}")

    return average_precision, average_recall


if __name__ == "__main__":
    fixed_results = evaluate_collection(
        fixed_collection,
        "Fixed-size with overlap"
    )

    sentence_results = evaluate_collection(
        sentence_collection,
        "Sentence-based"
    )

    print("\n========== FINAL COMPARISON ==========")
    print(
        f"Fixed-size: Precision={fixed_results[0]:.3f}, "
        f"Recall={fixed_results[1]:.3f}"
    )
    print(
        f"Sentence-based: Precision={sentence_results[0]:.3f}, "
        f"Recall={sentence_results[1]:.3f}"
    )
