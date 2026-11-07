
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import load_documents, create_chunks


# Project configuration
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"

FIXED_COLLECTION = "fixed_size_overlap"
SENTENCE_COLLECTION = "sentence_based"


def create_embeddings(chunks, model):
    """
    Convert text chunks into numerical embeddings.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


def store_chunks(collection, chunks, embeddings):
    """
    Store chunk text, embeddings and metadata in ChromaDB.
    """

    collection.add(
        ids=[chunk["id"] for chunk in chunks],

        documents=[chunk["text"] for chunk in chunks],

        embeddings=embeddings,

        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "strategy": chunk["strategy"]
            }
            for chunk in chunks
        ]
    )


def recreate_collection(client, collection_name):
    """
    Delete an existing collection before rebuilding it.
    This prevents duplicate IDs when running the script again.
    """

    existing_collections = client.list_collections()

    for item in existing_collections:

        # Handles both collection objects and collection names
        existing_name = (
            item if isinstance(item, str) else item.name
        )

        if existing_name == collection_name:
            client.delete_collection(name=collection_name)
            break

    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def main():

    print("\n========== BUILDING VECTOR DATABASE ==========")

    # 1. Load knowledge-base documents
    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents)}")

    # 2. Generate both chunking strategies
    fixed_chunks, sentence_chunks = create_chunks(documents)

    print(f"Fixed-size chunks: {len(fixed_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")

    # 3. Load the embedding model
    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully!")

    # 4. Generate embeddings
    print("\nGenerating fixed-size embeddings...")

    fixed_embeddings = create_embeddings(fixed_chunks, model)

    print("\nGenerating sentence-based embeddings...")

    sentence_embeddings = create_embeddings(sentence_chunks, model)

    print("\nEmbedding generation completed!")

    # 5. Create persistent ChromaDB database
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # 6. Create separate collections
    fixed_collection = recreate_collection(
        client,
        FIXED_COLLECTION
    )

    sentence_collection = recreate_collection(
        client,
        SENTENCE_COLLECTION
    )

    # 7. Store embeddings and their metadata
    store_chunks(
        fixed_collection,
        fixed_chunks,
        fixed_embeddings
    )

    store_chunks(
        sentence_collection,
        sentence_chunks,
        sentence_embeddings
    )

    # 8. Verify stored records
    print("\n========== DATABASE VERIFICATION ==========")

    print(
        "Fixed-size collection records:",
        fixed_collection.count()
    )

    print(
        "Sentence-based collection records:",
        sentence_collection.count()
    )

    print("\nCollections available:")

    for collection in client.list_collections():
        print(collection if isinstance(collection, str) else collection.name)

    print("\nTask 3 vector database created successfully!")


if __name__ == "__main__":
    main()
