
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

query = "What is the company's probation policy?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()

for collection_name in [
    "fixed_size_overlap",
    "sentence_based"
]:

    collection = client.get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\n===== {collection_name} =====")

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        print("\nDocument:", metadata["source"])
        print("Text:", document)
        print("Cosine distance:", round(distance, 4))
