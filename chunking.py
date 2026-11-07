
from pathlib import Path
import re

# Project paths
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"

# Fixed-size chunking settings
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def load_documents():
    """
    Read all knowledge-base text files.
    Returns a dictionary containing filename and document text.
    """

    if not KB_DIR.exists():
        raise FileNotFoundError(
            "knowledge_base folder not found. "
            "Run create_knowledge_base.py first."
        )

    documents = {}

    for file_path in sorted(KB_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents[file_path.name] = text

    if len(documents) < 12:
        raise ValueError(
            f"Expected at least 12 documents, found {len(documents)}."
        )

    return documents


def fixed_size_chunking(text):
    """
    Split text into chunks of approximately 150 characters.
    Consecutive chunks overlap by 30 characters.
    """

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def sentence_based_chunking(text):
    """
    Split text into individual sentences.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_chunks(documents):

    fixed_chunks = []
    sentence_chunks = []

    for filename, text in documents.items():

        # Method A: Fixed-size chunks
        for index, chunk in enumerate(fixed_size_chunking(text)):

            fixed_chunks.append({
                "id": f"fixed_{filename}_{index}",
                "text": chunk,
                "source": filename,
                "chunk_index": index,
                "strategy": "fixed_size_overlap"
            })

        # Method B: Sentence-based chunks
        for index, chunk in enumerate(sentence_based_chunking(text)):

            sentence_chunks.append({
                "id": f"sentence_{filename}_{index}",
                "text": chunk,
                "source": filename,
                "chunk_index": index,
                "strategy": "sentence_based"
            })

    return fixed_chunks, sentence_chunks


if __name__ == "__main__":

    documents = load_documents()

    fixed_chunks, sentence_chunks = create_chunks(documents)

    print("\n========== CHUNKING RESULTS ==========")

    print("Documents loaded:", len(documents))

    print("Fixed-size chunks:", len(fixed_chunks))

    print("Sentence-based chunks:", len(sentence_chunks))

    print("\nFixed-size example:")
    print(fixed_chunks[0]["text"])

    print("\nSentence-based example:")
    print(sentence_chunks[0]["text"])

    print("\nTask 3 chunking completed successfully!")
