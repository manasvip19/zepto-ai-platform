from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

model = SentenceTransformer(EMBEDDING_MODEL)

docs_path = Path("docs")

files = sorted(docs_path.glob("*.txt"))

for file in files:

    text = file.read_text(
        encoding="utf-8"
    )

    embedding = model.encode(text).tolist()

    collection.add(
        ids=[file.stem],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "source": file.name
            }
        ],
    )

print("=" * 50)
print(f"Ingested {collection.count()} documents.")
print("=" * 50)