import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    COLLECTION_NAME
)

model = SentenceTransformer(
    EMBEDDING_MODEL
)


def retrieve(query: str, top_k: int = 3):

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    return documents, ids, metadatas