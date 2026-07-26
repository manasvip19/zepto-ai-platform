import os

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "zepto_docs"

CHROMA_PATH = "chroma_db"