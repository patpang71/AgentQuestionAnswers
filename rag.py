from __future__ import annotations
import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_store: InMemoryVectorStore | None = None
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ingest_text(text: str) -> int:
    global _store
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    _store = InMemoryVectorStore(embedding=embeddings)
    _store.add_texts(chunks)
    return len(chunks)


def get_retriever(k: int = 4):
    if _store is None:
        raise RuntimeError("No knowledge base loaded. Call POST /ingest first.")
    return _store.as_retriever(search_kwargs={"k": k})


def is_loaded() -> bool:
    return _store is not None


def clear() -> None:
    global _store
    _store = None
