from __future__ import annotations
import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_store: InMemoryVectorStore | None = None
_status: str = "idle"       # idle | processing | ready | error
_status_detail: str = ""
_chunks_ingested: int = 0

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ingest_text(text: str) -> int:
    global _store
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    if _store is None:
        _store = InMemoryVectorStore(embedding=embeddings)
    _store.add_texts(chunks)
    return len(chunks)


def ingest_background(text: str) -> None:
    global _status, _status_detail, _chunks_ingested
    _status = "processing"
    _status_detail = ""
    try:
        count = ingest_text(text)
        _chunks_ingested += count
        _status = "ready"
    except Exception as exc:
        _status = "error"
        _status_detail = str(exc)


def get_ingestion_status() -> dict:
    return {
        "status": _status,
        "chunks_ingested": _chunks_ingested,
        "detail": _status_detail,
    }


def get_retriever(k: int = 4):
    if _store is None:
        raise RuntimeError("No knowledge base loaded. Call POST /ingest first.")
    return _store.as_retriever(search_kwargs={"k": k})


def is_loaded() -> bool:
    return _store is not None


def clear() -> None:
    global _store, _status, _status_detail, _chunks_ingested
    _store = None
    _status = "idle"
    _status_detail = ""
    _chunks_ingested = 0
