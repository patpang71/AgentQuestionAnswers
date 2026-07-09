import threading
import time

import pytest
from unittest.mock import MagicMock, patch

import rag


@pytest.fixture(autouse=True)
def reset_store():
    rag.clear()
    yield
    rag.clear()


# --- ingest_text ---

@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_returns_chunk_count(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    count = rag.ingest_text("Some text about security policies and data centers.")
    assert isinstance(count, int)
    assert count >= 1


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_calls_add_texts(mock_embeddings, mock_store_class):
    mock_store = MagicMock()
    mock_store_class.return_value = mock_store
    rag.ingest_text("Some content.")
    mock_store.add_texts.assert_called_once()


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_sets_store(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_text("Some content.")
    assert rag.is_loaded()


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_accumulates_into_existing_store(mock_embeddings, mock_store_class):
    mock_store = MagicMock()
    mock_store_class.return_value = mock_store
    rag.ingest_text("First document.")
    rag.ingest_text("Second document.")
    # Store created only once; add_texts called twice
    mock_store_class.assert_called_once()
    assert mock_store.add_texts.call_count == 2


# --- ingest_background ---

@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_background_sets_status_ready(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_background("Some content.")
    assert rag.get_ingestion_status()["status"] == "ready"


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_background_updates_chunks_ingested(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_background("Some content.")
    assert rag.get_ingestion_status()["chunks_ingested"] >= 1


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_ingest_background_accumulates_chunk_count(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_background("First document.")
    first_count = rag.get_ingestion_status()["chunks_ingested"]
    rag.ingest_background("Second document.")
    assert rag.get_ingestion_status()["chunks_ingested"] >= first_count


@patch("rag.ingest_text", side_effect=Exception("Embedding failed"))
def test_ingest_background_sets_status_error_on_failure(mock_ingest):
    rag.ingest_background("Some content.")
    result = rag.get_ingestion_status()
    assert result["status"] == "error"
    assert "Embedding failed" in result["detail"]


# --- get_ingestion_status ---

def test_status_idle_initially():
    assert rag.get_ingestion_status() == {"status": "idle", "chunks_ingested": 0, "detail": ""}


# --- get_retriever ---

def test_get_retriever_raises_when_not_loaded():
    with pytest.raises(RuntimeError, match="No knowledge base loaded"):
        rag.get_retriever()


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_get_retriever_uses_default_k(mock_embeddings, mock_store_class):
    mock_store = MagicMock()
    mock_store_class.return_value = mock_store
    rag.ingest_text("Some content.")
    rag.get_retriever()
    mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 4})


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_get_retriever_custom_k(mock_embeddings, mock_store_class):
    mock_store = MagicMock()
    mock_store_class.return_value = mock_store
    rag.ingest_text("Some content.")
    rag.get_retriever(k=8)
    mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 8})


# --- is_loaded / clear ---

def test_is_loaded_false_initially():
    assert not rag.is_loaded()


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_is_loaded_true_after_ingest(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_text("Some content.")
    assert rag.is_loaded()


@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_clear_resets_store(mock_embeddings, mock_store_class):
    mock_store_class.return_value = MagicMock()
    rag.ingest_text("Some content.")
    rag.clear()
    assert not rag.is_loaded()


def test_clear_resets_status_and_chunks():
    rag._status = "ready"
    rag._chunks_ingested = 42
    rag.clear()
    assert rag.get_ingestion_status() == {"status": "idle", "chunks_ingested": 0, "detail": ""}


def test_clear_when_already_empty():
    rag.clear()
    assert not rag.is_loaded()


# --- concurrency ---

@patch("rag.InMemoryVectorStore")
@patch("rag.OpenAIEmbeddings")
def test_concurrent_ingest_background_no_lost_updates(mock_embeddings, mock_store_class):
    # Simulate slow add_texts calls so overlapping ingests are likely to
    # interleave without the lock, exposing the chunks_ingested race.
    mock_store = MagicMock()

    def slow_add_texts(chunks):
        time.sleep(0.01)

    mock_store.add_texts.side_effect = slow_add_texts
    mock_store_class.return_value = mock_store

    texts = [f"Document number {i} with some content to chunk." for i in range(20)]
    threads = [threading.Thread(target=rag.ingest_background, args=(t,)) for t in texts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status = rag.get_ingestion_status()
    assert status["status"] == "ready"
    assert mock_store.add_texts.call_count == 20
    expected_chunks = sum(len(call.args[0]) for call in mock_store.add_texts.call_args_list)
    assert status["chunks_ingested"] == expected_chunks
