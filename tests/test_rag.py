import pytest
from unittest.mock import MagicMock, patch

import rag


@pytest.fixture(autouse=True)
def reset_store():
    rag.clear()
    yield
    rag.clear()


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
def test_ingest_overwrites_previous_store(mock_embeddings, mock_store_class):
    store1, store2 = MagicMock(), MagicMock()
    mock_store_class.side_effect = [store1, store2]
    rag.ingest_text("First document.")
    rag.ingest_text("Second document.")
    rag.get_retriever()
    store2.as_retriever.assert_called_once()
    store1.as_retriever.assert_not_called()


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


def test_clear_when_already_empty():
    rag.clear()
    assert not rag.is_loaded()
