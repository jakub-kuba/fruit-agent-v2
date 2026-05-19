import pytest
from unittest.mock import patch, MagicMock
from agent.rag import retrieve


def test_retrieve_returns_context():
    """RAG retrieve returns relevant text for a fruit query."""
    mock_doc = MagicMock()
    mock_doc.page_content = "Apple is a fruit harvested in autumn in Europe."

    with patch("agent.rag.get_vector_store") as mock_store:
        mock_store.return_value.similarity_search.return_value = [mock_doc]
        result = retrieve("apple seasonality")

    assert "Apple" in result
    assert isinstance(result, str)


def test_retrieve_no_results():
    """RAG retrieve returns fallback message when no results found."""
    with patch("agent.rag.get_vector_store") as mock_store:
        mock_store.return_value.similarity_search.return_value = []
        result = retrieve("some completely unrelated query")

    assert "No relevant information" in result


def test_retrieve_multiple_chunks():
    """RAG retrieve joins multiple chunks with double newline."""
    mock_docs = [
        MagicMock(page_content="Chunk about apples."),
        MagicMock(page_content="Chunk about oranges."),
    ]

    with patch("agent.rag.get_vector_store") as mock_store:
        mock_store.return_value.similarity_search.return_value = mock_docs
        result = retrieve("fruit knowledge")

    assert "apples" in result
    assert "oranges" in result
    assert "\n\n" in result