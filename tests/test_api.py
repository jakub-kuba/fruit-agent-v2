import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app

client = TestClient(app)


def test_root():
    """Root endpoint returns ok status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    """Health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_generates_session_id():
    """Chat endpoint generates session_id when not provided."""
    with patch("main.graph.ainvoke", new_callable=AsyncMock) as mock_graph, \
         patch("main.save_message"):

        mock_graph.return_value = {
            "messages": [
                type("msg", (), {"content": "The price of Apple is 2.50 PLN."})()
            ],
            "intent": "tool"
        }

        response = client.post("/chat", json={"message": "price of apple"})

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID format


def test_chat_uses_provided_session_id():
    """Chat endpoint uses session_id when provided."""
    session_id = "test-session-123"

    with patch("main.graph.ainvoke", new_callable=AsyncMock) as mock_graph, \
         patch("main.save_message"):

        mock_graph.return_value = {
            "messages": [
                type("msg", (), {"content": "The price of Apple is 2.50 PLN."})()
            ],
            "intent": "tool"
        }

        response = client.post("/chat", json={
            "message": "price of apple",
            "session_id": session_id
        })

    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


def test_clear_history():
    """Delete endpoint clears conversation history."""
    with patch("main.delete_history"):
        response = client.delete("/chat/test-session-123")

    assert response.status_code == 200
    assert response.json()["status"] == "cleared"