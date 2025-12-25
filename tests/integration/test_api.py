import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

# Initialize the TestClient with your FastAPI app
client = TestClient(app)

# 1. Test Health Check (GET /)
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Sports RAG API running"}

# 2. Test Success Flow (POST /ask)
@patch("app.main.rag_answer")
def test_ask_question_success(mock_rag):
    # Setup mock behavior
    mock_rag.return_value = "The Lakers won the championship."
    
    # Simulate a POST request
    payload = {"question": "Who won the championship?"}
    response = client.post("/ask", json=payload)
    
    # Assertions
    assert response.status_code == 200
    # The API layer normalizes spaces, so we verify the final output
    assert response.json() == {"answer": "The Lakers won the championship."}
    mock_rag.assert_called_once_with("Who won the championship?")

# 3. Test Validation Error (Invalid JSON)
def test_ask_question_invalid_payload():
    # Sending a string instead of the required object
    response = client.post("/ask", json={"wrong_key": "some value"})
    
    # FastAPI automatically returns 422 for pydantic validation failures
    assert response.status_code == 422
    assert "detail" in response.json()

# 4. Test Empty Question
@patch("app.main.rag_answer")
def test_ask_question_empty(mock_rag):
    mock_rag.return_value = "I don't know"
    
    response = client.post("/ask", json={"question": ""})
    
    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know"
