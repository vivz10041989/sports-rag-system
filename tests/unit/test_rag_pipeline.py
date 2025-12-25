import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_community.llms.fake import FakeListLLM
from app.rag_pipeline import build_prompt, rag_answer

# ==============================
# 1. PURE LOGIC TESTS
# ==============================
def test_build_prompt():
    """Verifies the prompt structure without any external calls."""
    context = "The Lakers are based in Los Angeles."
    question = "Where are the Lakers from?"
    prompt = build_prompt(context, question)
    
    assert "Los Angeles" in prompt
    assert question in prompt
    assert "sports analytics assistant" in prompt

# ==============================
# 2. RAG FLOW TESTS (MOCKED)
# ==============================

@patch("app.rag_pipeline.retriever")  # Patch the retriever object itself
@patch("app.rag_pipeline.llm")        # Patch the llm object itself
def test_rag_answer_flow(mock_llm, mock_retriever):
    """
    Tests the full rag_answer flow by properly mocking the .invoke() 
    methods of both the retriever and the LLM.
    """
    # 1. Setup Mock Retriever: Ensure it HAS an .invoke method
    mock_docs = [
        Document(page_content="The Lakers won the 2020 NBA Title.", metadata={"source": "nba_stats.txt"})
    ]
    mock_retriever.invoke.return_value = mock_docs
    
    # 2. Setup Mock LLM: Ensure it HAS an .invoke method
    # We use FakeListLLM logic for reliability
    fake_llm = FakeListLLM(responses=["The Lakers"])
    mock_llm.invoke.side_effect = fake_llm.invoke
    
    # Execute
    question = "Who won in 2020?"
    answer = rag_answer(question)
    
    # Assertions
    assert answer == "The Lakers"
    # Verify .invoke was called on the retriever, not the mock itself
    mock_retriever.invoke.assert_called_once_with(question)
    mock_llm.invoke.assert_called_once()

@patch("app.rag_pipeline.retriever")
@patch("app.rag_pipeline.llm")
def test_rag_answer_no_context(mock_llm, mock_retriever):
    """Tests the 'I don't know' fallback when no documents are found."""
    # Simulate empty retrieval via the .invoke method
    mock_retriever.invoke.return_value = []
    
    # Setup Fake LLM response
    fake_llm = FakeListLLM(responses=["I don't know"])
    mock_llm.invoke.side_effect = fake_llm.invoke
    
    # Execute
    answer = rag_answer("What is the capital of Mars?")
    
    # Assertions
    assert answer == "I don't know"
    mock_retriever.invoke.assert_called_once()
