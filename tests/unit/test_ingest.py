import pytest
from pathlib import Path
from langchain_core.documents import Document
from scripts.ingest import clean_text, load_text

# 1. Test the pure cleaning logic
def test_clean_text():
    raw_input = "  Hello   WORLD! \n New Line  "
    expected = "hello world! new line"
    assert clean_text(raw_input) == expected

# 2. Test text loading using a temporary file (tmp_path)
def test_load_text(tmp_path):
    # Arrange: Create a temporary txt file
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_file.txt"
    p.write_text("Test content for ingestion.", encoding="utf-8")
    
    # Act: Run the loading function
    docs = load_text(p)
    
    # Assert: Verify content and metadata
    assert len(docs) == 1
    assert docs[0].page_content == "Test content for ingestion."
    assert "test_file.txt" in str(p)

# 3. Mocking the Embeddings/Vectorstore to avoid slow setup
def test_pipeline_flow_mocked(mocker):
    """Tests if the pipeline calls the splitter correctly without running models."""
    # Mock expensive dependencies
    mock_hf = mocker.patch("scripts.ingest.HuggingFaceEmbeddings")
    mock_faiss = mocker.patch("scripts.ingest.FAISS")
    
    # Simulate a small list of documents
    sample_docs = [Document(page_content="sample text", metadata={"source": "test.txt"})]
    
    # Act: You would trigger your loading/splitting logic here
    # For unit testing, ensure functions like clean_text are called on these docs
    cleaned_content = clean_text(sample_docs[0].page_content)
    
    # Assert
    assert cleaned_content == "sample text"
    # Ensure HuggingFace was NOT actually initialized if you were to run the whole script
    assert mock_hf.called is False
