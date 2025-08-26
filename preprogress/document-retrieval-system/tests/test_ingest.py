import os
import pytest
from rag_pipeline.ingest import process_markdown_files, DATA_PATH, VECTOR_STORE_PATH

@pytest.fixture(scope="module")
def setup_test_environment():
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, "test_file.md"), "w") as f:
        f.write("# Test Document\n\nThis is a test document for ingestion.\n")
    yield
    os.remove(os.path.join(DATA_PATH, "test_file.md"))

def test_process_markdown_files(setup_test_environment):
    process_markdown_files()
    assert os.path.exists(VECTOR_STORE_PATH), "FAISS vector store was not created."