import os
import pytest
from app.extract_texts import clean_text, load_hidden_documents

def test_clean_text():
    text = "This  is a   test.\nNewLine:Test"
    cleaned_text = clean_text(text)
    expected_text = "This is a test.\nNewLine:\nTest"
    assert cleaned_text == expected_text

def test_load_hidden_documents(tmpdir):
    # Create a temporary directory and files for testing
    test_dir = tmpdir.mkdir("hidden_docs")
    test_file = test_dir.join("test.txt")
    test_file.write("This is a test document.")

    # Load documents from the temporary directory
    documents = load_hidden_documents(str(test_dir))
    assert len(documents) == 1
    assert documents[0] == "This is a test document."