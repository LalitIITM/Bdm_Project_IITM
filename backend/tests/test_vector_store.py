import os
import pytest
from app.vector_store import create_vector_store, get_file_mod_times, reload_vector_store_if_needed

def test_create_vector_store():
    document_texts = ["This is a test document."]
    vector_store = create_vector_store(document_texts)
    assert vector_store is not None

def test_get_file_mod_times(tmpdir):
    test_dir = tmpdir.mkdir("hidden_docs")
    test_file = test_dir.join("test.txt")
    test_file.write("This is a test document.")
    mod_times = get_file_mod_times(str(test_dir))
    assert len(mod_times) == 1

def test_reload_vector_store_if_needed(tmpdir):
    test_dir = tmpdir.mkdir("hidden_docs")
    test_file = test_dir.join("test.txt")
    test_file.write("This is a test document.")
    vector_store = reload_vector_store_if_needed(str(test_dir))
    assert vector_store is not None