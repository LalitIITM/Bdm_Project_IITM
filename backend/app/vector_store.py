import os
from .extract_texts import logger
from threading import Lock
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .extract_texts import load_hidden_documents
from .embeddings import store_embeddings_in_supabase, load_embeddings_from_supabase


VECTOR_STORE_PATH = "faiss_index"
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# In-memory storage for vector store and file modification times
in_memory_store = {
    "vector_store": None,
    "file_mod_times": None
}

# Thread lock for thread safety
store_lock = Lock()

def create_vector_store(document_texts):
    """Create a FAISS vector store from the provided document texts."""
    vector_store = FAISS.from_texts(document_texts, embedder)
    logger.info("Vector store created successfully.")
    return vector_store

def get_file_mod_times(directory):
    """Get the modification times of all files in the directory."""
    return {
        f: os.path.getmtime(os.path.join(directory, f))
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))  # Ensure it's a file, not a directory
    }
def load_or_build_vector_store(directory,supabase_client, embedder):
    """Load the existing vector store if available; otherwise, build it from scratch."""
    if os.path.exists(VECTOR_STORE_PATH):
        logger.info("Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_PATH, embedder, allow_dangerous_deserialization=True)
    else:
        logger.info("Building new vector store...")
        document_texts = load_hidden_documents(directory)
        if not document_texts:
            logger.warning("No documents found in the directory. Vector store not created.")
            return None
        logger.info(f"Document texts extracted for vector store are of length {len(document_texts)}")
        store_embeddings_in_supabase(supabase_client, document_texts, embedder)
        logger.info(f"Embeddings stored in supabase")
        vector_store = create_vector_store(document_texts)
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store
    
def reload_vector_store_if_needed(directory, supabase_client):
    """Reload the vector store if any files in the directory have been modified."""
    with store_lock:  # Ensure thread safety
        current_mod_times = get_file_mod_times(directory)

        if in_memory_store["file_mod_times"] != current_mod_times:
            in_memory_store["file_mod_times"] = current_mod_times
            vector_store = load_or_build_vector_store(directory, supabase_client, embedder)
            in_memory_store["vector_store"] = vector_store
            logger.info("Vector store reloaded and stored in memory.")
        else:
            vector_store = in_memory_store.get("vector_store", None)
            if vector_store:
                logger.info("Vector store loaded from memory.")
            else:
                logger.error("Vector store not found in memory.")

    return vector_store
