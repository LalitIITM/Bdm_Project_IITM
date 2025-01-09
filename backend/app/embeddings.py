import hashlib
from .extract_texts import logger

def generate_hash(text):
    """Generate a SHA-256 hash for the text."""
    return hashlib.sha256(text.encode()).hexdigest()

def store_embeddings_in_supabase(supabase_client, cleaned_texts, embedder):
    """Generate embeddings for cleaned text and store them in Supabase."""
    
    for text in cleaned_texts:
        # Generate hash for the text
        text_hash = generate_hash(text)

        # Check if embedding with the same hash already exists
        response = supabase_client.table("embeddings").select("*").eq("hash", text_hash).execute()
        if response.data:
            logger.info(f"Embedding already exists for text: {text[:30]}...")
            continue

        # Generate embedding if it doesn't exist
        embedding = embedder.embed_documents([text])[0]
        data = {
            "text": text,
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
            "hash": text_hash
        }
        response = supabase_client.table("embeddings").insert(data).execute()
        if response.status_code == 201:
            logger.info(f"Successfully stored embedding for text: {text[:30]}...")
        else:
            logger.error(f"Failed to store embedding for text: {text[:30]}... | Response: {response}")

def load_embeddings_from_supabase(supabase_client):
    """Load embeddings from Supabase."""
    response = supabase_client.table("embeddings").select("*").execute()
    if response.status_code == 200:
        logger.info("Successfully loaded embeddings from Supabase.")
        return response.data
    else:
        logger.error(f"Failed to load embeddings from Supabase. | Response: {response}")
        return None