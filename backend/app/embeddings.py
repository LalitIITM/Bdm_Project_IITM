import os
from .extract_texts import logger

def store_embeddings_in_supabase(supabase_client, cleaned_texts, embedder):
    """Generate embeddings for cleaned text and store them in Supabase."""
    
    for text in cleaned_texts:
        # Check if embedding for the text already exists
        response = supabase_client.table("embeddings").select("*").eq("text", text).execute()
        if response.data:
            logger.info(f"Embedding already exists for text: {text[:30]}...")
            continue

        # Generate embedding if it doesn't exist
        embedding = embedder.embed_documents([text])[0]
        data = {
            "text": text,
            "embedding": embedding  # Convert numpy array to list for JSON serialization
        }
        response = supabase_client.table("embeddings").insert(data).execute()
        if response:
            logger.info(f"Successfully stored embedding for text: {text[:30]}...")
        else:
            logger.error(f"Failed to store embedding for text: {text[:30]}...")

def load_embeddings_from_supabase(supabase_client):
    """Load embeddings from Supabase."""
    response = supabase_client.table("embeddings").select("*").execute()
    if response:
        logger.info("Successfully loaded embeddings from Supabase.")
        return response.data
    else:
        logger.error("Failed to load embeddings from Supabase.")
        return None