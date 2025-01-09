import tiktoken
from .extract_texts import logger

def count_tokens(text):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        num_tokens = len(tokens)
        logger.info(f"Text: {text[:30]}... | Tokens: {num_tokens}")
        return num_tokens
    except Exception as e:
        logger.error(f"Error counting tokens for text: {text[:30]}... | Error: {e}")
        return 0