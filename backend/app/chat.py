import os, re
from datetime import datetime, timedelta
import pytz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .extract_texts import logger
from .tokens import count_tokens

# Email validation regex
email_regex = re.compile(r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$")

def is_valid_email(email):
    """Validate email format."""
    return email_regex.match(email) is not None or email == "nitin@ee.iitm.ac.in"

def save_session_to_supabase(supabase, email, name, chat_history):
    """Save the chat session to Supabase."""
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
    
    data_list = []
    for question, answer in chat_history:
        data = {
            "email": email,
            "name": name if name else None,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
        }
        data_list.append(data)
    
    response = supabase.table("chat_sessions_2").insert(data_list).execute()
    if response:
        logger.info("Session data saved to Supabase.")
    else:
        logger.error(f"Error saving session data to Supabase")
        return False
    return True

def find_similar_question(user_input, chat_history):
    """Find a similar question in the chat history."""
    questions = [q for q, _ in chat_history]
    if not questions:
        return None, None

    vectorizer = TfidfVectorizer().fit_transform([user_input] + questions)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    most_similar_index = cosine_similarities.argmax()
    most_similar_score = cosine_similarities[most_similar_index]

    # Define a similarity threshold (e.g., 0.8)
    if most_similar_score > 0.8:
        return questions[most_similar_index], chat_history[most_similar_index][1]
    return None, None

def process_user_input(supabase, retrieval_chain,email, name, user_input, chat_history=None, start_time=None):
    """Process the user's input and return the chatbot's response."""
    logger.info(f"Processing user input: {user_input}")

    if chat_history is None:
        chat_history = []

    if start_time is None:
        start_time = datetime.now()

    current_time = datetime.now()
    elapsed_time = current_time - start_time    

    # Check for similar questions in the chat history
    similar_question, similar_answer = find_similar_question(user_input, chat_history)
    if similar_question:
        logger.info(f"Found similar question: {similar_question}")
        # Return similar answer with 0 token count (no API call)
        return similar_answer, 0

    # Process the question and get the answer
    response = retrieval_chain.invoke({"question": user_input, "chat_history": chat_history})
    answer = response["answer"]
    chat_history.append((user_input, answer))
    
    logger.info(f"Chatbot response: {answer}")
    tokens_count = count_tokens(user_input)
    logger.info(f"Number of tokens sent to API: {tokens_count}")
        
    if user_input.lower() == "stop" or elapsed_time > timedelta(minutes=30):  # Adjusted session timeout
        save_session_to_supabase(supabase, email, name, chat_history)
        logger.info("Session data successfully saved to Supabase. Please refresh to start a new session.")
        return "Session data successfully saved to Supabase. Please refresh to start a new session", tokens_count

    return answer, tokens_count
