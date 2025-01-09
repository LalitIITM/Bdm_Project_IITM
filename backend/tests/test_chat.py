import pytest
from app.chat import is_valid_email, process_user_input

def test_is_valid_email():
    valid_email = "22f1234567@ds.study.iitm.ac.in"
    invalid_email = "invalid.email@example.com"
    assert is_valid_email(valid_email) is True
    assert is_valid_email(invalid_email) is False

def test_process_user_input(mocker):
    mock_supabase = mocker.Mock()
    mock_model = mocker.Mock()
    email = "22f1234567@ds.study.iitm.ac.in"
    name = "Test User"
    user_input = "Hello"
    chat_history = []

    mock_model.invoke.return_value = {"answer": "Hello, how can I help you?"}
    answer, updated_chat_history = process_user_input(mock_supabase, mock_model, email, name, user_input, chat_history)
    assert answer == "Hello, how can I help you?"
    assert len(updated_chat_history) == 1