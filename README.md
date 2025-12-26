# Bdm_Project_IITM - Agentic RAG System

An advanced web application featuring an **Agentic Retrieval-Augmented Generation (RAG)** system to develop intelligent chatbots that answer queries on any subject based on documents.

## Overview

This project implements a sophisticated agentic RAG system where AI agents collaborate to provide enhanced question-answering capabilities. The system goes beyond traditional RAG by incorporating:

- **Multi-Agent Architecture**: Specialized agents for retrieval, reasoning, and orchestration
- **Intelligent Query Processing**: Adaptive strategies based on query complexity
- **Multi-Step Reasoning**: Breaking down complex queries into manageable sub-tasks
- **Tool-Using Agents**: Agents can use various tools for text analysis and processing
- **Memory & State Management**: Agents maintain context across interactions

## Agentic RAG Architecture

### Core Components

#### 1. **Base Agent System** (`backend/app/agents/base_agent.py`)
- Abstract base class for all agents
- Implements Perceive-Reason-Act cycle
- Manages agent memory and tool registration

#### 2. **Retrieval Agent** (`backend/app/agents/retrieval_agent.py`)
- Specialized in document retrieval
- Supports multiple retrieval strategies (similarity, MMR)
- Adaptive retrieval based on query complexity

#### 3. **Reasoning Agent** (`backend/app/agents/reasoning_agent.py`)
- Handles multi-step reasoning
- Query classification and intent detection
- Answer synthesis from multiple sources

#### 4. **Agent Orchestrator** (`backend/app/agents/orchestrator.py`)
- Coordinates multiple agents
- Supports sequential, parallel, and adaptive orchestration
- Tracks agent performance and statistics

#### 5. **Tool Registry** (`backend/app/agents/tools.py`)
- Provides reusable tools for agents
- Includes text analysis, entity extraction, and calculation tools
- Extensible framework for adding new tools

### How It Works

```
User Query → Orchestrator → Retrieval Agent → Reasoning Agent → Response
                ↓              ↓                    ↓
           Strategy        Documents            Multi-step
           Selection       Retrieved            Reasoning
```

1. **Query Reception**: User submits a query to the system
2. **Orchestration**: Orchestrator selects processing strategy (sequential/adaptive)
3. **Retrieval**: Retrieval agent fetches relevant documents from vector store
4. **Reasoning**: Reasoning agent analyzes context and generates comprehensive response
5. **Response Delivery**: Final answer returned with metadata

## Features

### Traditional RAG Features
- Document ingestion (PDF, DOCX, TXT, CSV, PPTX, ZIP)
- Vector store with FAISS
- Conversational chat with history
- Token counting and session management

### Agentic RAG Enhancements
- **Adaptive Query Processing**: Automatically adjusts strategy based on query complexity
- **Multi-Step Reasoning**: Breaks down complex queries for better answers
- **Query Classification**: Identifies query type (factual, comparison, analysis)
- **Tool Integration**: Agents use specialized tools for enhanced capabilities
- **Agent Memory**: Maintains context and learns from interactions
- **Performance Monitoring**: Track agent usage and success rates

## API Endpoints

### Standard Endpoints
- `POST /validate_email` - Validate user email
- `POST /chat` - Traditional chat endpoint
- `POST /get_token_count_from_input` - Get token count for input

### Agentic RAG Endpoints
- `POST /agentic_chat` - Enhanced chat with agentic capabilities
  - Parameters: `email`, `name`, `question`, `chat_history`, `strategy` (optional)
  - Returns: Answer with metadata including agents used and reasoning strategy
  
- `GET /agentic_stats` - Get agent performance statistics
  - Returns: Total executions, success rate, agent usage stats

## Setup and Installation

1. **Clone the repository**
```bash
git clone https://github.com/LalitIITM/Bdm_Project_IITM.git
cd Bdm_Project_IITM/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file in the backend directory:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
```

4. **Add documents**
Place your documents in the `backend/hidden_docs/` directory

5. **Run the application**
```bash
python main.py
```

## Usage Examples

### Using Agentic Chat

```python
import requests

# Simple query with adaptive strategy
response = requests.post('http://localhost:5000/agentic_chat', json={
    'email': 'user@example.com',
    'name': 'John Doe',
    'question': 'What are the key differences between supervised and unsupervised learning?',
    'chat_history': [],
    'strategy': 'adaptive'  # or 'sequential'
})

print(response.json())
# Output includes answer and metadata about agents used
```

### Getting Agent Statistics

```python
response = requests.get('http://localhost:5000/agentic_stats')
stats = response.json()['stats']
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']}")
```

## Architecture Comparison

### Traditional RAG
- Query → Retrieval → LLM → Answer
- Simple, direct processing
- Limited reasoning capabilities

### Agentic RAG
- Query → Orchestrator → Multiple Agents → Collaborative Answer
- Adaptive strategy selection
- Multi-step reasoning
- Tool usage for enhanced capabilities
- Memory and context management

## Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: LangChain, Groq (LLaMA 3), HuggingFace
- **Vector Store**: FAISS
- **Embeddings**: Sentence Transformers
- **Database**: Supabase
- **Document Processing**: PyPDF, python-docx, python-pptx, BeautifulSoup

## Project Structure

```
backend/
├── app/
│   ├── agents/              # Agentic RAG components
│   │   ├── __init__.py
│   │   ├── base_agent.py    # Base agent class
│   │   ├── retrieval_agent.py
│   │   ├── reasoning_agent.py
│   │   ├── orchestrator.py
│   │   ├── tools.py         # Tool registry
│   │   └── agentic_rag.py   # Main integration
│   ├── chat.py              # Chat logic
│   ├── embeddings.py        # Embedding management
│   ├── extract_texts.py     # Document processing
│   ├── tokens.py            # Token counting
│   └── vector_store.py      # Vector store management
├── tests/                   # Test suite
├── main.py                  # Flask application
└── requirements.txt         # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.
