# Quick Start Guide: Agentic RAG System

## Introduction

This guide will help you get started with the Agentic RAG system in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/LalitIITM/Bdm_Project_IITM.git
cd Bdm_Project_IITM/backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the `backend` directory:

```bash
# backend/.env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Add Documents

Place your documents in the `backend/hidden_docs/` directory:

```bash
mkdir -p hidden_docs
# Copy your PDF, DOCX, TXT, CSV, or PPTX files here
```

Supported file formats:
- PDF (.pdf)
- Word (.docx)
- Text (.txt)
- CSV (.csv)
- PowerPoint (.pptx)
- ZIP archives (.zip)

### 5. Run the Application

```bash
python main.py
```

The server will start on `http://localhost:5000`

## Your First Query

### Using Traditional Chat

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "question": "What is machine learning?",
    "chat_history": []
  }'
```

### Using Agentic Chat (Enhanced)

```bash
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "question": "Compare supervised and unsupervised learning",
    "chat_history": [],
    "strategy": "adaptive"
  }'
```

Response format:
```json
{
  "status": "success",
  "answer": "Detailed answer...",
  "metadata": {
    "agents_used": ["retrieval", "reasoning"],
    "retrieval_count": 5,
    "reasoning_strategy": "multi_step",
    "orchestration": "adaptive"
  }
}
```

## Understanding the Response

The agentic chat response includes:

- **answer**: The generated response to your query
- **agents_used**: Which agents participated (retrieval, reasoning)
- **retrieval_count**: Number of documents retrieved
- **reasoning_strategy**: How the query was processed (multi_step, direct, synthesis)
- **orchestration**: Strategy used (sequential, adaptive)

## Orchestration Strategies

### Sequential (Default)
Best for: Standard queries
```json
{
  "strategy": "sequential"
}
```

### Adaptive (Recommended)
Best for: All types of queries, automatically optimizes
```json
{
  "strategy": "adaptive"
}
```

## Common Use Cases

### 1. Simple Factual Questions

```bash
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "email": "user@example.com",
    "strategy": "adaptive"
  }'
```

Expected agents: Retrieval (simple) → Reasoning (direct)

### 2. Complex Analytical Questions

```bash
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze the trade-offs between different neural network architectures for image classification",
    "email": "user@example.com",
    "strategy": "adaptive"
  }'
```

Expected agents: Retrieval (enhanced) → Reasoning (multi-step)

### 3. Conversational Context

```bash
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you explain that in simpler terms?",
    "email": "user@example.com",
    "chat_history": [
      ["What is quantum computing?", "Quantum computing is..."]
    ],
    "strategy": "adaptive"
  }'
```

Expected agents: Retrieval (context-aware) → Reasoning (synthesis)

## Monitoring and Statistics

Get system statistics:

```bash
curl http://localhost:5000/agentic_stats
```

Response:
```json
{
  "status": "success",
  "stats": {
    "mode": "agentic",
    "agents_enabled": true,
    "total_executions": 42,
    "success_rate": 0.95,
    "agents": {
      "retrieval": 42,
      "reasoning": 42
    },
    "tools_available": 7
  }
}
```

## Python Integration Example

```python
import requests

# Configure endpoint
url = "http://localhost:5000/agentic_chat"

# Prepare query
data = {
    "email": "user@example.com",
    "name": "John Doe",
    "question": "What are the benefits of deep learning?",
    "chat_history": [],
    "strategy": "adaptive"
}

# Send request
response = requests.post(url, json=data)
result = response.json()

# Process response
if result["status"] == "success":
    print(f"Answer: {result['answer']}")
    print(f"Agents used: {result['metadata']['agents_used']}")
    print(f"Documents retrieved: {result['metadata']['retrieval_count']}")
else:
    print(f"Error: {result['message']}")
```

## Advanced Features

### Custom Tools

You can add custom tools in your code:

```python
from app.agents.tools import tool_registry

def custom_analyzer(text):
    # Your custom analysis
    return result

tool_registry.register_tool(
    name="custom_analyzer",
    func=custom_analyzer,
    description="Custom text analyzer"
)
```

### Direct Agent Usage

For more control, use agents directly:

```python
from app.agents.agentic_rag import create_agentic_rag

agentic_rag = create_agentic_rag(vector_store, model)
result = agentic_rag.query("Your question")
```

## Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "Vector store not initialized"
**Solution**: Check that documents exist in `hidden_docs/`
```bash
ls -la hidden_docs/
```

### Issue: "API key error"
**Solution**: Verify `.env` file contains correct keys
```bash
cat .env
```

### Issue: Slow responses
**Solution**: 
- Reduce number of documents in `hidden_docs/`
- Use simpler orchestration strategy
- Check your API rate limits

## Next Steps

1. **Read the Full Documentation**: See `backend/app/agents/README.md`
2. **Explore Examples**: Check `backend/app/agents/examples.py`
3. **Run Tests**: `python -m unittest tests.test_agents`
4. **Customize Configuration**: Edit `backend/app/agents/config.yaml`

## Best Practices

1. **Use Adaptive Strategy**: Let the system optimize automatically
2. **Keep Documents Focused**: Better quality over quantity
3. **Monitor Statistics**: Check success rates regularly
4. **Clear Chat History**: Don't let history grow too large (>10 exchanges)
5. **Handle Errors**: Always check response status

## API Reference Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Traditional chat (backward compatible) |
| `/agentic_chat` | POST | Enhanced agentic chat |
| `/agentic_stats` | GET | System statistics |
| `/validate_email` | POST | Email validation |

## Support

For more information:
- **Documentation**: `backend/app/agents/README.md`
- **Examples**: `backend/app/agents/examples.py`
- **Issues**: GitHub Issues
- **Tests**: `backend/tests/test_agents.py`

## What's Next?

Now that you have the basics working, explore:

1. **Multi-turn conversations** with context
2. **Different orchestration strategies** for optimization
3. **Custom tools** for specialized tasks
4. **Performance monitoring** with statistics
5. **Integration** with your own applications

Happy coding! 🚀
