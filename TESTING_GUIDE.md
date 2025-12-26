# Testing the Agentic RAG System

## Quick Test Options

### Option 1: Run Unit Tests (Fastest - 5 seconds)

```bash
cd backend
python -m unittest tests.test_agents -v
```

**Expected Output:**
```
test_adaptive_processing ... ok
test_agent_stats ... ok
test_orchestrator_initialization ... ok
... (25 tests total)
----------------------------------------------------------------------
Ran 25 tests in 0.005s

OK
```

### Option 2: Test Agent Imports

```bash
cd backend
python -c "
from app.agents.agentic_rag import create_agentic_rag
from app.agents.tools import tool_registry
print('✓ All imports successful')
print(f'✓ {len(tool_registry.list_tools())} tools available')
print('✓ System ready')
"
```

### Option 3: Test with Mock Data (No Documents Required)

```bash
cd backend
python << 'EOF'
from unittest.mock import Mock
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.orchestrator import AgentOrchestrator

# Create mock vector store
mock_vector_store = Mock()
mock_retriever = Mock()
mock_vector_store.as_retriever.return_value = mock_retriever
mock_doc = Mock(page_content="Machine learning is a subset of AI.")
mock_retriever.get_relevant_documents.return_value = [mock_doc]

# Create mock model
mock_model = Mock()
mock_model.invoke.return_value = Mock(content="Machine learning is a method...")

# Initialize agents
retrieval_agent = RetrievalAgent("TestRetriever", mock_vector_store)
reasoning_agent = ReasoningAgent("TestReasoner", mock_model)
orchestrator = AgentOrchestrator(retrieval_agent, reasoning_agent)

# Test query processing
result = orchestrator.process_query("What is machine learning?")
print(f"✓ Query processed successfully")
print(f"✓ Answer generated: {result['answer'][:50]}...")
print(f"✓ Agents used: {result['agents_used']}")
print(f"✓ Orchestration: {result['orchestration']}")

# Get stats
stats = orchestrator.get_agent_stats()
print(f"✓ Total executions: {stats['total_executions']}")
print(f"✓ Success rate: {stats['success_rate']}")
EOF
```

### Option 4: Test Flask Application Endpoints

#### Step 1: Start the Server (Terminal 1)

```bash
cd backend

# Create minimal test setup (if .env doesn't exist)
cat > .env << 'EOF'
SUPABASE_URL=http://test.supabase.co
SUPABASE_KEY=test_key
GROQ_API_KEY=test_api_key
EOF

# Create a test document
mkdir -p hidden_docs
echo "This is a test document about machine learning." > hidden_docs/test.txt

# Start the server
python main.py
```

**Note:** If you get errors about missing API keys, the server will still start but may not process real queries. This is fine for testing the endpoints exist.

#### Step 2: Test Endpoints (Terminal 2)

```bash
# Test agentic stats endpoint (no auth needed)
curl http://localhost:5000/agentic_stats

# Test agentic chat endpoint structure
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "question": "What is AI?",
    "strategy": "adaptive"
  }'
```

### Option 5: Interactive Python Session

```bash
cd backend
python
```

Then run:

```python
from unittest.mock import Mock
from app.agents.agentic_rag import create_agentic_rag

# Create mock components
mock_vector_store = Mock()
mock_retriever = Mock()
mock_vector_store.as_retriever.return_value = mock_retriever
mock_doc = Mock(page_content="Test content about AI")
mock_retriever.get_relevant_documents.return_value = [mock_doc]

mock_model = Mock()
mock_model.invoke.return_value = Mock(content="AI is artificial intelligence...")

# Create agentic RAG
agentic_rag = create_agentic_rag(mock_vector_store, mock_model, enable_agents=True)

# Test query
result = agentic_rag.query("What is AI?")
print("Answer:", result['answer'])
print("Agents used:", result.get('agents_used', []))
print("Orchestration:", result.get('orchestration', 'N/A'))

# Get stats
stats = agentic_rag.get_stats()
print("\nSystem Stats:")
print(f"  Mode: {stats['mode']}")
print(f"  Agents enabled: {stats['agents_enabled']}")
print(f"  Total executions: {stats.get('total_executions', 0)}")
```

### Option 6: Test Individual Components

#### Test Tools
```bash
cd backend
python << 'EOF'
from app.agents.tools import tool_registry

# List all tools
tools = tool_registry.list_tools()
print(f"✓ {len(tools)} tools registered:")
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}")

# Test word count
wc_tool = tool_registry.get_tool("word_count")
result = wc_tool("This is a test sentence")
print(f"\n✓ Word count test: {result} words")

# Test keyword extraction
kw_tool = tool_registry.get_tool("extract_keywords")
keywords = kw_tool("Machine learning is a subset of artificial intelligence")
print(f"✓ Keywords extracted: {keywords}")

# Test query classification
classify_tool = tool_registry.get_tool("classify_query_intent")
intent = classify_tool("What is machine learning?")
print(f"✓ Query intent: {intent}")

print("\n✓ All tools working correctly!")
EOF
```

#### Test Agents Individually
```bash
cd backend
python << 'EOF'
from unittest.mock import Mock
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent

# Test Retrieval Agent
print("Testing Retrieval Agent...")
mock_vs = Mock()
mock_ret = Mock()
mock_vs.as_retriever.return_value = mock_ret
mock_ret.get_relevant_documents.return_value = [Mock(page_content="Test doc")]

retrieval_agent = RetrievalAgent("TestRetriever", mock_vs)
result = retrieval_agent.run({"query": "test query", "k": 5})
print(f"✓ Retrieved {result['count']} documents")
print(f"✓ Strategy: {result['strategy']}")

# Test Reasoning Agent
print("\nTesting Reasoning Agent...")
mock_model = Mock()
mock_model.invoke.return_value = Mock(content="Test answer")

reasoning_agent = ReasoningAgent("TestReasoner", mock_model)
result = reasoning_agent.run({
    "query": "What is AI?",
    "documents": [Mock(page_content="AI info")],
    "chat_history": []
})
print(f"✓ Answer generated: {result['answer'][:30]}...")
print(f"✓ Strategy used: {result['strategy_used']}")

print("\n✓ All agents working correctly!")
EOF
```

## Full Integration Test (With Real Setup)

If you have documents and valid API keys:

### Step 1: Setup
```bash
cd backend

# Ensure .env file has valid credentials
cat .env  # Check SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY

# Ensure documents exist
ls -la hidden_docs/
```

### Step 2: Run Application
```bash
python main.py
```

### Step 3: Test in Another Terminal
```bash
# Test traditional chat
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "name": "Your Name",
    "question": "What is the main topic of the documents?"
  }' | jq

# Test agentic chat with adaptive strategy
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "name": "Your Name",
    "question": "Compare and contrast the key concepts in the documents",
    "strategy": "adaptive"
  }' | jq

# Get system statistics
curl http://localhost:5000/agentic_stats | jq
```

## Comparing Traditional vs Agentic

Run both endpoints with the same question to see the difference:

```bash
QUESTION="What are the main advantages of the approach?"

# Traditional endpoint
echo "=== Traditional Chat ==="
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"test@example.com\",
    \"question\": \"$QUESTION\"
  }" | jq '.answer'

# Agentic endpoint
echo "=== Agentic Chat ==="
curl -X POST http://localhost:5000/agentic_chat \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"test@example.com\",
    \"question\": \"$QUESTION\",
    \"strategy\": \"adaptive\"
  }" | jq '.answer, .metadata'
```

## Troubleshooting

### If tests fail with "Module not found"
```bash
cd backend
pip install -r requirements.txt
```

### If vector store initialization fails
The application needs at least one document in `hidden_docs/`:
```bash
mkdir -p hidden_docs
echo "Sample document content" > hidden_docs/sample.txt
```

### If Supabase connection fails
For testing purposes, you can modify main.py to skip Supabase temporarily, but the agentic agents should still work with mock data (see Option 3 above).

## Expected Test Results

✓ **Unit Tests**: All 25 tests passing
✓ **Import Tests**: No errors, system ready message
✓ **Mock Tests**: Successful query processing with agent coordination
✓ **Endpoint Tests**: JSON responses with answers and metadata
✓ **Tool Tests**: All 7 tools functioning correctly
✓ **Agent Tests**: Individual agents working independently

## Quick Validation Checklist

- [ ] Unit tests pass (25/25)
- [ ] Imports work without errors
- [ ] Tools are registered and functional
- [ ] Agents can process queries
- [ ] Orchestrator coordinates agents
- [ ] Flask endpoints respond correctly
- [ ] Statistics endpoint shows metrics
- [ ] Agentic chat returns metadata

## Need Help?

Check the comprehensive documentation:
- **QUICKSTART.md** - Setup and first steps
- **backend/app/agents/README.md** - Detailed agent documentation
- **backend/app/agents/examples.py** - 16 usage examples
- **ARCHITECTURE.md** - System design details
