# Agentic RAG System Documentation

## Overview

The Agentic RAG system implements a sophisticated multi-agent architecture for enhanced document-based question answering. Unlike traditional RAG systems that follow a linear retrieve-and-generate pattern, our agentic approach uses specialized AI agents that can perceive, reason, and act collaboratively.

## Core Concepts

### What is Agentic RAG?

Agentic RAG extends traditional Retrieval-Augmented Generation by incorporating autonomous agents that:
- Make intelligent decisions about retrieval strategies
- Perform multi-step reasoning
- Use tools to enhance their capabilities
- Maintain memory and context
- Adapt to query complexity

### Agent Architecture

Each agent follows the **Perceive-Reason-Act** cycle:

1. **Perceive**: Process and understand input data
2. **Reason**: Analyze the situation and plan actions
3. **Act**: Execute the planned actions

## Components

### 1. BaseAgent (`base_agent.py`)

The foundation for all agents in the system.

**Key Features:**
- Abstract interface defining the perceive-reason-act cycle
- Memory management for maintaining context
- Tool registration and usage
- Event tracking and history

**Example Usage:**
```python
from app.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def perceive(self, input_data):
        # Process input
        return processed_data
    
    def reason(self, perception):
        # Make decisions
        return plan
    
    def act(self, reasoning):
        # Execute actions
        return result
```

### 2. RetrievalAgent (`retrieval_agent.py`)

Specialized agent for document retrieval and context extraction.

**Capabilities:**
- Similarity-based retrieval
- Maximum Marginal Relevance (MMR) for diverse results
- Adaptive k-value selection based on query complexity
- Scored retrieval for ranking

**Usage:**
```python
from app.agents.retrieval_agent import RetrievalAgent

retrieval_agent = RetrievalAgent(
    name="DocumentRetriever",
    vector_store=vector_store,
    retrieval_strategy="similarity"
)

result = retrieval_agent.run({
    "query": "What is machine learning?",
    "k": 5
})

documents = result["documents"]
```

**Retrieval Strategies:**
- `similarity`: Standard similarity search
- `mmr`: Maximum Marginal Relevance for diversity

### 3. ReasoningAgent (`reasoning_agent.py`)

Handles complex reasoning and answer generation.

**Capabilities:**
- Query classification (factual, comparison, analysis, etc.)
- Multi-step reasoning for complex queries
- Query decomposition
- Answer synthesis from multiple sources
- Context-aware responses

**Query Types Detected:**
- **Factual**: "What is X?", "When did Y happen?"
- **Comparison**: "Compare X and Y", "What's the difference?"
- **Analysis**: "Why does X happen?", "Explain Y"
- **Calculation**: "How many?", "Calculate X"
- **General**: Open-ended queries

**Usage:**
```python
from app.agents.reasoning_agent import ReasoningAgent

reasoning_agent = ReasoningAgent(
    name="QueryReasoner",
    model=model,
    temperature=0.7
)

result = reasoning_agent.run({
    "query": "Compare supervised and unsupervised learning",
    "documents": retrieved_docs,
    "chat_history": []
})

answer = result["answer"]
```

**Reasoning Strategies:**
- `multi_step`: For complex, analytical queries
- `direct`: For simple factual questions
- `synthesis`: For general queries requiring information combination

### 4. AgentOrchestrator (`orchestrator.py`)

Coordinates multiple agents to work together.

**Orchestration Strategies:**

1. **Sequential**: Agents work one after another
   ```
   Query → Retrieval → Reasoning → Answer
   ```

2. **Parallel**: Multiple operations happen simultaneously (where possible)
   ```
   Query → [Retrieval1, Retrieval2] → Reasoning → Answer
   ```

3. **Adaptive**: Strategy changes based on query characteristics
   - Simple queries: Minimal retrieval + direct reasoning
   - Complex queries: Enhanced retrieval + multi-step reasoning

**Usage:**
```python
from app.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    retrieval_agent=retrieval_agent,
    reasoning_agent=reasoning_agent
)

result = orchestrator.process_query(
    query="Explain the benefits of agentic RAG",
    chat_history=[],
    strategy="adaptive"
)

print(result["answer"])
print(result["agents_used"])  # List of agents that participated
```

**Monitoring:**
```python
stats = orchestrator.get_agent_stats()
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']}")
print(f"Agent usage: {stats['agents']}")
```

### 5. Tool Registry (`tools.py`)

Provides reusable tools that agents can use.

**Built-in Tools:**

- `word_count`: Count words in text
- `extract_keywords`: Extract key terms
- `summarize_text`: Create brief summaries
- `classify_query_intent`: Classify query type
- `extract_entities`: Extract dates, numbers, names
- `calculate`: Perform basic calculations
- `format_timestamp`: Get formatted timestamps

**Creating Custom Tools:**
```python
from app.agents.tools import tool_registry

def custom_tool(text):
    # Your tool logic
    return result

tool_registry.register_tool(
    name="my_tool",
    func=custom_tool,
    description="Description of what the tool does"
)
```

**Using Tools in Agents:**
```python
# Register tool with agent
agent.register_tool("word_count", tool_registry.get_tool("word_count"))

# Use tool
word_count = agent.use_tool("word_count", "This is a test sentence")
```

### 6. AgenticRAG Integration (`agentic_rag.py`)

Main interface for using the agentic RAG system.

**Features:**
- Unified API for agentic and simple RAG modes
- Automatic fallback to simple mode on errors
- Tool registration with agents
- System statistics and monitoring

**Usage:**
```python
from app.agents.agentic_rag import create_agentic_rag

# Create agentic RAG system
agentic_rag = create_agentic_rag(
    vector_store=vector_store,
    model=model,
    enable_agents=True
)

# Query with agentic processing
result = agentic_rag.query(
    question="What are the advantages of multi-agent systems?",
    chat_history=[],
    strategy="adaptive"
)

print(result["answer"])
print(result["metadata"])

# Get system stats
stats = agentic_rag.get_stats()
```

## Workflow Examples

### Example 1: Simple Factual Query

**Query:** "What is machine learning?"

**Flow:**
1. **Orchestrator** receives query, classifies as "simple"
2. **Retrieval Agent**:
   - Perceives: Query length = 4 words, complexity = low
   - Reasons: Use k=4 documents, standard similarity
   - Acts: Retrieves 4 relevant documents
3. **Reasoning Agent**:
   - Perceives: Query type = factual
   - Reasons: Strategy = direct answer
   - Acts: Generates concise factual response

### Example 2: Complex Comparison Query

**Query:** "Compare the advantages and disadvantages of supervised learning versus unsupervised learning in the context of medical diagnosis."

**Flow:**
1. **Orchestrator** receives query, classifies as "complex"
2. **Retrieval Agent**:
   - Perceives: Query length = 18 words, complexity = high
   - Reasons: Use k=7 documents, needs comprehensive retrieval
   - Acts: Retrieves 7 diverse documents using MMR
3. **Reasoning Agent**:
   - Perceives: Query type = comparison, needs analysis
   - Reasons: Strategy = multi_step, decompose into sub-questions
   - Acts: 
     - Step 1: Identify supervised learning concepts
     - Step 2: Identify unsupervised learning concepts
     - Step 3: Compare in medical context
     - Step 4: Synthesize comprehensive answer

### Example 3: Conversational Query with Context

**Previous Context:**
- User asked about neural networks
- Discussed activation functions

**Current Query:** "How does that relate to deep learning?"

**Flow:**
1. **Orchestrator** detects chat history present
2. **Retrieval Agent**:
   - Perceives: Query + chat context
   - Reasons: Need context-aware retrieval
   - Acts: Retrieves documents relevant to both current and previous context
3. **Reasoning Agent**:
   - Perceives: Needs context from chat history
   - Reasons: Strategy = synthesis with context
   - Acts: Generates answer that references previous discussion

## Configuration

### Agent Parameters

**RetrievalAgent:**
```python
RetrievalAgent(
    name="DocumentRetriever",
    vector_store=vector_store,
    model=model,
    retrieval_strategy="similarity"  # or "mmr"
)
```

**ReasoningAgent:**
```python
ReasoningAgent(
    name="QueryReasoner",
    model=model,
    temperature=0.7  # 0.0-1.0, higher = more creative
)
```

**AgenticRAG:**
```python
create_agentic_rag(
    vector_store=vector_store,
    model=model,
    enable_agents=True  # False for simple RAG mode
)
```

## Best Practices

### 1. Choose the Right Strategy

- **Sequential**: Default, reliable, good for most queries
- **Adaptive**: Best for production, automatically optimizes
- **Parallel**: Use when you need maximum speed (future enhancement)

### 2. Memory Management

Agents maintain memory of their operations. Clear memory periodically:

```python
orchestrator.reset_agents()  # Clears all agent memory
```

### 3. Error Handling

The system automatically falls back to simpler modes on errors:
- Agentic mode fails → Simple RAG mode
- Multi-step reasoning fails → Direct answer mode

### 4. Monitoring

Regularly check agent statistics:

```python
stats = agentic_rag.get_stats()
if stats['success_rate'] < 0.8:
    # Investigate issues
    pass
```

### 5. Tool Usage

Only register tools that agents actually need:

```python
# Good: Register relevant tools
agent.register_tool("classify_query_intent", tool_func)

# Avoid: Registering too many unused tools
```

## Performance Considerations

### Memory Usage

- Agent memory is capped at 100 entries per agent
- Orchestrator history is capped at 1000 executions
- Clear memory periodically in long-running applications

### Latency

- Sequential processing: ~2-5 seconds per query
- Multi-step reasoning adds: ~1-2 seconds
- Document retrieval: ~0.5-1 second

### Optimization Tips

1. Use adaptive strategy for automatic optimization
2. Limit retrieval k-value for simple queries
3. Cache frequently accessed documents
4. Use temperature=0.0 for faster, more deterministic responses

## Extending the System

### Adding New Agent Types

```python
from app.agents.base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    def perceive(self, input_data):
        return {"answer": input_data["answer"]}
    
    def reason(self, perception):
        # Check answer quality
        return {"needs_improvement": False}
    
    def act(self, reasoning):
        if reasoning["needs_improvement"]:
            return {"validated": False}
        return {"validated": True}
```

### Adding New Tools

```python
from app.agents.tools import tool_registry

def sentiment_analysis(text):
    # Your sentiment analysis logic
    return "positive"  # or "negative", "neutral"

tool_registry.register_tool(
    name="sentiment_analysis",
    func=sentiment_analysis,
    description="Analyze sentiment of text"
)
```

### Custom Orchestration Strategies

```python
class CustomOrchestrator(AgentOrchestrator):
    def _custom_strategy(self, query, chat_history):
        # Your custom orchestration logic
        pass
```

## Troubleshooting

### Issue: Low Success Rate

**Solution:** Check agent memory and execution history
```python
stats = orchestrator.get_agent_stats()
# Analyze which agent is failing
```

### Issue: Slow Response Times

**Solution:** 
- Reduce k-value in retrieval
- Use temperature=0.0 for faster inference
- Clear agent memory if too large

### Issue: Irrelevant Answers

**Solution:**
- Check document quality in vector store
- Adjust retrieval strategy to MMR for diversity
- Tune reasoning agent temperature

## Future Enhancements

Planned features for the agentic RAG system:

1. **Parallel Processing**: True parallel agent execution
2. **Learning Agents**: Agents that improve from feedback
3. **Tool Learning**: Automatic tool selection based on query
4. **Multi-Modal Agents**: Support for images and other media
5. **Agent Communication**: Direct agent-to-agent messaging
6. **Hierarchical Agents**: Parent-child agent relationships

## References

- LangChain Documentation: https://python.langchain.com/
- Agent Architectures: https://arxiv.org/abs/2304.03442
- RAG Systems: https://arxiv.org/abs/2005.11401
