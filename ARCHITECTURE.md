# Agentic RAG Architecture

## System Overview

The Agentic RAG system is a sophisticated multi-agent architecture that enhances traditional Retrieval-Augmented Generation with autonomous agents capable of perception, reasoning, and action.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                              │
│                     (Flask REST API Endpoints)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AgenticRAG Controller                           │
│                   (agentic_rag.py)                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  - Query processing                                            │ │
│  │  - Strategy selection                                          │ │
│  │  - Fallback management                                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Agent Orchestrator                               │
│                   (orchestrator.py)                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Orchestration Strategies:                                     │ │
│  │  - Sequential: Agent1 → Agent2 → Result                        │ │
│  │  - Parallel: [Agent1, Agent2] → Merge → Result                │ │
│  │  - Adaptive: Dynamic strategy based on query                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────┬────────────────────────┬────────────────────────────────┘
             │                        │
             ▼                        ▼
┌──────────────────────┐    ┌──────────────────────┐
│  Retrieval Agent     │    │  Reasoning Agent     │
│ (retrieval_agent.py) │    │ (reasoning_agent.py) │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           │                           │
    ┌──────┴───────┐           ┌──────┴───────┐
    │   Perceive   │           │   Perceive   │
    │   Reason     │           │   Reason     │
    │   Act        │           │   Act        │
    └──────┬───────┘           └──────┬───────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Vector Store       │    │  Language Model      │
│   (FAISS)            │    │  (Groq/LLaMA 3)      │
└──────────────────────┘    └──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Document Store     │
│   (hidden_docs/)     │
└──────────────────────┘
```

## Component Details

### 1. User Interface Layer

**Components:**
- Flask REST API
- Request/Response handlers
- Authentication middleware

**Endpoints:**
- `/agentic_chat` - Enhanced agentic processing
- `/chat` - Traditional RAG (backward compatible)
- `/agentic_stats` - System monitoring

### 2. AgenticRAG Controller

**Responsibilities:**
- Route queries to appropriate processing mode
- Manage agent lifecycle
- Handle fallback scenarios
- Provide unified API

**Flow:**
```
Query → Enable Agents? 
        ├─ Yes → Orchestrator → Agents → Result
        └─ No  → Simple RAG → Result
```

### 3. Agent Orchestrator

**Strategies:**

#### Sequential Processing
```
User Query
    ↓
Retrieval Agent
    ├─ Perceive: Analyze query
    ├─ Reason: Determine retrieval strategy
    └─ Act: Retrieve documents
    ↓
Reasoning Agent
    ├─ Perceive: Analyze query + documents
    ├─ Reason: Plan response strategy
    └─ Act: Generate answer
    ↓
Final Answer
```

#### Adaptive Processing
```
User Query
    ↓
Complexity Analysis
    ├─ Simple → k=4, direct reasoning
    ├─ Medium → k=5, synthesis
    └─ Complex → k=7, multi-step
    ↓
Dynamic Agent Configuration
    ↓
Optimized Processing
    ↓
Final Answer
```

### 4. Agent Architecture

#### Base Agent (Perceive-Reason-Act Cycle)

```
┌─────────────────────────────────────────┐
│            Base Agent                    │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  PERCEIVE                          │ │
│  │  - Process input data              │ │
│  │  - Extract features                │ │
│  │  - Store in working memory         │ │
│  └────────────┬───────────────────────┘ │
│               │                          │
│               ▼                          │
│  ┌────────────────────────────────────┐ │
│  │  REASON                            │ │
│  │  - Analyze situation               │ │
│  │  - Plan actions                    │ │
│  │  - Select strategy                 │ │
│  └────────────┬───────────────────────┘ │
│               │                          │
│               ▼                          │
│  ┌────────────────────────────────────┐ │
│  │  ACT                               │ │
│  │  - Execute plan                    │ │
│  │  - Use tools                       │ │
│  │  - Generate output                 │ │
│  └────────────────────────────────────┘ │
│                                          │
│  Memory: [perception, reasoning, action] │
│  Tools: {tool1, tool2, ...}             │
└─────────────────────────────────────────┘
```

#### Retrieval Agent Workflow

```
Input: Query + Parameters
    ↓
[PERCEIVE]
    ├─ Query length
    ├─ Complexity estimation
    └─ Filter requirements
    ↓
[REASON]
    ├─ Query type: simple | complex
    ├─ Strategy: similarity | mmr
    ├─ Optimal k: 4-7 documents
    └─ Diversity needed?
    ↓
[ACT]
    ├─ Configure retriever
    ├─ Execute search
    └─ Return documents
    ↓
Output: Documents + Metadata
```

#### Reasoning Agent Workflow

```
Input: Query + Documents + History
    ↓
[PERCEIVE]
    ├─ Query classification
    │   ├─ Factual: "What is X?"
    │   ├─ Comparison: "Compare X vs Y"
    │   ├─ Analysis: "Why/How X?"
    │   └─ Calculation: "How many X?"
    ├─ Document analysis
    └─ Context requirements
    ↓
[REASON]
    ├─ Strategy selection
    │   ├─ Direct: Simple factual
    │   ├─ Multi-step: Complex analysis
    │   └─ Synthesis: General queries
    ├─ Query decomposition (if complex)
    └─ Response planning
    ↓
[ACT]
    ├─ Build context from documents
    ├─ Generate prompt
    ├─ Call LLM
    └─ Format response
    ↓
Output: Answer + Reasoning Metadata
```

### 5. Tool Registry

```
┌────────────────────────────────────────┐
│          Tool Registry                  │
├────────────────────────────────────────┤
│                                         │
│  Text Analysis Tools:                  │
│  ├─ word_count(text)                   │
│  ├─ extract_keywords(text, n)          │
│  ├─ summarize_text(text, length)       │
│  └─ extract_entities(text)             │
│                                         │
│  Query Analysis Tools:                 │
│  ├─ classify_query_intent(query)       │
│  └─ detect_query_complexity(query)     │
│                                         │
│  Utility Tools:                        │
│  ├─ calculate(expression)              │
│  └─ format_timestamp(format)           │
│                                         │
│  Custom Tools:                         │
│  └─ register_tool(name, func, desc)    │
│                                         │
└────────────────────────────────────────┘
```

## Data Flow

### Simple Query Flow

```
"What is machine learning?"
    ↓
Orchestrator (Adaptive)
    ↓
Retrieval Agent
    ├─ Complexity: Low
    ├─ k = 4
    └─ Strategy: similarity
    ↓
Retrieved: 4 documents
    ↓
Reasoning Agent
    ├─ Type: Factual
    ├─ Strategy: Direct
    └─ Context: 4 docs
    ↓
LLM: "Machine learning is..."
    ↓
User: Answer + Metadata
```

### Complex Query Flow

```
"Compare supervised vs unsupervised learning
 for medical diagnosis applications"
    ↓
Orchestrator (Adaptive)
    ↓
Retrieval Agent
    ├─ Complexity: High (18 words)
    ├─ k = 7
    └─ Strategy: mmr (diversity)
    ↓
Retrieved: 7 diverse documents
    ↓
Reasoning Agent
    ├─ Type: Comparison
    ├─ Strategy: Multi-step
    ├─ Steps:
    │   ├─ 1. Identify supervised concepts
    │   ├─ 2. Identify unsupervised concepts
    │   ├─ 3. Compare in medical context
    │   └─ 4. Synthesize conclusion
    └─ Context: 7 docs
    ↓
LLM (multiple calls):
    ├─ Step 1 output
    ├─ Step 2 output
    ├─ Step 3 output
    └─ Final synthesis
    ↓
User: Comprehensive Answer + Full Metadata
```

### Conversational Flow

```
Previous: "What is deep learning?"
    ↓
Current: "How does that relate to CNNs?"
    ↓
Orchestrator (Context-aware)
    ↓
Retrieval Agent
    ├─ Query + Previous context
    ├─ Enhanced search
    └─ k = 5
    ↓
Reasoning Agent
    ├─ History: Last 3 exchanges
    ├─ Strategy: Synthesis with context
    └─ Reference: Previous answer
    ↓
User: Contextual Answer
```

## Memory Management

```
┌─────────────────────────────────────────┐
│         Agent Memory System              │
├─────────────────────────────────────────┤
│                                          │
│  Per-Agent Memory:                      │
│  ┌────────────────────────────────────┐ │
│  │  Entry: {                          │ │
│  │    timestamp: "2024-01-01...",     │ │
│  │    type: "perception",             │ │
│  │    data: {...}                     │ │
│  │  }                                 │ │
│  └────────────────────────────────────┘ │
│  Max: 100 entries per agent            │
│                                          │
│  Orchestrator History:                  │
│  ┌────────────────────────────────────┐ │
│  │  Execution: {                      │ │
│  │    query: "...",                   │ │
│  │    strategy: "adaptive",           │ │
│  │    agents_used: [...],             │ │
│  │    success: true                   │ │
│  │  }                                 │ │
│  └────────────────────────────────────┘ │
│  Max: 1000 execution records           │
│                                          │
└─────────────────────────────────────────┘
```

## Performance Characteristics

### Latency Breakdown

```
Simple Query (5 words):
├─ Retrieval: 0.5s
├─ Reasoning: 1.0s
└─ Total: ~1.5s

Complex Query (20 words):
├─ Retrieval: 0.8s (more docs)
├─ Multi-step Reasoning: 3.0s
└─ Total: ~3.8s

With Context (follow-up):
├─ Retrieval: 0.7s (context-aware)
├─ Reasoning: 1.5s (synthesis)
└─ Total: ~2.2s
```

### Scalability

```
┌─────────────────────────────────────┐
│         Scaling Dimensions          │
├─────────────────────────────────────┤
│  Documents:                         │
│  └─ 10-1000 docs: Good             │
│  └─ 1000-10000 docs: Optimize FAISS│
│                                      │
│  Concurrent Users:                  │
│  └─ 1-10: No changes needed        │
│  └─ 10-100: Add caching            │
│  └─ 100+: Load balancing           │
│                                      │
│  Query Rate:                        │
│  └─ <10/min: Current setup         │
│  └─ 10-100/min: Rate limiting      │
│  └─ >100/min: Queue system         │
└─────────────────────────────────────┘
```

## Extensibility Points

### Adding New Agents

```python
from app.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def perceive(self, input_data):
        # Custom perception logic
        pass
    
    def reason(self, perception):
        # Custom reasoning logic
        pass
    
    def act(self, reasoning):
        # Custom action logic
        pass

# Register with orchestrator
orchestrator.agents["custom"] = CustomAgent(...)
```

### Adding New Tools

```python
from app.agents.tools import tool_registry

@tool_registry.register_tool(
    name="sentiment",
    description="Analyze sentiment"
)
def sentiment_tool(text):
    return analyze_sentiment(text)
```

### Custom Orchestration

```python
class CustomOrchestrator(AgentOrchestrator):
    def _custom_strategy(self, query, history):
        # Your orchestration logic
        # Can use any combination of agents
        pass
```

## Security Considerations

```
┌─────────────────────────────────────┐
│        Security Layers              │
├─────────────────────────────────────┤
│  1. Input Validation                │
│     └─ Email format check           │
│     └─ Query sanitization           │
│                                      │
│  2. Tool Safety                     │
│     └─ Calculation: No eval()       │
│     └─ File access: Restricted      │
│                                      │
│  3. LLM Safety                      │
│     └─ Prompt injection prevention  │
│     └─ Output filtering             │
│                                      │
│  4. Resource Limits                 │
│     └─ Max query length             │
│     └─ Max retrieval documents      │
│     └─ Memory limits                │
└─────────────────────────────────────┘
```

## Monitoring and Observability

```
Metrics Collected:
├─ Query metrics
│  ├─ Total queries
│  ├─ Success rate
│  ├─ Average latency
│  └─ Error rate
│
├─ Agent metrics
│  ├─ Agent usage count
│  ├─ Memory usage
│  └─ Tool calls
│
└─ Orchestration metrics
   ├─ Strategy distribution
   ├─ Document retrieval stats
   └─ Reasoning strategy usage
```

## Future Enhancements

```
Planned Features:
├─ Parallel agent execution
├─ Agent learning from feedback
├─ Multi-modal support (images, audio)
├─ Hierarchical agent systems
├─ Real-time streaming responses
└─ Advanced caching mechanisms
```

## Deployment Architecture

```
Production Deployment:

┌──────────────────────────────────┐
│       Load Balancer              │
└────────────┬─────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐      ┌─────────┐
│ Flask   │      │ Flask   │
│ Instance│      │ Instance│
│    1    │      │    2    │
└────┬────┘      └────┬────┘
     │                │
     └────────┬───────┘
              ▼
    ┌──────────────────┐
    │  Shared Vector   │
    │  Store (FAISS)   │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │   Supabase DB    │
    └──────────────────┘
```

This architecture provides a comprehensive, scalable, and maintainable foundation for an agentic RAG system.
