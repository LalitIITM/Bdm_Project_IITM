"""
Agentic RAG Examples
This file demonstrates various usage patterns for the agentic RAG system.
"""

# Example 1: Basic Usage
# =====================

from app.agents.agentic_rag import create_agentic_rag

# Assuming you have vector_store and model initialized
agentic_rag = create_agentic_rag(
    vector_store=vector_store,
    model=model,
    enable_agents=True
)

# Simple query
result = agentic_rag.query("What is machine learning?")
print(f"Answer: {result['answer']}")
print(f"Agents used: {result['agents_used']}")


# Example 2: Using Different Orchestration Strategies
# ===================================================

# Sequential processing (default)
result_seq = agentic_rag.query(
    question="Explain neural networks",
    strategy="sequential"
)

# Adaptive processing (recommended)
result_adaptive = agentic_rag.query(
    question="Compare supervised and unsupervised learning in detail",
    strategy="adaptive"
)


# Example 3: Conversational Chat
# ================================

chat_history = []
questions = [
    "What is deep learning?",
    "What are its main components?",
    "How does it differ from traditional machine learning?"
]

for question in questions:
    result = agentic_rag.query(
        question=question,
        chat_history=chat_history
    )
    
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
    
    # Update chat history
    chat_history.append((question, result['answer']))


# Example 4: Direct Agent Usage
# ==============================

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent

# Create individual agents
retrieval_agent = RetrievalAgent(
    name="MyRetriever",
    vector_store=vector_store
)

reasoning_agent = ReasoningAgent(
    name="MyReasoner",
    model=model,
    temperature=0.7
)

# Use retrieval agent
retrieval_result = retrieval_agent.run({
    "query": "machine learning",
    "k": 5
})
documents = retrieval_result['documents']

# Use reasoning agent with retrieved documents
reasoning_result = reasoning_agent.run({
    "query": "What is machine learning?",
    "documents": documents,
    "chat_history": []
})
answer = reasoning_result['answer']


# Example 5: Custom Orchestration
# =================================

from app.agents.orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(
    retrieval_agent=retrieval_agent,
    reasoning_agent=reasoning_agent
)

# Process query with custom strategy
result = orchestrator.process_query(
    query="Explain the differences between classification and regression",
    chat_history=[],
    strategy="adaptive"
)

# Get statistics
stats = orchestrator.get_agent_stats()
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.2%}")


# Example 6: Using Tools
# =======================

from app.agents.tools import tool_registry

# Use built-in tools
word_count_tool = tool_registry.get_tool("word_count")
count = word_count_tool("This is a test sentence")
print(f"Word count: {count}")

keywords_tool = tool_registry.get_tool("extract_keywords")
keywords = keywords_tool("Machine learning is a subset of artificial intelligence")
print(f"Keywords: {keywords}")

intent_tool = tool_registry.get_tool("classify_query_intent")
intent = intent_tool("How do I train a neural network?")
print(f"Query intent: {intent}")


# Example 7: Registering Custom Tools
# ====================================

def sentiment_analyzer(text):
    """Simple sentiment analysis."""
    positive_words = ['good', 'great', 'excellent', 'amazing']
    negative_words = ['bad', 'poor', 'terrible', 'awful']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"

# Register custom tool
tool_registry.register_tool(
    name="sentiment_analysis",
    func=sentiment_analyzer,
    description="Analyze sentiment of text"
)

# Use custom tool with agent
retrieval_agent.register_tool("sentiment_analysis", sentiment_analyzer)
sentiment = retrieval_agent.use_tool("sentiment_analysis", "This is great!")
print(f"Sentiment: {sentiment}")


# Example 8: Memory Management
# =============================

# Check agent memory
retrieval_memory = retrieval_agent.get_memory(limit=5)
print(f"Recent retrieval actions: {len(retrieval_memory)}")

reasoning_memory = reasoning_agent.get_memory(limit=5)
print(f"Recent reasoning actions: {len(reasoning_memory)}")

# Clear memory when needed
orchestrator.reset_agents()
print("All agent memory cleared")


# Example 9: Advanced Retrieval
# ==============================

# Retrieval with scores
scored_docs = retrieval_agent.retrieve_with_score(
    query="deep learning",
    k=5
)
for doc, score in scored_docs:
    print(f"Score: {score:.4f}, Content: {doc.page_content[:100]}...")

# Diverse retrieval using MMR
diverse_docs = retrieval_agent.retrieve_diverse(
    query="neural networks",
    k=5,
    lambda_mult=0.5  # Balance between relevance and diversity
)


# Example 10: System Statistics and Monitoring
# =============================================

# Get comprehensive statistics
stats = agentic_rag.get_stats()
print(f"Mode: {stats['mode']}")
print(f"Agents enabled: {stats['agents_enabled']}")
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Tools available: {stats['tools_available']}")

# Monitor specific agents
print(f"Retrieval agent usage: {stats['agents']['retrieval']}")
print(f"Reasoning agent usage: {stats['agents']['reasoning']}")


# Example 11: Error Handling and Fallback
# ========================================

try:
    # Attempt agentic processing
    result = agentic_rag.query("Complex query")
    print(f"Answer: {result['answer']}")
except Exception as e:
    print(f"Error: {e}")
    # System automatically falls back to simple mode

# Check if fallback occurred
if 'error' in result:
    print(f"Fallback occurred: {result['error']}")
    # Handle fallback scenario


# Example 12: Flask API Integration
# ==================================

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/agentic_chat', methods=['POST'])
def agentic_chat():
    data = request.json
    
    result = agentic_rag.query(
        question=data['question'],
        chat_history=data.get('chat_history', []),
        strategy=data.get('strategy', 'adaptive')
    )
    
    return jsonify({
        'answer': result['answer'],
        'metadata': {
            'agents_used': result.get('agents_used', []),
            'retrieval_count': result.get('retrieval_count', 0),
            'orchestration': result.get('orchestration', '')
        }
    })

@app.route('/agentic_stats', methods=['GET'])
def get_stats():
    stats = agentic_rag.get_stats()
    return jsonify(stats)


# Example 13: Query Type Specific Processing
# ===========================================

queries_by_type = {
    'factual': "What is the capital of France?",
    'comparison': "Compare Python and Java",
    'analysis': "Why is machine learning important?",
    'instructional': "How to train a neural network?"
}

for query_type, query in queries_by_type.items():
    result = agentic_rag.query(query)
    print(f"\nQuery Type: {query_type}")
    print(f"Query: {query}")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Strategy used: {result.get('reasoning_strategy', 'N/A')}")


# Example 14: Batch Processing
# =============================

questions = [
    "What is AI?",
    "What is ML?",
    "What is DL?",
    "Compare AI and ML"
]

results = []
for question in questions:
    result = agentic_rag.query(question)
    results.append({
        'question': question,
        'answer': result['answer'],
        'agents': result.get('agents_used', [])
    })

# Analyze batch results
for r in results:
    print(f"Q: {r['question']}")
    print(f"Agents: {r['agents']}\n")


# Example 15: Context-Aware Multi-Turn Conversation
# ==================================================

def interactive_chat():
    """Interactive chat session with context awareness."""
    chat_history = []
    print("Agentic RAG Chat (type 'quit' to exit)")
    
    while True:
        question = input("\nYou: ")
        if question.lower() == 'quit':
            break
        
        result = agentic_rag.query(
            question=question,
            chat_history=chat_history,
            strategy="adaptive"
        )
        
        answer = result['answer']
        print(f"Assistant: {answer}")
        
        # Show metadata
        print(f"[Agents: {result.get('agents_used', [])}]")
        print(f"[Strategy: {result.get('orchestration', 'N/A')}]")
        
        # Update history
        chat_history.append((question, answer))
        
        # Keep history manageable
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

# Uncomment to run interactive chat
# interactive_chat()


# Example 16: Performance Benchmarking
# =====================================

import time

def benchmark_strategies():
    """Compare performance of different strategies."""
    test_query = "Explain the concept of neural networks and their applications"
    strategies = ['sequential', 'adaptive']
    
    for strategy in strategies:
        start_time = time.time()
        result = agentic_rag.query(test_query, strategy=strategy)
        end_time = time.time()
        
        print(f"\nStrategy: {strategy}")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Retrieval count: {result.get('retrieval_count', 0)}")

# Uncomment to run benchmark
# benchmark_strategies()
