"""
Tests for the Agentic RAG System
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.base_agent import BaseAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.tools import ToolRegistry
from app.agents.agentic_rag import create_agentic_rag


class TestBaseAgent(unittest.TestCase):
    """Test the BaseAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        class TestAgent(BaseAgent):
            def perceive(self, input_data):
                return {"processed": input_data}
            
            def reason(self, perception):
                return {"plan": "test_plan"}
            
            def act(self, reasoning):
                return {"result": "success"}
        
        self.agent = TestAgent("TestAgent", "A test agent")
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.description, "A test agent")
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_agent_run_cycle(self):
        """Test the perceive-reason-act cycle."""
        result = self.agent.run({"test": "data"})
        self.assertEqual(result["result"], "success")
        self.assertEqual(len(self.agent.memory), 3)  # perceive, reason, act
    
    def test_tool_registration(self):
        """Test tool registration and usage."""
        def test_tool(x):
            return x * 2
        
        self.agent.register_tool("multiply", test_tool)
        result = self.agent.use_tool("multiply", 5)
        self.assertEqual(result, 10)
    
    def test_memory_management(self):
        """Test agent memory."""
        for i in range(5):
            self.agent.run({"iteration": i})
        
        memory = self.agent.get_memory()
        self.assertGreater(len(memory), 0)
        
        self.agent.clear_memory()
        self.assertEqual(len(self.agent.get_memory()), 0)


class TestRetrievalAgent(unittest.TestCase):
    """Test the RetrievalAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_retriever = Mock()
        self.mock_vector_store.as_retriever.return_value = self.mock_retriever
        
        # Mock documents
        self.mock_docs = [
            Mock(page_content="Document 1 content"),
            Mock(page_content="Document 2 content"),
            Mock(page_content="Document 3 content")
        ]
        self.mock_retriever.get_relevant_documents.return_value = self.mock_docs
        
        self.agent = RetrievalAgent(
            name="TestRetriever",
            vector_store=self.mock_vector_store
        )
    
    def test_retrieval_agent_initialization(self):
        """Test retrieval agent initialization."""
        self.assertEqual(self.agent.name, "TestRetriever")
        self.assertEqual(self.agent.retrieval_strategy, "similarity")
    
    def test_perceive(self):
        """Test perception of query."""
        input_data = {"query": "test query", "k": 5}
        perception = self.agent.perceive(input_data)
        
        self.assertEqual(perception["query"], "test query")
        self.assertEqual(perception["k"], 5)
        self.assertIn("query_length", perception)
    
    def test_reason(self):
        """Test reasoning about retrieval."""
        perception = {"query": "short query", "query_length": 2, "k": 5}
        reasoning = self.agent.reason(perception)
        
        self.assertIn("retrieval_method", reasoning)
        self.assertIn("optimal_k", reasoning)
        self.assertEqual(reasoning["query_complexity"], "low")
    
    def test_retrieval_run(self):
        """Test full retrieval cycle."""
        result = self.agent.run({"query": "test query", "k": 3})
        
        self.assertIn("documents", result)
        self.assertEqual(result["count"], 3)


class TestReasoningAgent(unittest.TestCase):
    """Test the ReasoningAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.invoke.return_value = Mock(content="Test answer")
        
        self.agent = ReasoningAgent(
            name="TestReasoner",
            model=self.mock_model
        )
    
    def test_reasoning_agent_initialization(self):
        """Test reasoning agent initialization."""
        self.assertEqual(self.agent.name, "TestReasoner")
        self.assertEqual(self.agent.temperature, 0.7)
    
    def test_query_classification(self):
        """Test query type classification."""
        test_cases = [
            ("What is machine learning?", "factual"),
            ("Compare X and Y", "comparison"),
            ("Why does this happen?", "analysis"),
            ("Calculate the sum", "calculation")
        ]
        
        for query, expected_type in test_cases:
            query_type = self.agent._classify_query(query)
            self.assertEqual(query_type, expected_type)
    
    def test_perceive(self):
        """Test perception of query and documents."""
        input_data = {
            "query": "What is AI?",
            "documents": [Mock(), Mock()],
            "chat_history": []
        }
        
        perception = self.agent.perceive(input_data)
        
        self.assertEqual(perception["document_count"], 2)
        self.assertEqual(perception["query_type"], "factual")
    
    def test_build_context(self):
        """Test context building from documents."""
        docs = [
            Mock(page_content="Content 1"),
            Mock(page_content="Content 2")
        ]
        
        context = self.agent._build_context(docs)
        
        self.assertIn("Document 1", context)
        self.assertIn("Content 1", context)


class TestAgentOrchestrator(unittest.TestCase):
    """Test the AgentOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_retriever = Mock()
        self.mock_vector_store.as_retriever.return_value = self.mock_retriever
        
        mock_docs = [Mock(page_content="Test content")]
        self.mock_retriever.get_relevant_documents.return_value = mock_docs
        
        self.mock_model = Mock()
        self.mock_model.invoke.return_value = Mock(content="Test answer")
        
        self.retrieval_agent = RetrievalAgent(
            name="Retriever",
            vector_store=self.mock_vector_store
        )
        
        self.reasoning_agent = ReasoningAgent(
            name="Reasoner",
            model=self.mock_model
        )
        
        self.orchestrator = AgentOrchestrator(
            retrieval_agent=self.retrieval_agent,
            reasoning_agent=self.reasoning_agent
        )
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIn("retrieval", self.orchestrator.agents)
        self.assertIn("reasoning", self.orchestrator.agents)
    
    def test_sequential_processing(self):
        """Test sequential query processing."""
        result = self.orchestrator.process_query(
            query="Test query",
            strategy="sequential"
        )
        
        self.assertIn("answer", result)
        self.assertIn("agents_used", result)
        self.assertEqual(result["orchestration"], "sequential")
    
    def test_adaptive_processing(self):
        """Test adaptive query processing."""
        result = self.orchestrator.process_query(
            query="This is a complex query with many words",
            strategy="adaptive"
        )
        
        self.assertIn("answer", result)
        self.assertEqual(result["orchestration"], "adaptive")
    
    def test_agent_stats(self):
        """Test agent statistics."""
        # Run a few queries
        for i in range(3):
            self.orchestrator.process_query(f"Query {i}")
        
        stats = self.orchestrator.get_agent_stats()
        
        self.assertEqual(stats["total_executions"], 3)
        self.assertIn("agents", stats)


class TestToolRegistry(unittest.TestCase):
    """Test the ToolRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
    
    def test_default_tools_registered(self):
        """Test that default tools are registered."""
        tools = self.registry.list_tools()
        self.assertGreater(len(tools), 0)
        
        tool_names = [t["name"] for t in tools]
        self.assertIn("word_count", tool_names)
        self.assertIn("extract_keywords", tool_names)
    
    def test_word_count_tool(self):
        """Test word count tool."""
        tool = self.registry.get_tool("word_count")
        result = tool("This is a test sentence")
        self.assertEqual(result, 5)
    
    def test_calculate_tool(self):
        """Test calculation tool."""
        tool = self.registry.get_tool("calculate")
        result = tool("2 + 2")
        self.assertEqual(result, 4.0)
    
    def test_classify_query_intent_tool(self):
        """Test query intent classification tool."""
        tool = self.registry.get_tool("classify_query_intent")
        
        test_cases = [
            ("What is AI?", "informational"),
            ("How to do X?", "instructional"),
            ("Compare A and B", "comparative")
        ]
        
        for query, expected in test_cases:
            result = tool(query)
            self.assertEqual(result, expected)
    
    def test_custom_tool_registration(self):
        """Test registering custom tools."""
        def custom_tool(x):
            return x.upper()
        
        self.registry.register_tool("uppercase", custom_tool, "Convert to uppercase")
        
        tool = self.registry.get_tool("uppercase")
        result = tool("hello")
        self.assertEqual(result, "HELLO")


class TestAgenticRAG(unittest.TestCase):
    """Test the AgenticRAG integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_retriever = Mock()
        self.mock_vector_store.as_retriever.return_value = self.mock_retriever
        
        mock_docs = [Mock(page_content="Test content")]
        self.mock_retriever.get_relevant_documents.return_value = mock_docs
        
        self.mock_model = Mock()
        self.mock_model.invoke.return_value = Mock(content="Test answer")
    
    def test_agentic_rag_creation_enabled(self):
        """Test creating AgenticRAG with agents enabled."""
        rag = create_agentic_rag(
            vector_store=self.mock_vector_store,
            model=self.mock_model,
            enable_agents=True
        )
        
        self.assertTrue(rag.enable_agents)
        self.assertIsNotNone(rag.orchestrator)
    
    def test_agentic_rag_creation_disabled(self):
        """Test creating AgenticRAG with agents disabled."""
        rag = create_agentic_rag(
            vector_store=self.mock_vector_store,
            model=self.mock_model,
            enable_agents=False
        )
        
        self.assertFalse(rag.enable_agents)
        self.assertIsNone(rag.orchestrator)
    
    def test_query_with_agents(self):
        """Test querying with agents enabled."""
        rag = create_agentic_rag(
            vector_store=self.mock_vector_store,
            model=self.mock_model,
            enable_agents=True
        )
        
        result = rag.query("Test query")
        
        self.assertIn("answer", result)
    
    def test_get_stats(self):
        """Test getting system statistics."""
        rag = create_agentic_rag(
            vector_store=self.mock_vector_store,
            model=self.mock_model,
            enable_agents=True
        )
        
        stats = rag.get_stats()
        
        self.assertEqual(stats["mode"], "agentic")
        self.assertTrue(stats["agents_enabled"])


if __name__ == "__main__":
    unittest.main()
