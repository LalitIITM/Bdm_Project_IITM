#!/usr/bin/env python3
"""
Quick Test Script for Agentic RAG System
Run this to verify the system is working correctly.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Checking Imports")
    print("=" * 60)
    try:
        from app.agents.agentic_rag import create_agentic_rag
        from app.agents.base_agent import BaseAgent
        from app.agents.retrieval_agent import RetrievalAgent
        from app.agents.reasoning_agent import ReasoningAgent
        from app.agents.orchestrator import AgentOrchestrator
        from app.agents.tools import tool_registry
        
        print("✓ All agent modules imported successfully")
        print(f"✓ {len(tool_registry.list_tools())} tools registered")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tools():
    """Test the tool registry."""
    print("\n" + "=" * 60)
    print("TEST 2: Checking Tools")
    print("=" * 60)
    try:
        from app.agents.tools import tool_registry
        
        # List tools
        tools = tool_registry.list_tools()
        print(f"✓ {len(tools)} tools available:")
        for tool in tools:
            print(f"  - {tool['name']}")
        
        # Test word count
        wc_tool = tool_registry.get_tool("word_count")
        result = wc_tool("This is a test")
        print(f"✓ word_count tool: {result} words")
        
        # Test keyword extraction
        kw_tool = tool_registry.get_tool("extract_keywords")
        keywords = kw_tool("Machine learning is artificial intelligence")
        print(f"✓ extract_keywords tool: {keywords}")
        
        return True
    except Exception as e:
        print(f"✗ Tool test failed: {e}")
        return False


def test_agents_with_mocks():
    """Test agents with mock data."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing Agents with Mock Data")
    print("=" * 60)
    try:
        from unittest.mock import Mock
        from app.agents.retrieval_agent import RetrievalAgent
        from app.agents.reasoning_agent import ReasoningAgent
        from app.agents.orchestrator import AgentOrchestrator
        
        # Create mock vector store
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_doc = Mock(page_content="Machine learning is a subset of artificial intelligence.")
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        
        # Create mock model
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content="Machine learning is a method of data analysis that automates analytical model building."
        )
        
        # Test Retrieval Agent
        print("Testing Retrieval Agent...")
        retrieval_agent = RetrievalAgent("TestRetriever", mock_vector_store)
        retrieval_result = retrieval_agent.run({"query": "What is machine learning?", "k": 5})
        print(f"✓ Retrieval Agent: Retrieved {retrieval_result['count']} documents")
        
        # Test Reasoning Agent
        print("Testing Reasoning Agent...")
        reasoning_agent = ReasoningAgent("TestReasoner", mock_model)
        reasoning_result = reasoning_agent.run({
            "query": "What is machine learning?",
            "documents": [mock_doc],
            "chat_history": []
        })
        print(f"✓ Reasoning Agent: Generated answer ({len(reasoning_result['answer'])} chars)")
        
        # Test Orchestrator
        print("Testing Agent Orchestrator...")
        orchestrator = AgentOrchestrator(retrieval_agent, reasoning_agent)
        result = orchestrator.process_query("What is artificial intelligence?")
        print(f"✓ Orchestrator: Processed query successfully")
        print(f"  - Agents used: {result['agents_used']}")
        print(f"  - Orchestration: {result['orchestration']}")
        
        # Get statistics
        stats = orchestrator.get_agent_stats()
        print(f"✓ Statistics: {stats['total_executions']} executions, {stats['success_rate']:.0%} success rate")
        
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agentic_rag():
    """Test the AgenticRAG integration."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing AgenticRAG Integration")
    print("=" * 60)
    try:
        from unittest.mock import Mock
        from app.agents.agentic_rag import create_agentic_rag
        
        # Create mock components
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_doc = Mock(page_content="Artificial intelligence enables machines to learn.")
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content="AI is the simulation of human intelligence processes by machines."
        )
        
        # Create agentic RAG (enabled)
        print("Testing with agents ENABLED...")
        agentic_rag = create_agentic_rag(mock_vector_store, mock_model, enable_agents=True)
        result = agentic_rag.query("What is AI?")
        print(f"✓ AgenticRAG (enabled): Answer generated ({len(result['answer'])} chars)")
        print(f"  - Agents used: {result.get('agents_used', [])}")
        
        # Get stats
        stats = agentic_rag.get_stats()
        print(f"✓ System stats: Mode={stats['mode']}, Agents={stats['agents_enabled']}")
        
        # Test with agents disabled
        print("Testing with agents DISABLED...")
        simple_rag = create_agentic_rag(mock_vector_store, mock_model, enable_agents=False)
        result = simple_rag.query("What is AI?")
        print(f"✓ AgenticRAG (disabled): Answer generated ({len(result['answer'])} chars)")
        
        return True
    except Exception as e:
        print(f"✗ AgenticRAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_unit_tests():
    """Run the unit test suite."""
    print("\n" + "=" * 60)
    print("TEST 5: Running Unit Tests")
    print("=" * 60)
    try:
        import unittest
        from tests.test_agents import (
            TestBaseAgent, TestRetrievalAgent, TestReasoningAgent,
            TestAgentOrchestrator, TestToolRegistry, TestAgenticRAG
        )
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        suite.addTests(loader.loadTestsFromTestCase(TestBaseAgent))
        suite.addTests(loader.loadTestsFromTestCase(TestRetrievalAgent))
        suite.addTests(loader.loadTestsFromTestCase(TestReasoningAgent))
        suite.addTests(loader.loadTestsFromTestCase(TestAgentOrchestrator))
        suite.addTests(loader.loadTestsFromTestCase(TestToolRegistry))
        suite.addTests(loader.loadTestsFromTestCase(TestAgenticRAG))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print(f"✓ All {result.testsRun} unit tests passed!")
            return True
        else:
            print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        print(f"✗ Unit test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🤖 AGENTIC RAG SYSTEM TEST SUITE 🤖".center(60))
    print()
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Tools", test_tools()))
    results.append(("Agents", test_agents_with_mocks()))
    results.append(("AgenticRAG", test_agentic_rag()))
    results.append(("Unit Tests", run_unit_tests()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use. 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
