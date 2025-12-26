"""
Agent Orchestrator
Coordinates multiple agents to work together in solving complex tasks.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
import logging

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple agents to collaboratively solve complex queries.
    Manages agent communication and task delegation.
    """
    
    def __init__(self, retrieval_agent: RetrievalAgent, reasoning_agent: ReasoningAgent):
        """
        Initialize the agent orchestrator.
        
        Args:
            retrieval_agent: Agent for document retrieval
            reasoning_agent: Agent for reasoning and answer generation
        """
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.agents = {
            "retrieval": retrieval_agent,
            "reasoning": reasoning_agent
        }
        self.execution_history = []
        logger.info("Agent orchestrator initialized")
    
    def process_query(self, query: str, chat_history: Optional[List] = None, 
                     strategy: str = "sequential") -> Dict[str, Any]:
        """
        Process a query using coordinated agents.
        
        Args:
            query: User query
            chat_history: Optional conversation history
            strategy: Orchestration strategy (sequential, parallel, adaptive)
            
        Returns:
            Final response with metadata
        """
        logger.info(f"Processing query with {strategy} strategy: {query[:50]}...")
        
        if chat_history is None:
            chat_history = []
        
        try:
            if strategy == "sequential":
                result = self._sequential_processing(query, chat_history)
            elif strategy == "parallel":
                result = self._parallel_processing(query, chat_history)
            else:  # adaptive
                result = self._adaptive_processing(query, chat_history)
            
            # Record execution
            self._record_execution(query, result, strategy)
            
            logger.info("Query processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "error": str(e),
                "agents_used": []
            }
    
    def _sequential_processing(self, query: str, chat_history: List) -> Dict[str, Any]:
        """
        Process query sequentially through agents.
        
        Args:
            query: User query
            chat_history: Conversation history
            
        Returns:
            Processing result
        """
        # Step 1: Retrieval
        retrieval_input = {
            "query": query,
            "k": 5
        }
        retrieval_result = self.retrieval_agent.run(retrieval_input)
        
        # Step 2: Reasoning
        reasoning_input = {
            "query": query,
            "documents": retrieval_result.get("documents", []),
            "chat_history": chat_history
        }
        reasoning_result = self.reasoning_agent.run(reasoning_input)
        
        return {
            "answer": reasoning_result.get("answer", ""),
            "agents_used": ["retrieval", "reasoning"],
            "retrieval_count": retrieval_result.get("count", 0),
            "reasoning_strategy": reasoning_result.get("strategy_used", ""),
            "orchestration": "sequential"
        }
    
    def _parallel_processing(self, query: str, chat_history: List) -> Dict[str, Any]:
        """
        Process query with parallel agent operations where possible.
        
        Args:
            query: User query
            chat_history: Conversation history
            
        Returns:
            Processing result
        """
        # For now, retrieval must come first, so this is similar to sequential
        # But could be extended for multiple retrievals or analyses in parallel
        return self._sequential_processing(query, chat_history)
    
    def _adaptive_processing(self, query: str, chat_history: List) -> Dict[str, Any]:
        """
        Adaptively choose processing strategy based on query characteristics.
        
        Args:
            query: User query
            chat_history: Conversation history
            
        Returns:
            Processing result
        """
        # Analyze query complexity
        query_length = len(query.split())
        has_history = len(chat_history) > 0
        
        # Choose strategy based on characteristics
        if query_length > 20 or has_history:
            # Complex query - use enhanced retrieval
            retrieval_input = {
                "query": query,
                "k": 7  # Get more documents for complex queries
            }
        else:
            # Simple query - standard retrieval
            retrieval_input = {
                "query": query,
                "k": 4
            }
        
        retrieval_result = self.retrieval_agent.run(retrieval_input)
        
        # Reasoning with adaptive parameters
        reasoning_input = {
            "query": query,
            "documents": retrieval_result.get("documents", []),
            "chat_history": chat_history
        }
        reasoning_result = self.reasoning_agent.run(reasoning_input)
        
        return {
            "answer": reasoning_result.get("answer", ""),
            "agents_used": ["retrieval", "reasoning"],
            "retrieval_count": retrieval_result.get("count", 0),
            "reasoning_strategy": reasoning_result.get("strategy_used", ""),
            "orchestration": "adaptive",
            "query_complexity": "high" if query_length > 20 else "low"
        }
    
    def _record_execution(self, query: str, result: Dict[str, Any], strategy: str):
        """
        Record execution for monitoring and improvement.
        
        Args:
            query: Processed query
            result: Execution result
            strategy: Strategy used
        """
        execution_record = {
            "query": query[:100],  # Truncate for storage
            "strategy": strategy,
            "agents_used": result.get("agents_used", []),
            "success": "error" not in result
        }
        self.execution_history.append(execution_record)
        
        # Keep history manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get statistics about agent usage and performance.
        
        Returns:
            Agent statistics
        """
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "agents": {}
            }
        
        successful = sum(1 for e in self.execution_history if e["success"])
        
        agent_usage = {}
        for agent_name in self.agents.keys():
            usage_count = sum(
                1 for e in self.execution_history 
                if agent_name in e.get("agents_used", [])
            )
            agent_usage[agent_name] = usage_count
        
        return {
            "total_executions": total_executions,
            "success_rate": successful / total_executions,
            "agents": agent_usage,
            "retrieval_agent_memory": len(self.retrieval_agent.get_memory()),
            "reasoning_agent_memory": len(self.reasoning_agent.get_memory())
        }
    
    def reset_agents(self):
        """Reset all agents and clear execution history."""
        for agent in self.agents.values():
            agent.clear_memory()
        self.execution_history = []
        logger.info("All agents reset")
