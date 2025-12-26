"""
Agentic RAG Integration
Integrates the agentic system with the existing RAG infrastructure.
"""

from typing import Dict, Any, List, Optional
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .orchestrator import AgentOrchestrator
from .tools import tool_registry
import logging

logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Main interface for agentic RAG system.
    Provides a unified API for using agents with the RAG pipeline.
    """
    
    def __init__(self, vector_store, model, enable_agents: bool = True):
        """
        Initialize the agentic RAG system.
        
        Args:
            vector_store: Vector store for document retrieval
            model: Language model for reasoning
            enable_agents: Whether to use agentic system or fall back to simple RAG
        """
        self.vector_store = vector_store
        self.model = model
        self.enable_agents = enable_agents
        
        if enable_agents:
            # Initialize agents
            self.retrieval_agent = RetrievalAgent(
                name="DocumentRetriever",
                vector_store=vector_store,
                model=model
            )
            
            self.reasoning_agent = ReasoningAgent(
                name="QueryReasoner",
                model=model
            )
            
            # Register tools with agents
            self._register_agent_tools()
            
            # Create orchestrator
            self.orchestrator = AgentOrchestrator(
                retrieval_agent=self.retrieval_agent,
                reasoning_agent=self.reasoning_agent
            )
            
            logger.info("Agentic RAG system initialized with agents")
        else:
            self.orchestrator = None
            logger.info("Agentic RAG system initialized in simple mode")
    
    def query(self, question: str, chat_history: Optional[List] = None, 
             strategy: str = "adaptive") -> Dict[str, Any]:
        """
        Process a query using the agentic RAG system.
        
        Args:
            question: User question
            chat_history: Optional conversation history
            strategy: Orchestration strategy (sequential, parallel, adaptive)
            
        Returns:
            Response with answer and metadata
        """
        if not self.enable_agents or self.orchestrator is None:
            return self._simple_query(question, chat_history)
        
        try:
            result = self.orchestrator.process_query(
                query=question,
                chat_history=chat_history or [],
                strategy=strategy
            )
            return result
        except Exception as e:
            logger.error(f"Agentic query failed, falling back to simple mode: {e}")
            return self._simple_query(question, chat_history)
    
    def _simple_query(self, question: str, chat_history: Optional[List]) -> Dict[str, Any]:
        """
        Process query using simple RAG without agents.
        
        Args:
            question: User question
            chat_history: Conversation history
            
        Returns:
            Simple response
        """
        try:
            # Simple retrieval
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            documents = retriever.get_relevant_documents(question)
            
            # Simple context building
            context = "\n\n".join([
                doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500]
                for doc in documents[:3]
            ])
            
            # Simple prompt
            prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
            
            response = self.model.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "agents_used": [],
                "retrieval_count": len(documents),
                "orchestration": "simple"
            }
        except Exception as e:
            logger.error(f"Simple query failed: {e}")
            return {
                "answer": "I apologize, but I couldn't process your query.",
                "error": str(e)
            }
    
    def _register_agent_tools(self):
        """Register tools with agents."""
        if not self.enable_agents:
            return
        
        # Register text analysis tools
        for tool_name in ["word_count", "extract_keywords", "classify_query_intent"]:
            tool_func = tool_registry.get_tool(tool_name)
            self.retrieval_agent.register_tool(tool_name, tool_func)
            self.reasoning_agent.register_tool(tool_name, tool_func)
        
        logger.info("Tools registered with agents")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agentic RAG system.
        
        Returns:
            System statistics
        """
        if not self.enable_agents or self.orchestrator is None:
            return {
                "mode": "simple",
                "agents_enabled": False
            }
        
        stats = self.orchestrator.get_agent_stats()
        stats["mode"] = "agentic"
        stats["agents_enabled"] = True
        stats["tools_available"] = len(tool_registry.list_tools())
        
        return stats
    
    def reset(self):
        """Reset the agentic RAG system."""
        if self.enable_agents and self.orchestrator:
            self.orchestrator.reset_agents()
            logger.info("Agentic RAG system reset")


def create_agentic_rag(vector_store, model, enable_agents: bool = True) -> AgenticRAG:
    """
    Factory function to create an AgenticRAG instance.
    
    Args:
        vector_store: Vector store for document retrieval
        model: Language model for reasoning
        enable_agents: Whether to enable agentic features
        
    Returns:
        AgenticRAG instance
    """
    return AgenticRAG(vector_store, model, enable_agents)
