"""
Reasoning Agent
Specialized agent for multi-step reasoning and query analysis.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import logging
import re

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Agent specialized in reasoning, query decomposition, and answer synthesis.
    Performs multi-step reasoning to improve response quality.
    """
    
    def __init__(self, name: str, model, temperature: float = 0.7):
        """
        Initialize the reasoning agent.
        
        Args:
            name: Agent name
            model: Language model for reasoning
            temperature: Temperature for model inference
        """
        super().__init__(
            name=name,
            description="Performs multi-step reasoning and query analysis",
            model=model
        )
        self.temperature = temperature
        logger.info(f"Reasoning agent initialized with temperature: {temperature}")
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive the query and retrieved documents.
        
        Args:
            input_data: Contains query, documents, and chat history
            
        Returns:
            Processed perception with query analysis
        """
        query = input_data.get("query", "")
        documents = input_data.get("documents", [])
        chat_history = input_data.get("chat_history", [])
        
        perception = {
            "query": query,
            "documents": documents,
            "document_count": len(documents),
            "chat_history": chat_history,
            "query_type": self._classify_query(query),
            "needs_context": len(chat_history) > 0
        }
        
        logger.info(f"Reasoning agent perceived {perception['query_type']} query")
        return perception
    
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason about the query and plan response strategy.
        
        Args:
            perception: Processed query and document perception
            
        Returns:
            Reasoning plan and strategy
        """
        query_type = perception["query_type"]
        document_count = perception["document_count"]
        needs_context = perception["needs_context"]
        
        # Determine reasoning strategy
        if query_type in ["comparison", "analysis"]:
            strategy = "multi_step"
            steps = self._decompose_complex_query(perception["query"])
        elif query_type == "factual":
            strategy = "direct"
            steps = ["extract_answer"]
        else:
            strategy = "synthesis"
            steps = ["gather_info", "synthesize"]
        
        reasoning = {
            "strategy": strategy,
            "steps": steps,
            "use_chat_context": needs_context,
            "confidence_threshold": 0.7 if document_count > 3 else 0.5,
            "requires_verification": query_type in ["comparison", "calculation"]
        }
        
        logger.info(f"Reasoning strategy: {reasoning['strategy']} with {len(steps)} steps")
        return reasoning
    
    def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response based on reasoning strategy.
        
        Args:
            reasoning: Reasoning plan and strategy
            
        Returns:
            Generated response and metadata
        """
        try:
            perception = self.memory[-2]["data"]  # Get from perception memory
            query = perception["query"]
            documents = perception["documents"]
            chat_history = perception["chat_history"]
            
            # Build context from documents
            context = self._build_context(documents)
            
            # Generate response based on strategy
            if reasoning["strategy"] == "multi_step":
                response = self._multi_step_reasoning(query, context, reasoning["steps"], chat_history)
            elif reasoning["strategy"] == "direct":
                response = self._direct_answer(query, context, chat_history)
            else:
                response = self._synthesize_answer(query, context, chat_history)
            
            result = {
                "answer": response,
                "strategy_used": reasoning["strategy"],
                "steps_executed": reasoning["steps"],
                "context_length": len(context)
            }
            
            logger.info(f"Generated response using {reasoning['strategy']} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Reasoning action failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "error": str(e)
            }
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query.
        
        Args:
            query: User query
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        comparison_keywords = ["compare", "difference", "versus", "vs", "better", "worse"]
        if any(keyword in query_lower for keyword in comparison_keywords):
            return "comparison"
        
        analysis_keywords = ["analyze", "explain", "why", "how"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            return "analysis"
        
        factual_keywords = ["what", "when", "where", "who", "which"]
        if any(query_lower.startswith(keyword) for keyword in factual_keywords):
            return "factual"
        
        calculation_keywords = ["calculate", "compute", "how many", "how much"]
        if any(keyword in query_lower for keyword in calculation_keywords):
            return "calculation"
        
        return "general"
    
    def _decompose_complex_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-questions.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-questions
        """
        # Simple decomposition - can be enhanced with LLM
        steps = [
            "identify_key_concepts",
            "gather_relevant_info",
            "compare_and_analyze",
            "synthesize_conclusion"
        ]
        return steps
    
    def _build_context(self, documents: List) -> str:
        """
        Build context string from documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Combined context string
        """
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            context_parts.append(f"Document {i+1}: {content[:500]}")
        
        return "\n\n".join(context_parts)
    
    def _multi_step_reasoning(self, query: str, context: str, steps: List[str], chat_history: List) -> str:
        """
        Perform multi-step reasoning to answer complex queries.
        
        Args:
            query: User query
            context: Retrieved context
            steps: Reasoning steps
            chat_history: Conversation history
            
        Returns:
            Generated answer
        """
        if not self.model:
            return self._synthesize_answer(query, context, chat_history)
        
        prompt = f"""You are a helpful assistant performing multi-step reasoning.

Query: {query}

Context: {context}

Reasoning Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}

Please analyze the query step by step and provide a comprehensive answer based on the context."""
        
        try:
            response = self.model.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Multi-step reasoning failed: {e}")
            return self._synthesize_answer(query, context, chat_history)
    
    def _direct_answer(self, query: str, context: str, chat_history: List) -> str:
        """
        Generate direct answer for factual queries.
        
        Args:
            query: User query
            context: Retrieved context
            chat_history: Conversation history
            
        Returns:
            Direct answer
        """
        if not self.model:
            return f"Based on the available information: {context[:200]}..."
        
        prompt = f"""Provide a direct, concise answer to the following question based on the context.

Question: {query}

Context: {context}

Answer:"""
        
        try:
            response = self.model.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Direct answer generation failed: {e}")
            return f"Based on the context: {context[:200]}..."
    
    def _synthesize_answer(self, query: str, context: str, chat_history: List) -> str:
        """
        Synthesize answer from multiple sources.
        
        Args:
            query: User query
            context: Retrieved context
            chat_history: Conversation history
            
        Returns:
            Synthesized answer
        """
        if not self.model:
            return f"Based on the available information: {context[:300]}..."
        
        chat_context = ""
        if chat_history:
            recent_history = chat_history[-3:]  # Last 3 exchanges
            chat_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent_history])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Previous Conversation:
{chat_context}

Current Question: {query}

Context: {context}

Please provide a comprehensive answer that synthesizes information from the context."""
        
        try:
            response = self.model.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return "I apologize, but I couldn't generate a proper response based on the available context."
