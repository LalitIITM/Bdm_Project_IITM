"""
Retrieval Agent
Specialized agent for document retrieval and context extraction.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """
    Agent specialized in retrieving relevant documents and context.
    Uses vector store and advanced retrieval strategies.
    """
    
    def __init__(self, name: str, vector_store, model=None, retrieval_strategy: str = "similarity"):
        """
        Initialize the retrieval agent.
        
        Args:
            name: Agent name
            vector_store: Vector store for document retrieval
            model: Language model for agent reasoning
            retrieval_strategy: Strategy for retrieval (similarity, mmr, etc.)
        """
        super().__init__(
            name=name,
            description="Retrieves relevant documents and context for queries",
            model=model
        )
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        logger.info(f"Retrieval agent initialized with strategy: {retrieval_strategy}")
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive the query and extract retrieval parameters.
        
        Args:
            input_data: Contains query and optional filters
            
        Returns:
            Processed perception with query analysis
        """
        query = input_data.get("query", "")
        filters = input_data.get("filters", {})
        k = input_data.get("k", 5)
        
        perception = {
            "query": query,
            "filters": filters,
            "k": k,
            "query_length": len(query.split()),
            "has_filters": bool(filters)
        }
        
        logger.info(f"Retrieval agent perceived query with {perception['query_length']} words")
        return perception
    
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason about the best retrieval approach.
        
        Args:
            perception: Processed query perception
            
        Returns:
            Retrieval plan and strategy
        """
        query = perception["query"]
        k = perception["k"]
        
        # Determine if query needs decomposition
        needs_decomposition = perception["query_length"] > 15
        
        # Determine optimal k value
        optimal_k = min(k * 2, 10) if needs_decomposition else k
        
        reasoning = {
            "retrieval_method": self.retrieval_strategy,
            "needs_decomposition": needs_decomposition,
            "optimal_k": optimal_k,
            "query_complexity": "high" if needs_decomposition else "low"
        }
        
        logger.info(f"Retrieval reasoning: {reasoning}")
        return reasoning
    
    def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document retrieval.
        
        Args:
            reasoning: Retrieval plan
            
        Returns:
            Retrieved documents and metadata
        """
        try:
            # Get retriever with appropriate parameters
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": reasoning["optimal_k"]}
            )
            
            # Perform retrieval
            query = self.memory[-2]["data"]["query"]  # Get from perception memory
            documents = retriever.get_relevant_documents(query)
            
            result = {
                "documents": documents,
                "count": len(documents),
                "strategy": reasoning["retrieval_method"],
                "query_complexity": reasoning["query_complexity"]
            }
            
            logger.info(f"Retrieved {result['count']} documents")
            return result
            
        except Exception as e:
            logger.error(f"Retrieval action failed: {e}")
            return {
                "documents": [],
                "count": 0,
                "error": str(e)
            }
    
    def retrieve_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        except Exception as e:
            logger.error(f"Scored retrieval failed: {e}")
            return []
    
    def retrieve_diverse(self, query: str, k: int = 5, lambda_mult: float = 0.5) -> List:
        """
        Retrieve diverse documents using MMR.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse documents
        """
        try:
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": lambda_mult}
            )
            documents = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(documents)} diverse documents")
            return documents
        except Exception as e:
            logger.error(f"Diverse retrieval failed: {e}")
            return []
