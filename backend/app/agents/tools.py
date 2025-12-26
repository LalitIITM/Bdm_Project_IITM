"""
Tool Registry for Agents
Provides a collection of tools that agents can use to perform actions.
"""

from typing import Dict, Any, Callable, List
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry of tools available to agents.
    Tools are functions that agents can call to perform specific actions.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self._register_default_tools()
        logger.info("Tool registry initialized")
    
    def register_tool(self, name: str, func: Callable, description: str):
        """
        Register a new tool.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        self.tools[name] = func
        self.tool_descriptions[name] = description
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Callable:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool function
        """
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self.tools[name]
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all available tools.
        
        Returns:
            List of tool information
        """
        return [
            {"name": name, "description": desc}
            for name, desc in self.tool_descriptions.items()
        ]
    
    def _register_default_tools(self):
        """Register default tools."""
        
        # Text analysis tools
        self.register_tool(
            "word_count",
            self._word_count,
            "Count words in a text"
        )
        
        self.register_tool(
            "extract_keywords",
            self._extract_keywords,
            "Extract keywords from text"
        )
        
        self.register_tool(
            "summarize_text",
            self._summarize_text,
            "Create a brief summary of text"
        )
        
        # Query analysis tools
        self.register_tool(
            "classify_query_intent",
            self._classify_query_intent,
            "Classify the intent of a user query"
        )
        
        self.register_tool(
            "extract_entities",
            self._extract_entities,
            "Extract named entities from text"
        )
        
        # Utility tools
        self.register_tool(
            "calculate",
            self._calculate,
            "Perform basic calculations"
        )
        
        self.register_tool(
            "format_timestamp",
            self._format_timestamp,
            "Format current timestamp"
        )
    
    def _word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from text (simple implementation).
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'their'}
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def _summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Create a simple summary of text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if len(text) <= max_length:
            return text
        
        # Simple summarization: take first sentences up to max_length
        sentences = re.split(r'[.!?]+', text)
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of a query.
        
        Args:
            query: User query
            
        Returns:
            Query intent classification
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return "informational"
        elif any(word in query_lower for word in ['how', 'tutorial', 'guide']):
            return "instructional"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return "comparative"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return "causal"
        elif any(word in query_lower for word in ['when', 'where', 'who']):
            return "factual"
        else:
            return "general"
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text (simple implementation).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {
            "dates": [],
            "numbers": [],
            "capitalized": []
        }
        
        # Extract dates (simple patterns)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        entities["dates"] = re.findall(date_pattern, text)
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, text)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities["capitalized"] = re.findall(capitalized_pattern, text)
        
        return entities
    
    def _calculate(self, expression: str) -> float:
        """
        Perform basic calculation.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Calculation result
        """
        try:
            # Only allow basic math operations for safety
            allowed_chars = set('0123456789+-*/.()')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            raise ValueError(f"Cannot calculate: {expression}")
    
    def _format_timestamp(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format current timestamp.
        
        Args:
            format_str: Format string
            
        Returns:
            Formatted timestamp
        """
        return datetime.now().strftime(format_str)


# Global tool registry instance
tool_registry = ToolRegistry()
