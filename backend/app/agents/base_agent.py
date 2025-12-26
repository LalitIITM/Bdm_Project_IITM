"""
Base Agent Class
Provides the foundation for all agents in the agentic RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Agents can perceive, reason, and act within the RAG system.
    """
    
    def __init__(self, name: str, description: str, model=None):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            description: Agent description and purpose
            model: Language model for agent reasoning
        """
        self.name = name
        self.description = description
        self.model = model
        self.memory = []
        self.tools = {}
        logger.info(f"Initialized agent: {name}")
    
    @abstractmethod
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive and process input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Processed perception data
        """
        pass
    
    @abstractmethod
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason about the perceived data and plan actions.
        
        Args:
            perception: Processed perception data
            
        Returns:
            Reasoning results and action plan
        """
        pass
    
    @abstractmethod
    def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on reasoning.
        
        Args:
            reasoning: Reasoning results and action plan
            
        Returns:
            Action results
        """
        pass
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent through perceive-reason-act cycle.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Final agent output
        """
        logger.info(f"Agent {self.name} starting run cycle")
        
        # Perceive
        perception = self.perceive(input_data)
        self._add_to_memory("perception", perception)
        
        # Reason
        reasoning = self.reason(perception)
        self._add_to_memory("reasoning", reasoning)
        
        # Act
        result = self.act(reasoning)
        self._add_to_memory("action", result)
        
        logger.info(f"Agent {self.name} completed run cycle")
        return result
    
    def register_tool(self, tool_name: str, tool_func):
        """
        Register a tool that the agent can use.
        
        Args:
            tool_name: Name of the tool
            tool_func: Function implementing the tool
        """
        self.tools[tool_name] = tool_func
        logger.info(f"Agent {self.name} registered tool: {tool_name}")
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Use a registered tool.
        
        Args:
            tool_name: Name of the tool to use
            *args, **kwargs: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered for agent {self.name}")
        
        logger.info(f"Agent {self.name} using tool: {tool_name}")
        return self.tools[tool_name](*args, **kwargs)
    
    def _add_to_memory(self, event_type: str, data: Dict[str, Any]):
        """
        Add an event to agent memory.
        
        Args:
            event_type: Type of event (perception, reasoning, action)
            data: Event data
        """
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.memory.append(memory_entry)
        
        # Keep memory size manageable
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def get_memory(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve agent memory.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of memory entries
        """
        if limit:
            return self.memory[-limit:]
        return self.memory
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory = []
        logger.info(f"Agent {self.name} memory cleared")
