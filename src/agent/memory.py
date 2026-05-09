"""Conversation memory management"""
import logging
from typing import List, Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum number of exchanges to keep
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        
        logger.info(f"Initialized conversation memory with max_history={max_history}")
    
    def add_exchange(
        self,
        question: str,
        answer: str,
        retrieved_context: str = "",
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a question-answer exchange to memory
        
        Args:
            question: User's question
            answer: Agent's answer
            retrieved_context: Context retrieved from CV
            metadata: Additional metadata
        """
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "retrieved_context": retrieved_context,
            "metadata": metadata or {}
        }
        
        self.history.append(exchange)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            removed = self.history.pop(0)
            logger.debug(f"Removed oldest exchange from memory: {removed['question'][:50]}...")
        
        logger.info(f"Added exchange to memory. Total exchanges: {len(self.history)}")
    
    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation history
        
        Args:
            n: Number of recent exchanges to retrieve
            
        Returns:
            List of recent exchanges
        """
        return self.history[-n:] if self.history else []
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete conversation history"""
        return self.history.copy()
    
    def format_history_for_context(self, n: int = 3) -> str:
        """
        Format recent history as context string for the LLM
        
        Args:
            n: Number of recent exchanges to include
            
        Returns:
            Formatted history string
        """
        recent = self.get_recent_history(n)
        
        if not recent:
            return "No previous conversation history."
        
        formatted = ["Previous conversation:"]
        for i, exchange in enumerate(recent, 1):
            formatted.append(f"\nQ{i}: {exchange['question']}")
            formatted.append(f"A{i}: {exchange['answer']}")
        
        return "\n".join(formatted)
    
    def clear(self) -> None:
        """Clear all conversation history"""
        self.history.clear()
        self.session_start = datetime.now()
        logger.info("Cleared conversation memory")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_exchanges": len(self.history),
            "session_start": self.session_start.isoformat(),
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60
        }
    
    def export_to_file(self, filepath: str) -> None:
        """
        Export conversation history to a file
        
        Args:
            filepath: Path to output file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"CV Question Answering Session\n")
                f.write(f"Session Start: {self.session_start.isoformat()}\n")
                f.write(f"Total Exchanges: {len(self.history)}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, exchange in enumerate(self.history, 1):
                    f.write(f"Exchange {i}\n")
                    f.write(f"Timestamp: {exchange['timestamp']}\n")
                    f.write(f"Question: {exchange['question']}\n")
                    f.write(f"Answer: {exchange['answer']}\n")
                    
                    if exchange.get('retrieved_context'):
                        f.write(f"\nRetrieved Context:\n{exchange['retrieved_context']}\n")
                    
                    f.write("\n" + "-" * 80 + "\n\n")
            
            logger.info(f"Exported conversation history to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting conversation history: {str(e)}")
            raise

