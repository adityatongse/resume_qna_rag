"""CV QA Agent using LangChain"""
import logging
from typing import Optional, List, Dict, Any
from openai import AuthenticationError, PermissionDeniedError, RateLimitError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


logger = logging.getLogger(__name__)


class CVQAAgent:
    """CV Question Answering Agent"""
    
    def __init__(
        self,
        llm,
        retrieval_tool,
        grounding_validator=None,
        max_iterations: int = 5
    ):
        """
        Initialize the CV QA Agent
        
        Args:
            llm: Language model instance
            retrieval_tool: Tool for retrieving CV content
            grounding_validator: Optional validator for grounding
            max_iterations: Maximum agent iterations
        """
        self.llm = llm
        self.retrieval_tool = retrieval_tool
        self.grounding_validator = grounding_validator
        self.max_iterations = max_iterations
        
        # Create system prompt
        self.system_prompt = self._create_system_prompt()
        
        logger.info("Initialized CVQAAgent")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent"""
        return """You are a helpful AI assistant that answers questions about a person's CV (resume).

IMPORTANT RULES:
1. You MUST first retrieve relevant information from the CV using the available context
2. Base your answers ONLY on the retrieved CV content provided to you
3. If the retrieved content doesn't contain the answer, clearly state: "This information is not available in the provided CV."
4. Never make up or infer information that isn't explicitly in the CV
5. When answering, reference that the information comes from the CV (e.g., "Based on your CV..." or "According to the CV...")
6. Be concise and direct in your answers
7. For follow-up questions, use the conversation history for context

Remember: Accuracy and grounding in the CV content are more important than providing a complete answer."""
    
    def _format_user_friendly_error(self, error: Exception) -> str:
        """Map provider errors to user-friendly messages."""
        error_text = str(error).lower()

        if isinstance(error, PermissionDeniedError):
            if "key limit exceeded" in error_text or "total limit" in error_text:
                return (
                    "The configured OpenRouter API key has exceeded its usage limit. "
                    "Update your OpenRouter key or billing settings, then try again."
                )
            return (
                "The configured API key does not have permission to access the selected model. "
                "Verify the provider key and model configuration, then try again."
            )

        if isinstance(error, AuthenticationError):
            return (
                "The configured API key is invalid or missing. "
                "Update your provider credentials in the environment configuration and try again."
            )

        if isinstance(error, RateLimitError):
            return (
                "The language model provider rate limit has been reached. "
                "Wait a moment and try again."
            )

        if "key limit exceeded" in error_text or "total limit" in error_text:
            return (
                "The configured OpenRouter API key has exceeded its usage limit. "
                "Update your OpenRouter key or billing settings, then try again."
            )

        if "rate limit" in error_text or "too many requests" in error_text:
            return (
                "The language model provider rate limit has been reached. "
                "Wait a moment and try again."
            )

        if "authentication" in error_text or "invalid api key" in error_text:
            return (
                "The configured API key is invalid or missing. "
                "Update your provider credentials in the environment configuration and try again."
            )

        return (
            "I encountered a language model provider error while processing your question. "
            "Check the configured provider credentials, model access, and billing limits."
        )

    def query(
        self,
        question: str,
        chat_history: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question
        
        Args:
            question: User's question
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing query: '{question[:50]}...'")
            
            # Step 1: Retrieve relevant context from CV
            logger.info("Retrieving CV content...")
            retrieved_context = self.retrieval_tool.func(question)
            
            logger.info(f"Retrieved context length: {len(retrieved_context)} characters")
            
            # Step 2: Build messages for LLM
            messages: List[BaseMessage] = [
                SystemMessage(content=self.system_prompt)
            ]
            
            # Add chat history if available
            if chat_history:
                for role, content in chat_history[-6:]:  # Last 3 exchanges
                    if role == "human":
                        messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        messages.append(AIMessage(content=content))
            
            # Add current question with retrieved context
            user_message = f"""Question: {question}

Retrieved CV Content:
{retrieved_context}

Please answer the question based ONLY on the retrieved CV content above. If the information is not in the retrieved content, say "This information is not available in the provided CV." """
            
            messages.append(HumanMessage(content=user_message))
            
            # Step 3: Get LLM response
            logger.info("Generating answer with LLM...")
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info(f"Generated answer length: {len(answer)} characters")
            
            # Step 4: Validate grounding if validator is provided
            is_grounded = True
            grounding_message = ""
            
            if self.grounding_validator and retrieved_context:
                is_grounded, grounding_message = self.grounding_validator.validate(
                    answer, retrieved_context
                )
                
                if not is_grounded:
                    logger.warning(f"Answer failed grounding validation: {grounding_message}")
                    # Override answer if not grounded
                    answer = "This information is not available in the provided CV."
            
            logger.info(f"Query processed successfully. Grounded: {is_grounded}")
            
            return {
                "answer": answer,
                "retrieved_context": retrieved_context,
                "is_grounded": is_grounded,
                "grounding_message": grounding_message,
                "intermediate_steps": [
                    ("retrieve", retrieved_context),
                    ("generate", answer)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            user_friendly_error = self._format_user_friendly_error(e)
            return {
                "answer": user_friendly_error,
                "retrieved_context": "",
                "is_grounded": False,
                "grounding_message": f"Error: {str(e)}",
                "intermediate_steps": []
            }
    
    def stream_query(self, question: str, chat_history: Optional[List] = None):
        """
        Stream the agent's response (for future implementation)
        
        Args:
            question: User's question
            chat_history: Optional conversation history
        """
        # Placeholder for streaming implementation
        result = self.query(question, chat_history)
        yield result["answer"]

# Made with Bob
