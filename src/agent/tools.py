"""Tools for the CV QA agent"""
import logging
from typing import List, Dict, Any
from langchain_core.tools import Tool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class RetrievalInput(BaseModel):
    """Input schema for retrieval tool"""
    query: str = Field(description="The question or query to search for in the CV")


def create_retrieval_tool(vector_store, top_k: int = 3) -> Tool:
    """
    Create a retrieval tool for the agent
    
    Args:
        vector_store: ChromaStore instance
        top_k: Number of chunks to retrieve
        
    Returns:
        LangChain Tool for CV retrieval
    """
    
    def retrieve_cv_content(query: str) -> str:
        """
        Retrieve relevant CV content based on query
        
        Args:
            query: Search query
            
        Returns:
            Formatted string with retrieved content
        """
        try:
            logger.info(f"Retrieving CV content for query: '{query[:50]}...'")
            
            # Search vector store
            results = vector_store.search(query, top_k=top_k)
            
            if not results:
                logger.warning(f"No results found for query: {query}")
                return "No relevant information found in the CV for this query."
            
            # Format results
            formatted_content = []
            for i, result in enumerate(results, 1):
                text = result['text']
                metadata = result.get('metadata', {})
                distance = result.get('distance', 0.0)
                
                formatted_content.append(
                    f"[Chunk {i}] (Relevance: {1 - distance:.2f})\n{text}\n"
                )
            
            retrieved_text = "\n---\n".join(formatted_content)
            
            logger.info(f"Retrieved {len(results)} chunks for query")
            return retrieved_text
            
        except Exception as e:
            logger.error(f"Error in retrieval tool: {str(e)}")
            return f"Error retrieving CV content: {str(e)}"
    
    # Create the tool
    tool = Tool(
        name="retrieve_cv_content",
        description=(
            "Retrieves relevant information from the CV based on a query. "
            "Use this tool to find specific information about work experience, "
            "education, skills, projects, or any other CV content. "
            "Always use this tool before answering questions about the CV."
        ),
        func=retrieve_cv_content,
        args_schema=RetrievalInput
    )
    
    logger.info("Created retrieval tool")
    return tool


def format_retrieved_context(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved results into a readable context string
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        text = result.get('text', '')
        metadata = result.get('metadata', {})
        
        context_parts.append(f"Context {i}:\n{text}")
    
    return "\n\n".join(context_parts)


