"""Text chunking module for splitting CV content"""
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into chunks for vector storage"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    "text": chunk,
                    "chunk_id": i,
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk),
                        **(metadata or {})
                    }
                }
                chunk_objects.append(chunk_obj)
            
            logger.info(f"Created {len(chunk_objects)} chunks from text")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def chunk_cv_data(self, cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk CV data with file metadata
        
        Args:
            cv_data: Parsed CV data from CVParser
            
        Returns:
            List of chunk dictionaries
        """
        text = cv_data.get("text", "")
        file_metadata = cv_data.get("metadata", {})
        
        if not text:
            logger.error("No text found in CV data")
            raise ValueError("CV data contains no text content")
        
        # Add source information to metadata
        enhanced_metadata = {
            **file_metadata,
            "source": "cv",
            "file_name": file_metadata.get("file_name", "unknown")
        }
        
        chunks = self.chunk_text(text, enhanced_metadata)
        
        logger.info(f"Chunked CV into {len(chunks)} pieces")
        return chunks


