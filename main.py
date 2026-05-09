"""
CV Question Answering Agent - Main Entry Point

This is an agentic AI solution that answers questions about a CV using:
- LangGraph for agent orchestration
- ChromaDB for vector storage
- Multiple LLM providers (OpenAI, Anthropic, OpenRouter)
- Grounding validation to prevent hallucinations
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.utils.logging_config import setup_logging
from src.ingestion import CVParser, TextChunker
from src.vectorstore import ChromaStore
from src.llm import LLMProvider
from src.agent import create_retrieval_tool, CVQAAgent, ConversationMemory
from src.grounding import GroundingValidator
from src.cli import CVQACLI


logger = logging.getLogger(__name__)


def initialize_system():
    """Initialize all system components"""
    try:
        # Load settings
        logger.info("Loading configuration...")
        settings = get_settings()
        
        # Setup logging
        setup_logging(settings.log_level, settings.log_file)
        logger.info("=" * 80)
        logger.info("CV Question Answering Agent - Starting")
        logger.info("=" * 80)
        
        # Check if CV file exists
        cv_path = Path(settings.cv_file_path)
        if not cv_path.exists():
            logger.error(f"CV file not found: {settings.cv_file_path}")
            print(f"\n❌ Error: CV file not found at {settings.cv_file_path}")
            print(f"Please place your CV (PDF or DOCX) at: {settings.cv_file_path}")
            print(f"Or update the CV_FILE_PATH in your .env file\n")
            sys.exit(1)
        
        logger.info(f"CV file found: {settings.cv_file_path}")
        
        # Initialize vector store
        logger.info("Initializing ChromaDB vector store...")
        vector_store = ChromaStore(
            persist_directory=settings.chroma_db_path,
            collection_name="cv_collection",
            embedding_model=settings.embedding_model
        )
        
        # Check if we need to ingest CV
        if vector_store.is_empty():
            logger.info("Vector store is empty. Ingesting CV...")
            ingest_cv(settings, vector_store)
        else:
            logger.info(f"Vector store already populated with {vector_store.collection.count()} documents")
            print(f"\n✓ Using existing CV embeddings ({vector_store.collection.count()} chunks)")
            
            # Ask if user wants to re-ingest
            response = input("\nDo you want to re-ingest the CV? (y/N): ").strip().lower()
            if response == 'y':
                logger.info("Re-ingesting CV as requested by user")
                vector_store.clear_collection()
                ingest_cv(settings, vector_store)
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {settings.llm_provider} - {settings.model_name}")
        llm = LLMProvider.create_llm(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.get_api_key(),
            temperature=settings.temperature
        )
        
        # Create retrieval tool
        logger.info("Creating retrieval tool...")
        retrieval_tool = create_retrieval_tool(vector_store, settings.retrieval_top_k)
        
        # Initialize grounding validator
        logger.info("Initializing grounding validator...")
        grounding_validator = GroundingValidator(strict_mode=True)
        
        # Initialize agent
        logger.info("Initializing CV QA Agent...")
        agent = CVQAAgent(
            llm=llm,
            retrieval_tool=retrieval_tool,
            grounding_validator=grounding_validator,
            max_iterations=settings.max_iterations
        )
        
        # Initialize memory
        logger.info("Initializing conversation memory...")
        memory = ConversationMemory(max_history=10)
        
        logger.info("System initialization complete!")
        return agent, memory, settings
        
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}", exc_info=True)
        print(f"\n❌ Error initializing system: {str(e)}\n")
        sys.exit(1)


def ingest_cv(settings, vector_store):
    """Ingest CV into vector store"""
    try:
        print(f"\n📄 Ingesting CV from: {settings.cv_file_path}")
        
        # Parse CV
        parser = CVParser()
        cv_data = parser.parse(settings.cv_file_path)
        
        if not parser.validate_content(cv_data):
            raise ValueError("CV content validation failed - content too short or empty")
        
        print(f"✓ Parsed CV: {cv_data['metadata']['char_count']} characters")
        
        # Chunk text
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = chunker.chunk_cv_data(cv_data)
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Add to vector store
        print("⏳ Generating embeddings and storing in ChromaDB...")
        vector_store.add_chunks(chunks)
        
        print(f"✓ CV successfully ingested into vector store!\n")
        logger.info(f"CV ingestion complete: {len(chunks)} chunks stored")
        
    except Exception as e:
        logger.error(f"Error ingesting CV: {str(e)}", exc_info=True)
        print(f"\n❌ Error ingesting CV: {str(e)}\n")
        raise


def main():
    """Main entry point"""
    try:
        # Initialize system
        agent, memory, settings = initialize_system()
        
        # Create and run CLI
        cli = CVQACLI(agent, memory, settings)
        cli.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()


