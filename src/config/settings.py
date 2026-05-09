"""Configuration settings using Pydantic"""
import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Provider Configuration
    llm_provider: Literal["openai", "anthropic", "openrouter"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name to use"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses"
    )
    
    # API Keys
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key"
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key"
    )
    openrouter_model: str = Field(
        default="openai/gpt-4-turbo-preview",
        description="OpenRouter model identifier"
    )
    
    # Vector Store Configuration
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # File Paths
    cv_file_path: str = Field(
        default="data/cv/resume.pdf",
        description="Path to CV file"
    )
    output_dir: str = Field(
        default="outputs/conversations",
        description="Directory for output files"
    )
    chroma_db_path: str = Field(
        default="data/chroma_db",
        description="Path to ChromaDB storage"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="logs/agent.log",
        description="Log file path"
    )
    
    # Agent Configuration
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum agent iterations"
    )
    
    def get_api_key(self) -> str:
        """Get the appropriate API key based on provider"""
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return self.anthropic_api_key
        elif self.llm_provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            return self.openrouter_api_key
        else:
            raise ValueError(f"Unknown provider: {self.llm_provider}")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cv_file_path).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


