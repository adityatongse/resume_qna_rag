"""Multi-provider LLM support"""
import logging
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


class LLMProvider:
    """Factory for creating LLM instances from different providers"""
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance based on provider
        
        Args:
            provider: Provider name (openai, anthropic, openrouter)
            model_name: Model identifier
            api_key: API key for the provider
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LangChain chat model instance
            
        Raises:
            ValueError: If provider is not supported
        """
        logger.info(f"Creating LLM instance: provider={provider}, model={model_name}")
        
        try:
            if provider == "openai":
                return LLMProvider._create_openai(model_name, api_key, temperature, **kwargs)
            elif provider == "anthropic":
                return LLMProvider._create_anthropic(model_name, api_key, temperature, **kwargs)
            elif provider == "openrouter":
                return LLMProvider._create_openrouter(model_name, api_key, temperature, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Supported providers: openai, anthropic, openrouter"
                )
        except Exception as e:
            logger.error(f"Error creating LLM instance: {str(e)}")
            raise
    
    @staticmethod
    def _create_openai(
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI chat model"""
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_anthropic(
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs
    ) -> ChatAnthropic:
        """Create Anthropic Claude chat model"""
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_openrouter(
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create OpenRouter chat model (uses OpenAI-compatible API)
        
        OpenRouter uses OpenAI's API format, so we use ChatOpenAI
        with custom base_url
        """
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/cv-qa-agent",
                "X-Title": "CV QA Agent"
            },
            **kwargs
        )
    
    @staticmethod
    def validate_provider(provider: str) -> bool:
        """Check if provider is supported"""
        supported = ["openai", "anthropic", "openrouter"]
        return provider.lower() in supported
    
    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported providers"""
        return ["openai", "anthropic", "openrouter"]


