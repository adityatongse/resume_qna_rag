"""CV ingestion and processing module"""
from .parser import CVParser
from .chunker import TextChunker

__all__ = ["CVParser", "TextChunker"]


