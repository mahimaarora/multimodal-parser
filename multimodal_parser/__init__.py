"""Multimodal Parser - Parse and query PDF documents with RAG."""

__version__ = "0.1.0"

from .chunk_models import (
    ChunkType,
    ImageType,
    BaseChunk,
    TextChunk,
    TableChunk,
    ImageChunk,
)
from .docling_parser import DoclingParser
from .indexer import DocumentIndexer
from .agent import RAGAgent

__all__ = [
    "DoclingParser",
    "DocumentIndexer",
    "RAGAgent",
    "ChunkType",
    "ImageType",
    "BaseChunk",
    "TextChunk",
    "TableChunk",
    "ImageChunk",
]
