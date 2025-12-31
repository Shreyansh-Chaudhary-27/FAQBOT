"""
RAG System Package
Retrieval-Augmented Generation components for intelligent FAQ processing

This package provides a complete RAG system for processing DOCX documents,
extracting FAQ data, and providing intelligent responses to user queries
using semantic search and AI-powered generation.
"""

# Core system components
from .core.rag_system import RAGSystem
from .core.factory import RAGSystemFactory, rag_factory

# Configuration
from .config.settings import RAGConfig, RAGConfigManager, rag_config

# Base interfaces and data models
from .interfaces.base import (
    FAQEntry, ProcessedQuery, Response, ConversationContext,
    DocumentStructure, ValidationResult, SimilarityMatch,
    DOCXScraperInterface, QueryProcessorInterface,
    FAQVectorizerInterface, VectorStoreInterface,
    ResponseGeneratorInterface, ConversationManagerInterface,
    RAGSystemInterface
)

# Utilities
from .utils.logging import RAGLogger, get_rag_logger, log_performance, log_system_event
from .utils.text_processing import (
    clean_text, extract_keywords, calculate_text_similarity,
    detect_question_patterns, split_into_sentences, extract_text_features
)
from .utils.validation import (
    validate_file_path, validate_faq_entry, validate_query,
    validate_embedding, validate_similarity_score
)

__version__ = "1.0.0"
__author__ = "RAG System Development Team"

# Package-level exports
__all__ = [
    # Core system
    'RAGSystem',
    'RAGSystemFactory',
    'rag_factory',
    
    # Configuration
    'RAGConfig',
    'RAGConfigManager', 
    'rag_config',
    
    # Data models
    'FAQEntry',
    'ProcessedQuery',
    'Response',
    'ConversationContext',
    'DocumentStructure',
    'ValidationResult',
    'SimilarityMatch',
    
    # Interfaces
    'DOCXScraperInterface',
    'QueryProcessorInterface',
    'FAQVectorizerInterface',
    'VectorStoreInterface',
    'ResponseGeneratorInterface',
    'ConversationManagerInterface',
    'RAGSystemInterface',
    
    # Utilities
    'RAGLogger',
    'get_rag_logger',
    'log_performance',
    'log_system_event',
    'clean_text',
    'extract_keywords',
    'calculate_text_similarity',
    'detect_question_patterns',
    'split_into_sentences',
    'extract_text_features',
    'validate_file_path',
    'validate_faq_entry',
    'validate_query',
    'validate_embedding',
    'validate_similarity_score',
]