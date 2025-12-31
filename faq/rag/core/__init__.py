"""
RAG System Core Module

This module provides the main entry points and orchestration for the RAG system,
including system initialization, component coordination, and high-level interfaces.
"""

from .rag_system import RAGSystem, RAGSystemError
from .factory import RAGSystemFactory, rag_factory
from .initializer import RAGSystemInitializer, RAGInitializationError, rag_initializer

# Main entry points for the RAG system
__all__ = [
    'RAGSystem',
    'RAGSystemError', 
    'RAGSystemFactory',
    'rag_factory',
    'RAGSystemInitializer',
    'RAGInitializationError',
    'rag_initializer',
    'create_rag_system',
    'initialize_rag_system'
]


def create_rag_system(**components) -> RAGSystem:
    """
    Create a RAG system with specified components.
    
    Args:
        **components: Component implementations to use
        
    Returns:
        RAGSystem: Configured RAG system
        
    Example:
        >>> from faq.rag.core import create_rag_system
        >>> rag_system = create_rag_system()
    """
    return rag_factory.create_rag_system(**components)


def initialize_rag_system(validate_config: bool = True, 
                         perform_health_check: bool = True) -> RAGSystem:
    """
    Initialize a complete RAG system with validation and health checks.
    
    Args:
        validate_config: Whether to validate configuration
        perform_health_check: Whether to perform health check
        
    Returns:
        RAGSystem: Fully initialized RAG system
        
    Example:
        >>> from faq.rag.core import initialize_rag_system
        >>> rag_system = initialize_rag_system()
    """
    return rag_initializer.initialize_system(
        validate_config=validate_config,
        perform_health_check=perform_health_check
    )