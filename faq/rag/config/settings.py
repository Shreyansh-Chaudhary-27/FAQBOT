"""
RAG System Configuration Management

This module handles configuration for API keys, system settings, and RAG component parameters.
Supports environment variables and Django settings integration.
"""

import os
from typing import Dict, Any, Optional
from django.conf import settings
from dataclasses import dataclass
from pathlib import Path

# Base directory for the backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class RAGConfig:
    """Configuration container for RAG system settings."""
    
    # Gemini AI Configuration
    gemini_api_key: str
    gemini_model: str = "gemini-pro"
    gemini_embedding_model: str = "models/embedding-001"
    
    # Embedding Strategy
    embedding_type: str = "local" # "gemini" or "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    vector_store_path: str = str(BASE_DIR / "vector_store_data")
    vector_dimension: int = 384
    similarity_threshold: float = 0.5
    max_results: int = 10
    
    # Query Processing Configuration
    typo_correction_enabled: bool = True
    query_expansion_enabled: bool = True
    multi_language_support: bool = True
    
    # Response Generation Configuration
    max_response_length: int = 500
    confidence_threshold: float = 0.6
    context_window_size: int = 5
    
    # Document Processing Configuration
    max_document_size_mb: int = 10
    supported_formats: tuple = ("docx",)
    batch_processing_size: int = 100
    
    # Conversation Context Configuration
    session_timeout_minutes: int = 30
    max_conversation_history: int = 20
    context_relevance_threshold: float = 0.5
    
    # Logging Configuration
    min_log_confidence: float = 0.2 # Minimum confidence to log a response


class RAGConfigManager:
    """Manages RAG system configuration from various sources."""
    
    _instance: Optional['RAGConfigManager'] = None
    _config: Optional[RAGConfig] = None
    
    def __new__(cls) -> 'RAGConfigManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> RAGConfig:
        """Load configuration from environment variables and Django settings."""
        
        # Get Gemini API key from environment or Django settings
        gemini_api_key = (
            os.getenv('GEMINI_API_KEY') or 
            getattr(settings, 'GEMINI_API_KEY', '')
        )
        
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set in environment variables or Django settings"
            )
        
        return RAGConfig(
            # Gemini AI Configuration
            gemini_api_key=gemini_api_key,
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
            gemini_embedding_model=os.getenv('GEMINI_EMBEDDING_MODEL', 'models/embedding-001'),
            
            # Embedding Strategy
            embedding_type=os.getenv('RAG_EMBEDDING_TYPE', 'local').lower(),
            local_embedding_model=os.getenv('RAG_LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            
            # Vector Store Configuration
            vector_store_path=os.getenv('RAG_VECTOR_STORE_PATH', str(BASE_DIR / "vector_store_data")),
            vector_dimension=int(os.getenv('RAG_VECTOR_DIMENSION', '384')), # Default MiniLM dim is 384
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.5')),
            max_results=int(os.getenv('RAG_MAX_RESULTS', '10')),
            
            # Query Processing Configuration
            typo_correction_enabled=os.getenv('RAG_TYPO_CORRECTION', 'true').lower() == 'true',
            query_expansion_enabled=os.getenv('RAG_QUERY_EXPANSION', 'true').lower() == 'true',
            multi_language_support=os.getenv('RAG_MULTI_LANGUAGE', 'true').lower() == 'true',
            
            # Response Generation Configuration
            max_response_length=int(os.getenv('RAG_MAX_RESPONSE_LENGTH', '500')),
            confidence_threshold=float(os.getenv('RAG_CONFIDENCE_THRESHOLD', '0.6')),
            context_window_size=int(os.getenv('RAG_CONTEXT_WINDOW_SIZE', '5')),
            
            # Document Processing Configuration
            max_document_size_mb=int(os.getenv('RAG_MAX_DOCUMENT_SIZE_MB', '10')),
            batch_processing_size=int(os.getenv('RAG_BATCH_PROCESSING_SIZE', '100')),
            
            # Conversation Context Configuration
            session_timeout_minutes=int(os.getenv('RAG_SESSION_TIMEOUT_MINUTES', '30')),
            max_conversation_history=int(os.getenv('RAG_MAX_CONVERSATION_HISTORY', '20')),
            context_relevance_threshold=float(os.getenv('RAG_CONTEXT_RELEVANCE_THRESHOLD', '0.5')),
            
            # Logging Configuration
            min_log_confidence=float(os.getenv('RAG_MIN_LOG_CONFIDENCE', '0.2')),
        )
    
    @property
    def config(self) -> RAGConfig:
        """Get the current configuration."""
        return self._config
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini AI specific configuration."""
        return {
            'api_key': self._config.gemini_api_key,
            'model': self._config.gemini_model,
            'embedding_model': self._config.gemini_embedding_model,
        }
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector store specific configuration."""
        return {
            'dimension': self._config.vector_dimension,
            'similarity_threshold': self._config.similarity_threshold,
            'max_results': self._config.max_results,
        }
    

    
    def get_response_config(self) -> Dict[str, Any]:
        """Get response generation specific configuration."""
        return {
            'max_response_length': self._config.max_response_length,
            'confidence_threshold': self._config.confidence_threshold,
            'context_window_size': self._config.context_window_size,
        }
    
    def reload_config(self) -> None:
        """Reload configuration from sources."""
        self._config = self._load_config()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Dynamically update configuration parameters.
        Args:
            updates: A dictionary of configuration parameters to update.
                     Keys should match RAGConfig attribute names.
        """
        if not self._config:
            self._config = self._load_config() # Ensure config is loaded

        for key, value in updates.items():
            if hasattr(self._config, key):
                current_value = getattr(self._config, key)
                # Type conversion for consistency, especially for env var loaded types
                if isinstance(current_value, bool):
                    setattr(self._config, key, str(value).lower() == 'true')
                elif isinstance(current_value, int):
                    setattr(self._config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(self._config, key, float(value))
                else:
                    setattr(self._config, key, value)
                print(f"Updated RAG config: {key} = {getattr(self._config, key)}")
            else:
                print(f"Warning: Attempted to update non-existent config key: {key}")


# Global configuration instance
rag_config = RAGConfigManager()