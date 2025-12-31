"""
Logging Utilities for RAG System

Centralized logging configuration and utilities for RAG components.
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class RAGLogger:
    """Centralized logger for RAG system components."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure_logging(cls, log_level: str = "INFO", log_file: Optional[str] = None) -> None:
        """Configure logging for the RAG system."""
        if cls._configured:
            return
        
        # Set up root logger
        root_logger = logging.getLogger('rag')
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific component."""
        if not cls._configured:
            cls.configure_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(f'rag.{name}')
        
        return cls._loggers[name]
    
    @classmethod
    def log_query_processing(cls, 
                           original_query: str, 
                           corrected_query: str, 
                           processing_time: float,
                           metadata: Dict[str, Any] = None) -> None:
        """Log query processing information."""
        logger = cls.get_logger('query_processor')
        
        log_data = {
            'original_query': original_query,
            'corrected_query': corrected_query,
            'processing_time_ms': processing_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            log_data.update(metadata)
        
        logger.info(f"Query processed: {log_data}")
    
    @classmethod
    def log_document_processing(cls,
                              document_path: str,
                              faqs_extracted: int,
                              processing_time: float,
                              success: bool,
                              errors: list = None) -> None:
        """Log document processing information."""
        logger = cls.get_logger('docx_scraper')
        
        log_data = {
            'document_path': document_path,
            'faqs_extracted': faqs_extracted,
            'processing_time_ms': processing_time * 1000,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if errors:
            log_data['errors'] = errors
        
        level = logging.INFO if success else logging.ERROR
        logger.log(level, f"Document processed: {log_data}")
    
    @classmethod
    def log_embedding_generation(cls,
                               text_length: int,
                               embedding_dimension: int,
                               generation_time: float,
                               success: bool) -> None:
        """Log embedding generation information."""
        logger = cls.get_logger('vectorizer')
        
        log_data = {
            'text_length': text_length,
            'embedding_dimension': embedding_dimension,
            'generation_time_ms': generation_time * 1000,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        level = logging.INFO if success else logging.ERROR
        logger.log(level, f"Embedding generated: {log_data}")
    
    @classmethod
    def log_similarity_search(cls,
                            query_length: int,
                            results_found: int,
                            search_time: float,
                            top_similarity: float = None) -> None:
        """Log similarity search information."""
        logger = cls.get_logger('vector_store')
        
        log_data = {
            'query_length': query_length,
            'results_found': results_found,
            'search_time_ms': search_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        if top_similarity is not None:
            log_data['top_similarity'] = top_similarity
        
        logger.info(f"Similarity search: {log_data}")
    
    @classmethod
    def log_response_generation(cls,
                              query: str,
                              response_length: int,
                              confidence: float,
                              generation_time: float,
                              sources_used: int) -> None:
        """Log response generation information."""
        logger = cls.get_logger('response_generator')
        
        log_data = {
            'query_length': len(query),
            'response_length': response_length,
            'confidence': confidence,
            'generation_time_ms': generation_time * 1000,
            'sources_used': sources_used,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Response generated: {log_data}")
    
    @classmethod
    def log_error(cls, component: str, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error information."""
        logger = cls.get_logger(component)
        
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        logger.error(f"Error in {component}: {log_data}", exc_info=True)


# Convenience functions for common logging operations
def get_rag_logger(component: str) -> logging.Logger:
    """Get a logger for a RAG component."""
    return RAGLogger.get_logger(component)


def log_performance(component: str, operation: str, duration: float, metadata: Dict[str, Any] = None) -> None:
    """Log performance information."""
    logger = RAGLogger.get_logger(component)
    
    log_data = {
        'operation': operation,
        'duration_ms': duration * 1000,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        log_data.update(metadata)
    
    logger.info(f"Performance: {log_data}")


def log_system_event(event: str, details: Dict[str, Any] = None) -> None:
    """Log system-level events."""
    logger = RAGLogger.get_logger('system')
    
    log_data = {
        'event': event,
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"System event: {log_data}")