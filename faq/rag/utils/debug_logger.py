"""
Debug Logger for RAG System Investigation

This module provides enhanced debug logging to track query processing
and response generation for identifying the content dump issue.
"""

import logging
import json
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Create debug log directory
DEBUG_LOG_DIR = Path("debug_logs")
DEBUG_LOG_DIR.mkdir(exist_ok=True)

class RAGDebugLogger:
    """Enhanced debug logger for RAG system investigation."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f"rag_debug.{component_name}")
        
        # Set up file handler for debug logs
        debug_file = DEBUG_LOG_DIR / f"rag_debug_{component_name}.log"
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def log_query_processing(self, query: str, processed_query: Any = None, 
                           retrieved_faqs: List[Any] = None, response: Any = None,
                           step: str = "unknown") -> None:
        """Log detailed query processing information."""
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "step": step,
            "query": query,
            "processed_query": self._serialize_object(processed_query),
            "retrieved_faqs_count": len(retrieved_faqs) if retrieved_faqs else 0,
            "retrieved_faqs": self._serialize_faqs(retrieved_faqs) if retrieved_faqs else [],
            "response": self._serialize_object(response)
        }
        
        self.logger.debug(f"QUERY_PROCESSING: {json.dumps(debug_data, indent=2)}")
    
    def log_response_generation(self, query: str, faqs: List[Any], 
                              response_text: str, confidence: float,
                              generation_method: str, metadata: Dict[str, Any] = None) -> None:
        """Log response generation details."""
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "query": query,
            "input_faqs_count": len(faqs) if faqs else 0,
            "input_faqs": self._serialize_faqs(faqs) if faqs else [],
            "response_text": response_text,
            "response_length": len(response_text),
            "confidence": confidence,
            "generation_method": generation_method,
            "metadata": metadata or {},
            "is_content_dump": self._detect_content_dump(response_text)
        }
        
        self.logger.debug(f"RESPONSE_GENERATION: {json.dumps(debug_data, indent=2)}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with context."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(error_data, indent=2)}")
    
    def log_vector_search(self, query: str, query_embedding: Any, 
                         search_results: List[Any], threshold: float) -> None:
        """Log vector search details."""
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "query": query,
            "embedding_shape": getattr(query_embedding, 'shape', None) if query_embedding is not None else None,
            "embedding_type": type(query_embedding).__name__ if query_embedding is not None else None,
            "search_results_count": len(search_results) if search_results else 0,
            "threshold": threshold,
            "search_results": self._serialize_search_results(search_results) if search_results else []
        }
        
        self.logger.debug(f"VECTOR_SEARCH: {json.dumps(debug_data, indent=2)}")
    
    def _serialize_object(self, obj: Any) -> Dict[str, Any]:
        """Serialize object for logging."""
        if obj is None:
            return None
        
        try:
            if hasattr(obj, '__dict__'):
                return {
                    "type": type(obj).__name__,
                    "attributes": {k: str(v) for k, v in obj.__dict__.items()}
                }
            else:
                return {"type": type(obj).__name__, "value": str(obj)}
        except Exception:
            return {"type": type(obj).__name__, "serialization_error": True}
    
    def _serialize_faqs(self, faqs: List[Any]) -> List[Dict[str, Any]]:
        """Serialize FAQ list for logging."""
        if not faqs:
            return []
        
        serialized = []
        for faq in faqs[:5]:  # Limit to first 5 FAQs to avoid huge logs
            try:
                faq_data = {
                    "id": getattr(faq, 'id', 'unknown'),
                    "question": getattr(faq, 'question', 'unknown')[:100],  # Truncate long questions
                    "answer": getattr(faq, 'answer', 'unknown')[:200],  # Truncate long answers
                    "confidence_score": getattr(faq, 'confidence_score', 0),
                    "category": getattr(faq, 'category', 'unknown'),
                    "source_document": getattr(faq, 'source_document', 'unknown')
                }
                serialized.append(faq_data)
            except Exception as e:
                serialized.append({"serialization_error": str(e)})
        
        if len(faqs) > 5:
            serialized.append({"truncated": f"... and {len(faqs) - 5} more FAQs"})
        
        return serialized
    
    def _serialize_search_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Serialize search results for logging."""
        if not results:
            return []
        
        serialized = []
        for result in results[:5]:  # Limit to first 5 results
            try:
                result_data = {
                    "similarity_score": getattr(result, 'similarity_score', 'unknown'),
                    "faq_id": getattr(getattr(result, 'faq_entry', None), 'id', 'unknown'),
                    "faq_question": getattr(getattr(result, 'faq_entry', None), 'question', 'unknown')[:100]
                }
                serialized.append(result_data)
            except Exception as e:
                serialized.append({"serialization_error": str(e)})
        
        if len(results) > 5:
            serialized.append({"truncated": f"... and {len(results) - 5} more results"})
        
        return serialized
    
    def _detect_content_dump(self, response_text: str) -> Dict[str, Any]:
        """Detect if response appears to be a content dump."""
        if not response_text:
            return {"is_dump": False, "reason": "empty_response"}
        
        # Check for multiple Q: A: patterns (indicates FAQ dump)
        qa_pattern_count = response_text.count('Q:')
        
        # Check for excessive length
        is_too_long = len(response_text) > 1000
        
        # Check for multiple FAQ-like structures
        faq_indicators = ['Question:', 'Answer:', 'FAQ:', 'Q.', 'A.']
        indicator_count = sum(response_text.count(indicator) for indicator in faq_indicators)
        
        # Check for repetitive patterns
        lines = response_text.split('\n')
        long_lines_count = len([line for line in lines if len(line) > 100])
        
        is_dump = (
            qa_pattern_count > 1 or 
            is_too_long or 
            indicator_count > 3 or 
            long_lines_count > 10
        )
        
        return {
            "is_dump": is_dump,
            "qa_pattern_count": qa_pattern_count,
            "response_length": len(response_text),
            "indicator_count": indicator_count,
            "long_lines_count": long_lines_count,
            "is_too_long": is_too_long
        }


def get_debug_logger(component_name: str) -> RAGDebugLogger:
    """Get a debug logger for the specified component."""
    return RAGDebugLogger(component_name)