"""
Validation Utilities for RAG System

Common validation functions used across RAG components.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ..interfaces.base import FAQEntry, ValidationResult


def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> ValidationResult:
    """Validate file path and extension."""
    errors = []
    warnings = []
    
    if not file_path:
        errors.append("File path cannot be empty")
        return ValidationResult(False, errors, warnings, {})
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        errors.append(f"File does not exist: {file_path}")
    
    # Check if it's a file (not directory)
    if path.exists() and not path.is_file():
        errors.append(f"Path is not a file: {file_path}")
    
    # Check file extension
    if allowed_extensions:
        extension = path.suffix.lower().lstrip('.')
        if extension not in [ext.lower().lstrip('.') for ext in allowed_extensions]:
            errors.append(f"Unsupported file extension: {extension}. Allowed: {allowed_extensions}")
    
    # Check file size (warn if > 10MB)
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 10:
            warnings.append(f"Large file size: {size_mb:.1f}MB")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={'file_size_mb': size_mb if path.exists() else 0}
    )


def validate_faq_entry(faq: FAQEntry) -> ValidationResult:
    """Validate FAQ entry completeness and quality."""
    errors = []
    warnings = []
    metadata = {}
    
    # Check required fields
    if not faq.id:
        errors.append("FAQ ID cannot be empty")
    
    if not faq.question or not faq.question.strip():
        errors.append("FAQ question cannot be empty")
    
    if not faq.answer or not faq.answer.strip():
        errors.append("FAQ answer cannot be empty")
    
    # Check question quality
    if faq.question:
        question = faq.question.strip()
        
        # Check minimum length
        if len(question) < 5:
            warnings.append("Question is very short (< 5 characters)")
        
        # Check if it looks like a question
        if not (question.endswith('?') or any(word in question.lower() for word in 
                ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'do', 'is', 'are'])):
            warnings.append("Text may not be a proper question")
        
        metadata['question_length'] = len(question)
        metadata['question_word_count'] = len(question.split())
    
    # Check answer quality
    if faq.answer:
        answer = faq.answer.strip()
        
        # Check minimum length
        if len(answer) < 10:
            warnings.append("Answer is very short (< 10 characters)")
        
        metadata['answer_length'] = len(answer)
        metadata['answer_word_count'] = len(answer.split())
    
    # Check confidence score
    if not (0.0 <= faq.confidence_score <= 1.0):
        errors.append("Confidence score must be between 0.0 and 1.0")
    
    # Check keywords
    if not faq.keywords:
        warnings.append("No keywords provided")
    elif len(faq.keywords) > 20:
        warnings.append("Too many keywords (> 20)")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


def validate_query(query: str) -> ValidationResult:
    """Validate user query."""
    errors = []
    warnings = []
    metadata = {}
    
    if not query or not query.strip():
        errors.append("Query cannot be empty")
        return ValidationResult(False, errors, warnings, metadata)
    
    query = query.strip()
    
    # Check length
    if len(query) < 2:
        errors.append("Query too short (< 2 characters)")
    elif len(query) > 1000:
        errors.append("Query too long (> 1000 characters)")
    
    # Check for suspicious patterns
    if re.search(r'[<>{}[\]\\]', query):
        warnings.append("Query contains suspicious characters")
    
    # Check word count
    word_count = len(query.split())
    if word_count > 100:
        warnings.append("Query is very long (> 100 words)")
    
    metadata['query_length'] = len(query)
    metadata['word_count'] = word_count
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


def validate_embedding(embedding: Any) -> ValidationResult:
    """Validate embedding vector."""
    errors = []
    warnings = []
    metadata = {}
    
    if embedding is None:
        errors.append("Embedding cannot be None")
        return ValidationResult(False, errors, warnings, metadata)
    
    try:
        import numpy as np
        
        if not isinstance(embedding, np.ndarray):
            errors.append("Embedding must be a numpy array")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check dimensions
        if embedding.ndim != 1:
            errors.append("Embedding must be 1-dimensional")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)):
            errors.append("Embedding contains NaN values")
        
        if np.any(np.isinf(embedding)):
            errors.append("Embedding contains infinite values")
        
        # Check vector norm (should be reasonable)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            warnings.append("Embedding has zero norm")
        elif norm > 100:
            warnings.append("Embedding has very large norm")
        
        metadata['dimension'] = embedding.shape[0]
        metadata['norm'] = float(norm)
        metadata['mean'] = float(np.mean(embedding))
        metadata['std'] = float(np.std(embedding))
        
    except ImportError:
        errors.append("NumPy not available for embedding validation")
    except Exception as e:
        errors.append(f"Error validating embedding: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


def validate_similarity_score(score: float) -> ValidationResult:
    """Validate similarity score."""
    errors = []
    warnings = []
    
    if not isinstance(score, (int, float)):
        errors.append("Similarity score must be a number")
        return ValidationResult(False, errors, warnings, {})
    
    if not (0.0 <= score <= 1.0):
        errors.append("Similarity score must be between 0.0 and 1.0")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={'score': float(score)}
    )


def validate_batch_size(batch_size: int, max_size: int = 1000) -> ValidationResult:
    """Validate batch processing size."""
    errors = []
    warnings = []
    
    if not isinstance(batch_size, int):
        errors.append("Batch size must be an integer")
        return ValidationResult(False, errors, warnings, {})
    
    if batch_size <= 0:
        errors.append("Batch size must be positive")
    elif batch_size > max_size:
        errors.append(f"Batch size too large (> {max_size})")
    elif batch_size > 100:
        warnings.append("Large batch size may impact performance")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={'batch_size': batch_size}
    )