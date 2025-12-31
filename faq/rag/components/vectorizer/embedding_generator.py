"""
Embedding Generator for FAQ Vectorization

This module handles the generation of embeddings for FAQ entries, including
questions, answers, and keywords, with batch processing and validation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from faq.rag.interfaces.base import FAQEntry, ValidationResult
from faq.rag.components.vectorizer.gemini_service import GeminiEmbeddingService, GeminiServiceError
from faq.rag.components.vectorizer.local_service import LocalEmbeddingService
from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger
from faq.rag.utils.text_processing import clean_text, extract_keywords


logger = get_rag_logger(__name__)


class EmbeddingGenerationError(Exception):
    """Custom exception for embedding generation errors."""
    pass


class FAQEmbeddingGenerator:
    """
    Generates embeddings for FAQ entries using Gemini AI.
    
    Handles embedding generation for questions, answers, and keywords
    with batch processing, validation, and quality checks.
    """
    
    def __init__(self):
        """Initialize the embedding generator with configured service."""
        try:
            self.config = rag_config.config
            if self.config.embedding_type == 'local':
                self.service = LocalEmbeddingService()
                logger.info("Using LOCAL embedding service")
            else:
                self.service = GeminiEmbeddingService()
                logger.info("Using GEMINI embedding service")
                
            self.embedding_dimension = self.service.get_embedding_dimension()
            logger.info(f"FAQ embedding generator initialized with dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            raise EmbeddingGenerationError(f"Initialization failed: {e}")
    
    def generate_faq_embedding(self, faq: FAQEntry) -> FAQEntry:
        """
        Generate embeddings for a single FAQ entry.
        
        Creates a composite embedding from question, answer, and keywords
        that represents the semantic content of the entire FAQ entry.
        
        Args:
            faq: FAQ entry to generate embedding for
            
        Returns:
            FAQ entry with embedding populated
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            logger.debug(f"Generating embedding for FAQ: {faq.id}")
            
            # Prepare text components for embedding
            components = self._prepare_text_components(faq)
            
            # Generate embeddings for each component
            embeddings = {}
            for component_name, text in components.items():
                if text:
                    try:
                        if self.config.embedding_type == 'local':
                            embedding = self.service.generate_embedding(text)
                        else:
                            embedding = self.service.generate_embedding(
                                text, 
                                task_type="retrieval_document"
                            )
                        embeddings[component_name] = embedding
                        logger.debug(f"Generated {component_name} embedding: {embedding.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to generate {component_name} embedding: {e}")
                        embeddings[component_name] = np.zeros(self.embedding_dimension)
            
            # Create composite embedding
            composite_embedding = self._create_composite_embedding(embeddings)
            
            # Validate embedding quality
            if not self._validate_embedding_quality(composite_embedding, faq):
                logger.warning(f"Low quality embedding detected for FAQ: {faq.id}")
            
            # Update FAQ entry with embedding
            faq.embedding = composite_embedding
            faq.updated_at = datetime.now()
            
            logger.debug(f"Successfully generated embedding for FAQ: {faq.id}")
            return faq
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for FAQ {faq.id}: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed for FAQ {faq.id}: {e}")
    
    def generate_batch_embeddings(self, faqs: List[FAQEntry]) -> List[FAQEntry]:
        """
        Generate embeddings for multiple FAQ entries with batch processing.
        
        Args:
            faqs: List of FAQ entries to process
            
        Returns:
            List of FAQ entries with embeddings populated
            
        Raises:
            EmbeddingGenerationError: If batch processing fails
        """
        if not faqs:
            logger.warning("Empty FAQ list provided for batch embedding generation")
            return []
        
        logger.info(f"Starting batch embedding generation for {len(faqs)} FAQs")
        
        try:
            # Prepare all text components for batch processing
            all_texts = []
            text_to_faq_mapping = []
            
            for faq in faqs:
                components = self._prepare_text_components(faq)
                faq_texts = []
                
                for component_name, text in components.items():
                    if text:
                        all_texts.append(text)
                        faq_texts.append((len(all_texts) - 1, component_name))
                
                text_to_faq_mapping.append((faq, faq_texts))
            
            # Generate embeddings in batch
            logger.info(f"Generating {len(all_texts)} embeddings in batch")
            if self.config.embedding_type == 'local':
                batch_embeddings = self.service.generate_embeddings_batch(all_texts)
            else:
                batch_embeddings = self.service.generate_embeddings_batch(
                    all_texts, 
                    task_type="retrieval_document"
                )
            
            # Assign embeddings back to FAQ entries
            processed_faqs = []
            for faq, faq_texts in text_to_faq_mapping:
                try:
                    # Collect embeddings for this FAQ
                    faq_embeddings = {}
                    for text_index, component_name in faq_texts:
                        if text_index < len(batch_embeddings):
                            faq_embeddings[component_name] = batch_embeddings[text_index]
                        else:
                            logger.warning(f"Missing embedding for {component_name} in FAQ {faq.id}")
                            faq_embeddings[component_name] = np.zeros(self.embedding_dimension)
                    
                    # Create composite embedding
                    composite_embedding = self._create_composite_embedding(faq_embeddings)
                    
                    # Update FAQ entry
                    faq.embedding = composite_embedding
                    faq.updated_at = datetime.now()
                    
                    processed_faqs.append(faq)
                    
                except Exception as e:
                    logger.error(f"Failed to process embedding for FAQ {faq.id}: {e}")
                    # Add FAQ with zero embedding as fallback
                    faq.embedding = np.zeros(self.embedding_dimension)
                    faq.updated_at = datetime.now()
                    processed_faqs.append(faq)
            
            logger.info(f"Successfully processed embeddings for {len(processed_faqs)} FAQs")
            return processed_faqs
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingGenerationError(f"Batch processing failed: {e}")
    
    def update_faq_embedding(self, faq: FAQEntry) -> FAQEntry:
        """
        Update embedding for an existing FAQ entry.
        
        Args:
            faq: FAQ entry to update
            
        Returns:
            FAQ entry with updated embedding
        """
        logger.info(f"Updating embedding for FAQ: {faq.id}")
        return self.generate_faq_embedding(faq)
    
    def validate_embeddings(self, faqs: List[FAQEntry]) -> ValidationResult:
        """
        Validate embeddings for a list of FAQ entries.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        valid_count = 0
        
        for faq in faqs:
            if faq.embedding is None:
                errors.append(f"FAQ {faq.id} has no embedding")
                continue
            
            if not self.service.validate_embedding(faq.embedding):
                errors.append(f"FAQ {faq.id} has invalid embedding")
                continue
            
            if not self._validate_embedding_quality(faq.embedding, faq):
                warnings.append(f"FAQ {faq.id} has low quality embedding")
            
            valid_count += 1
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metadata={
                'total_faqs': len(faqs),
                'valid_embeddings': valid_count,
                'error_count': len(errors),
                'warning_count': len(warnings)
            }
        )
    
    def _prepare_text_components(self, faq: FAQEntry) -> Dict[str, str]:
        """
        Prepare text components from FAQ entry for embedding generation.
        Optimized to only return the composite text to reduce API calls.
        """
        # Create composite text for main embedding
        composite_parts = []
        if faq.question:
            composite_parts.append(f"Question: {clean_text(faq.question)}")
        if faq.answer:
            composite_parts.append(f"Answer: {clean_text(faq.answer)}")
        if faq.keywords:
            composite_parts.append(f"Keywords: {' '.join(faq.keywords)}")
        
        return {'composite': ' '.join(composite_parts)}
    
    def _create_composite_embedding(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a composite embedding from component embeddings.
        """
        if 'composite' in embeddings and self.service.validate_embedding(embeddings['composite']):
            return embeddings['composite']
        
        # Fallback to weighted combination of components
        valid_embeddings = []
        weights = []
        
        # Question gets highest weight (0.5)
        if 'question' in embeddings and self.service.validate_embedding(embeddings['question']):
            valid_embeddings.append(embeddings['question'])
            weights.append(0.5)
        
        # Answer gets medium weight (0.4)
        if 'answer' in embeddings and self.service.validate_embedding(embeddings['answer']):
            valid_embeddings.append(embeddings['answer'])
            weights.append(0.4)
        
        # Keywords get lower weight (0.1)
        if 'keywords' in embeddings and self.service.validate_embedding(embeddings['keywords']):
            valid_embeddings.append(embeddings['keywords'])
            weights.append(0.1)
        
        if not valid_embeddings:
            logger.warning("No valid embeddings found for composite creation")
            return np.zeros(self.embedding_dimension)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create weighted average
        composite = np.zeros(self.embedding_dimension)
        for embedding, weight in zip(valid_embeddings, weights):
            composite += weight * embedding
        
        # Normalize the composite embedding
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite = composite / norm
        
        return composite
    
    def _validate_embedding_quality(self, embedding: np.ndarray, faq: FAQEntry) -> bool:
        """
        Validate the quality of a generated embedding.
        """
        if not self.service.validate_embedding(embedding):
            return False
        
        # Check embedding magnitude (should not be too small)
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1:
            logger.warning(f"Very small embedding magnitude ({magnitude}) for FAQ {faq.id}")
            return False
        
        # Check for reasonable variance (not all values the same)
        variance = np.var(embedding)
        if variance < 1e-6:
            logger.warning(f"Very low embedding variance ({variance}) for FAQ {faq.id}")
            return False
        
        return True
    
    def get_embedding_stats(self, faqs: List[FAQEntry]) -> Dict[str, Any]:
        """
        Get statistics about embeddings in a list of FAQs.
        
        Args:
            faqs: List of FAQ entries to analyze
            
        Returns:
            Dictionary containing embedding statistics
        """
        stats = {
            'total_faqs': len(faqs),
            'faqs_with_embeddings': 0,
            'faqs_without_embeddings': 0,
            'average_magnitude': 0.0,
            'average_variance': 0.0,
            'dimension': self.embedding_dimension
        }
        
        magnitudes = []
        variances = []
        
        for faq in faqs:
            if faq.embedding is not None and self.service.validate_embedding(faq.embedding):
                stats['faqs_with_embeddings'] += 1
                magnitude = np.linalg.norm(faq.embedding)
                variance = np.var(faq.embedding)
                magnitudes.append(magnitude)
                variances.append(variance)
            else:
                stats['faqs_without_embeddings'] += 1
        
        if magnitudes:
            stats['average_magnitude'] = np.mean(magnitudes)
            stats['average_variance'] = np.mean(variances)
            stats['min_magnitude'] = np.min(magnitudes)
            stats['max_magnitude'] = np.max(magnitudes)
        
        return stats