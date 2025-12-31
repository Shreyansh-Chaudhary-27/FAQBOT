"""
Main FAQ Vectorizer Component

This module provides the main interface for FAQ vectorization, integrating
embedding generation, similarity matching, and vector operations for the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from faq.rag.interfaces.base import (
    FAQVectorizerInterface, FAQEntry, SimilarityMatch, ValidationResult
)
from faq.rag.components.vectorizer.embedding_generator import (
    FAQEmbeddingGenerator, EmbeddingGenerationError
)
from faq.rag.components.vectorizer.similarity_matcher import (
    VectorSimilarityMatcher, AdvancedSimilarityMatcher, SimilarityMatchingError
)
from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger


logger = get_rag_logger(__name__)


class FAQVectorizerError(Exception):
    """Custom exception for FAQ vectorizer errors."""
    pass


class FAQVectorizer(FAQVectorizerInterface):
    """
    Main FAQ Vectorizer component that integrates embedding generation
    and similarity matching for semantic FAQ retrieval.
    
    This class implements the FAQVectorizerInterface and provides a unified
    interface for all vectorization operations in the RAG system.
    """
    
    def __init__(self, use_advanced_matching: bool = False):
        """
        Initialize the FAQ vectorizer with embedding and similarity components.
        
        Args:
            use_advanced_matching: Whether to use advanced similarity matching features
        """
        try:
            # Initialize embedding generator
            self.embedding_generator = FAQEmbeddingGenerator()
            
            # Initialize similarity matcher
            if use_advanced_matching:
                self.similarity_matcher = AdvancedSimilarityMatcher()
                logger.info("FAQ vectorizer initialized with advanced similarity matching")
            else:
                self.similarity_matcher = VectorSimilarityMatcher()
                logger.info("FAQ vectorizer initialized with standard similarity matching")
            
            # Configuration
            self.config = rag_config.get_vector_config()
            self.embedding_dimension = self.embedding_generator.embedding_dimension
            
            # Vector index for efficient similarity search
            self._vector_index: List[FAQEntry] = []
            self._index_dirty = False
            
            logger.info(f"FAQ vectorizer initialized successfully with dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAQ vectorizer: {e}")
            raise FAQVectorizerError(f"Vectorizer initialization failed: {e}")
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text using configured model.
        """
        try:
            if self.embedding_generator.config.embedding_type == 'local':
                return self.embedding_generator.service.generate_embedding(text)
            else:
                return self.embedding_generator.service.generate_embedding(
                    text, task_type="retrieval_query"
                )
        except Exception as e:
            logger.error(f"Failed to generate embeddings for text: {e}")
            raise FAQVectorizerError(f"Embedding generation failed: {e}")
    
    def vectorize_faq_entry(self, faq: FAQEntry) -> FAQEntry:
        """
        Generate embeddings for FAQ entry components.
        
        Args:
            faq: FAQ entry to vectorize
            
        Returns:
            FAQ entry with embeddings populated
            
        Raises:
            FAQVectorizerError: If vectorization fails
        """
        try:
            logger.debug(f"Vectorizing FAQ entry: {faq.id}")
            vectorized_faq = self.embedding_generator.generate_faq_embedding(faq)
            
            # Mark index as dirty since we have new embeddings
            self._index_dirty = True
            
            return vectorized_faq
            
        except EmbeddingGenerationError as e:
            logger.error(f"Failed to vectorize FAQ entry {faq.id}: {e}")
            raise FAQVectorizerError(f"FAQ vectorization failed: {e}")
    
    def vectorize_faq_batch(self, faqs: List[FAQEntry]) -> List[FAQEntry]:
        """
        Generate embeddings for multiple FAQ entries with batch processing.
        
        Args:
            faqs: List of FAQ entries to vectorize
            
        Returns:
            List of FAQ entries with embeddings populated
            
        Raises:
            FAQVectorizerError: If batch vectorization fails
        """
        try:
            logger.info(f"Batch vectorizing {len(faqs)} FAQ entries")
            vectorized_faqs = self.embedding_generator.generate_batch_embeddings(faqs)
            
            # Mark index as dirty since we have new embeddings
            self._index_dirty = True
            
            return vectorized_faqs
            
        except EmbeddingGenerationError as e:
            logger.error(f"Failed to batch vectorize FAQ entries: {e}")
            raise FAQVectorizerError(f"Batch vectorization failed: {e}")
    
    def update_vector_index(self, vectors: List[FAQEntry]) -> None:
        """
        Update the vector index with new embeddings.
        
        Args:
            vectors: List of FAQ entries with embeddings to add to index
            
        Raises:
            FAQVectorizerError: If index update fails
        """
        try:
            logger.info(f"Updating vector index with {len(vectors)} entries")
            
            # Validate that all entries have embeddings
            valid_vectors = []
            for faq in vectors:
                if faq.embedding is not None and self._validate_embedding(faq.embedding):
                    valid_vectors.append(faq)
                else:
                    logger.warning(f"FAQ {faq.id} has invalid embedding, skipping index update")
            
            # Update the index
            # For now, we use a simple list-based index
            # In production, this could be replaced with a more sophisticated vector database
            self._vector_index.extend(valid_vectors)
            self._index_dirty = False
            
            logger.info(f"Vector index updated with {len(valid_vectors)} valid entries. Total index size: {len(self._vector_index)}")
            
        except Exception as e:
            logger.error(f"Failed to update vector index: {e}")
            raise FAQVectorizerError(f"Vector index update failed: {e}")
    
    def find_similar_vectors(self, query_vector: np.ndarray, top_k: int) -> List[SimilarityMatch]:
        """
        Find similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of similar vectors to return
            
        Returns:
            List of similarity matches sorted by relevance
            
        Raises:
            FAQVectorizerError: If similarity search fails
        """
        try:
            logger.debug(f"Finding {top_k} similar vectors from index of size {len(self._vector_index)}")
            
            if not self._vector_index:
                logger.warning("Vector index is empty, no similar vectors found")
                return []
            
            # Use similarity matcher to find similar FAQs
            matches = self.similarity_matcher.find_similar_faqs(
                query_vector=query_vector,
                faq_entries=self._vector_index,
                top_k=top_k
            )
            
            logger.debug(f"Found {len(matches)} similar vectors")
            return matches
            
        except SimilarityMatchingError as e:
            logger.error(f"Failed to find similar vectors: {e}")
            raise FAQVectorizerError(f"Similarity search failed: {e}")
    
    def find_similar_faqs(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[SimilarityMatch]:
        """
        Find FAQ entries similar to a text query.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similarity matches
            
        Raises:
            FAQVectorizerError: If search fails
        """
        try:
            # Generate query embedding
            query_vector = self.generate_embeddings(query)
            
            # Find similar vectors
            matches = self.similarity_matcher.find_similar_faqs(
                query_vector=query_vector,
                faq_entries=self._vector_index,
                top_k=top_k,
                threshold=threshold
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to find similar FAQs for query: {e}")
            raise FAQVectorizerError(f"FAQ search failed: {e}")
    
    def rank_faqs_by_relevance(
        self, 
        query: str, 
        faq_entries: List[FAQEntry],
        ranking_factors: Optional[Dict[str, float]] = None
    ) -> List[Tuple[FAQEntry, float]]:
        """
        Rank FAQ entries by relevance to a query.
        
        Args:
            query: Text query for ranking
            faq_entries: List of FAQ entries to rank
            ranking_factors: Weights for different ranking factors
            
        Returns:
            List of (FAQ, relevance_score) tuples sorted by relevance
            
        Raises:
            FAQVectorizerError: If ranking fails
        """
        try:
            # Generate query embedding
            query_vector = self.generate_embeddings(query)
            
            # Use similarity matcher to rank FAQs
            ranked_faqs = self.similarity_matcher.rank_faqs_by_relevance(
                query_vector=query_vector,
                faq_entries=faq_entries,
                ranking_factors=ranking_factors
            )
            
            return ranked_faqs
            
        except Exception as e:
            logger.error(f"Failed to rank FAQs by relevance: {e}")
            raise FAQVectorizerError(f"FAQ ranking failed: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1
            
        Raises:
            FAQVectorizerError: If similarity calculation fails
        """
        try:
            # Generate embeddings for both texts
            embedding1 = self.generate_embeddings(text1)
            embedding2 = self.generate_embeddings(text2)
            
            # Calculate cosine similarity
            similarity = self.similarity_matcher.calculate_cosine_similarity(
                embedding1, embedding2
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity between texts: {e}")
            raise FAQVectorizerError(f"Similarity calculation failed: {e}")
    
    def find_diverse_results(
        self, 
        query: str, 
        diversity_threshold: float = 0.8,
        max_results: Optional[int] = None
    ) -> List[SimilarityMatch]:
        """
        Find diverse FAQ results to avoid redundant answers.
        
        Args:
            query: Text query to search for
            diversity_threshold: Minimum similarity between results to consider diverse
            max_results: Maximum number of diverse results
            
        Returns:
            List of diverse similarity matches
            
        Raises:
            FAQVectorizerError: If diverse search fails
        """
        try:
            # Generate query embedding
            query_vector = self.generate_embeddings(query)
            
            # Find diverse results using similarity matcher
            diverse_matches = self.similarity_matcher.find_diverse_results(
                query_vector=query_vector,
                faq_entries=self._vector_index,
                diversity_threshold=diversity_threshold,
                max_results=max_results
            )
            
            return diverse_matches
            
        except Exception as e:
            logger.error(f"Failed to find diverse results: {e}")
            raise FAQVectorizerError(f"Diverse search failed: {e}")
    
    def validate_embeddings(self, faqs: List[FAQEntry]) -> ValidationResult:
        """
        Validate embeddings for a list of FAQ entries.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        try:
            return self.embedding_generator.validate_embeddings(faqs)
        except Exception as e:
            logger.error(f"Failed to validate embeddings: {e}")
            raise FAQVectorizerError(f"Embedding validation failed: {e}")
    
    def get_vectorizer_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vectorizer state.
        
        Returns:
            Dictionary containing vectorizer statistics
        """
        try:
            # Get embedding stats
            embedding_stats = self.embedding_generator.get_embedding_stats(self._vector_index)
            
            # Get similarity stats from recent matches (if any)
            similarity_stats = {
                'index_size': len(self._vector_index),
                'index_dirty': self._index_dirty,
                'embedding_dimension': self.embedding_dimension
            }
            
            # Combine stats
            stats = {
                'vectorizer_type': 'advanced' if isinstance(self.similarity_matcher, AdvancedSimilarityMatcher) else 'standard',
                'embedding_stats': embedding_stats,
                'similarity_stats': similarity_stats,
                'config': self.config
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vectorizer stats: {e}")
            return {'error': str(e)}
    
    def clear_index(self) -> None:
        """Clear the vector index to free memory."""
        self._vector_index.clear()
        self._index_dirty = False
        logger.info("Vector index cleared")
    
    def rebuild_index(self, faqs: List[FAQEntry]) -> None:
        """
        Rebuild the vector index from scratch.
        
        Args:
            faqs: List of FAQ entries to build index from
            
        Raises:
            FAQVectorizerError: If index rebuild fails
        """
        try:
            logger.info(f"Rebuilding vector index with {len(faqs)} FAQ entries")
            
            # Clear existing index
            self.clear_index()
            
            # Vectorize all FAQs if needed
            faqs_to_vectorize = [faq for faq in faqs if faq.embedding is None]
            if faqs_to_vectorize:
                logger.info(f"Vectorizing {len(faqs_to_vectorize)} FAQ entries without embeddings")
                vectorized_faqs = self.vectorize_faq_batch(faqs_to_vectorize)
                
                # Update original FAQs with embeddings
                faq_dict = {faq.id: faq for faq in vectorized_faqs}
                for faq in faqs:
                    if faq.id in faq_dict:
                        faq.embedding = faq_dict[faq.id].embedding
            
            # Update index
            self.update_vector_index(faqs)
            
            logger.info(f"Vector index rebuilt successfully with {len(self._vector_index)} entries")
            
        except Exception as e:
            logger.error(f"Failed to rebuild vector index: {e}")
            raise FAQVectorizerError(f"Index rebuild failed: {e}")
    
    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding is properly formatted.
        """
        return self.embedding_generator.service.validate_embedding(embedding)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the vectorizer.
        """
        try:
            # Check embedding generator health
            embedding_health = self.embedding_generator.service.health_check()
            
            # Check similarity matcher functionality
            try:
                test_vector = np.random.rand(self.embedding_dimension)
                test_similarity = self.similarity_matcher.calculate_cosine_similarity(
                    test_vector, test_vector
                )
                similarity_healthy = abs(test_similarity - 1.0) < 0.001
            except Exception as e:
                similarity_healthy = False
                logger.error(f"Similarity matcher health check failed: {e}")
            
            return {
                'status': 'healthy' if embedding_health.get('status') == 'healthy' and similarity_healthy else 'unhealthy',
                'embedding_service': embedding_health,
                'similarity_matcher': {
                    'status': 'healthy' if similarity_healthy else 'unhealthy',
                    'test_similarity_correct': similarity_healthy
                },
                'vector_index': {
                    'size': len(self._vector_index),
                    'dirty': self._index_dirty
                },
                'embedding_dimension': self.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Vectorizer health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }