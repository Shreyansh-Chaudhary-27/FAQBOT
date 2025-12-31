"""
Vector Similarity Matching for FAQ Retrieval

This module implements cosine similarity calculation and efficient similarity
search algorithms for finding relevant FAQ entries based on semantic embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import heapq

from faq.rag.interfaces.base import FAQEntry, SimilarityMatch
from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger


logger = get_rag_logger(__name__)


class SimilarityMatchingError(Exception):
    """Custom exception for similarity matching errors."""
    pass


class VectorSimilarityMatcher:
    """
    Handles vector similarity matching for FAQ retrieval.
    
    Implements cosine similarity calculation, efficient search algorithms,
    and ranking mechanisms for semantic FAQ matching.
    """
    
    def __init__(self):
        """Initialize the similarity matcher with configuration."""
        self.config = rag_config.get_vector_config()
        self.similarity_threshold = self.config['similarity_threshold']
        self.max_results = self.config['max_results']
        
        # Cache for normalized embeddings to improve performance
        self._embedding_cache = {}
        self._cache_enabled = True
        
        logger.info(f"Vector similarity matcher initialized with threshold: {self.similarity_threshold}")
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
            
        Raises:
            SimilarityMatchingError: If vectors are invalid or incompatible
        """
        try:
            # Validate input vectors
            if not self._validate_vector(vector1) or not self._validate_vector(vector2):
                raise SimilarityMatchingError("Invalid input vectors for similarity calculation")
            
            # Check dimension compatibility
            if vector1.shape != vector2.shape:
                raise SimilarityMatchingError(
                    f"Vector dimension mismatch: {vector1.shape} vs {vector2.shape}"
                )
            
            # Calculate cosine similarity
            # Reshape to 2D arrays for sklearn compatibility
            v1_2d = vector1.reshape(1, -1)
            v2_2d = vector2.reshape(1, -1)
            
            similarity = cosine_similarity(v1_2d, v2_2d)[0, 0]
            
            # Handle potential numerical issues
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise SimilarityMatchingError(f"Similarity calculation failed: {e}")
    
    def find_similar_faqs(
        self, 
        query_vector: np.ndarray, 
        faq_entries: List[FAQEntry],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[SimilarityMatch]:
        """
        Find FAQ entries most similar to the query vector.
        
        Args:
            query_vector: Query embedding vector
            faq_entries: List of FAQ entries to search
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similarity matches sorted by relevance
            
        Raises:
            SimilarityMatchingError: If search fails
        """
        if not self._validate_vector(query_vector):
            raise SimilarityMatchingError("Invalid query vector provided")
        
        if not faq_entries:
            logger.warning("Empty FAQ entries list provided for similarity search")
            return []
        
        top_k = top_k or self.max_results
        threshold = threshold or self.similarity_threshold
        
        logger.debug(f"Searching for similar FAQs: top_k={top_k}, threshold={threshold}")
        
        try:
            # Filter FAQs with valid embeddings
            valid_faqs = [faq for faq in faq_entries if self._has_valid_embedding(faq)]
            
            if not valid_faqs:
                logger.warning("No FAQs with valid embeddings found")
                return []
            
            # Calculate similarities efficiently
            similarities = self._calculate_batch_similarities(query_vector, valid_faqs)
            
            # Create similarity matches
            matches = []
            for faq, similarity in zip(valid_faqs, similarities):
                if similarity >= threshold:
                    match = SimilarityMatch(
                        faq_entry=faq,
                        similarity_score=similarity,
                        match_type="semantic",
                        matched_components=["composite"]  # Since we use composite embeddings
                    )
                    matches.append(match)
            
            # Sort by similarity score (descending) and limit results
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            matches = matches[:top_k]
            
            logger.debug(f"Found {len(matches)} similar FAQs above threshold")
            return matches
            
        except Exception as e:
            logger.error(f"Error finding similar FAQs: {e}")
            raise SimilarityMatchingError(f"Similarity search failed: {e}")
    
    def rank_faqs_by_relevance(
        self, 
        query_vector: np.ndarray, 
        faq_entries: List[FAQEntry],
        ranking_factors: Optional[Dict[str, float]] = None
    ) -> List[Tuple[FAQEntry, float]]:
        """
        Rank FAQ entries by relevance using multiple factors.
        
        Args:
            query_vector: Query embedding vector
            faq_entries: List of FAQ entries to rank
            ranking_factors: Weights for different ranking factors
            
        Returns:
            List of (FAQ, relevance_score) tuples sorted by relevance
        """
        if not self._validate_vector(query_vector):
            raise SimilarityMatchingError("Invalid query vector provided")
        
        if not faq_entries:
            return []
        
        # Default ranking factors
        default_factors = {
            'semantic_similarity': 0.7,
            'confidence_score': 0.2,
            'recency': 0.1
        }
        ranking_factors = ranking_factors or default_factors
        
        logger.debug(f"Ranking {len(faq_entries)} FAQs with factors: {ranking_factors}")
        
        try:
            ranked_faqs = []
            
            for faq in faq_entries:
                if not self._has_valid_embedding(faq):
                    continue
                
                # Calculate semantic similarity
                semantic_score = self.calculate_cosine_similarity(query_vector, faq.embedding)
                
                # Calculate confidence factor
                confidence_factor = min(faq.confidence_score, 1.0)
                
                # Calculate recency factor (newer FAQs get slight boost)
                recency_factor = self._calculate_recency_factor(faq)
                
                # Combine factors into final relevance score
                relevance_score = (
                    ranking_factors['semantic_similarity'] * semantic_score +
                    ranking_factors['confidence_score'] * confidence_factor +
                    ranking_factors['recency'] * recency_factor
                )
                
                ranked_faqs.append((faq, relevance_score))
            
            # Sort by relevance score (descending)
            ranked_faqs.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Ranked {len(ranked_faqs)} FAQs by relevance")
            return ranked_faqs
            
        except Exception as e:
            logger.error(f"Error ranking FAQs by relevance: {e}")
            raise SimilarityMatchingError(f"FAQ ranking failed: {e}")
    
    def find_diverse_results(
        self, 
        query_vector: np.ndarray, 
        faq_entries: List[FAQEntry],
        diversity_threshold: float = 0.8,
        max_results: Optional[int] = None
    ) -> List[SimilarityMatch]:
        """
        Find diverse FAQ results to avoid redundant answers.
        
        Uses similarity threshold between results to ensure diversity
        while maintaining relevance to the query.
        
        Args:
            query_vector: Query embedding vector
            faq_entries: List of FAQ entries to search
            diversity_threshold: Minimum similarity between results to consider diverse
            max_results: Maximum number of diverse results
            
        Returns:
            List of diverse similarity matches
        """
        max_results = max_results or self.max_results
        
        # First get all relevant matches
        all_matches = self.find_similar_faqs(query_vector, faq_entries, top_k=max_results * 2)
        
        if not all_matches:
            return []
        
        # Select diverse results
        diverse_matches = [all_matches[0]]  # Always include the best match
        
        for match in all_matches[1:]:
            if len(diverse_matches) >= max_results:
                break
            
            # Check if this match is diverse enough from existing matches
            is_diverse = True
            for existing_match in diverse_matches:
                similarity = self.calculate_cosine_similarity(
                    match.faq_entry.embedding,
                    existing_match.faq_entry.embedding
                )
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_matches.append(match)
        
        logger.debug(f"Selected {len(diverse_matches)} diverse results from {len(all_matches)} candidates")
        return diverse_matches
    
    def _calculate_batch_similarities(self, query_vector: np.ndarray, faqs: List[FAQEntry]) -> np.ndarray:
        """
        Calculate similarities between query vector and multiple FAQ embeddings efficiently.
        
        Args:
            query_vector: Query embedding vector
            faqs: List of FAQ entries with embeddings
            
        Returns:
            Array of similarity scores
        """
        # Stack all FAQ embeddings into a matrix
        faq_embeddings = np.stack([faq.embedding for faq in faqs])
        
        # Reshape query vector for batch computation
        query_matrix = query_vector.reshape(1, -1)
        
        # Calculate cosine similarities in batch
        similarities = cosine_similarity(query_matrix, faq_embeddings)[0]
        
        # Clip to valid range
        similarities = np.clip(similarities, -1.0, 1.0)
        
        return similarities
    
    def _validate_vector(self, vector: np.ndarray) -> bool:
        """
        Validate that a vector is suitable for similarity calculation.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if vector is valid
        """
        if vector is None:
            return False
        
        if not isinstance(vector, np.ndarray):
            return False
        
        if vector.size == 0:
            return False
        
        if not np.isfinite(vector).all():
            return False
        
        # Check if vector is not all zeros
        if np.allclose(vector, 0):
            return False
        
        return True
    
    def _has_valid_embedding(self, faq: FAQEntry) -> bool:
        """
        Check if FAQ entry has a valid embedding.
        
        Args:
            faq: FAQ entry to check
            
        Returns:
            True if FAQ has valid embedding
        """
        return faq.embedding is not None and self._validate_vector(faq.embedding)
    
    def _calculate_recency_factor(self, faq: FAQEntry) -> float:
        """
        Calculate recency factor for FAQ ranking.
        
        Args:
            faq: FAQ entry to calculate recency for
            
        Returns:
            Recency factor between 0 and 1
        """
        try:
            from datetime import datetime, timedelta
            
            now = datetime.now()
            age_days = (now - faq.updated_at).days
            
            # Newer FAQs get higher scores, with decay over time
            # Maximum boost for FAQs updated within last 30 days
            if age_days <= 30:
                return 1.0
            elif age_days <= 90:
                return 0.8
            elif age_days <= 180:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            # Default recency factor if calculation fails
            return 0.5
    
    def get_similarity_stats(self, matches: List[SimilarityMatch]) -> Dict[str, Any]:
        """
        Get statistics about similarity matching results.
        
        Args:
            matches: List of similarity matches to analyze
            
        Returns:
            Dictionary containing similarity statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'average_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'above_threshold': 0
            }
        
        similarities = [match.similarity_score for match in matches]
        
        return {
            'total_matches': len(matches),
            'average_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'above_threshold': len([s for s in similarities if s >= self.similarity_threshold]),
            'threshold_used': self.similarity_threshold
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable embedding caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        logger.debug(f"Embedding cache {'enabled' if enabled else 'disabled'}")


class AdvancedSimilarityMatcher(VectorSimilarityMatcher):
    """
    Advanced similarity matcher with additional features like
    multi-component matching and weighted similarity calculation.
    """
    
    def __init__(self):
        """Initialize advanced similarity matcher."""
        super().__init__()
        self.component_weights = {
            'question': 0.5,
            'answer': 0.4,
            'keywords': 0.1
        }
    
    def find_multi_component_matches(
        self, 
        query_components: Dict[str, np.ndarray], 
        faq_entries: List[FAQEntry]
    ) -> List[SimilarityMatch]:
        """
        Find matches using multiple query components (question, intent, keywords).
        
        Args:
            query_components: Dictionary of component embeddings
            faq_entries: List of FAQ entries to search
            
        Returns:
            List of weighted similarity matches
        """
        if not query_components or not faq_entries:
            return []
        
        logger.debug(f"Multi-component matching with {len(query_components)} components")
        
        matches = []
        
        for faq in faq_entries:
            if not self._has_valid_embedding(faq):
                continue
            
            # Calculate weighted similarity across components
            total_similarity = 0.0
            total_weight = 0.0
            
            # Use composite embedding as primary component
            if 'composite' in query_components:
                similarity = self.calculate_cosine_similarity(
                    query_components['composite'], 
                    faq.embedding
                )
                total_similarity += similarity
                total_weight += 1.0
            
            # Add component-specific similarities if available
            # This would require storing component embeddings in FAQ entries
            # For now, we use the composite embedding
            
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                
                if final_similarity >= self.similarity_threshold:
                    match = SimilarityMatch(
                        faq_entry=faq,
                        similarity_score=final_similarity,
                        match_type="multi_component",
                        matched_components=list(query_components.keys())
                    )
                    matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:self.max_results]