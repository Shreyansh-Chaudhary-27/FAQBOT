"""
Test Vector Search and Retrieval Functionality

This module tests the enhanced vector search capabilities including
batch search, filtering, and ranking functionality.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from faq.rag.components.vector_store.vector_store import VectorStore
from faq.rag.interfaces.base import FAQEntry


class TestVectorSearch(unittest.TestCase):
    """Test vector search and retrieval functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vector_store = VectorStore(storage_path="test_vector_store")
        
        # Create test FAQ entries with embeddings
        self.test_faqs = [
            FAQEntry(
                id="faq-1",
                question="What is machine learning?",
                answer="Machine learning is a subset of AI that enables computers to learn.",
                keywords=["machine learning", "AI", "artificial intelligence"],
                category="technology",
                confidence_score=0.9,
                source_document="tech_faq.docx",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            ),
            FAQEntry(
                id="faq-2",
                question="How does deep learning work?",
                answer="Deep learning uses neural networks with multiple layers.",
                keywords=["deep learning", "neural networks", "AI"],
                category="technology",
                confidence_score=0.8,
                source_document="tech_faq.docx",
                created_at=datetime.now(),
                updated_at=datetime.now() - timedelta(days=45),
                embedding=np.array([0.2, 0.3, 0.4, 0.5, 0.6])
            ),
            FAQEntry(
                id="faq-3",
                question="What is customer service?",
                answer="Customer service is the assistance provided to customers.",
                keywords=["customer service", "support", "help"],
                category="business",
                confidence_score=0.7,
                source_document="business_faq.docx",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                embedding=np.array([0.8, 0.7, 0.6, 0.5, 0.4])
            )
        ]
        
        # Store test FAQs
        self.vector_store.store_vectors(self.test_faqs)
    
    def test_basic_similarity_search(self):
        """Test basic similarity search functionality."""
        # Query vector similar to first FAQ
        query_vector = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        
        matches = self.vector_store.search_similar(query_vector, threshold=0.5, top_k=2)
        
        self.assertGreater(len(matches), 0, "Should find at least one match")
        self.assertLessEqual(len(matches), 2, "Should not exceed top_k limit")
        
        # Check that results are sorted by similarity (descending)
        if len(matches) > 1:
            self.assertGreaterEqual(matches[0].similarity_score, matches[1].similarity_score)
    
    def test_batch_search_similar(self):
        """Test batch similarity search functionality."""
        # Multiple query vectors
        query_vectors = [
            np.array([0.15, 0.25, 0.35, 0.45, 0.55]),  # Similar to faq-1
            np.array([0.75, 0.65, 0.55, 0.45, 0.35])   # Similar to faq-3
        ]
        
        batch_results = self.vector_store.batch_search_similar(
            query_vectors, threshold=0.5, top_k=2
        )
        
        self.assertEqual(len(batch_results), 2, "Should return results for each query")
        
        # Each result should be a list of matches
        for matches in batch_results:
            self.assertIsInstance(matches, list)
            self.assertLessEqual(len(matches), 2, "Should not exceed top_k limit")
    
    def test_search_with_filters(self):
        """Test search with filtering options."""
        query_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Test category filter
        tech_matches = self.vector_store.search_with_filters(
            query_vector, threshold=0.1, top_k=10, category_filter="technology"
        )
        
        for match in tech_matches:
            self.assertEqual(match.faq_entry.category, "technology")
        
        # Test confidence filter
        high_conf_matches = self.vector_store.search_with_filters(
            query_vector, threshold=0.1, top_k=10, confidence_filter=0.85
        )
        
        for match in high_conf_matches:
            self.assertGreaterEqual(match.faq_entry.confidence_score, 0.85)
        
        # Test keyword filter
        ai_matches = self.vector_store.search_with_filters(
            query_vector, threshold=0.1, top_k=10, keyword_filter=["AI"]
        )
        
        for match in ai_matches:
            faq_keywords = [kw.lower() for kw in match.faq_entry.keywords]
            self.assertIn("ai", faq_keywords)
    
    def test_search_with_ranking(self):
        """Test search with advanced ranking."""
        query_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Test with recent boost
        recent_matches = self.vector_store.search_with_ranking(
            query_vector, threshold=0.1, top_k=10, boost_recent=True
        )
        
        # Test with confidence boost
        conf_matches = self.vector_store.search_with_ranking(
            query_vector, threshold=0.1, top_k=10, boost_high_confidence=True
        )
        
        # Both should return results
        self.assertIsInstance(recent_matches, list)
        self.assertIsInstance(conf_matches, list)
    
    def test_empty_query_handling(self):
        """Test handling of edge cases."""
        # Zero vector
        zero_vector = np.zeros(5)
        matches = self.vector_store.search_similar(zero_vector, threshold=0.5)
        self.assertEqual(len(matches), 0, "Zero vector should return no matches")
        
        # Empty batch search
        empty_batch = self.vector_store.batch_search_similar([], threshold=0.5)
        self.assertEqual(len(empty_batch), 0, "Empty batch should return empty results")
    
    def test_threshold_filtering(self):
        """Test that threshold filtering works correctly."""
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # High threshold should return fewer results
        high_threshold_matches = self.vector_store.search_similar(
            query_vector, threshold=0.95, top_k=10
        )
        
        # Low threshold should return more results
        low_threshold_matches = self.vector_store.search_similar(
            query_vector, threshold=0.1, top_k=10
        )
        
        self.assertLessEqual(
            len(high_threshold_matches), 
            len(low_threshold_matches),
            "High threshold should return fewer or equal results"
        )
        
        # All results should meet threshold
        for match in high_threshold_matches:
            self.assertGreaterEqual(match.similarity_score, 0.95)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test storage
        import shutil
        import os
        if os.path.exists("test_vector_store"):
            shutil.rmtree("test_vector_store")


if __name__ == '__main__':
    unittest.main()