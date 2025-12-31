"""
Integration Test for RAG System Setup

Test that verifies the RAG system can be properly initialized and
basic functionality works as expected.
"""

import unittest
from unittest.mock import Mock
from datetime import datetime

from faq.rag.interfaces.base import (
    FAQEntry, ProcessedQuery, Response,
    DOCXScraperInterface, QueryProcessorInterface,
    FAQVectorizerInterface, VectorStoreInterface,
    ResponseGeneratorInterface, ConversationManagerInterface
)
from faq.rag.core.rag_system import RAGSystem
from faq.rag.config.settings import RAGConfig


class TestRAGSystemIntegration(unittest.TestCase):
    """Test RAG system integration with mock components."""
    
    def setUp(self):
        """Set up test fixtures with mock components."""
        # Create mock components
        self.mock_docx_scraper = Mock(spec=DOCXScraperInterface)
        self.mock_query_processor = Mock(spec=QueryProcessorInterface)
        self.mock_vectorizer = Mock(spec=FAQVectorizerInterface)
        self.mock_vector_store = Mock(spec=VectorStoreInterface)
        self.mock_response_generator = Mock(spec=ResponseGeneratorInterface)
        self.mock_conversation_manager = Mock(spec=ConversationManagerInterface)
        
        # Create RAG system with mock components
        self.rag_system = RAGSystem(
            docx_scraper=self.mock_docx_scraper,
            query_processor=self.mock_query_processor,
            vectorizer=self.mock_vectorizer,
            vector_store=self.mock_vector_store,
            response_generator=self.mock_response_generator,
            conversation_manager=self.mock_conversation_manager
        )
    
    def test_rag_system_initialization(self):
        """Test that RAG system initializes correctly with all components."""
        self.assertIsNotNone(self.rag_system)
        self.assertEqual(self.rag_system.docx_scraper, self.mock_docx_scraper)
        self.assertEqual(self.rag_system.query_processor, self.mock_query_processor)
        self.assertEqual(self.rag_system.vectorizer, self.mock_vectorizer)
        self.assertEqual(self.rag_system.vector_store, self.mock_vector_store)
        self.assertEqual(self.rag_system.response_generator, self.mock_response_generator)
        self.assertEqual(self.rag_system.conversation_manager, self.mock_conversation_manager)
    
    def test_process_document_workflow(self):
        """Test document processing workflow."""
        # Setup mock returns
        test_faqs = [
            FAQEntry(
                id="test-1",
                question="What is RAG?",
                answer="Retrieval-Augmented Generation",
                keywords=["rag", "ai"],
                category="AI",
                confidence_score=0.9,
                source_document="test.docx",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        self.mock_docx_scraper.extract_faqs.return_value = test_faqs
        self.mock_vectorizer.vectorize_faq_entry.return_value = test_faqs[0]
        
        # Test document processing
        result = self.rag_system.process_document("test.docx")
        
        # Verify calls were made
        self.mock_docx_scraper.extract_faqs.assert_called_once_with("test.docx")
        self.mock_vectorizer.vectorize_faq_entry.assert_called_once_with(test_faqs[0])
        self.mock_vector_store.store_vectors.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].question, "What is RAG?")
    
    def test_answer_query_workflow(self):
        """Test query answering workflow."""
        import numpy as np
        from faq.rag.interfaces.base import SimilarityMatch
        
        # Setup mock returns
        processed_query = ProcessedQuery(
            original_query="what is rag?",
            corrected_query="What is RAG?",
            intent="information_request",
            expanded_queries=["What is RAG?"],
            language="en",
            confidence=0.95
        )
        
        test_embedding = np.array([0.1, 0.2, 0.3])
        
        test_faq = FAQEntry(
            id="test-1",
            question="What is RAG?",
            answer="Retrieval-Augmented Generation",
            keywords=["rag", "ai"],
            category="AI",
            confidence_score=0.9,
            source_document="test.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        similarity_match = SimilarityMatch(
            faq_entry=test_faq,
            similarity_score=0.95,
            match_type="semantic",
            matched_components=["question"]
        )
        
        test_response = Response(
            text="RAG stands for Retrieval-Augmented Generation",
            confidence=0.9,
            source_faqs=[test_faq],
            context_used=False,
            generation_method="rag",
            metadata={}
        )
        
        self.mock_query_processor.preprocess_query.return_value = processed_query
        self.mock_vectorizer.generate_embeddings.return_value = test_embedding
        self.mock_vector_store.search_similar.return_value = [similarity_match]
        self.mock_response_generator.generate_response.return_value = test_response
        
        # Test query answering
        result = self.rag_system.answer_query("what is rag?")
        
        # Verify calls were made
        self.mock_query_processor.preprocess_query.assert_called_once_with("what is rag?")
        self.mock_vectorizer.generate_embeddings.assert_called_once_with("What is RAG?")
        self.mock_vector_store.search_similar.assert_called_once()
        self.mock_response_generator.generate_response.assert_called_once()
        
        # Verify result
        self.assertEqual(result.text, "RAG stands for Retrieval-Augmented Generation")
        self.assertEqual(result.confidence, 0.9)
    
    def test_update_knowledge_base(self):
        """Test knowledge base update functionality."""
        test_faqs = [
            FAQEntry(
                id="test-2",
                question="How does RAG work?",
                answer="RAG combines retrieval and generation",
                keywords=["rag", "retrieval", "generation"],
                category="AI",
                confidence_score=0.85,
                source_document="test2.docx",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        self.mock_vectorizer.vectorize_faq_entry.return_value = test_faqs[0]
        
        # Test knowledge base update
        self.rag_system.update_knowledge_base(test_faqs)
        
        # Verify calls were made
        self.mock_vectorizer.vectorize_faq_entry.assert_called_once_with(test_faqs[0])
        self.mock_vector_store.store_vectors.assert_called_once()
    
    def test_get_system_stats(self):
        """Test system statistics retrieval."""
        mock_stats = {
            'total_vectors': 100,
            'index_size': 1024,
            'last_updated': datetime.now().isoformat()
        }
        
        self.mock_vector_store.get_vector_stats.return_value = mock_stats
        
        # Test stats retrieval
        result = self.rag_system.get_system_stats()
        
        # Verify result structure
        self.assertIn('vector_store', result)
        self.assertIn('config', result)
        self.assertIn('system_status', result)
        self.assertEqual(result['system_status'], 'operational')
        self.assertEqual(result['vector_store'], mock_stats)


if __name__ == '__main__':
    unittest.main()