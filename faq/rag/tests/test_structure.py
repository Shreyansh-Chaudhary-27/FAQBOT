"""
Test RAG System Structure and Dependencies

Basic tests to verify that the RAG system structure is properly set up
and all dependencies are available.
"""

import unittest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestRAGSystemStructure(unittest.TestCase):
    """Test RAG system structure and basic imports."""
    
    def test_dependencies_import(self):
        """Test that all required dependencies can be imported."""
        try:
            import docx
            import numpy
            import sklearn
            import google.generativeai
        except ImportError as e:
            self.fail(f"Failed to import required dependency: {e}")
    
    def test_rag_package_import(self):
        """Test that RAG package can be imported."""
        try:
            from faq import rag
        except ImportError as e:
            self.fail(f"Failed to import RAG package: {e}")
    
    def test_rag_interfaces_import(self):
        """Test that RAG interfaces can be imported."""
        try:
            from faq.rag.interfaces.base import (
                FAQEntry, ProcessedQuery, Response,
                DOCXScraperInterface, QueryProcessorInterface,
                FAQVectorizerInterface, VectorStoreInterface,
                ResponseGeneratorInterface, RAGSystemInterface
            )
        except ImportError as e:
            self.fail(f"Failed to import RAG interfaces: {e}")
    
    def test_rag_config_import(self):
        """Test that RAG configuration can be imported."""
        try:
            from faq.rag.config.settings import RAGConfig, rag_config
        except ImportError as e:
            self.fail(f"Failed to import RAG configuration: {e}")
    
    def test_rag_utilities_import(self):
        """Test that RAG utilities can be imported."""
        try:
            from faq.rag.utils.logging import RAGLogger
            from faq.rag.utils.text_processing import clean_text
            from faq.rag.utils.validation import validate_query
        except ImportError as e:
            self.fail(f"Failed to import RAG utilities: {e}")
    
    def test_rag_core_import(self):
        """Test that RAG core components can be imported."""
        try:
            from faq.rag.core.rag_system import RAGSystem
            from faq.rag.core.factory import RAGSystemFactory
        except ImportError as e:
            self.fail(f"Failed to import RAG core components: {e}")
    
    def test_data_models_creation(self):
        """Test that data models can be created."""
        from faq.rag.interfaces.base import FAQEntry, ProcessedQuery, Response
        from datetime import datetime
        import numpy as np
        
        # Test FAQEntry creation
        faq = FAQEntry(
            id="test-1",
            question="What is RAG?",
            answer="Retrieval-Augmented Generation",
            keywords=["rag", "ai", "generation"],
            category="AI",
            confidence_score=0.9,
            source_document="test.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.assertEqual(faq.id, "test-1")
        self.assertEqual(faq.question, "What is RAG?")
        
        # Test ProcessedQuery creation
        query = ProcessedQuery(
            original_query="what is rag?",
            corrected_query="What is RAG?",
            intent="information_request",
            expanded_queries=["What is RAG?", "Define RAG"],
            language="en",
            confidence=0.95
        )
        
        self.assertEqual(query.original_query, "what is rag?")
        self.assertEqual(query.language, "en")
    
    def test_configuration_loading(self):
        """Test that configuration can be loaded (without API key requirement)."""
        from faq.rag.config.settings import RAGConfig
        
        # Test default configuration values
        config = RAGConfig(
            gemini_api_key="test-key",  # Use test key for structure test
            vector_dimension=768,
            similarity_threshold=0.7,
            max_results=10
        )
        
        self.assertEqual(config.vector_dimension, 768)
        self.assertEqual(config.similarity_threshold, 0.7)
        self.assertEqual(config.max_results, 10)


if __name__ == '__main__':
    unittest.main()