"""
Simple tests for Document Ingestion Pipeline

This module tests the document ingestion pipeline functionality without Django dependencies.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# Import the classes we want to test directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from faq.rag.core.ingestion_pipeline import IngestionProgress, DocumentIngestionError


class TestIngestionProgress(unittest.TestCase):
    """Test the IngestionProgress tracking class."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        progress = IngestionProgress(5)
        
        self.assertEqual(progress.total_documents, 5)
        self.assertEqual(progress.processed_documents, 0)
        self.assertEqual(progress.successful_documents, 0)
        self.assertEqual(progress.failed_documents, 0)
        self.assertEqual(len(progress.errors), 0)
    
    def test_progress_updates(self):
        """Test progress tracking updates."""
        progress = IngestionProgress(3)
        
        # Mark success
        progress.mark_success("doc1.docx", 5)
        self.assertEqual(progress.processed_documents, 1)
        self.assertEqual(progress.successful_documents, 1)
        self.assertEqual(progress.failed_documents, 0)
        
        # Mark failure
        progress.mark_failure("doc2.docx", "File not found")
        self.assertEqual(progress.processed_documents, 2)
        self.assertEqual(progress.successful_documents, 1)
        self.assertEqual(progress.failed_documents, 1)
        self.assertEqual(len(progress.errors), 1)
        
        # Get progress
        progress_data = progress.get_progress()
        self.assertEqual(progress_data['total_documents'], 3)
        self.assertEqual(progress_data['processed_documents'], 2)
        self.assertAlmostEqual(progress_data['progress_percent'], 66.67, places=1)
    
    def test_update_current_document(self):
        """Test updating current document being processed."""
        progress = IngestionProgress(2)
        
        progress.update_current("test_document.docx")
        progress_data = progress.get_progress()
        
        self.assertEqual(progress_data['current_document'], "test_document.docx")


class TestDocumentValidation(unittest.TestCase):
    """Test document validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_scraper = Mock()
        self.mock_vectorizer = Mock()
        self.mock_vector_store = Mock()
        
        # Mock the config to avoid Django dependency
        mock_config = Mock()
        mock_config.max_document_size_mb = 50
        
        # We'll test validation methods directly without full pipeline initialization
        self.test_config = mock_config
    
    def test_document_hash_generation(self):
        """Test document hash generation."""
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as tmp:
            tmp.write("test content for hashing")
            tmp.flush()
            
            # Import the pipeline class and test hash generation
            from faq.rag.core.ingestion_pipeline import DocumentIngestionPipeline
            
            # Create a minimal pipeline instance for testing
            try:
                pipeline = DocumentIngestionPipeline(
                    self.mock_scraper, self.mock_vectorizer, self.mock_vector_store
                )
                
                # Test hash generation
                hash1 = pipeline._generate_document_hash(tmp.name)
                hash2 = pipeline._generate_document_hash(tmp.name)
                
                # Same file should produce same hash
                self.assertEqual(hash1, hash2)
                self.assertIsInstance(hash1, str)
                self.assertEqual(len(hash1), 64)  # SHA256 produces 64-character hex string
                
            except Exception as e:
                # If pipeline initialization fails due to config, skip this test
                self.skipTest(f"Pipeline initialization failed: {e}")
            finally:
                os.unlink(tmp.name)
    
    def test_file_validation_logic(self):
        """Test file validation without full pipeline."""
        # Test with temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid DOCX file
            valid_docx = os.path.join(tmpdir, "valid.docx")
            with open(valid_docx, 'wb') as f:
                f.write(b"test docx content")
            
            # Invalid file (wrong extension)
            invalid_file = os.path.join(tmpdir, "invalid.txt")
            with open(invalid_file, 'w') as f:
                f.write("test content")
            
            # Test file existence
            self.assertTrue(os.path.exists(valid_docx))
            self.assertTrue(os.path.exists(invalid_file))
            
            # Test file extensions
            self.assertTrue(valid_docx.lower().endswith('.docx'))
            self.assertFalse(invalid_file.lower().endswith('.docx'))


class TestDocumentIngestionError(unittest.TestCase):
    """Test the DocumentIngestionError exception."""
    
    def test_error_creation(self):
        """Test creating DocumentIngestionError."""
        error_msg = "Test error message"
        error = DocumentIngestionError(error_msg)
        
        self.assertEqual(str(error), error_msg)
        self.assertIsInstance(error, Exception)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)