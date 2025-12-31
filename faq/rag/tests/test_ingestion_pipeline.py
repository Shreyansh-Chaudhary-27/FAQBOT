"""
Tests for Document Ingestion Pipeline

This module tests the document ingestion pipeline functionality including
automated DOCX processing, incremental updates, and batch processing.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from faq.rag.core.ingestion_pipeline import DocumentIngestionPipeline, DocumentIngestionError, IngestionProgress
from faq.rag.interfaces.base import FAQEntry


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


class TestDocumentIngestionPipeline(unittest.TestCase):
    """Test the DocumentIngestionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_scraper = Mock()
        self.mock_vectorizer = Mock()
        self.mock_vector_store = Mock()
        
        self.pipeline = DocumentIngestionPipeline(
            docx_scraper=self.mock_scraper,
            vectorizer=self.mock_vectorizer,
            vector_store=self.mock_vector_store,
            max_workers=2,
            batch_size=2
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.docx_scraper)
        self.assertIsNotNone(self.pipeline.vectorizer)
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertEqual(self.pipeline.max_workers, 2)
        self.assertEqual(self.pipeline.batch_size, 2)
    
    def test_pipeline_initialization_validation(self):
        """Test pipeline initialization with invalid components."""
        with self.assertRaises(ValueError):
            DocumentIngestionPipeline(None, self.mock_vectorizer, self.mock_vector_store)
        
        with self.assertRaises(ValueError):
            DocumentIngestionPipeline(self.mock_scraper, None, self.mock_vector_store)
        
        with self.assertRaises(ValueError):
            DocumentIngestionPipeline(self.mock_scraper, self.mock_vectorizer, None)
    
    def test_document_validation(self):
        """Test document validation."""
        # Test non-existent file
        self.assertFalse(self.pipeline._validate_document("/nonexistent/file.docx"))
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()
            self.assertFalse(self.pipeline._validate_document(tmp.name))
            os.unlink(tmp.name)
        
        # Test valid DOCX file (mock)
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(b"test docx content")
            tmp.flush()
            self.assertTrue(self.pipeline._validate_document(tmp.name))
            os.unlink(tmp.name)
    
    def test_preprocess_document_paths(self):
        """Test document path preprocessing."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid DOCX file
            valid_docx = os.path.join(tmpdir, "valid.docx")
            with open(valid_docx, 'wb') as f:
                f.write(b"test content")
            
            # Invalid file (wrong extension)
            invalid_file = os.path.join(tmpdir, "invalid.txt")
            with open(invalid_file, 'w') as f:
                f.write("test content")
            
            # Non-existent file
            nonexistent = os.path.join(tmpdir, "nonexistent.docx")
            
            paths = [valid_docx, invalid_file, nonexistent]
            valid_paths = self.pipeline._preprocess_document_paths(paths)
            
            self.assertEqual(len(valid_paths), 1)
            self.assertIn(os.path.abspath(valid_docx), valid_paths)
    
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_ingest_document_success(self, mock_open, mock_exists):
        """Test successful document ingestion."""
        # Setup mocks
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = b"test content"
        
        # Mock FAQ entry
        mock_faq = FAQEntry(
            id="test-1",
            question="Test question?",
            answer="Test answer",
            keywords=["test"],
            category="general",
            confidence_score=0.9,
            source_document="test.docx",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding=None
        )
        
        # Configure mocks
        self.mock_vector_store.is_document_processed.return_value = False
        self.mock_scraper.extract_faqs.return_value = [mock_faq]
        self.mock_vectorizer.vectorize_faq_batch.return_value = [mock_faq]
        
        # Test ingestion
        result = self.pipeline.ingest_document("test.docx")
        
        # Verify calls
        self.mock_scraper.extract_faqs.assert_called_once()
        self.mock_vectorizer.vectorize_faq_batch.assert_called_once()
        self.mock_vector_store.store_vectors.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "test-1")
    
    @patch('os.path.exists')
    def test_ingest_document_already_processed(self, mock_exists):
        """Test document ingestion when document is already processed."""
        mock_exists.return_value = True
        
        # Mock that document is already processed
        self.mock_vector_store.is_document_processed.return_value = True
        self.mock_vector_store.get_document_faqs.return_value = ["faq-1", "faq-2"]
        
        # Test ingestion
        result = self.pipeline.ingest_document("test.docx")
        
        # Verify that extraction was skipped
        self.mock_scraper.extract_faqs.assert_not_called()
        self.mock_vectorizer.vectorize_faq_batch.assert_not_called()
    
    def test_ingest_document_invalid_path(self):
        """Test document ingestion with invalid path."""
        with self.assertRaises(DocumentIngestionError):
            self.pipeline.ingest_document("/nonexistent/file.docx")
    
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_ingest_documents_batch_sequential(self, mock_open, mock_exists):
        """Test batch document ingestion in sequential mode."""
        # Setup mocks
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = b"test content"
        
        # Mock FAQ entries
        mock_faq1 = FAQEntry(
            id="test-1", question="Q1?", answer="A1", keywords=["test"],
            category="general", confidence_score=0.9, source_document="doc1.docx",
            created_at=datetime.now(), updated_at=datetime.now(), embedding=None
        )
        mock_faq2 = FAQEntry(
            id="test-2", question="Q2?", answer="A2", keywords=["test"],
            category="general", confidence_score=0.9, source_document="doc2.docx",
            created_at=datetime.now(), updated_at=datetime.now(), embedding=None
        )
        
        # Configure mocks
        self.mock_vector_store.is_document_processed.return_value = False
        self.mock_scraper.extract_faqs.side_effect = [[mock_faq1], [mock_faq2]]
        self.mock_vectorizer.vectorize_faq_batch.side_effect = [[mock_faq1], [mock_faq2]]
        
        # Test batch ingestion
        results = self.pipeline.ingest_documents_batch(["doc1.docx", "doc2.docx"], parallel=False)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertIn("doc1.docx", results)
        self.assertIn("doc2.docx", results)
        self.assertEqual(len(results["doc1.docx"]), 1)
        self.assertEqual(len(results["doc2.docx"]), 1)
    
    def test_ingest_documents_batch_empty_list(self):
        """Test batch ingestion with empty document list."""
        results = self.pipeline.ingest_documents_batch([])
        self.assertEqual(results, {})
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def test_callback(progress_data):
            callback_calls.append(progress_data)
        
        # Add callback
        self.pipeline.add_progress_callback(test_callback)
        
        # Simulate progress update
        self.pipeline._current_progress = IngestionProgress(1)
        self.pipeline._notify_progress()
        
        # Verify callback was called
        self.assertEqual(len(callback_calls), 1)
        self.assertIn('total_documents', callback_calls[0])
        
        # Remove callback
        self.pipeline.remove_progress_callback(test_callback)
        self.pipeline._notify_progress()
        
        # Verify callback was not called again
        self.assertEqual(len(callback_calls), 1)
    
    def test_get_ingestion_stats(self):
        """Test ingestion statistics retrieval."""
        # Update some stats
        self.pipeline._update_stats(2, 10, 5.0)
        
        stats = self.pipeline.get_ingestion_stats()
        
        self.assertEqual(stats['total_documents_processed'], 2)
        self.assertEqual(stats['total_faqs_extracted'], 10)
        self.assertEqual(stats['total_processing_time'], 5.0)
        self.assertEqual(stats['average_faqs_per_document'], 5.0)
    
    def test_reset_stats(self):
        """Test statistics reset."""
        # Update stats
        self.pipeline._update_stats(2, 10, 5.0)
        
        # Reset
        self.pipeline.reset_stats()
        
        stats = self.pipeline.get_ingestion_stats()
        self.assertEqual(stats['total_documents_processed'], 0)
        self.assertEqual(stats['total_faqs_extracted'], 0)
        self.assertEqual(stats['total_processing_time'], 0.0)


if __name__ == '__main__':
    unittest.main()