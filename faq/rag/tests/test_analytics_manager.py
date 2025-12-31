"""
Tests for AnalyticsManager - Comprehensive query analytics and logging functionality.

Tests cover query pattern logging, performance metrics collection, and system event tracking
as required by task 10.1.
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from faq.rag.core.analytics_manager import AnalyticsManager
from faq.rag.interfaces.base import ProcessedQuery, Response, FAQEntry


class TestAnalyticsManager(unittest.TestCase):
    """Test suite for AnalyticsManager comprehensive functionality."""
    
    def setUp(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.analytics_manager = AnalyticsManager(storage_path=self.temp_dir)
        
        # Create sample data
        self.sample_processed_query = ProcessedQuery(
            original_query="What is machine learning?",
            corrected_query="What is machine learning?",
            intent="information_request",
            expanded_queries=["What is ML?", "Define machine learning"],
            language="en",
            confidence=0.9,
            embedding=None
        )
        
        self.sample_faq = FAQEntry(
            id="faq_1",
            question="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            keywords=["machine learning", "AI", "algorithms"],
            category="technology",
            confidence_score=0.95,
            source_document="ml_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.sample_response = Response(
            text="Machine learning is a subset of artificial intelligence...",
            confidence=0.85,
            source_faqs=[self.sample_faq],
            context_used=False,
            generation_method="rag",
            metadata={"response_time": 1.2, "embedding_time": 0.3}
        )
    
    def tearDown(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test AnalyticsManager initialization."""
        self.assertIsInstance(self.analytics_manager.query_logs, list)
        self.assertIsInstance(self.analytics_manager.ingestion_logs, list)
        self.assertIsInstance(self.analytics_manager.system_event_logs, list)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_log_query_comprehensive(self):
        """Test comprehensive query logging functionality."""
        query_id = "test_query_1"
        query_text = "How does neural network work?"
        timestamp = datetime.now()
        
        # Log the query
        self.analytics_manager.log_query(
            query_id, query_text, self.sample_processed_query, self.sample_response, timestamp
        )
        
        # Verify query was logged
        self.assertEqual(len(self.analytics_manager.query_logs), 1)
        
        logged_entry = self.analytics_manager.query_logs[0]
        self.assertEqual(logged_entry["query_id"], query_id)
        self.assertEqual(logged_entry["query_text"], query_text)
        self.assertEqual(logged_entry["query_length"], len(query_text))
        self.assertIn("processed_query", logged_entry)
        self.assertIn("response", logged_entry)
        self.assertIn("processing_metadata", logged_entry)
        
        # Verify analytics patterns were updated
        self.assertGreater(self.analytics_manager.query_patterns['common_intents']['information_request'], 0)
        self.assertGreater(self.analytics_manager.query_patterns['language_distribution']['en'], 0)
        self.assertGreater(len(self.analytics_manager.performance_metrics['confidence_scores']), 0)
    
    def test_log_query_with_typo_correction(self):
        """Test query logging with typo correction tracking."""
        # Create query with typo correction
        typo_query = ProcessedQuery(
            original_query="Wat is machien lerning?",  # Typos
            corrected_query="What is machine learning?",  # Corrected
            intent="information_request",
            expanded_queries=["What is ML?"],
            language="en",
            confidence=0.8,
            embedding=None
        )
        
        self.analytics_manager.log_query(
            "typo_test", "Wat is machien lerning?", typo_query, self.sample_response, datetime.now()
        )
        
        # Verify typo correction was tracked
        self.assertEqual(len(self.analytics_manager.query_patterns['typo_corrections']), 1)
        correction = self.analytics_manager.query_patterns['typo_corrections'][0]
        self.assertEqual(correction['original'], "Wat is machien lerning?")
        self.assertEqual(correction['corrected'], "What is machine learning?")
    
    def test_log_query_failed_query_tracking(self):
        """Test tracking of failed queries (low confidence)."""
        low_confidence_response = Response(
            text="I'm not sure about that...",
            confidence=0.2,  # Low confidence
            source_faqs=[],
            context_used=False,
            generation_method="fallback",
            metadata={}
        )
        
        self.analytics_manager.log_query(
            "failed_test", "Complex unclear question", self.sample_processed_query, 
            low_confidence_response, datetime.now()
        )
        
        # Verify failed query was tracked
        self.assertEqual(len(self.analytics_manager.query_patterns['failed_queries']), 1)
        failed_query = self.analytics_manager.query_patterns['failed_queries'][0]
        self.assertEqual(failed_query['confidence'], 0.2)
        self.assertEqual(failed_query['reason'], 'low_confidence')
    
    def test_log_document_ingestion(self):
        """Test document ingestion logging."""
        doc_path = "/test/document.docx"
        timestamp = datetime.now()
        
        self.analytics_manager.log_document_ingestion(
            doc_path, 15, timestamp, "success"
        )
        
        # Verify ingestion was logged
        self.assertEqual(len(self.analytics_manager.ingestion_logs), 1)
        
        logged_entry = self.analytics_manager.ingestion_logs[0]
        self.assertEqual(logged_entry["document_path"], doc_path)
        self.assertEqual(logged_entry["document_name"], "document.docx")
        self.assertEqual(logged_entry["faqs_ingested"], 15)
        self.assertEqual(logged_entry["status"], "success")
    
    def test_log_document_ingestion_failure(self):
        """Test logging of failed document ingestion."""
        error_msg = "File format not supported"
        
        self.analytics_manager.log_document_ingestion(
            "/test/bad_doc.docx", 0, datetime.now(), "failed", error_msg
        )
        
        # Verify error tracking
        self.assertGreater(self.analytics_manager.system_health['component_errors']['document_ingestion'], 0)
        self.assertIn("ingestion_error", str(self.analytics_manager.system_health['error_patterns']))
    
    def test_log_system_event(self):
        """Test system event logging with categorization."""
        event_details = {"component": "vectorizer", "memory_usage": "85%"}
        
        self.analytics_manager.log_system_event(
            "vectorizer_high_memory", event_details, datetime.now()
        )
        
        # Verify event was logged
        self.assertEqual(len(self.analytics_manager.system_event_logs), 1)
        
        logged_event = self.analytics_manager.system_event_logs[0]
        self.assertEqual(logged_event["event_type"], "vectorizer_high_memory")
        self.assertEqual(logged_event["component"], "vectorizer")
        self.assertIn("severity", logged_event)
    
    def test_get_query_patterns_comprehensive(self):
        """Test comprehensive query pattern analysis."""
        # Add multiple queries with different patterns
        queries_data = [
            ("query1", "What is AI?", "information_request", "en", 0.9),
            ("query2", "How to use ML?", "how_to", "en", 0.8),
            ("query3", "¿Qué es IA?", "information_request", "es", 0.7),
            ("query4", "AI definition", "information_request", "en", 0.6)
        ]
        
        for query_id, query_text, intent, language, confidence in queries_data:
            processed_query = ProcessedQuery(
                original_query=query_text,
                corrected_query=query_text,
                intent=intent,
                expanded_queries=[],
                language=language,
                confidence=confidence,
                embedding=None
            )
            
            response = Response(
                text="Response text",
                confidence=confidence,
                source_faqs=[self.sample_faq] if confidence > 0.7 else [],
                context_used=False,
                generation_method="rag",
                metadata={}
            )
            
            self.analytics_manager.log_query(
                query_id, query_text, processed_query, response, datetime.now()
            )
        
        # Get query patterns
        patterns = self.analytics_manager.get_query_patterns()
        
        # Verify pattern analysis
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Check for expected pattern types
        pattern_types = [p["pattern_type"] for p in patterns]
        expected_types = ["intent_distribution", "language_distribution", "query_complexity", 
                         "typo_correction", "response_quality", "popular_topics", "temporal_patterns"]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, pattern_types)
        
        # Verify intent distribution
        intent_pattern = next(p for p in patterns if p["pattern_type"] == "intent_distribution")
        self.assertIn("information_request", intent_pattern["data"])
        self.assertEqual(intent_pattern["total_queries"], 4)
    
    def test_get_performance_metrics_comprehensive(self):
        """Test comprehensive performance metrics collection."""
        # Add sample queries with performance data
        for i in range(5):
            response = Response(
                text=f"Response {i}",
                confidence=0.8 + (i * 0.05),  # Varying confidence
                source_faqs=[self.sample_faq],
                context_used=i % 2 == 0,  # Alternating context usage
                generation_method="rag",
                metadata={"response_time": 1.0 + (i * 0.2)}
            )
            
            self.analytics_manager.log_query(
                f"perf_query_{i}", f"Test query {i}", self.sample_processed_query, 
                response, datetime.now()
            )
        
        # Get performance metrics
        metrics = self.analytics_manager.get_performance_metrics()
        
        # Verify comprehensive metrics structure
        self.assertIn("analysis_period", metrics)
        self.assertIn("query_processing", metrics)
        self.assertIn("response_quality", metrics)
        self.assertIn("performance_timing", metrics)
        self.assertIn("context_usage", metrics)
        self.assertIn("system_health", metrics)
        
        # Verify query processing metrics
        query_metrics = metrics["query_processing"]
        self.assertEqual(query_metrics["total_queries"], 5)
        self.assertGreater(query_metrics["success_rate"], 0)
        
        # Verify response quality metrics
        quality_metrics = metrics["response_quality"]
        self.assertGreater(quality_metrics["average_confidence"], 0)
        self.assertIn("confidence_distribution", quality_metrics)
    
    def test_get_performance_metrics_with_date_filter(self):
        """Test performance metrics with date filtering."""
        # Add queries with different timestamps
        old_date = datetime.now() - timedelta(days=10)
        recent_date = datetime.now() - timedelta(days=1)
        
        # Old query
        self.analytics_manager.log_query(
            "old_query", "Old test", self.sample_processed_query, self.sample_response, old_date
        )
        
        # Recent query
        self.analytics_manager.log_query(
            "recent_query", "Recent test", self.sample_processed_query, self.sample_response, recent_date
        )
        
        # Get metrics for recent period only
        start_date = datetime.now() - timedelta(days=2)
        metrics = self.analytics_manager.get_performance_metrics(start_date=start_date)
        
        # Should only include recent query
        self.assertEqual(metrics["query_processing"]["total_queries"], 1)
    
    def test_persistent_storage(self):
        """Test persistent storage of analytics data."""
        # Add some data
        self.analytics_manager.log_query(
            "storage_test", "Test query", self.sample_processed_query, self.sample_response, datetime.now()
        )
        
        # Force save
        self.analytics_manager._save_persistent_data()
        
        # Verify files were created
        patterns_file = os.path.join(self.temp_dir, "query_patterns.json")
        metrics_file = os.path.join(self.temp_dir, "performance_metrics.json")
        
        self.assertTrue(os.path.exists(patterns_file))
        self.assertTrue(os.path.exists(metrics_file))
        
        # Verify data can be loaded
        with open(patterns_file, 'r') as f:
            patterns_data = json.load(f)
            self.assertIn("common_intents", patterns_data)
    
    def test_system_health_report(self):
        """Test system health report generation."""
        # Add some system events
        self.analytics_manager.log_system_event(
            "component_error", {"component": "vectorizer", "error": "timeout"}, datetime.now()
        )
        
        health_report = self.analytics_manager.get_system_health_report()
        
        # Verify report structure
        self.assertIn("timestamp", health_report)
        self.assertIn("component_errors", health_report)
        self.assertIn("error_patterns", health_report)
        self.assertIn("total_queries_processed", health_report)
        self.assertIn("data_storage_status", health_report)
    
    def test_export_analytics_data(self):
        """Test analytics data export functionality."""
        # Add sample data
        self.analytics_manager.log_query(
            "export_test", "Export test query", self.sample_processed_query, 
            self.sample_response, datetime.now()
        )
        
        export_path = os.path.join(self.temp_dir, "analytics_export.json")
        
        # Export data
        success = self.analytics_manager.export_analytics_data(export_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported data
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
            self.assertIn("query_logs", exported_data)
            self.assertIn("query_patterns", exported_data)
            self.assertIn("export_timestamp", exported_data)


if __name__ == '__main__':
    unittest.main()