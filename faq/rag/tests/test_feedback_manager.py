"""
Tests for FeedbackManager - Comprehensive user feedback tracking functionality.

Tests cover user feedback submission, analysis, and tracking as required by task 10.1.
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from faq.rag.core.feedback_manager import FeedbackManager


class TestFeedbackManager(unittest.TestCase):
    """Test suite for FeedbackManager comprehensive functionality."""
    
    def setUp(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_manager = FeedbackManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test FeedbackManager initialization."""
        self.assertIsInstance(self.feedback_manager.feedback_entries, list)
        self.assertIsInstance(self.feedback_manager.feedback_analytics, dict)
        self.assertIsInstance(self.feedback_manager.user_patterns, dict)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_submit_feedback_basic(self):
        """Test basic feedback submission."""
        query_id = "test_query_1"
        user_id = "user_123"
        rating = 4
        comments = "Great response, very helpful!"
        
        self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        # Verify feedback was stored
        self.assertEqual(len(self.feedback_manager.feedback_entries), 1)
        
        feedback = self.feedback_manager.feedback_entries[0]
        self.assertEqual(feedback["query_id"], query_id)
        self.assertEqual(feedback["user_id"], user_id)
        self.assertEqual(feedback["rating"], rating)
        self.assertEqual(feedback["comments"], comments)
        self.assertIn("feedback_id", feedback)
        self.assertIn("timestamp", feedback)
        self.assertIn("feedback_metadata", feedback)
    
    def test_submit_feedback_with_metadata_analysis(self):
        """Test feedback submission with comprehensive metadata analysis."""
        # Test positive feedback
        self.feedback_manager.submit_feedback(
            "query_1", "user_1", 5, "Excellent response! Very accurate and fast."
        )
        
        feedback = self.feedback_manager.feedback_entries[0]
        metadata = feedback["feedback_metadata"]
        
        # Verify metadata analysis
        self.assertEqual(metadata["sentiment"], "positive")
        self.assertTrue(metadata["has_detailed_feedback"])
        self.assertGreater(metadata["comment_length"], 20)
        self.assertIn("praise", metadata["feedback_category"])
        
        # Test negative feedback
        self.feedback_manager.submit_feedback(
            "query_2", "user_2", 2, "Response was wrong and confusing."
        )
        
        negative_feedback = self.feedback_manager.feedback_entries[1]
        negative_metadata = negative_feedback["feedback_metadata"]
        
        self.assertEqual(negative_metadata["sentiment"], "negative")
        self.assertIn("complaint", negative_metadata["feedback_category"])
    
    def test_submit_feedback_rating_validation(self):
        """Test rating validation and clamping."""
        # Test invalid high rating
        self.feedback_manager.submit_feedback("query_1", "user_1", 10, "Great!")
        self.assertEqual(self.feedback_manager.feedback_entries[0]["rating"], 5)
        
        # Test invalid low rating
        self.feedback_manager.submit_feedback("query_2", "user_2", -1, "Bad!")
        self.assertEqual(self.feedback_manager.feedback_entries[1]["rating"], 1)
    
    def test_feedback_analytics_updates(self):
        """Test that feedback analytics are properly updated."""
        # Submit various feedback
        feedback_data = [
            ("query_1", "user_1", 5, "Great performance!"),
            ("query_2", "user_2", 2, "Too slow and inaccurate"),
            ("query_3", "user_1", 4, "Good but could be better"),
            ("query_4", "user_3", 1, "Completely wrong answer")
        ]
        
        for query_id, user_id, rating, comments in feedback_data:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        # Verify analytics updates
        self.assertGreater(len(self.feedback_manager.feedback_analytics['rating_trends']), 0)
        self.assertGreater(len(self.feedback_manager.feedback_analytics['feedback_categories']), 0)
        self.assertGreater(len(self.feedback_manager.feedback_analytics['improvement_suggestions']), 0)
        
        # Verify user patterns
        self.assertEqual(self.feedback_manager.user_patterns['frequent_users']['user_1'], 2)
        self.assertEqual(self.feedback_manager.user_patterns['frequent_users']['user_2'], 1)
    
    def test_get_feedback_filtering(self):
        """Test feedback retrieval with filtering."""
        # Add sample feedback
        self.feedback_manager.submit_feedback("query_1", "user_1", 5, "Great!")
        self.feedback_manager.submit_feedback("query_2", "user_1", 3, "Okay")
        self.feedback_manager.submit_feedback("query_1", "user_2", 4, "Good")
        
        # Test query filtering
        query_feedback = self.feedback_manager.get_feedback(query_id="query_1")
        self.assertEqual(len(query_feedback), 2)
        
        # Test user filtering
        user_feedback = self.feedback_manager.get_feedback(user_id="user_1")
        self.assertEqual(len(user_feedback), 2)
        
        # Test combined filtering
        specific_feedback = self.feedback_manager.get_feedback(query_id="query_1", user_id="user_1")
        self.assertEqual(len(specific_feedback), 1)
    
    def test_analyze_feedback_comprehensive(self):
        """Test comprehensive feedback analysis."""
        # Add diverse feedback data
        feedback_samples = [
            ("query_1", "user_1", 5, "Excellent accuracy and speed!"),
            ("query_2", "user_2", 4, "Good response, helpful"),
            ("query_3", "user_3", 3, "Average, could be better"),
            ("query_4", "user_4", 2, "Slow response time"),
            ("query_5", "user_5", 1, "Completely wrong and confusing"),
            ("query_6", "user_1", 5, "Perfect answer again!"),
            ("query_7", "user_2", 2, "Still having accuracy issues")
        ]
        
        for query_id, user_id, rating, comments in feedback_samples:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        # Analyze feedback
        analysis = self.feedback_manager.analyze_feedback()
        
        # Verify analysis structure
        self.assertIn("analysis_period", analysis)
        self.assertIn("overall_statistics", analysis)
        self.assertIn("rating_distribution", analysis)
        self.assertIn("sentiment_analysis", analysis)
        self.assertIn("feedback_categories", analysis)
        self.assertIn("temporal_analysis", analysis)
        self.assertIn("improvement_opportunities", analysis)
        self.assertIn("user_satisfaction_trends", analysis)
        
        # Verify statistics
        stats = analysis["overall_statistics"]
        self.assertEqual(stats["total_feedback_entries"], 7)
        self.assertGreater(stats["average_rating"], 0)
        self.assertEqual(stats["unique_users"], 5)
        
        # Verify rating distribution
        rating_dist = analysis["rating_distribution"]
        self.assertIn("detailed", rating_dist)
        self.assertIn("summary", rating_dist)
        self.assertGreater(rating_dist["summary"]["positive_feedback_count"], 0)
        self.assertGreater(rating_dist["summary"]["negative_feedback_count"], 0)
    
    def test_analyze_feedback_with_date_filter(self):
        """Test feedback analysis with date filtering."""
        # Add old feedback
        old_date = datetime.now() - timedelta(days=10)
        with patch('faq.rag.core.feedback_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_date
            mock_datetime.fromisoformat = datetime.fromisoformat
            self.feedback_manager.submit_feedback("old_query", "user_1", 3, "Old feedback")
        
        # Add recent feedback
        self.feedback_manager.submit_feedback("recent_query", "user_2", 5, "Recent feedback")
        
        # Analyze recent period only
        start_date = datetime.now() - timedelta(days=2)
        analysis = self.feedback_manager.analyze_feedback(start_date=start_date)
        
        # Should only include recent feedback
        self.assertEqual(analysis["overall_statistics"]["total_feedback_entries"], 1)
    
    def test_feedback_trends_analysis(self):
        """Test feedback trend analysis."""
        # Add feedback with improving trend
        timestamps = [
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=8),
            datetime.now() - timedelta(days=6),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2),
            datetime.now()
        ]
        
        ratings = [2, 2, 3, 4, 4, 5]  # Improving trend
        
        for i, (timestamp, rating) in enumerate(zip(timestamps, ratings)):
            with patch('faq.rag.core.feedback_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value = timestamp
                mock_datetime.fromisoformat = datetime.fromisoformat
                self.feedback_manager.submit_feedback(f"query_{i}", "user_1", rating, f"Comment {i}")
        
        analysis = self.feedback_manager.analyze_feedback()
        
        # Verify trend analysis
        temporal_analysis = analysis["temporal_analysis"]
        self.assertIn("trend", temporal_analysis)
        # Should detect improving trend
        self.assertEqual(temporal_analysis["trend"], "improving")
    
    def test_comment_analysis(self):
        """Test textual comment analysis."""
        comments_data = [
            ("query_1", "user_1", 5, "Fast and accurate response with great details"),
            ("query_2", "user_2", 2, "Slow response time and poor accuracy"),
            ("query_3", "user_3", 4, "Good answer but could be more detailed"),
            ("query_4", "user_4", 1, "Wrong information and confusing explanation")
        ]
        
        for query_id, user_id, rating, comments in comments_data:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        analysis = self.feedback_manager.analyze_feedback()
        comments_analysis = analysis["comments_analysis"]
        
        # Verify comment analysis
        self.assertEqual(comments_analysis["total_comments"], 4)
        self.assertGreater(comments_analysis["negative_comments"], 0)
        self.assertGreater(comments_analysis["positive_comments"], 0)
        self.assertGreater(comments_analysis["average_comment_length"], 0)
        self.assertIn("common_negative_themes", comments_analysis)
        self.assertIn("common_positive_themes", comments_analysis)
    
    def test_improvement_opportunities_identification(self):
        """Test identification of improvement opportunities."""
        # Add feedback with specific complaint patterns
        complaints = [
            ("query_1", "user_1", 2, "Response was too slow"),
            ("query_2", "user_2", 2, "Performance issues again"),
            ("query_3", "user_3", 1, "System is very slow"),
            ("query_4", "user_4", 2, "Accuracy problems persist"),
            ("query_5", "user_5", 1, "Wrong answers provided")
        ]
        
        for query_id, user_id, rating, comments in complaints:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        analysis = self.feedback_manager.analyze_feedback()
        opportunities = analysis["improvement_opportunities"]
        
        # Verify improvement opportunities are identified
        self.assertIsInstance(opportunities, list)
        self.assertGreater(len(opportunities), 0)
        
        # Check for high priority issues
        high_priority = [opp for opp in opportunities if opp.get("priority") == "high"]
        self.assertGreater(len(high_priority), 0)
    
    def test_user_satisfaction_trends(self):
        """Test user satisfaction trend analysis."""
        # Add feedback for multiple users with different patterns
        user_feedback_patterns = [
            # User 1: Improving satisfaction
            ("query_1", "user_1", 2, "Not great"),
            ("query_2", "user_1", 3, "Better"),
            ("query_3", "user_1", 4, "Good improvement"),
            
            # User 2: Declining satisfaction
            ("query_4", "user_2", 5, "Excellent!"),
            ("query_5", "user_2", 3, "Getting worse"),
            ("query_6", "user_2", 2, "Disappointed"),
            
            # User 3: Consistent satisfaction
            ("query_7", "user_3", 4, "Consistent quality"),
            ("query_8", "user_3", 4, "Still good")
        ]
        
        for query_id, user_id, rating, comments in user_feedback_patterns:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        analysis = self.feedback_manager.analyze_feedback()
        satisfaction_trends = analysis["user_satisfaction_trends"]
        
        # Verify trend analysis
        self.assertIn("users_with_multiple_feedback", satisfaction_trends)
        self.assertIn("improving_users", satisfaction_trends)
        self.assertIn("declining_users", satisfaction_trends)
        self.assertIn("consistent_users", satisfaction_trends)
        self.assertIn("most_active_users", satisfaction_trends)
        
        # Should detect at least one improving and one declining user
        self.assertGreaterEqual(satisfaction_trends["improving_users"], 1)
        self.assertGreaterEqual(satisfaction_trends["declining_users"], 1)
    
    def test_get_user_feedback_summary(self):
        """Test user-specific feedback summary."""
        user_id = "test_user"
        
        # Add multiple feedback entries for user
        feedback_entries = [
            ("query_1", 3, "Okay response"),
            ("query_2", 4, "Better this time"),
            ("query_3", 5, "Excellent improvement!")
        ]
        
        for query_id, rating, comments in feedback_entries:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
        
        summary = self.feedback_manager.get_user_feedback_summary(user_id)
        
        # Verify summary structure
        self.assertEqual(summary["user_id"], user_id)
        self.assertEqual(summary["total_feedback"], 3)
        self.assertGreater(summary["average_rating"], 0)
        self.assertIn("rating_range", summary)
        self.assertIn("recent_feedback", summary)
        self.assertIn("satisfaction_trend", summary)
        
        # Should detect improving trend
        self.assertEqual(summary["satisfaction_trend"], "improving")
    
    def test_get_user_feedback_summary_no_data(self):
        """Test user feedback summary with no data."""
        summary = self.feedback_manager.get_user_feedback_summary("nonexistent_user")
        
        self.assertEqual(summary["total_feedback"], 0)
        self.assertIn("message", summary)
    
    def test_persistent_storage(self):
        """Test persistent storage of feedback data."""
        # Add sample feedback
        self.feedback_manager.submit_feedback("storage_test", "user_1", 4, "Test feedback")
        
        # Force save
        self.feedback_manager._save_persistent_feedback()
        
        # Verify files were created
        feedback_file = os.path.join(self.temp_dir, "feedback_entries.json")
        analytics_file = os.path.join(self.temp_dir, "feedback_analytics.json")
        
        self.assertTrue(os.path.exists(feedback_file))
        self.assertTrue(os.path.exists(analytics_file))
        
        # Verify data can be loaded
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
            self.assertEqual(len(feedback_data), 1)
            self.assertEqual(feedback_data[0]["query_id"], "storage_test")
    
    def test_export_feedback_data(self):
        """Test feedback data export functionality."""
        # Add sample feedback
        self.feedback_manager.submit_feedback("export_test", "user_1", 5, "Export test feedback")
        
        export_path = os.path.join(self.temp_dir, "feedback_export.json")
        
        # Export with comments
        success = self.feedback_manager.export_feedback_data(export_path, include_comments=True)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported data
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
            self.assertIn("feedback_entries", exported_data)
            self.assertIn("analytics_summary", exported_data)
            self.assertIn("export_metadata", exported_data)
            
            # Verify comments are included
            self.assertEqual(exported_data["feedback_entries"][0]["comments"], "Export test feedback")
        
        # Test export without comments
        export_path_no_comments = os.path.join(self.temp_dir, "feedback_export_no_comments.json")
        success = self.feedback_manager.export_feedback_data(export_path_no_comments, include_comments=False)
        
        self.assertTrue(success)
        
        with open(export_path_no_comments, 'r') as f:
            exported_data = json.load(f)
            # Comments should be excluded
            self.assertNotIn("comments", exported_data["feedback_entries"][0])
    
    def test_sentiment_analysis(self):
        """Test comment sentiment analysis."""
        # Test positive sentiment
        self.feedback_manager.submit_feedback("pos_test", "user_1", 5, "Great job! Excellent and amazing response!")
        pos_feedback = self.feedback_manager.feedback_entries[0]
        self.assertEqual(pos_feedback["feedback_metadata"]["sentiment"], "positive")
        
        # Test negative sentiment
        self.feedback_manager.submit_feedback("neg_test", "user_2", 1, "Terrible and awful response, very bad!")
        neg_feedback = self.feedback_manager.feedback_entries[1]
        self.assertEqual(neg_feedback["feedback_metadata"]["sentiment"], "negative")
        
        # Test neutral sentiment
        self.feedback_manager.submit_feedback("neu_test", "user_3", 3, "The system provided information.")
        neu_feedback = self.feedback_manager.feedback_entries[2]
        self.assertEqual(neu_feedback["feedback_metadata"]["sentiment"], "neutral")
    
    def test_feedback_categorization(self):
        """Test feedback categorization logic."""
        test_cases = [
            (5, "Fast and accurate response", "performance_praise"),
            (4, "Correct answer provided", "accuracy_praise"),
            (5, "Good system overall", "general_satisfaction"),
            (2, "Too slow to respond", "performance_complaint"),
            (1, "Wrong information given", "accuracy_complaint"),
            (2, "Confusing interface", "usability_complaint"),
            (1, "Not satisfied", "general_dissatisfaction"),
            (3, "Average response", "neutral_feedback")
        ]
        
        for rating, comments, expected_category in test_cases:
            self.feedback_manager.submit_feedback(f"cat_test_{rating}", "user_1", rating, comments)
            feedback = self.feedback_manager.feedback_entries[-1]
            actual_category = feedback["feedback_metadata"]["feedback_category"]
            self.assertEqual(actual_category, expected_category, 
                           f"Expected {expected_category}, got {actual_category} for rating {rating}, comments '{comments}'")


if __name__ == '__main__':
    unittest.main()