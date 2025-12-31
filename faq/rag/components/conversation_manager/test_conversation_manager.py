"""
Tests for Conversation Manager Component

Tests the conversation state management functionality including session tracking,
context storage, and cleanup mechanisms.
"""

import unittest
import time
from datetime import datetime, timedelta
from faq.rag.components.conversation_manager import ConversationManager


class TestConversationManager(unittest.TestCase):
    """Test cases for ConversationManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use short timeouts for testing
        self.manager = ConversationManager(
            session_timeout_minutes=1,  # 1 minute timeout for testing
            max_history_length=3,       # Small history for testing
            cleanup_interval_minutes=0  # No cleanup interval for testing
        )
    
    def test_create_session(self):
        """Test session creation."""
        # Test creating session with auto-generated ID
        context1 = self.manager.create_session()
        self.assertIsNotNone(context1.session_id)
        self.assertEqual(len(context1.history), 0)
        self.assertIsNone(context1.current_topic)
        
        # Test creating session with specific ID
        session_id = "test-session-123"
        context2 = self.manager.create_session(session_id)
        self.assertEqual(context2.session_id, session_id)
        
        # Test creating duplicate session returns existing
        context3 = self.manager.create_session(session_id)
        self.assertEqual(context2.session_id, context3.session_id)
    
    def test_update_context(self):
        """Test updating conversation context."""
        session_id = "test-session"
        self.manager.create_session(session_id)
        
        # Add first interaction
        interaction1 = {
            'query': 'What is Python?',
            'response': 'Python is a programming language.',
            'confidence': 0.9,
            'context_used': False,
            'metadata': {'topic': 'programming'}
        }
        
        self.manager.update_context(session_id, interaction1)
        
        context = self.manager.get_context(session_id)
        self.assertEqual(len(context.history), 1)
        self.assertEqual(context.current_topic, 'programming')
        self.assertEqual(context.history[0]['query'], 'What is Python?')
        
        # Add second interaction
        interaction2 = {
            'query': 'How do I install it?',
            'response': 'You can install Python from python.org',
            'confidence': 0.8,
            'context_used': True,
            'metadata': {'preferences': {'language': 'en'}}
        }
        
        self.manager.update_context(session_id, interaction2)
        
        context = self.manager.get_context(session_id)
        self.assertEqual(len(context.history), 2)
        self.assertEqual(context.user_preferences['language'], 'en')
    
    def test_history_length_limit(self):
        """Test that history is limited to max_history_length."""
        session_id = "test-session"
        self.manager.create_session(session_id)
        
        # Add more interactions than the limit (3)
        for i in range(5):
            interaction = {
                'query': f'Question {i}',
                'response': f'Answer {i}',
                'confidence': 0.8,
                'context_used': False
            }
            self.manager.update_context(session_id, interaction)
        
        context = self.manager.get_context(session_id)
        self.assertEqual(len(context.history), 3)  # Should be limited to max_history_length
        
        # Check that the most recent interactions are kept
        self.assertEqual(context.history[-1]['query'], 'Question 4')
        self.assertEqual(context.history[0]['query'], 'Question 2')
    
    def test_get_context_nonexistent_session(self):
        """Test getting context for non-existent session."""
        context = self.manager.get_context("nonexistent-session")
        self.assertIsNone(context)
    
    def test_session_expiration(self):
        """Test session expiration and cleanup."""
        session_id = "test-session"
        context = self.manager.create_session(session_id)
        
        # Manually set last_activity to past time to simulate expiration
        context.last_activity = datetime.now() - timedelta(minutes=2)
        
        # Getting expired context should return None and remove session
        result = self.manager.get_context(session_id)
        self.assertIsNone(result)
        
        # Session should be removed from internal storage
        self.assertNotIn(session_id, self.manager._contexts)
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        # Create multiple sessions
        session1 = self.manager.create_session("session1")
        session2 = self.manager.create_session("session2")
        session3 = self.manager.create_session("session3")
        
        # Make some sessions expired
        session1.last_activity = datetime.now() - timedelta(minutes=2)
        session2.last_activity = datetime.now() - timedelta(minutes=2)
        # session3 remains active
        
        # Run cleanup
        cleaned_count = self.manager.cleanup_expired_sessions()
        
        self.assertEqual(cleaned_count, 2)
        self.assertIsNone(self.manager.get_context("session1"))
        self.assertIsNone(self.manager.get_context("session2"))
        self.assertIsNotNone(self.manager.get_context("session3"))
    
    def test_get_session_stats(self):
        """Test getting session statistics."""
        # Create sessions and add interactions
        self.manager.create_session("session1")
        self.manager.create_session("session2")
        
        interaction = {
            'query': 'Test query',
            'response': 'Test response',
            'confidence': 0.8,
            'context_used': False
        }
        
        self.manager.update_context("session1", interaction)
        self.manager.update_context("session1", interaction)
        self.manager.update_context("session2", interaction)
        
        stats = self.manager.get_session_stats()
        
        self.assertEqual(stats['active_sessions'], 2)
        self.assertEqual(stats['total_interactions'], 3)
        self.assertEqual(stats['max_history_length'], 3)
        self.assertEqual(stats['session_timeout_minutes'], 1)
    
    def test_reset_session(self):
        """Test resetting a session."""
        session_id = "test-session"
        self.manager.create_session(session_id)
        
        # Add some interactions and preferences
        interaction = {
            'query': 'Test query',
            'response': 'Test response',
            'confidence': 0.8,
            'context_used': False,
            'metadata': {
                'topic': 'test',
                'preferences': {'lang': 'en'}
            }
        }
        
        self.manager.update_context(session_id, interaction)
        
        # Verify context has data
        context = self.manager.get_context(session_id)
        self.assertEqual(len(context.history), 1)
        self.assertEqual(context.current_topic, 'test')
        self.assertEqual(context.user_preferences['lang'], 'en')
        
        # Reset session
        result = self.manager.reset_session(session_id)
        self.assertTrue(result)
        
        # Verify context is cleared
        context = self.manager.get_context(session_id)
        self.assertEqual(len(context.history), 0)
        self.assertIsNone(context.current_topic)
        self.assertEqual(len(context.user_preferences), 0)
    
    def test_get_recent_interactions(self):
        """Test getting recent interactions."""
        session_id = "test-session"
        self.manager.create_session(session_id)
        
        # Add multiple interactions
        for i in range(5):
            interaction = {
                'query': f'Question {i}',
                'response': f'Answer {i}',
                'confidence': 0.8,
                'context_used': False
            }
            self.manager.update_context(session_id, interaction)
        
        # Get recent interactions (should be limited by max_history_length=3)
        recent = self.manager.get_recent_interactions(session_id, count=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[-1]['query'], 'Question 4')  # Most recent
        
        # Test with non-existent session
        recent_empty = self.manager.get_recent_interactions("nonexistent", count=5)
        self.assertEqual(len(recent_empty), 0)
    
    def test_update_user_preferences(self):
        """Test updating user preferences."""
        session_id = "test-session"
        self.manager.create_session(session_id)
        
        # Update preferences
        preferences = {'language': 'en', 'theme': 'dark'}
        result = self.manager.update_user_preferences(session_id, preferences)
        self.assertTrue(result)
        
        context = self.manager.get_context(session_id)
        self.assertEqual(context.user_preferences['language'], 'en')
        self.assertEqual(context.user_preferences['theme'], 'dark')
        
        # Update with additional preferences
        more_preferences = {'notifications': True}
        self.manager.update_user_preferences(session_id, more_preferences)
        
        context = self.manager.get_context(session_id)
        self.assertEqual(context.user_preferences['language'], 'en')  # Should still be there
        self.assertEqual(context.user_preferences['notifications'], True)  # New preference
        
        # Test with non-existent session
        result = self.manager.update_user_preferences("nonexistent", preferences)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()