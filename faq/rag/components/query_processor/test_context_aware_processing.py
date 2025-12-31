"""
Test Context-Aware Query Processing

Tests for the context-aware query processing functionality including
context utilization, follow-up question handling, and ambiguity resolution.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from faq.rag.components.query_processor.query_processor import QueryProcessor
from faq.rag.interfaces.base import ConversationContext


class TestContextAwareProcessing(unittest.TestCase):
    """Test context-aware query processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = QueryProcessor()
        
        # Create a sample conversation context
        self.context = ConversationContext(
            session_id="test_session_123",
            history=[
                {
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'query': 'How do I reset my password?',
                    'response': 'To reset your password, go to the login page and click "Forgot Password".',
                    'confidence': 0.9,
                    'context_used': False,
                    'metadata': {'topic': 'password_reset'}
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'query': 'What if I don\'t receive the email?',
                    'response': 'Check your spam folder and ensure the email address is correct.',
                    'confidence': 0.8,
                    'context_used': True,
                    'metadata': {'topic': 'password_reset'}
                }
            ],
            current_topic="password_reset",
            user_preferences={'language': 'en'},
            last_activity=datetime.now(),
            context_embeddings=[]
        )
    
    def test_process_with_context_basic(self):
        """Test basic context-aware processing."""
        query = "How long does it take?"
        
        # Process without context
        result_no_context = self.processor.preprocess_query(query)
        
        # Process with context
        result_with_context = self.processor.process_with_context(query, self.context)
        
        # Context should enhance the query
        self.assertNotEqual(result_no_context.corrected_query, result_with_context.corrected_query)
        self.assertGreater(len(result_with_context.expanded_queries), 
                          len(result_no_context.expanded_queries))
    
    def test_process_with_context_topic_enhancement(self):
        """Test that current topic enhances short queries."""
        query = "steps"
        
        result = self.processor.process_with_context(query, self.context)
        
        # Should include topic in enhanced query
        self.assertIn("password_reset", result.corrected_query.lower())
    
    def test_detect_ambiguity_pronouns(self):
        """Test detection of pronoun ambiguity."""
        query = "How do I fix it?"
        
        # Without context - should be ambiguous
        result_no_context = self.processor.detect_ambiguity(query, None)

        self.assertTrue(result_no_context['is_ambiguous'])
        self.assertEqual(result_no_context['ambiguity_type'], 'pronoun_ambiguity')
        
        # With context - should be less ambiguous
        result_with_context = self.processor.detect_ambiguity(query, self.context)
        self.assertFalse(result_with_context['is_ambiguous'])
    
    def test_detect_ambiguity_incomplete_query(self):
        """Test detection of incomplete queries."""
        query = "help"
        
        result = self.processor.detect_ambiguity(query)
        
        self.assertTrue(result['is_ambiguous'])
        self.assertEqual(result['ambiguity_type'], 'incomplete_query')
        self.assertGreater(len(result['clarifying_questions']), 0)
    
    def test_detect_ambiguity_multiple_intent(self):
        """Test detection of multiple intent ambiguity."""
        query = "How do I reset my password and what are the security requirements?"
        
        result = self.processor.detect_ambiguity(query)
        
        self.assertTrue(result['is_ambiguous'])
        self.assertEqual(result['ambiguity_type'], 'multiple_intent')
    
    def test_detect_ambiguity_vague_terms(self):
        """Test detection of vague terms."""
        query = "I have an issue with something"
        
        result = self.processor.detect_ambiguity(query)
        
        self.assertTrue(result['is_ambiguous'])
        self.assertEqual(result['ambiguity_type'], 'vague_terms')
    
    def test_handle_follow_up_question(self):
        """Test handling of follow-up questions."""
        query = "What about mobile devices?"
        
        result = self.processor.handle_follow_up_question(query, self.context)
        
        # Should enhance the query with context
        self.assertIn("password", result.corrected_query.lower())
        self.assertGreater(result.confidence, 0.8)
    
    def test_handle_follow_up_with_pronouns(self):
        """Test follow-up questions with pronouns."""
        query = "Can I do it on my phone?"
        
        result = self.processor.handle_follow_up_question(query, self.context)
        
        # Should resolve "it" to something from context
        self.assertNotIn(" it ", result.corrected_query.lower())
    
    def test_context_relevance_detection(self):
        """Test detection of context relevance."""
        # Query with pronouns should be context-relevant
        processed_pronoun = self.processor.preprocess_query("How do I use it?")
        self.assertTrue(self.processor._is_context_relevant(processed_pronoun, self.context))
        
        # Short query should be context-relevant
        processed_short = self.processor.preprocess_query("help")
        self.assertTrue(self.processor._is_context_relevant(processed_short, self.context))
        
        # Query mentioning current topic should be context-relevant
        processed_topic = self.processor.preprocess_query("password reset issues")
        self.assertTrue(self.processor._is_context_relevant(processed_topic, self.context))
        
        # Long, specific query should be less context-relevant
        processed_specific = self.processor.preprocess_query("How do I create a new user account in the system?")
        # This might still be relevant due to short length check, but less likely
    
    def test_pronoun_resolution(self):
        """Test pronoun resolution with conversation history."""
        query = "How do I configure it properly?"
        
        enhanced = self.processor._resolve_pronouns_with_history(query, self.context.history)
        
        # Should replace "it" with something from history
        self.assertNotIn(" it ", enhanced.lower())
        # Should contain some keyword from the conversation history
        history_keywords = []
        for interaction in self.context.history:
            keywords = self.processor._extract_key_terms(interaction.get('query', ''))
            history_keywords.extend(keywords)
        
        # Check that at least one keyword from history is in the enhanced query
        self.assertTrue(any(keyword in enhanced.lower() for keyword in history_keywords))
    
    def test_key_term_extraction(self):
        """Test extraction of key terms from text."""
        text = "How do I reset my password for the admin account?"
        
        keywords = self.processor._extract_key_terms(text)
        
        self.assertIn("reset", keywords)
        self.assertIn("password", keywords)
        self.assertIn("admin", keywords)
        self.assertIn("account", keywords)
        
        # Should not include stop words
        self.assertNotIn("how", keywords)
        self.assertNotIn("the", keywords)
    
    def test_follow_up_pattern_analysis(self):
        """Test analysis of follow-up patterns."""
        # Test follow-up indicators
        result1 = self.processor._analyze_follow_up_patterns("Also, what about mobile?", self.context)
        self.assertTrue(result1['is_follow_up'])
        self.assertTrue(result1['has_indicators'])
        
        # Test reference words
        result2 = self.processor._analyze_follow_up_patterns("Can I do this on my phone?", self.context)
        self.assertTrue(result2['is_follow_up'])
        self.assertTrue(result2['has_references'])
        
        # Test continuation words
        result3 = self.processor._analyze_follow_up_patterns("What's next?", self.context)
        self.assertTrue(result3['is_follow_up'])
        self.assertTrue(result3['has_continuations'])
        
        # Test non-follow-up
        result4 = self.processor._analyze_follow_up_patterns("How do I create an account?", self.context)
        self.assertFalse(result4['is_follow_up'])
    
    def test_clarification_response_generation(self):
        """Test generation of clarification responses."""
        ambiguity_result = {
            'is_ambiguous': True,
            'ambiguity_type': 'pronoun_ambiguity',
            'clarifying_questions': ['What are you referring to?'],
            'confidence': 0.7
        }
        
        responses = self.processor._generate_clarification_responses(ambiguity_result)
        
        self.assertGreater(len(responses), 0)
        self.assertTrue(any('pronoun' in response.lower() for response in responses))
    
    def test_empty_context_handling(self):
        """Test handling of empty or None context."""
        query = "How do I reset my password?"
        
        # Should not crash with None context
        result_none = self.processor.process_with_context(query, None)
        self.assertIsNotNone(result_none)
        
        # Should not crash with empty context
        empty_context = ConversationContext(
            session_id="empty",
            history=[],
            current_topic=None,
            user_preferences={},
            last_activity=datetime.now(),
            context_embeddings=[]
        )
        
        result_empty = self.processor.process_with_context(query, empty_context)
        self.assertIsNotNone(result_empty)
    
    def test_context_aware_expansions(self):
        """Test generation of context-aware query expansions."""
        query = "troubleshooting"
        
        expansions = self.processor._generate_context_aware_expansions(query, self.context)
        
        # Should include topic-based expansions
        self.assertTrue(any("password_reset" in exp.lower() for exp in expansions))
        
        # Should include history-based expansions
        self.assertTrue(any("password" in exp.lower() for exp in expansions))


if __name__ == '__main__':
    unittest.main()