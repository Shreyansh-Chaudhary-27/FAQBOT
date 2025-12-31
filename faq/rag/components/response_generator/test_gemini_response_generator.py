"""
Tests for Gemini AI-Powered Response Generator

This module contains tests for the GeminiResponseGenerator class,
verifying AI-powered contextual response generation capabilities.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np

from faq.rag.components.response_generator.gemini_response_generator import (
    GeminiResponseGenerator, 
    GeminiResponseGeneratorError
)
from faq.rag.interfaces.base import FAQEntry, Response, ConversationContext
from faq.rag.components.vectorizer.gemini_service import GeminiServiceError


class TestGeminiResponseGenerator(unittest.TestCase):
    """Test cases for Gemini AI response generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the Gemini service to avoid API calls during testing
        with patch('faq.rag.components.response_generator.gemini_response_generator.GeminiGenerationService'):
            self.generator = GeminiResponseGenerator()
        
        # Create sample FAQ entries
        self.sample_faq = FAQEntry(
            id="faq_1",
            question="How do I reset my password?",
            answer="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your email.",
            keywords=["password", "reset", "login"],
            category="account",
            confidence_score=0.9,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.sample_faqs = [
            self.sample_faq,
            FAQEntry(
                id="faq_2",
                question="How do I change my password?",
                answer="You can change your password in the account settings. Navigate to Profile > Security > Change Password.",
                keywords=["password", "change", "security"],
                category="account",
                confidence_score=0.8,
                source_document="user_guide.docx",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.gemini_service)
        self.assertIsNotNone(self.generator.basic_generator)
        self.assertIsNotNone(self.generator.prompt_templates)
        self.assertIsNotNone(self.generator.tone_analysis_patterns)
    
    def test_analyze_user_tone_formal(self):
        """Test formal tone detection."""
        formal_query = "Could you please help me understand how to reset my password? Thank you."
        tone = self.generator._analyze_user_tone(formal_query)
        self.assertEqual(tone, 'formal')
    
    def test_analyze_user_tone_casual(self):
        """Test casual tone detection."""
        casual_query = "Hey, how do I reset my password?"
        tone = self.generator._analyze_user_tone(casual_query)
        self.assertEqual(tone, 'casual')
    
    def test_analyze_user_tone_urgent(self):
        """Test urgent tone detection."""
        urgent_query = "HELP! I can't access my account!"
        tone = self.generator._analyze_user_tone(urgent_query)
        self.assertEqual(tone, 'urgent')
    
    def test_extract_style_characteristics(self):
        """Test style characteristic extraction."""
        formal_query = "Could you please provide detailed information about password reset procedures?"
        characteristics = self.generator._extract_style_characteristics(formal_query)
        
        self.assertIn('formality', characteristics)
        self.assertIn('verbosity', characteristics)
        self.assertIn('technical_level', characteristics)
        self.assertEqual(characteristics['formality'], 'high')
    
    @patch('faq.rag.components.response_generator.gemini_response_generator.GeminiGenerationService')
    def test_generate_response_single_match(self, mock_service_class):
        """Test response generation for single FAQ match."""
        # Mock the Gemini service
        mock_service = Mock()
        mock_service.generate_response.return_value = "Based on your question about password reset, you can reset your password by going to the login page and clicking 'Forgot Password'."
        mock_service_class.return_value = mock_service
        
        # Create generator with mocked service
        generator = GeminiResponseGenerator()
        generator.gemini_service = mock_service
        
        # Test single match response
        response = generator.generate_response("How do I reset my password?", [self.sample_faq])
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.generation_method, 'rag')
        self.assertTrue(response.metadata['ai_generated'])
        self.assertEqual(len(response.source_faqs), 1)
        self.assertGreater(response.confidence, 0.0)
    
    @patch('faq.rag.components.response_generator.gemini_response_generator.GeminiGenerationService')
    def test_generate_response_multiple_matches(self, mock_service_class):
        """Test response generation for multiple FAQ matches."""
        # Mock the Gemini service
        mock_service = Mock()
        mock_service.generate_response.return_value = "There are two ways to handle your password: you can reset it if you forgot it, or change it if you know the current one."
        mock_service_class.return_value = mock_service
        
        # Create generator with mocked service
        generator = GeminiResponseGenerator()
        generator.gemini_service = mock_service
        
        # Test multiple match response
        response = generator.generate_response("password help", self.sample_faqs)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.generation_method, 'rag')
        self.assertTrue(response.metadata['ai_generated'])
        self.assertEqual(len(response.source_faqs), 2)
        self.assertIn('num_sources', response.metadata)
    
    @patch('faq.rag.components.response_generator.gemini_response_generator.GeminiGenerationService')
    def test_generate_response_no_match(self, mock_service_class):
        """Test response generation when no FAQs match."""
        # Mock the Gemini service
        mock_service = Mock()
        mock_service.generate_response.return_value = "I couldn't find specific information about your question. You might want to contact support for more help."
        mock_service_class.return_value = mock_service
        
        # Create generator with mocked service
        generator = GeminiResponseGenerator()
        generator.gemini_service = mock_service
        
        # Test no match response
        response = generator.generate_response("random question", [])
        
        self.assertIsInstance(response, Response)
        self.assertEqual(len(response.source_faqs), 0)
        self.assertTrue(response.metadata.get('no_match', False))
        self.assertLess(response.confidence, 0.5)
    
    def test_synthesize_multiple_sources_fallback(self):
        """Test multi-source synthesis with fallback to basic generator."""
        # Test with Gemini service error to trigger fallback
        self.generator.gemini_service.generate_response = Mock(side_effect=GeminiServiceError("API error"))
        
        result = self.generator.synthesize_multiple_sources(self.sample_faqs)
        
        # Should fall back to basic generator
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_maintain_context_basic(self):
        """Test basic context maintenance."""
        conversation_history = [
            {"query": "How do I reset my password?", "response": "You can reset it on the login page."},
            {"query": "What about changing it?", "response": "Go to account settings."}
        ]
        
        context = self.generator.maintain_context(conversation_history)
        
        self.assertIsInstance(context, ConversationContext)
        self.assertEqual(len(context.history), 2)
        self.assertIsNotNone(context.session_id)
    
    def test_calculate_confidence_ai_boost(self):
        """Test confidence calculation with AI enhancements."""
        response = Response(
            text="This is a detailed AI-generated response with good content.",
            confidence=0.7,
            source_faqs=[self.sample_faq],
            context_used=True,
            generation_method='rag',
            metadata={'ai_generated': True}
        )
        
        confidence = self.generator.calculate_confidence(response)
        
        # Should be boosted from original 0.7
        self.assertGreaterEqual(confidence, 0.7)
        self.assertLessEqual(confidence, 1.0)
    
    @patch('faq.rag.components.response_generator.gemini_response_generator.GeminiGenerationService')
    def test_generate_contextual_response(self, mock_service_class):
        """Test contextual response generation with conversation history."""
        # Mock the Gemini service
        mock_service = Mock()
        mock_service.generate_response.return_value = "Following up on our previous discussion about passwords, here's how to reset it..."
        mock_service_class.return_value = mock_service
        
        # Create generator with mocked service
        generator = GeminiResponseGenerator()
        generator.gemini_service = mock_service
        
        # Create conversation context
        context = ConversationContext(
            session_id="test_session",
            history=[{"query": "password help", "response": "I can help with passwords."}],
            current_topic="password management",
            user_preferences={},
            last_activity=datetime.now(),
            context_embeddings=[]
        )
        
        response = generator.generate_contextual_response(
            "How do I reset it?", 
            [self.sample_faq], 
            context
        )
        
        self.assertIsInstance(response, Response)
        self.assertTrue(response.context_used)
        self.assertEqual(response.generation_method, 'rag')
        self.assertIn('context_items', response.metadata)
    
    def test_format_faq_sources_for_synthesis(self):
        """Test FAQ source formatting for synthesis."""
        formatted = self.generator._format_faq_sources_for_synthesis(self.sample_faqs)
        
        self.assertIn("Source 1:", formatted)
        self.assertIn("Source 2:", formatted)
        self.assertIn(self.sample_faqs[0].question, formatted)
        self.assertIn(self.sample_faqs[1].answer, formatted)
    
    def test_format_conversation_context(self):
        """Test conversation context formatting."""
        context = ConversationContext(
            session_id="test",
            history=[
                {"query": "first question", "response": "first answer"},
                {"query": "second question", "response": "second answer"}
            ],
            current_topic="test topic",
            user_preferences={},
            last_activity=datetime.now(),
            context_embeddings=[]
        )
        
        formatted = self.generator._format_conversation_context(context)
        
        self.assertIn("Current Topic: test topic", formatted)
        self.assertIn("Exchange 1:", formatted)
        self.assertIn("first question", formatted)
    
    def test_extract_topic_with_ai_fallback(self):
        """Test AI topic extraction with fallback."""
        conversation_history = [
            {"query": "How do I reset my password?"},
            {"query": "What about changing passwords?"}
        ]
        
        # Mock AI service to fail
        self.generator.gemini_service.generate_response = Mock(side_effect=Exception("AI error"))
        
        topic = self.generator._extract_topic_with_ai(conversation_history)
        
        # Should return None on failure
        self.assertIsNone(topic)
    
    def test_analyze_conversation_patterns(self):
        """Test conversation pattern analysis."""
        conversation_history = [
            {"query": "Could you please help me with passwords?"},
            {"query": "Thank you, could you also explain security?"},
            {"query": "I appreciate your detailed explanations."}
        ]
        
        patterns = self.generator._analyze_conversation_patterns(conversation_history)
        
        self.assertIn('preferred_tone', patterns)
        self.assertIn('detail_preference', patterns)
        self.assertIn('technical_level', patterns)
        self.assertEqual(patterns['preferred_tone'], 'formal')
    
    def test_health_check(self):
        """Test health check functionality."""
        # Mock successful health check
        self.generator.gemini_service.health_check = Mock(return_value={'status': 'healthy'})
        
        health = self.generator.health_check()
        
        self.assertIn('status', health)
        self.assertIn('ai_service', health)
        self.assertIn('fallback_available', health)
        self.assertTrue(health['fallback_available'])
    
    def test_fallback_on_gemini_error(self):
        """Test fallback to basic generator on Gemini service error."""
        # Mock Gemini service to raise error
        self.generator.gemini_service.generate_response = Mock(side_effect=GeminiServiceError("API error"))
        
        response = self.generator.generate_response("test query", [self.sample_faq])
        
        # Should still return a valid response via fallback
        self.assertIsInstance(response, Response)
        self.assertIsNotNone(response.text)


if __name__ == '__main__':
    unittest.main()