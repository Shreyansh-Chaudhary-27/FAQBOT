"""
Tests for Basic Response Generator

This module contains unit tests for the BasicResponseGenerator class,
testing template-based response generation, FAQ content integration,
and confidence scoring functionality.
"""

import unittest
from datetime import datetime
import numpy as np

from faq.rag.interfaces.base import FAQEntry, Response
from faq.rag.components.response_generator.response_generator import BasicResponseGenerator


class TestBasicResponseGenerator(unittest.TestCase):
    """Test cases for BasicResponseGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = BasicResponseGenerator()
        
        # Create sample FAQ entries for testing
        self.sample_faq1 = FAQEntry(
            id="faq_1",
            question="How do I reset my password?",
            answer="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your email.",
            keywords=["password", "reset", "login", "email"],
            category="account",
            confidence_score=0.9,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding=np.random.rand(768)
        )
        
        self.sample_faq2 = FAQEntry(
            id="faq_2",
            question="What should I do if I can't access my account?",
            answer="If you can't access your account, first try resetting your password. If that doesn't work, contact our support team at support@example.com.",
            keywords=["account", "access", "support", "password"],
            category="account",
            confidence_score=0.8,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding=np.random.rand(768)
        )
        
        self.sample_faq3 = FAQEntry(
            id="faq_3",
            question="How do I update my profile information?",
            answer="To update your profile, log in to your account and go to Settings > Profile. Make your changes and click Save.",
            keywords=["profile", "update", "settings", "account"],
            category="profile",
            confidence_score=0.85,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding=np.random.rand(768)
        )
    
    def test_single_match_response_generation(self):
        """Test response generation for a single FAQ match."""
        query = "How to reset password?"
        response = self.generator.generate_response(query, [self.sample_faq1])
        
        # Verify response structure
        self.assertIsInstance(response, Response)
        self.assertIn("reset your password", response.text.lower())
        self.assertEqual(len(response.source_faqs), 1)
        self.assertEqual(response.source_faqs[0].id, "faq_1")
        self.assertEqual(response.generation_method, "direct_match")
        self.assertGreater(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
    
    def test_multiple_match_response_generation(self):
        """Test response generation for multiple FAQ matches."""
        query = "Account access issues"
        faqs = [self.sample_faq1, self.sample_faq2]
        response = self.generator.generate_response(query, faqs)
        
        # Verify response structure
        self.assertIsInstance(response, Response)
        self.assertEqual(len(response.source_faqs), 2)
        self.assertEqual(response.generation_method, "synthesized")
        self.assertGreater(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        
        # Should contain information from both FAQs
        self.assertTrue(
            "password" in response.text.lower() or 
            "support" in response.text.lower()
        )
    
    def test_no_match_response_generation(self):
        """Test response generation when no FAQs match."""
        query = "How to delete the universe?"
        response = self.generator.generate_response(query, [])
        
        # Verify fallback response with enhanced uncertainty handling
        self.assertIsInstance(response, Response)
        self.assertEqual(len(response.source_faqs), 0)
        self.assertEqual(response.generation_method, "fallback")
        self.assertLess(response.confidence, 0.5)
        # Updated to match enhanced uncertainty response
        self.assertTrue(
            "don't have" in response.text.lower() or 
            "couldn't find" in response.text.lower() or
            "wasn't able to locate" in response.text.lower()
        )
        # Verify uncertainty handling metadata
        self.assertTrue(response.metadata.get('uncertainty_handled', False))
        self.assertTrue(response.metadata.get('suggestions_provided', False))
    
    def test_synthesize_multiple_sources(self):
        """Test synthesis of information from multiple FAQ sources."""
        faqs = [self.sample_faq1, self.sample_faq2, self.sample_faq3]
        synthesized = self.generator.synthesize_multiple_sources(faqs)
        
        # Verify synthesis
        self.assertIsInstance(synthesized, str)
        self.assertGreater(len(synthesized), 0)
        
        # Should contain information from multiple sources
        # (This is a basic check - in practice, we'd verify semantic content)
        self.assertGreater(len(synthesized.split()), 10)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Create a response with known properties
        response = Response(
            text="Test response",
            confidence=0.0,  # Will be calculated
            source_faqs=[self.sample_faq1],
            context_used=False,
            generation_method="direct_match",
            metadata={}
        )
        
        confidence = self.generator.calculate_confidence(response)
        
        # Verify confidence is within valid range
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(confidence, float)
    
    def test_confidence_calculation_no_sources(self):
        """Test confidence calculation with no source FAQs."""
        response = Response(
            text="Fallback response",
            confidence=0.0,
            source_faqs=[],
            context_used=False,
            generation_method="fallback",
            metadata={}
        )
        
        confidence = self.generator.calculate_confidence(response)
        
        # Should have very low confidence
        self.assertLess(confidence, 0.2)
        self.assertGreater(confidence, 0.0)
    
    def test_maintain_context_basic(self):
        """Test basic conversation context maintenance."""
        conversation_history = [
            {"query": "How to reset password?", "response": "Follow these steps..."},
            {"query": "What if email doesn't arrive?", "response": "Check spam folder..."}
        ]
        
        context = self.generator.maintain_context(conversation_history)
        
        # Verify context structure
        self.assertIsNotNone(context.session_id)
        self.assertEqual(len(context.history), 2)
        self.assertIsInstance(context.last_activity, datetime)
    
    def test_format_response_with_context(self):
        """Test response formatting with conversation context."""
        response = Response(
            text="Here's how to update your profile settings.",
            confidence=0.8,
            source_faqs=[self.sample_faq3],
            context_used=False,
            generation_method="direct_match",
            metadata={}
        )
        
        formatted = self.generator.format_response_with_context(response)
        
        # Should include confidence indicator (updated to match new format)
        self.assertTrue(
            "confidence" in formatted.lower() or 
            "confident" in formatted.lower()
        )
        self.assertGreater(len(formatted), len(response.text))
    
    def test_response_templates_loaded(self):
        """Test that response templates are properly loaded."""
        templates = self.generator.response_templates
        
        # Verify template structure
        self.assertIn('single_match', templates)
        self.assertIn('multiple_matches', templates)
        self.assertIn('no_match', templates)
        
        # Verify confidence levels
        self.assertIn('high_confidence', templates['single_match'])
        self.assertIn('medium_confidence', templates['single_match'])
        self.assertIn('low_confidence', templates['single_match'])
    
    def test_confidence_weights_loaded(self):
        """Test that confidence weights are properly loaded."""
        weights = self.generator.confidence_weights
        
        # Verify all required weights are present
        required_weights = [
            'similarity_score', 'match_completeness', 
            'source_quality', 'content_relevance'
        ]
        
        for weight in required_weights:
            self.assertIn(weight, weights)
            self.assertIsInstance(weights[weight], float)
            self.assertGreaterEqual(weights[weight], 0.0)
            self.assertLessEqual(weights[weight], 1.0)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_formatting_rules_loaded(self):
        """Test that formatting rules are properly loaded."""
        rules = self.generator.formatting_rules
        
        # Verify required rules are present
        required_rules = [
            'max_answer_length', 'max_total_length', 'bullet_threshold',
            'source_attribution', 'confidence_display', 'preserve_formatting'
        ]
        
        for rule in required_rules:
            self.assertIn(rule, rules)
    
    def test_generator_stats(self):
        """Test generator statistics retrieval."""
        stats = self.generator.get_generator_stats()
        
        # Verify stats structure
        self.assertIn('generator_type', stats)
        self.assertIn('templates_loaded', stats)
        self.assertIn('confidence_weights', stats)
        self.assertIn('formatting_rules', stats)
        
        # Verify values
        self.assertEqual(stats['generator_type'], 'basic_template')
        self.assertGreater(stats['templates_loaded'], 0)


if __name__ == '__main__':
    unittest.main()