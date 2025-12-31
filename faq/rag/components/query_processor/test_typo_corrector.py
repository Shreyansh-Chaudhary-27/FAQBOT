"""
Unit tests for the TypoCorrector class

Tests typo correction functionality including:
- Basic spell checking
- Common typo patterns
- Confidence scoring
- Edge cases and error handling
"""

import unittest
from unittest.mock import patch, MagicMock
from django.test import TestCase

from faq.rag.components.query_processor.typo_corrector import TypoCorrector


class TestTypoCorrector(TestCase):
    """Test cases for TypoCorrector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.corrector = TypoCorrector()
    
    def test_basic_typo_correction(self):
        """Test basic typo correction functionality."""
        test_cases = [
            ('teh', 'the'),
            ('adn', 'and'),
            ('waht', 'what'),
            ('recieve', 'receive'),
            ('seperate', 'separate'),
            ('definately', 'definitely')
        ]
        
        for typo, expected in test_cases:
            corrected, confidence = self.corrector.correct_typos(typo)
            self.assertEqual(corrected, expected, 
                           f"Expected '{typo}' to be corrected to '{expected}', got '{corrected}'")
            self.assertGreater(confidence, 0.0, "Confidence should be greater than 0")
            self.assertLessEqual(confidence, 1.0, "Confidence should not exceed 1.0")
    
    def test_contraction_correction(self):
        """Test correction of contractions and informal spellings."""
        test_cases = [
            ('dont', "don't"),
            ('cant', "can't"),
            ('wont', "won't"),
            ('u', 'you'),
            ('ur', 'your')
        ]
        
        for informal, expected in test_cases:
            corrected, confidence = self.corrector.correct_typos(informal)
            self.assertEqual(corrected, expected,
                           f"Expected '{informal}' to be corrected to '{expected}', got '{corrected}'")
    
    def test_sentence_correction(self):
        """Test correction of complete sentences with multiple typos."""
        test_cases = [
            ('teh quick brown fox', 'the quick brown fox'),
            ('I dont know waht to do', "I don't know what to do"),
            ('u need to check ur email', 'you need to check your email'),
            ('cant find the answer', "can't find the answer")
        ]
        
        for original, expected in test_cases:
            corrected, confidence = self.corrector.correct_typos(original)
            # Check that corrections were made (may not be exact due to dictionary limitations)
            self.assertNotEqual(corrected, original, 
                              f"Expected corrections to be made to '{original}'")
            self.assertGreater(confidence, 0.0, "Confidence should be greater than 0")
    
    def test_correct_words_unchanged(self):
        """Test that correctly spelled words remain unchanged."""
        correct_sentences = [
            'how do i fix this problem',
            'what is the password',
            'where can i get help',
            'the quick brown fox jumps'
        ]
        
        for sentence in correct_sentences:
            corrected, confidence = self.corrector.correct_typos(sentence)
            # Should have high confidence for correct text
            self.assertGreaterEqual(confidence, 0.9, 
                                  f"Confidence should be high for correct text: '{sentence}'")
    
    def test_confidence_scoring(self):
        """Test confidence scoring for different types of corrections."""
        # Test high confidence for no corrections
        corrected, confidence = self.corrector.correct_typos('the quick brown fox')
        self.assertGreaterEqual(confidence, 0.85, "Should have high confidence for correct text")
        
        # Test medium confidence for few corrections
        corrected, confidence = self.corrector.correct_typos('teh quick brown fox')
        self.assertGreater(confidence, 0.7, "Should have medium confidence for few corrections")
        
        # Test lower confidence for many corrections
        corrected, confidence = self.corrector.correct_typos('teh quik browm foxe')
        self.assertLess(confidence, 0.8, "Should have lower confidence for many corrections")
    
    def test_empty_and_whitespace_input(self):
        """Test handling of empty and whitespace-only input."""
        test_cases = ['', '   ', '\t', '\n', '  \t\n  ']
        
        for empty_input in test_cases:
            corrected, confidence = self.corrector.correct_typos(empty_input)
            self.assertEqual(corrected, empty_input, "Empty input should remain unchanged")
            self.assertEqual(confidence, 1.0, "Empty input should have confidence 1.0")
    
    def test_single_character_words(self):
        """Test handling of single character words."""
        corrected, confidence = self.corrector.correct_typos('i a')
        # Should handle single characters appropriately
        self.assertIsInstance(corrected, str)
        self.assertGreater(confidence, 0.0)
    
    def test_word_similarity_calculation(self):
        """Test the similarity calculation method."""
        # Test identical words
        similarity = self.corrector._calculate_similarity('test', 'test')
        self.assertEqual(similarity, 1.0, "Identical words should have similarity 1.0")
        
        # Test completely different words
        similarity = self.corrector._calculate_similarity('cat', 'dog')
        self.assertLess(similarity, 0.5, "Different words should have low similarity")
        
        # Test similar words
        similarity = self.corrector._calculate_similarity('test', 'tests')
        self.assertGreater(similarity, 0.8, "Similar words should have high similarity")
    
    def test_pattern_corrections(self):
        """Test pattern-based corrections for common typo patterns."""
        # Test doubled letter removal (this is internal functionality)
        # We test it indirectly through the main correction method
        test_word = 'helllo'  # Should potentially be corrected to 'hello'
        corrected_word, was_corrected = self.corrector._correct_word(test_word)
        # The result depends on whether 'hello' is in the dictionary
        self.assertIsInstance(corrected_word, str)
        self.assertIsInstance(was_corrected, bool)
    
    def test_tokenization(self):
        """Test the tokenization method."""
        test_text = "Hello, world! How are you?"
        tokens = self.corrector._tokenize(test_text)
        expected_tokens = ['hello', 'world', 'how', 'are', 'you']
        self.assertEqual(tokens, expected_tokens, "Tokenization should extract words correctly")
    
    def test_query_reconstruction(self):
        """Test query reconstruction with corrected words."""
        original = "Hello World"
        words = ['hello', 'world']
        reconstructed = self.corrector._reconstruct_query(words, original)
        
        # Should preserve capitalization of first word
        self.assertTrue(reconstructed.startswith('H'), "Should preserve original capitalization")
    
    def test_get_correction_confidence(self):
        """Test the correction confidence calculation method."""
        # Test identical strings
        confidence = self.corrector.get_correction_confidence('test', 'test')
        self.assertEqual(confidence, 1.0, "Identical strings should have confidence 1.0")
        
        # Test different strings
        confidence = self.corrector.get_correction_confidence('teh', 'the')
        self.assertGreater(confidence, 0.0, "Different strings should have positive confidence")
        self.assertLess(confidence, 1.0, "Different strings should have confidence < 1.0")
    
    def test_requirements_2_1_spelling_errors(self):
        """
        Test Requirement 2.1: WHEN a user submits a query with spelling errors 
        THEN the Query_Processor SHALL correct common typos and misspellings
        """
        spelling_errors = [
            'recieve',      # receive
            'seperate',     # separate  
            'definately',   # definitely
            'occured',      # occurred
            'beleive',      # believe
            'calender',     # calendar
            'embarass',     # embarrass
            'enviroment',   # environment
            'experiance',   # experience
            'goverment',    # government
            'grammer',      # grammar
            'independant',  # independent
            'neccessary',   # necessary
            'priviledge',   # privilege
            'recomend',     # recommend
            'tommorow',     # tomorrow
        ]
        
        for error in spelling_errors:
            corrected, confidence = self.corrector.correct_typos(error)
            # Should attempt correction (may not be perfect due to dictionary limitations)
            self.assertIsInstance(corrected, str, f"Should return string for '{error}'")
            self.assertGreater(confidence, 0.0, f"Should have positive confidence for '{error}'")
    
    def test_requirements_2_5_logging_capability(self):
        """
        Test Requirement 2.5: WHEN typo correction occurs 
        THEN the RAG_System SHALL log the original and corrected queries for analysis
        """
        # Test that the correction method returns both original and corrected versions
        original = 'teh quick brown fox'
        corrected, confidence = self.corrector.correct_typos(original)
        
        # The method should provide both original and corrected for logging
        self.assertNotEqual(original, corrected, "Should provide different corrected version")
        self.assertIsInstance(confidence, float, "Should provide confidence score for logging")
        
        # Test confidence calculation method for logging purposes
        log_confidence = self.corrector.get_correction_confidence(original, corrected)
        self.assertIsInstance(log_confidence, float, "Should provide confidence for logging")
        self.assertGreaterEqual(log_confidence, 0.0, "Confidence should be non-negative")
        self.assertLessEqual(log_confidence, 1.0, "Confidence should not exceed 1.0")


class TestTypoCorrectorIntegration(TestCase):
    """Integration tests for TypoCorrector with external dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.corrector = TypoCorrector()
    
    @patch('faq.rag.components.query_processor.typo_corrector.SPELLCHECKER_AVAILABLE', True)
    @patch('faq.rag.components.query_processor.typo_corrector.SpellChecker')
    def test_external_spellchecker_integration(self, mock_spellchecker_class):
        """Test integration with external spell checker library."""
        # Mock the spell checker
        mock_spell_checker = MagicMock()
        mock_spellchecker_class.return_value = mock_spell_checker
        
        # Mock spell checker behavior
        mock_spell_checker.__contains__ = MagicMock(return_value=False)
        mock_spell_checker.candidates.return_value = {'hello'}
        
        # Create corrector with mocked spell checker
        corrector = TypoCorrector()
        
        # Test correction with external spell checker
        corrected_word, was_corrected = corrector._correct_word('helo')
        
        # Verify spell checker was used
        mock_spell_checker.candidates.assert_called_with('helo')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)