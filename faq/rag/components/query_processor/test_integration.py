"""
Integration tests for Query Processor components

Tests the integration between typo correction and other query processing components.
"""

from django.test import TestCase
from faq.rag.components.query_processor.typo_corrector import TypoCorrector


class TestQueryProcessorIntegration(TestCase):
    """Integration tests for query processor components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.typo_corrector = TypoCorrector()
    
    def test_typo_correction_integration(self):
        """Test that typo correction works as expected for integration."""
        # Test cases that demonstrate the functionality working end-to-end
        test_cases = [
            {
                'input': 'teh quick brown fox',
                'description': 'Basic typo correction'
            },
            {
                'input': 'I dont know waht to do',
                'description': 'Multiple corrections including contractions'
            },
            {
                'input': 'recieve the package',
                'description': 'Common misspelling'
            },
            {
                'input': 'u need to check ur email',
                'description': 'Informal language'
            },
            {
                'input': 'how do i fix this problem',
                'description': 'Correct text (no changes needed)'
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['description']):
                corrected, confidence = self.typo_corrector.correct_typos(case['input'])
                
                # Basic assertions
                self.assertIsInstance(corrected, str, "Should return a string")
                self.assertIsInstance(confidence, float, "Should return a float confidence")
                self.assertGreaterEqual(confidence, 0.0, "Confidence should be non-negative")
                self.assertLessEqual(confidence, 1.0, "Confidence should not exceed 1.0")
                
                # Log the results for verification
                print(f"\nTest: {case['description']}")
                print(f"Input: '{case['input']}'")
                print(f"Output: '{corrected}' (confidence: {confidence:.2f})")
    
    def test_requirements_compliance_integration(self):
        """Test that the implementation meets the specified requirements."""
        
        # Requirement 2.1: Correct common typos and misspellings
        typos_to_test = [
            'teh', 'adn', 'waht', 'recieve', 'seperate', 'definately',
            'dont', 'cant', 'wont', 'u', 'ur'
        ]
        
        corrections_made = 0
        for typo in typos_to_test:
            corrected, confidence = self.typo_corrector.correct_typos(typo)
            if corrected != typo:
                corrections_made += 1
        
        # Should correct most common typos
        correction_rate = corrections_made / len(typos_to_test)
        self.assertGreater(correction_rate, 0.7, 
                          f"Should correct most typos, got {correction_rate:.1%}")
        
        # Requirement 2.5: Provide logging information
        original = "teh quick brown fox"
        corrected, confidence = self.typo_corrector.correct_typos(original)
        
        # Should provide all necessary information for logging
        self.assertNotEqual(original, corrected, "Should make corrections for logging")
        self.assertIsInstance(confidence, float, "Should provide confidence for logging")
        
        # Test the logging confidence method
        log_confidence = self.typo_corrector.get_correction_confidence(original, corrected)
        self.assertIsInstance(log_confidence, float, "Should provide log confidence")
        self.assertGreaterEqual(log_confidence, 0.0, "Log confidence should be non-negative")
        self.assertLessEqual(log_confidence, 1.0, "Log confidence should not exceed 1.0")
    
    def test_edge_cases_integration(self):
        """Test edge cases to ensure robustness."""
        edge_cases = [
            ('', 'Empty string'),
            ('   ', 'Whitespace only'),
            ('a', 'Single character'),
            ('ABC123', 'Mixed case with numbers'),
            ('hello@example.com', 'Email address'),
            ('http://example.com', 'URL'),
            ('!@#$%', 'Special characters only')
        ]
        
        for text, description in edge_cases:
            with self.subTest(case=description):
                try:
                    corrected, confidence = self.typo_corrector.correct_typos(text)
                    
                    # Should handle gracefully without errors
                    self.assertIsInstance(corrected, str, f"Should handle {description}")
                    self.assertIsInstance(confidence, float, f"Should provide confidence for {description}")
                    self.assertGreaterEqual(confidence, 0.0, f"Confidence should be valid for {description}")
                    self.assertLessEqual(confidence, 1.0, f"Confidence should be valid for {description}")
                    
                except Exception as e:
                    self.fail(f"Should handle {description} without error, got: {e}")