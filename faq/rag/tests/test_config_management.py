import os
import django
import sys
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the directory containing the 'backend' Django project to the Python path
DJANGO_PROJECT_ROOT = "c:\\Users\\patel\\FINAL\\backend\\backend"
if DJANGO_PROJECT_ROOT not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    sys.path.append(DJANGO_PROJECT_ROOT)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.config.settings import rag_config, RAGConfigManager
from faq.rag.components.response_generator.gemini_response_generator import GeminiResponseGenerator
from faq.rag.interfaces.base import FAQEntry, Response

class RAGConfigManagementTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Store initial config to restore later
        cls.initial_rag_config_state = rag_config._config
        # Ensure a clean state for tests
        rag_config._config = None
        rag_config.__init__() # Re-initialize to load default/env config

    @classmethod
    def tearDownClass(cls):
        # Restore initial config state
        rag_config._config = cls.initial_rag_config_state

    def setUp(self):
        # Reset config to a known state before each test
        rag_config._config = None
        rag_config.__init__()
        # Set a dummy API key for tests
        os.environ['GEMINI_API_KEY'] = 'test_api_key_123'
        rag_config.reload_config() # Reload with dummy API key

    def tearDown(self):
        # Clean up environment variables set for tests
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']

    @patch('faq.rag.components.vectorizer.gemini_service.GeminiGenerationService.generate_response')
    def test_dynamic_max_response_length_update(self, mock_gemini_generate_response):
        """
        Verify that dynamic updates to max_response_length in RAGConfigManager
        influence GeminiResponseGenerator's output.
        """
        print("\n--- Verifying Dynamic RAG Configuration Update ---")

        # Mock the GeminiService to control its output and check parameters
        mock_gemini_generate_response.return_value = MagicMock(text="Mocked response text.")

        # 1. Get initial max_response_length
        initial_config = rag_config.get_response_config()
        initial_max_response_length = initial_config['max_response_length']
        print(f"Initial max_response_length: {initial_max_response_length}")

        # 2. Instantiate GeminiResponseGenerator
        generator = GeminiResponseGenerator()
        print("GeminiResponseGenerator instantiated.")

        # 3. Simulate a configuration update
        new_max_response_length = initial_max_response_length + 50 if initial_max_response_length < 500 else 100
        print(f"Updating max_response_length to: {new_max_response_length}")
        rag_config.update_config({'max_response_length': new_max_response_length})

        # 4. Get updated config and verify
        updated_config = rag_config.get_response_config()
        print(f"Config after update: {updated_config['max_response_length']}")
        self.assertEqual(updated_config['max_response_length'], new_max_response_length, "Config update failed!")

        # 5. Prepare dummy data for response generation
        query = "What is the capital of France?"
        retrieved_faqs = [
            FAQEntry(
                id="1",
                question="Capital of France",
                answer="The capital of France is Paris. It is known for its art, fashion, gastronomy, and culture.",
                keywords=["Paris", "France", "capital"],
                category="Geography",
                confidence_score=0.9,
                source_document="Wikipedia",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]

        # 6. Generate response and check its length
        print("Generating response with updated configuration...")
        response: Response = generator.generate_response(query, retrieved_faqs)

        # Verify that generate_response was called with the updated max_tokens
        mock_gemini_generate_response.assert_called_once()
        call_args, call_kwargs = mock_gemini_generate_response.call_args
        self.assertIn('max_tokens', call_kwargs)
        self.assertEqual(call_kwargs['max_tokens'], new_max_response_length,
                         "GeminiService.generate_response was not called with the updated max_tokens.")

        print(f"Generated response text: {response.text}")
        print("\n--- Dynamic RAG Configuration Update Verification Complete ---\n")

        # Revert configuration change (optional, but good practice for tests)
        print(f"Reverting max_response_length to initial value: {initial_max_response_length}")
        rag_config.update_config({'max_response_length': initial_max_response_length})
        reverted_config = rag_config.get_response_config()
        print(f"Config after revert: {reverted_config['max_response_length']}")
        self.assertEqual(reverted_config['max_response_length'], initial_max_response_length, "Config revert failed!")

if __name__ == '__main__':
    unittest.main()