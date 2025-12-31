"""
Django management command to demonstrate context-aware query processing.
"""

from django.core.management.base import BaseCommand
from datetime import datetime, timedelta

from faq.rag.components.query_processor.query_processor import QueryProcessor
from faq.rag.interfaces.base import ConversationContext


class Command(BaseCommand):
    help = 'Demonstrate context-aware query processing capabilities'

    def handle(self, *args, **options):
        """Run the context-aware processing demonstration."""
        
        self.stdout.write("=== Context-Aware Query Processing Demo ===\n")
        
        # Initialize the query processor
        processor = QueryProcessor()
        
        # Create a conversation context
        context = ConversationContext(
            session_id="demo_session",
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
        
        # Example 1: Context-aware processing
        self.stdout.write("1. Context-Aware Processing")
        self.stdout.write("-" * 30)
        
        query1 = "How long does it take?"
        
        # Process without context
        result_no_context = processor.preprocess_query(query1)
        self.stdout.write(f"Query: '{query1}'")
        self.stdout.write(f"Without context:")
        self.stdout.write(f"  Corrected: '{result_no_context.corrected_query}'")
        self.stdout.write(f"  Intent: {result_no_context.intent}")
        self.stdout.write(f"  Expansions: {result_no_context.expanded_queries[:3]}...")
        
        # Process with context
        result_with_context = processor.process_with_context(query1, context)
        self.stdout.write(f"With context:")
        self.stdout.write(f"  Corrected: '{result_with_context.corrected_query}'")
        self.stdout.write(f"  Intent: {result_with_context.intent}")
        self.stdout.write(f"  Expansions: {result_with_context.expanded_queries[:3]}...")
        self.stdout.write(f"  Confidence boost: {result_with_context.confidence - result_no_context.confidence:.2f}")
        self.stdout.write("")
        
        # Example 2: Ambiguity detection
        self.stdout.write("2. Ambiguity Detection")
        self.stdout.write("-" * 22)
        
        ambiguous_queries = [
            "How do I fix it?",  # Pronoun ambiguity
            "help",  # Incomplete query
            "What about security and privacy?",  # Multiple intent
            "I have an issue with something"  # Vague terms
        ]
        
        for query in ambiguous_queries:
            result = processor.detect_ambiguity(query, context)
            self.stdout.write(f"Query: '{query}'")
            self.stdout.write(f"  Ambiguous: {result['is_ambiguous']}")
            if result['is_ambiguous']:
                self.stdout.write(f"  Type: {result['ambiguity_type']}")
                self.stdout.write(f"  Confidence: {result['confidence']:.2f}")
                if result['clarifying_questions']:
                    self.stdout.write(f"  Clarification: {result['clarifying_questions'][0]}")
            self.stdout.write("")
        
        # Example 3: Follow-up question handling
        self.stdout.write("3. Follow-up Question Handling")
        self.stdout.write("-" * 30)
        
        follow_up_queries = [
            "What about mobile devices?",
            "Can I do it on my phone?",
            "Also, how secure is this process?"
        ]
        
        for query in follow_up_queries:
            result = processor.handle_follow_up_question(query, context)
            self.stdout.write(f"Follow-up: '{query}'")
            self.stdout.write(f"  Enhanced: '{result.corrected_query}'")
            self.stdout.write(f"  Intent: {result.intent}")
            self.stdout.write(f"  Confidence: {result.confidence:.2f}")
            self.stdout.write("")
        
        # Example 4: Pronoun resolution
        self.stdout.write("4. Pronoun Resolution")
        self.stdout.write("-" * 21)
        
        pronoun_query = "How do I configure it properly?"
        enhanced = processor._resolve_pronouns_with_history(pronoun_query, context.history)
        
        self.stdout.write(f"Original: '{pronoun_query}'")
        self.stdout.write(f"Resolved: '{enhanced}'")
        self.stdout.write("")
        
        # Example 5: Context relevance detection
        self.stdout.write("5. Context Relevance Detection")
        self.stdout.write("-" * 29)
        
        test_queries = [
            "How do I use it?",  # Pronoun - should be relevant
            "help",  # Short - should be relevant
            "password reset issues",  # Topic mention - should be relevant
            "How do I create a new user account?"  # Specific - less relevant
        ]
        
        for query in test_queries:
            processed = processor.preprocess_query(query)
            is_relevant = processor._is_context_relevant(processed, context)
            self.stdout.write(f"Query: '{query}'")
            self.stdout.write(f"  Context relevant: {is_relevant}")
            self.stdout.write("")
        
        self.stdout.write(self.style.SUCCESS("Context-aware processing demonstration completed!"))