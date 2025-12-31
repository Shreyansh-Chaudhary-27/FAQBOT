#!/usr/bin/env python
"""
Example Usage of Context-Aware Query Processing

This example demonstrates how to use the context-aware query processing
functionality including context utilization, follow-up question handling,
and ambiguity resolution.

Run with: python manage.py shell < faq/rag/components/query_processor/example_context_aware_usage.py
"""

from datetime import datetime, timedelta
from faq.rag.components.query_processor.query_processor import QueryProcessor
from faq.rag.interfaces.base import ConversationContext


def demonstrate_context_aware_processing():
    """Demonstrate context-aware query processing capabilities."""
    
    print("=== Context-Aware Query Processing Demo ===\n")
    
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
    print("1. Context-Aware Processing")
    print("-" * 30)
    
    query1 = "How long does it take?"
    
    # Process without context
    result_no_context = processor.preprocess_query(query1)
    print(f"Query: '{query1}'")
    print(f"Without context:")
    print(f"  Corrected: '{result_no_context.corrected_query}'")
    print(f"  Intent: {result_no_context.intent}")
    print(f"  Expansions: {result_no_context.expanded_queries[:3]}...")
    
    # Process with context
    result_with_context = processor.process_with_context(query1, context)
    print(f"With context:")
    print(f"  Corrected: '{result_with_context.corrected_query}'")
    print(f"  Intent: {result_with_context.intent}")
    print(f"  Expansions: {result_with_context.expanded_queries[:3]}...")
    print(f"  Confidence boost: {result_with_context.confidence - result_no_context.confidence:.2f}")
    print()
    
    # Example 2: Ambiguity detection
    print("2. Ambiguity Detection")
    print("-" * 22)
    
    ambiguous_queries = [
        "How do I fix it?",  # Pronoun ambiguity
        "help",  # Incomplete query
        "What about security and privacy?",  # Multiple intent
        "I have an issue with something"  # Vague terms
    ]
    
    for query in ambiguous_queries:
        result = processor.detect_ambiguity(query, context)
        print(f"Query: '{query}'")
        print(f"  Ambiguous: {result['is_ambiguous']}")
        if result['is_ambiguous']:
            print(f"  Type: {result['ambiguity_type']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            if result['clarifying_questions']:
                print(f"  Clarification: {result['clarifying_questions'][0]}")
        print()
    
    # Example 3: Follow-up question handling
    print("3. Follow-up Question Handling")
    print("-" * 30)
    
    follow_up_queries = [
        "What about mobile devices?",
        "Can I do it on my phone?",
        "Also, how secure is this process?"
    ]
    
    for query in follow_up_queries:
        result = processor.handle_follow_up_question(query, context)
        print(f"Follow-up: '{query}'")
        print(f"  Enhanced: '{result.corrected_query}'")
        print(f"  Intent: {result.intent}")
        print(f"  Confidence: {result.confidence:.2f}")
        print()
    
    # Example 4: Pronoun resolution
    print("4. Pronoun Resolution")
    print("-" * 21)
    
    pronoun_query = "How do I configure it properly?"
    enhanced = processor._resolve_pronouns_with_history(pronoun_query, context.history)
    
    print(f"Original: '{pronoun_query}'")
    print(f"Resolved: '{enhanced}'")
    print()
    
    # Example 5: Context relevance detection
    print("5. Context Relevance Detection")
    print("-" * 29)
    
    test_queries = [
        "How do I use it?",  # Pronoun - should be relevant
        "help",  # Short - should be relevant
        "password reset issues",  # Topic mention - should be relevant
        "How do I create a new user account?"  # Specific - less relevant
    ]
    
    for query in test_queries:
        processed = processor.preprocess_query(query)
        is_relevant = processor._is_context_relevant(processed, context)
        print(f"Query: '{query}'")
        print(f"  Context relevant: {is_relevant}")
        print()


# Run the demonstration
demonstrate_context_aware_processing()