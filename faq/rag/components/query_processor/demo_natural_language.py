#!/usr/bin/env python3
"""
Demo script for Natural Language Understanding capabilities

This script demonstrates the natural language understanding features
implemented in task 3.2, including intent extraction, informal grammar
handling, and query expansion.
"""

import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.components.query_processor.query_processor import QueryProcessor
from faq.rag.components.query_processor.intent_extractor import IntentExtractor


def demo_natural_language_understanding():
    """Demonstrate natural language understanding capabilities."""
    
    print("=" * 80)
    print("RAG FAQ System - Natural Language Understanding Demo")
    print("=" * 80)
    print()
    
    # Initialize components
    query_processor = QueryProcessor()
    intent_extractor = IntentExtractor()
    
    # Test cases demonstrating different capabilities
    test_cases = [
        {
            'category': 'Intent Extraction',
            'queries': [
                'What is machine learning?',
                'How do I reset my password?',
                'I have a problem with my account',
                'Can you help me with this?',
                'Tell me about your services'
            ]
        },
        {
            'category': 'Informal Grammar Normalization',
            'queries': [
                'need help with login',
                'you going to fix this',
                'what about password reset',
                'this broken',
                'wanna know more about pricing'
            ]
        },
        {
            'category': 'Complete Pipeline (Typos + Grammar + Intent)',
            'queries': [
                'whats the best way to reset my pasword',
                'i dont know how fix this problem',
                'need help with login issue',
                'can u tell me about ur services',
                'gonna need some assitance with setup'
            ]
        },
        {
            'category': 'Query Expansion',
            'queries': [
                'reset password',
                'machine learning',
                'login not working'
            ]
        }
    ]
    
    for test_category in test_cases:
        print(f"\n{test_category['category']}")
        print("-" * len(test_category['category']))
        
        for query in test_category['queries']:
            print(f"\nOriginal Query: '{query}'")
            
            if test_category['category'] == 'Intent Extraction':
                # Just show intent extraction
                intent_result = intent_extractor.extract_intent(query)
                print(f"  Intent: {intent_result.intent.value}")
                print(f"  Confidence: {intent_result.confidence:.2f}")
                print(f"  Keywords: {intent_result.keywords[:5]}")  # Show first 5 keywords
                
            elif test_category['category'] == 'Informal Grammar Normalization':
                # Show grammar normalization
                normalized = query_processor.normalize_informal_grammar(query)
                print(f"  Normalized: '{normalized}'")
                
            elif test_category['category'] == 'Query Expansion':
                # Show query expansion
                variations = query_processor.expand_query(query)
                print(f"  Variations ({len(variations)}):")
                for i, variation in enumerate(variations[:5], 1):  # Show first 5
                    print(f"    {i}. {variation}")
                if len(variations) > 5:
                    print(f"    ... and {len(variations) - 5} more")
                    
            else:
                # Show complete pipeline
                processed = query_processor.preprocess_query(query)
                print(f"  Corrected: '{processed.corrected_query}'")
                print(f"  Intent: {processed.intent}")
                print(f"  Language: {processed.language}")
                print(f"  Confidence: {processed.confidence:.2f}")
                print(f"  Expansions: {len(processed.expanded_queries)} variations")
    
    print("\n" + "=" * 80)
    print("Demo completed! The natural language understanding system can:")
    print("✓ Extract intent from user queries")
    print("✓ Handle informal grammar and incomplete sentences")
    print("✓ Correct typos and spelling errors")
    print("✓ Generate query variations for better matching")
    print("✓ Support multi-language detection")
    print("✓ Provide confidence scores for all operations")
    print("=" * 80)


def interactive_demo():
    """Interactive demo where users can test their own queries."""
    
    print("\n" + "=" * 80)
    print("Interactive Natural Language Understanding Demo")
    print("Enter your queries to see how the system processes them.")
    print("Type 'quit' to exit.")
    print("=" * 80)
    
    query_processor = QueryProcessor()
    
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_query:
                print("Please enter a query.")
                continue
            
            print(f"\nProcessing: '{user_query}'")
            print("-" * 50)
            
            # Process the query
            processed = query_processor.preprocess_query(user_query)
            
            # Show results
            print(f"Original:    '{processed.original_query}'")
            print(f"Corrected:   '{processed.corrected_query}'")
            print(f"Intent:      {processed.intent}")
            print(f"Language:    {processed.language}")
            print(f"Confidence:  {processed.confidence:.2f}")
            print(f"Expansions:  {len(processed.expanded_queries)} variations")
            
            # Show some expansions
            if processed.expanded_queries:
                print("\nTop 3 Query Variations:")
                for i, variation in enumerate(processed.expanded_queries[:3], 1):
                    print(f"  {i}. {variation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == '__main__':
    # Run the demo
    demo_natural_language_understanding()
    
    # Ask if user wants interactive demo
    response = input("\nWould you like to try the interactive demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_demo()