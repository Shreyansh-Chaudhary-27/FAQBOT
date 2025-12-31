#!/usr/bin/env python3
"""
Simple RAG System Test (No External APIs)

This script tests the basic RAG system components without relying on external APIs.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')

import django
django.setup()

from faq.rag.interfaces.base import FAQEntry, ProcessedQuery
from faq.rag.components.query_processor.query_processor import QueryProcessor
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_query_processor():
    """Test the query processor component."""
    print("\n1. Testing Query Processor...")

    try:
        processor = QueryProcessor()

        # Test basic preprocessing
        query = "How do I reset my password?"
        processed = processor.preprocess_query(query)

        print(f"✓ Query processed: '{query}' -> '{processed.corrected_query}'")
        print(f"  Intent: {processed.intent}")
        print(f"  Language: {processed.language}")
        print(f"  Confidence: {processed.confidence:.2f}")

        # Test typo correction
        typo_query = "How do I resset my passwrd?"
        corrected = processor.correct_typos(typo_query)
        print(f"✓ Typo correction: '{typo_query}' -> '{corrected}'")

        # Test intent extraction
        intent = processor.extract_intent(query)
        print(f"✓ Intent extraction: '{query}' -> '{intent}'")

        return True

    except Exception as e:
        print(f"✗ Query processor test failed: {e}")
        logger.exception("Query processor test failed")
        return False


def test_faq_interfaces():
    """Test FAQ data structures and interfaces."""
    print("\n2. Testing FAQ Interfaces...")

    try:
        # Create a sample FAQ entry
        faq = FAQEntry(
            id="test_1",
            question="How do I reset my password?",
            answer="Go to login page and click 'Forgot Password'.",
            keywords=["password", "reset", "login"],
            category="authentication",
            confidence_score=0.9,
            source_document="test.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        print("✓ FAQ Entry created successfully")
        print(f"  ID: {faq.id}")
        print(f"  Question: {faq.question}")
        print(f"  Category: {faq.category}")
        print(f"  Keywords: {faq.keywords}")

        # Test ProcessedQuery
        processed = ProcessedQuery(
            original_query="How do I reset my password?",
            corrected_query="How do I reset my password?",
            intent="instruction",
            expanded_queries=["how to reset password", "password reset steps"],
            language="en",
            confidence=0.95,
            embedding=None
        )

        print("✓ ProcessedQuery created successfully")
        print(f"  Original: {processed.original_query}")
        print(f"  Intent: {processed.intent}")
        print(f"  Expansions: {len(processed.expanded_queries)}")

        return True

    except Exception as e:
        print(f"✗ FAQ interfaces test failed: {e}")
        logger.exception("FAQ interfaces test failed")
        return False


def test_imports():
    """Test that all RAG components can be imported."""
    print("\n3. Testing Component Imports...")

    try:
        # Test core imports
        from faq.rag.core.factory import RAGSystemFactory, rag_factory
        print("✓ Core factory imports successful")

        from faq.rag.core.rag_system import RAGSystem
        print("✓ RAG system import successful")

        # Test component imports
        from faq.rag.components.query_processor import ProcessedQuery
        print("✓ Query processor imports successful")

        from faq.rag.components.vectorizer.vectorizer import FAQVectorizer
        print("✓ Vectorizer import successful")

        from faq.rag.components.response_generator.response_generator import ResponseGenerator
        print("✓ Response generator import successful")

        from faq.rag.components.vector_store.vector_store import VectorStore
        print("✓ Vector store import successful")

        return True

    except Exception as e:
        print(f"✗ Component imports test failed: {e}")
        logger.exception("Component imports test failed")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("SIMPLE RAG SYSTEM TEST")
    print("=" * 50)
    print("Testing basic components without external API calls...")

    results = []

    # Run tests
    results.append(("Query Processor", test_query_processor()))
    results.append(("FAQ Interfaces", test_faq_interfaces()))
    results.append(("Component Imports", test_imports()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic RAG components are working correctly!")
        print("The system is ready for use (API rate limits may apply for full functionality).")
        return True
    else:
        print("❌ Some components have issues. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
