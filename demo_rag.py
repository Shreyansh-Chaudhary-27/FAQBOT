#!/usr/bin/env python3
"""
Simple RAG System Demo

This script provides a quick way to test if the RAG system is working properly.
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

from faq.rag.core import initialize_rag_system, RAGInitializationError
from faq.rag.interfaces.base import FAQEntry
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_faqs():
    """Create sample FAQ entries for testing."""
    return [
        FAQEntry(
            id="demo_1",
            question="How do I reset my password?",
            answer="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your email.",
            keywords=["password", "reset", "login", "email"],
            category="authentication",
            confidence_score=0.9,
            source_document="demo_document.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        FAQEntry(
            id="demo_2",
            question="How do I contact support?",
            answer="You can contact our support team by email at support@example.com or by phone at 1-800-SUPPORT. Our support hours are Monday-Friday 9AM-5PM EST.",
            keywords=["support", "contact", "email", "phone"],
            category="support",
            confidence_score=0.95,
            source_document="demo_document.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        FAQEntry(
            id="demo_3",
            question="What are your business hours?",
            answer="Our business hours are Monday through Friday, 9:00 AM to 6:00 PM Eastern Time. We are closed on weekends and major holidays.",
            keywords=["hours", "business", "schedule", "time"],
            category="general",
            confidence_score=0.85,
            source_document="demo_document.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]


def test_rag_system():
    """Test the RAG system with sample data and queries."""
    print("=" * 50)
    print("RAG SYSTEM DEMO")
    print("=" * 50)

    try:
        # Initialize RAG system
        print("\n1. Initializing RAG System...")
        rag_system = initialize_rag_system(
            validate_config=True,
            perform_health_check=True
        )
        print("✓ RAG system initialized successfully!")

        # Add sample FAQs
        print("\n2. Adding sample FAQ data...")
        sample_faqs = create_sample_faqs()
        rag_system.update_knowledge_base(sample_faqs)
        print(f"✓ Added {len(sample_faqs)} sample FAQs")

        # Test queries
        test_queries = [
            "How do I reset my password?",
            "How can I contact support?",
            "What are the business hours?",
            "How do I change my email?",  # Should have lower confidence
        ]

        print("\n3. Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            print("-" * 30)

            try:
                response = rag_system.answer_query(query)
                print(f"Response: {response.text[:150]}...")
                print(f"Confidence: {response.confidence:.2f}")
                print(f"Method: {response.generation_method}")
                print(f"Sources: {len(response.source_faqs)}")

            except Exception as e:
                print(f"✗ Query failed: {e}")

        # Get system stats
        print("\n4. System Statistics...")
        stats = rag_system.get_system_stats()
        print(f"System Status: {stats.get('system_status', 'Unknown')}")
        print(f"Queries Processed: {stats.get('performance_metrics', {}).get('queries_processed', 0)}")

        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("The RAG system is working properly.")
        print("=" * 50)

        return True

    except RAGInitializationError as e:
        print(f"✗ RAG initialization failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        logger.exception("Demo failed")
        return False


if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
