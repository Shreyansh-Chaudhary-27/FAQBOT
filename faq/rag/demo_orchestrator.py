#!/usr/bin/env python3
"""
RAG System Orchestrator Demo

This script demonstrates the complete RAG system orchestrator functionality,
including initialization, document processing, query answering, and system monitoring.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_system_initialization():
    """Demonstrate RAG system initialization and validation."""
    print("\n" + "="*60)
    print("RAG SYSTEM ORCHESTRATOR DEMO")
    print("="*60)
    
    print("\n1. Initializing RAG System...")
    print("-" * 40)
    
    try:
        # Initialize the complete RAG system
        rag_system = initialize_rag_system(
            validate_config=True,
            perform_health_check=True
        )
        
        print("✓ RAG system initialized successfully!")
        
        # Display component status
        component_status = rag_system.get_component_status()
        print("\nComponent Status:")
        for component, available in component_status.items():
            status = "✓ Available" if available else "✗ Not Available"
            print(f"  {component}: {status}")
        
        return rag_system
        
    except RAGInitializationError as e:
        print(f"✗ Initialization failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def demo_system_health_check(rag_system):
    """Demonstrate system health checking."""
    print("\n2. System Health Check...")
    print("-" * 40)
    
    try:
        health_results = rag_system.health_check()
        
        print(f"Overall Status: {health_results['overall_status']}")
        print(f"Timestamp: {health_results['timestamp']}")
        
        if health_results.get('issues'):
            print("Issues Found:")
            for issue in health_results['issues']:
                print(f"  - {issue}")
        else:
            print("No issues found!")
        
        print("\nComponent Health:")
        for component, health in health_results.get('components', {}).items():
            status = health.get('status', 'unknown')
            print(f"  {component}: {status}")
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")


def demo_system_statistics(rag_system):
    """Demonstrate system statistics and monitoring."""
    print("\n3. System Statistics...")
    print("-" * 40)
    
    try:
        stats = rag_system.get_system_stats()
        
        print("System Information:")
        system_info = stats.get('system_info', {})
        print(f"  Initialized: {system_info.get('initialized', 'Unknown')}")
        print(f"  Status: {stats.get('system_status', 'Unknown')}")
        
        performance = stats.get('performance_metrics', {})
        print(f"\nPerformance Metrics:")
        print(f"  Queries Processed: {performance.get('queries_processed', 0)}")
        print(f"  Documents Processed: {performance.get('documents_processed', 0)}")
        print(f"  Errors Encountered: {performance.get('errors_encountered', 0)}")
        print(f"  Error Rate: {performance.get('error_rate', 0):.2f}%")
        
        config = stats.get('configuration', {})
        print(f"\nConfiguration:")
        print(f"  Similarity Threshold: {config.get('similarity_threshold', 'N/A')}")
        print(f"  Max Results: {config.get('max_results', 'N/A')}")
        print(f"  Vector Dimension: {config.get('vector_dimension', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Statistics retrieval failed: {e}")


def demo_document_processing(rag_system):
    """Demonstrate document processing capabilities."""
    print("\n4. Document Processing Demo...")
    print("-" * 40)
    
    # Create sample FAQ entries for demonstration
    sample_faqs = [
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
    
    try:
        print(f"Processing {len(sample_faqs)} sample FAQ entries...")
        
        # Update knowledge base with sample FAQs
        rag_system.update_knowledge_base(sample_faqs)
        
        print("✓ Knowledge base updated successfully!")
        
        # Display updated statistics
        stats = rag_system.get_system_stats()
        vector_stats = stats.get('vector_store_stats', {})
        total_vectors = vector_stats.get('total_vectors', 0)
        print(f"  Total vectors in store: {total_vectors}")
        
    except Exception as e:
        print(f"✗ Document processing failed: {e}")


def demo_query_processing(rag_system):
    """Demonstrate query processing and response generation."""
    print("\n5. Query Processing Demo...")
    print("-" * 40)
    
    # Sample queries to test
    test_queries = [
        "How can I reset my password?",
        "I need help contacting support",
        "What time are you open?",
        "How do I change my email address?",  # This should have lower confidence
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 30)
        
        try:
            # Process query with session ID for conversation context
            session_id = f"demo_session_{i}"
            response = rag_system.answer_query(query, session_id=session_id)
            
            print(f"Response: {response.text}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Generation Method: {response.generation_method}")
            print(f"Sources Used: {len(response.source_faqs)}")
            print(f"Context Used: {response.context_used}")
            
        except Exception as e:
            print(f"✗ Query processing failed: {e}")


def demo_conversation_context(rag_system):
    """Demonstrate conversation context management."""
    print("\n6. Conversation Context Demo...")
    print("-" * 40)
    
    session_id = "demo_conversation"
    
    # Simulate a conversation
    conversation_queries = [
        "How do I reset my password?",
        "What if I don't receive the email?",
        "How long does it usually take?"
    ]
    
    for i, query in enumerate(conversation_queries, 1):
        print(f"\nConversation Turn {i}: '{query}'")
        print("-" * 25)
        
        try:
            response = rag_system.answer_query(query, session_id=session_id)
            print(f"Response: {response.text[:100]}...")
            print(f"Context Used: {response.context_used}")
            
        except Exception as e:
            print(f"✗ Conversation processing failed: {e}")


def demo_error_handling(rag_system):
    """Demonstrate error handling and fallback mechanisms."""
    print("\n7. Error Handling Demo...")
    print("-" * 40)
    
    # Test various error scenarios
    error_scenarios = [
        ("", "Empty query"),
        ("askjdhaksjdhaksjdh", "Nonsensical query"),
        ("What is the meaning of life, the universe, and everything?", "Out-of-domain query")
    ]
    
    for query, scenario in error_scenarios:
        print(f"\nScenario: {scenario}")
        print(f"Query: '{query}'")
        print("-" * 20)
        
        try:
            response = rag_system.answer_query(query)
            print(f"Response: {response.text[:100]}...")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Generation Method: {response.generation_method}")
            
        except Exception as e:
            print(f"✗ Error handling failed: {e}")


def main():
    """Main demo function."""
    try:
        # Initialize system
        rag_system = demo_system_initialization()
        
        if rag_system is None:
            print("Cannot continue demo without initialized system")
            return
        
        # Run demonstrations
        demo_system_health_check(rag_system)
        demo_system_statistics(rag_system)
        demo_document_processing(rag_system)
        demo_query_processing(rag_system)
        demo_conversation_context(rag_system)
        demo_error_handling(rag_system)
        
        # Final statistics
        print("\n8. Final System Statistics...")
        print("-" * 40)
        demo_system_statistics(rag_system)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()