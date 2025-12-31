import os
import sys
import django
import logging
from typing import List, Any

# Set up Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core.factory import RAGSystemFactory
from faq.rag.interfaces.base import FAQEntry, Response

def test_ngram_search():
    print("=" * 50)
    print("TESTING N-GRAM SEARCH AND SEMANTIC JUDGE")
    print("=" * 50)
    
    # Initialize RAG System
    print("\n1. Initializing RAG System...")
    factory = RAGSystemFactory()
    rag_system = factory.create_default_system()
    
    # Get stats
    stats = rag_system.vector_store.get_vector_stats()
    print(f"   Vector Store: {stats['total_vectors']} vectors")
    
    # Test Query
    query = "Is the Graphura internship paid?"
    print(f"\n2. Querying: '{query}'")
    
    # We expect:
    # - N-Gram search to find "Who owns Graphura?" (100% overlap)
    # - Gemini to validate it as "YES"
    # - RAG to return the answer
    
    response = rag_system.answer_query(query)
    
    print("\n3. Retrieval Results:")
    source_faqs = getattr(response, 'source_faqs', [])
    print(f"   Number of FAQs retrieved: {len(source_faqs)}")
    
    for i, faq in enumerate(source_faqs):
        print(f"   [{i+1}] {faq.question}")
        print(f"       Category: {faq.category}")
        print(f"       Keywords (N-Grams): {faq.keywords[:5]}... (total {len(faq.keywords)})")

    print("\n4. Final Answer:")
    print(f"   Confidence: {response.confidence:.2f}")
    print(f"   Answer: {response.text}")
    
    if len(source_faqs) > 0 and "Graphura" in response.text:
        print("\n✅ TEST PASSED: High-precision retrieval and validation worked.")
    else:
        print("\n❌ TEST FAILED: Retrieval did not return expected result.")

if __name__ == "__main__":
    # Ensure logs are visible
    logging.basicConfig(level=logging.INFO)
    test_ngram_search()
