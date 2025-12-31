import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core import initialize_rag_system

def test_rag():
    print("Initializing RAG System...")
    rag = initialize_rag_system(perform_health_check=False)
    if not rag:
        print("Failed to initialize RAG system.")
        return

    print("\nSystem Health Check:")
    health = rag.health_check()
    print(f"Status: {health.get('overall_status')}")
    for comp, h in health.get('components', {}).items():
        print(f" - {comp}: {h.get('status')}")

    print("\nTesting Query Response...")
    query = "How do I use the chatbot?"
    print(f"Query: {query}")
    try:
        response = rag.answer_query(query)
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence}")
        print(f"Source FAQs: {len(response.source_faqs)}")
    except Exception as e:
        print(f"Error answering query: {e}")

if __name__ == "__main__":
    test_rag()
