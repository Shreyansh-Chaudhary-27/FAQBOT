import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core import initialize_rag_system

print("Testing RAG with local embeddings...")
print("=" * 60)

# Initialize RAG
rag = initialize_rag_system(perform_health_check=False)

# Test query
query = "How do I apply for an internship?"
print(f"\nQuery: {query}\n")

try:
    response = rag.answer_query(query)
    print(f"✓ Response generated successfully!")
    print(f"\nAnswer: {response.text}")
    print(f"\nConfidence: {response.confidence}")
    print(f"Source FAQs: {len(response.source_faqs)}")
    
    if response.source_faqs:
        print("\nTop matched FAQs:")
        for i, faq in enumerate(response.source_faqs[:3], 1):
            print(f"{i}. {faq.question[:80]}...")
            
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
