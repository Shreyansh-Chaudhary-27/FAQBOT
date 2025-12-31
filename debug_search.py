import os
import django
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core import initialize_rag_system

print("Testing FAQ Search...")
print("=" * 60)

rag = initialize_rag_system(perform_health_check=False)

query = "How do I apply for an internship?"
print(f"\nQuery: {query}")

# Generate query embedding
query_vector = rag.vectorizer.generate_embeddings(query)
print(f"Query vector shape: {query_vector.shape}")
print(f"Query vector norm: {np.linalg.norm(query_vector):.4f}")

# Get all FAQs from vector store
all_faqs = list(rag.vector_store._metadata.values())
print(f"\nTotal FAQs in store: {len(all_faqs)}")

# Search for internship-related FAQs
internship_faqs = [faq for faq in all_faqs if 'internship' in faq.question.lower()]
print(f"FAQs with 'internship': {len(internship_faqs)}")

if internship_faqs:
    print("\nInternship FAQs found:")
    for faq in internship_faqs[:3]:
        print(f"  - {faq.question[:80]}...")
        if faq.embedding is not None:
            sim = cosine_similarity(query_vector.reshape(1, -1), faq.embedding.reshape(1, -1))[0, 0]
            print(f"    Similarity: {sim:.4f}")
            print(f"    Embedding shape: {faq.embedding.shape}")

# Try vector search
print(f"\nVector search with threshold {rag.vectorizer.config['similarity_threshold']}:")
matches = rag.vector_store.search_similar(query_vector, top_k=5)
print(f"Matches found: {len(matches)}")

for i, match in enumerate(matches[:5], 1):
    print(f"\n{i}. Similarity: {match.similarity_score:.4f}")
    print(f"   Question: {match.faq_entry.question[:80]}...")
