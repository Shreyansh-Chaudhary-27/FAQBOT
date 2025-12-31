import os
import django
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core import get_rag_system
from faq.rag.config.settings import rag_config

def debug_similarity():
    rag = get_rag_system()
    if not rag:
        print("RAG not initialized")
        return

    # Check config
    print(f"Embedding type: {rag_config.config.embedding_type}")
    print(f"Similarity threshold: {rag_config.config.similarity_threshold}")
    print(f"Vector dimension: {rag_config.config.vector_dimension}")

    # Get all FAQs from vector store
    all_faqs = rag.vector_store._metadata.values()
    print(f"Total FAQs in store metadata: {len(all_faqs)}")
    
    if not all_faqs:
        return

    query_text = "How do I use the chatbot?"
    print(f"\nQuery: {query_text}")
    
    # Generate query embedding
    query_vector = rag.vectorizer.generate_embeddings(query_text)
    print(f"Query vector shape: {query_vector.shape}")
    
    # Calculate similarities manually
    similarities = []
    for faq in all_faqs:
        if faq.embedding is not None:
            sim = cosine_similarity(query_vector.reshape(1, -1), faq.embedding.reshape(1, -1))[0, 0]
            similarities.append((faq.question[:50], sim))
    
    # Sort and show top 5
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Manual Similarities:")
    for q, sim in similarities[:5]:
        print(f" - {sim:.4f}: {q}")

if __name__ == "__main__":
    debug_similarity()
