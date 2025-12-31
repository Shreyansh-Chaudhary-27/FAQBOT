import os
import sys
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')

import django
django.setup()

from faq.rag.core import initialize_rag_system

print("Testing RAG initialization with local embeddings...")
print("=" * 60)

try:
    rag = initialize_rag_system(perform_health_check=False)
    print("✓ RAG system initialized successfully!")
    print(f"✓ Embedding type: {rag.vectorizer.embedding_generator.config.embedding_type}")
    print(f"✓ Vector dimension: {rag.vectorizer.embedding_generator.config.vector_dimension}")
    print(f"✓ Similarity threshold: {rag.vectorizer.embedding_generator.config.similarity_threshold}")
    
    # Check vector store
    stats = rag.vector_store.get_vector_stats()
    print(f"✓ Total vectors in store: {stats.get('total_vectors', 0)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
