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

from faq.rag.core import initialize_rag_system
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_docs():
    print("=" * 50)
    print("INGESTING DOCUMENTS TO RAG SYSTEM")
    print("=" * 50)

    try:
        # 1. Initialize RAG system
        print("\n1. Initializing RAG System...")
        rag_system = initialize_rag_system(
            validate_config=True,
            perform_health_check=False
        )
        print("✓ RAG system initialized successfully!")

        # 2. Path to document
        doc_path = os.path.join(project_root, "AI chatbot.docx")
        if not os.path.exists(doc_path):
            print(f"! Document NOT found at {doc_path}")
            return

        print(f"\n2. Processing document: {doc_path}")
        # 3. Process document (this uses the ingestion pipeline)
        faqs = rag_system.process_document(doc_path, force_update=True)
        print(f"✓ Extracted and ingested {len(faqs)} FAQs from the document.")

        print("\n3. Current Vector Store Stats:")
        stats = rag_system.vector_store.get_vector_stats()
        print(f"Total Vectors: {stats.get('total_vectors')}")
        print(f"Documents Tracked: {len(stats.get('document_ids', []))}")

        print("\n" + "=" * 50)
        print("INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        logger.exception("Ingestion failed")

if __name__ == "__main__":
    ingest_docs()
