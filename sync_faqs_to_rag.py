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

from faq.models import FAQEntry as DjangoFAQEntry
from faq.rag.core import initialize_rag_system
from faq.rag.interfaces.base import FAQEntry as RAGFAQInterface
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_faqs():
    print("=" * 50)
    print("SYNCING DJANGO FAQS TO RAG SYSTEM")
    print("=" * 50)

    try:
        # 1. Fetch FAQs from Django DB
        django_faqs = DjangoFAQEntry.objects.all()
        print(f"\n1. Found {django_faqs.count()} FAQs in Django database.")
        
        if not django_faqs.exists():
            print("! No FAQs found to sync.")
            return

        # 2. Initialize RAG system
        print("\n2. Initializing RAG System...")
        rag_system = initialize_rag_system(
            validate_config=True,
            perform_health_check=False
        )
        print("✓ RAG system initialized successfully!")

        # 3. Convert to RAG Interface objects
        print("\n3. Converting and Vectorizing FAQs (this may take a while)...")
        rag_faqs = []
        for df in django_faqs:
            rag_faqs.append(RAGFAQInterface(
                id=str(df.id),
                question=df.question,
                answer=df.answer,
                keywords=df.keywords.split(',') if df.keywords else [],
                category="General",
                confidence_score=1.0,
                source_document="django_db",
                created_at=df.created_at,
                updated_at=df.updated_at
            ))
        
        # 4. Update knowledge base (this will trigger embedding generation)
        rag_system.update_knowledge_base(rag_faqs)
        print(f"✓ Successfully ingested and vectorized {len(rag_faqs)} FAQs!")

        print("\n4. Verifying search...")
        # Test query
        test_query = "Where is the verification portal?"
        response = rag_system.answer_query(test_query)
        print(f"Test Query: '{test_query}'")
        print(f"Response: {response.text[:100]}...")
        print(f"Confidence: {response.confidence:.2f}")

        print("\n" + "=" * 50)
        print("SYNC COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"✗ Sync failed: {e}")
        logger.exception("Sync failed")

if __name__ == "__main__":
    sync_faqs()
