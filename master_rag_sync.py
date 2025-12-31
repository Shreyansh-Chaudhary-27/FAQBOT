import os
import sys
import hashlib
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')

import django
django.setup()

from faq.models import FAQEntry as DjangoFAQEntry, RAGDocument, RAGFAQEntry
from faq.rag.core import initialize_rag_system
from faq.rag.interfaces.base import FAQEntry as RAGFAQInterface
from faq.rag_django_service import RAGDjangoService
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def master_sync():
    print("=" * 60)
    print("MASTER RAG SYSTEM SYNCHRONIZATION")
    print("=" * 60)

    try:
        # 1. Initialize RAG system
        print("\n1. Initializing RAG System...")
        rag_system = initialize_rag_system(
            validate_config=True,
            perform_health_check=False
        )
        django_service = RAGDjangoService(rag_system)
        print("✓ RAG system initialized.")

        # 2. Ingest Document (AI chatbot.docx)
        print("\n2. Ingesting 'AI chatbot.docx'...")
        doc_path = os.path.join(project_root, "AI chatbot.docx")
        if os.path.exists(doc_path):
            # Create or get RAGDocument record
            file_hash = hashlib.sha256(open(doc_path, 'rb').read()).hexdigest()
            file_name = "AI chatbot.docx"
            file_size = os.path.getsize(doc_path)
            
            try:
                document_record = RAGDocument.objects.get(file_name=file_name)
                # If hash changed, force update
                force = (document_record.file_hash != file_hash)
            except RAGDocument.DoesNotExist:
                document_record = django_service.create_document_record(doc_path, file_name, file_hash, file_size)
                force = True
            
            print(f" - Processing document (force={force})...")
            # Ingest into RAG system
            extracted_faqs = rag_system.process_document(doc_path, force_update=force)
            print(f" ✓ Extracted {len(extracted_faqs)} FAQs from document.")
            
            # Sync to Django DB if new FAQs extracted
            if extracted_faqs:
                django_service.update_document_processing_status(document_record.id, 'processing')
                django_service.sync_faqs_to_django(extracted_faqs, document_record.id)
                django_service.update_document_processing_status(document_record.id, 'completed', faqs_extracted=len(extracted_faqs))
                print(f" ✓ Synced {len(extracted_faqs)} document FAQs to Django DB.")
        else:
            print(" ! 'AI chatbot.docx' not found. Skipping.")

        # 3. Sync Django FAQEntry Table
        print("\n3. Syncing Django 'FAQEntry' table to RAG...")
        django_faqs = DjangoFAQEntry.objects.all()
        if django_faqs.exists():
            print(f" - Found {django_faqs.count()} entries in FAQEntry.")
            rag_faqs = []
            for df in django_faqs:
                rag_faqs.append(RAGFAQInterface(
                    id=f"db_{df.id}",
                    question=df.question,
                    answer=df.answer,
                    keywords=df.keywords.split(',') if df.keywords else [],
                    category="DB_Import",
                    confidence_score=1.0,
                    source_document="django_db",
                    created_at=df.created_at,
                    updated_at=df.updated_at
                ))
            
            # Ingest into RAG system (Vectorize and Store)
            print(" - Vectorizing and storing DB FAQs (this may hit rate limits)...")
            try:
                rag_system.update_knowledge_base(rag_faqs)
                print(f" ✓ Ingested {len(rag_faqs)} DB entries into RAG Vector Store.")
            except Exception as e:
                print(f" ! DB Ingestion failed/interrupted: {e}")
        else:
            print(" - No entries in FAQEntry table.")

        # 4. Final Health Check
        print("\n4. Finalizing RAG System Health...")
        health = rag_system.health_check()
        print(f" Overall Status: {health.get('overall_status')}")
        if health.get('issues'):
            print(f" Issues: {health.get('issues')}")

        print("\n" + "=" * 60)
        print("MASTER SYNC COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Master Sync failed: {e}")
        logger.exception("Master Sync failed")

if __name__ == "__main__":
    master_sync()
