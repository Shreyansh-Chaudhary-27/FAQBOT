from django.core.management.base import BaseCommand
from faq.models import RAGFAQEntry, FAQEntry, RAGDocument
from faq.rag.core.factory import RAGSystemFactory
from django.conf import settings
import logging

class Command(BaseCommand):
    help = 'Clear all RAG data from database and vector store to prepare for re-ingestion with composite keys'

    def handle(self, *args, **options):
        self.stdout.write("Starting RAG data cleanup...")
        
        # 1. Clear Database
        try:
            self.stdout.write("Clearing RAGFAQEntry table...")
            rag_count = RAGFAQEntry.objects.all().count()
            RAGFAQEntry.objects.all().delete()
            self.stdout.write(f"Deleted {rag_count} RAGFAQEntry records.")
            
            self.stdout.write("Clearing FAQEntry table...")
            faq_count = FAQEntry.objects.all().count()
            FAQEntry.objects.all().delete()
            self.stdout.write(f"Deleted {faq_count} FAQEntry records.")
            
            self.stdout.write("Clearing RAGDocument table...")
            doc_count = RAGDocument.objects.all().count()
            RAGDocument.objects.all().delete()
            self.stdout.write(f"Deleted {doc_count} RAGDocument records.")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error clearing database: {e}"))

        # 2. Clear Vector Store
        try:
            self.stdout.write("Initializing RAG system to clear vector store...")
            gemini_api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not gemini_api_key:
                self.stdout.write(self.style.WARNING("GEMINI_API_KEY not found in settings. Some components may not initialize."))
            
            # The factory doesn't take arguments in __init__
            factory = RAGSystemFactory()
            # Use create_default_system to get a fully configured system
            rag_system = factory.create_default_system()
            
            if rag_system.vector_store:
                self.stdout.write("Calling clear_all() on vector store...")
                success = rag_system.vector_store.clear_all()
                if success:
                    self.stdout.write(self.style.SUCCESS("Successfully cleared vector store (memory and persistent files)."))
                else:
                    self.stdout.write(self.style.ERROR("Failed to clear vector store."))
            else:
                self.stdout.write(self.style.WARNING("Vector store not available, skipping vector cleanup."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error clearing vector store: {e}"))
            
        self.stdout.write(self.style.SUCCESS("RAG data cleanup operation complete."))
