"""
Django management command to sync existing FAQs to the RAG system.

This command vectorizes all FAQ entries that don't have embeddings yet
and ensures they are available for semantic search.
"""

from django.core.management.base import BaseCommand
from faq.models import RAGFAQEntry
from faq.rag.core import initialize_rag_system
from faq.rag.interfaces.base import FAQEntry as RAGFAQInterface
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Sync all FAQ entries to the RAG system with vectorization'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-vectorization of all FAQs, even those with existing embeddings',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of FAQs to process in each batch (default: 10)',
        )

    def handle(self, *args, **options):
        force = options['force']
        batch_size = options['batch_size']
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('FAQ to RAG System Synchronization'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        # Initialize RAG system
        self.stdout.write('\n1. Initializing RAG system...')
        try:
            rag_system = initialize_rag_system(
                validate_config=True,
                perform_health_check=False
            )
            self.stdout.write(self.style.SUCCESS('   ✓ RAG system initialized'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Failed to initialize RAG system: {e}'))
            return
        
        # Get FAQs to sync
        self.stdout.write('\n2. Identifying FAQs to sync...')
        if force:
            faqs_to_sync = RAGFAQEntry.objects.all()
            self.stdout.write(f'   - Force mode: syncing ALL {faqs_to_sync.count()} FAQs')
        else:
            faqs_to_sync = RAGFAQEntry.objects.filter(question_embedding__isnull=True)
            total_faqs = RAGFAQEntry.objects.count()
            self.stdout.write(f'   - Found {faqs_to_sync.count()} FAQs without embeddings (out of {total_faqs} total)')
        
        if not faqs_to_sync.exists():
            self.stdout.write(self.style.SUCCESS('\n✓ All FAQs already have embeddings. Nothing to do.'))
            return
        
        # Process FAQs in batches
        self.stdout.write(f'\n3. Processing FAQs in batches of {batch_size}...')
        total_synced = 0
        total_failed = 0
        
        # Convert to list for batching
        faq_list = list(faqs_to_sync)
        total_batches = (len(faq_list) + batch_size - 1) // batch_size
        
        for batch_num in range(0, len(faq_list), batch_size):
            batch = faq_list[batch_num:batch_num + batch_size]
            current_batch = (batch_num // batch_size) + 1
            
            self.stdout.write(f'\n   Batch {current_batch}/{total_batches} ({len(batch)} FAQs)...')
            
            # Convert Django FAQs to RAG FAQs
            rag_faqs = []
            for faq in batch:
                rag_faq = RAGFAQInterface(
                    id=faq.rag_id,
                    question=faq.question,
                    answer=faq.answer,
                    keywords=faq.keywords.split(',') if faq.keywords else [],
                    category=faq.category or "Manual Entry",
                    confidence_score=1.0,
                    source_document="manual_entry",
                    created_at=faq.created_at,
                    updated_at=faq.updated_at,
                    audience=faq.audience or "any",
                    intent=faq.intent or "info",
                    condition=faq.condition or "default"
                )
                rag_faqs.append(rag_faq)
            
            # Vectorize batch
            try:
                self.stdout.write('     - Vectorizing...')
                rag_system.update_knowledge_base(rag_faqs)
                
                # Update Django models with embeddings
                for i, faq in enumerate(batch):
                    if rag_faqs[i].embedding is not None:
                        faq.set_question_embedding_array(rag_faqs[i].embedding)
                        faq.embedding_model = rag_system.vectorizer.embedding_generator.service.__class__.__name__
                        faq.embedding_version = "1.0"
                        faq.save(update_fields=['question_embedding', 'embedding_model', 'embedding_version'])
                
                total_synced += len(batch)
                self.stdout.write(self.style.SUCCESS(f'     ✓ Synced {len(batch)} FAQs'))
                
            except Exception as e:
                total_failed += len(batch)
                self.stdout.write(self.style.ERROR(f'     ✗ Failed to sync batch: {e}'))
                logger.exception(f"Batch sync failed: {e}")
        
        # Summary
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('SYNC COMPLETE'))
        self.stdout.write('=' * 70)
        self.stdout.write(f'Total FAQs synced:  {total_synced}')
        self.stdout.write(f'Total FAQs failed:  {total_failed}')
        
        if total_failed == 0:
            self.stdout.write(self.style.SUCCESS('\n✓ All FAQs successfully synced to RAG system!'))
        else:
            self.stdout.write(self.style.WARNING(f'\n⚠ {total_failed} FAQs failed to sync. Check logs for details.'))
