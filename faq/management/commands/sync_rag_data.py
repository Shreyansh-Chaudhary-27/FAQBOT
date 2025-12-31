"""
Django management command to sync RAG system data with Django models.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
import os
import logging

from faq.models import RAGDocument, RAGFAQEntry
from faq.rag_django_service import RAGDjangoService


class Command(BaseCommand):
    help = 'Sync RAG system data with Django models'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--document-path',
            type=str,
            help='Process a specific document',
        )
        parser.add_argument(
            '--document-dir',
            type=str,
            help='Process all documents in a directory',
        )
        parser.add_argument(
            '--force-update',
            action='store_true',
            help='Force re-processing of already processed documents',
        )
        parser.add_argument(
            '--cleanup-orphaned',
            action='store_true',
            help='Clean up orphaned records',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting RAG data synchronization...'))
        
        try:
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Create Django service
            django_service = RAGDjangoService()
            
            # Get RAG system from cache or initialize
            rag_system = self._get_rag_system()
            if not rag_system:
                raise CommandError('RAG system not initialized. Run init_rag_system first.')
            
            django_service.rag_system = rag_system
            
            # Perform requested operations
            if options['cleanup_orphaned']:
                self._cleanup_orphaned_records(django_service, options['dry_run'])
            
            if options['document_path']:
                self._process_single_document(
                    django_service, options['document_path'], 
                    options['force_update'], options['dry_run']
                )
            elif options['document_dir']:
                self._process_document_directory(
                    django_service, options['document_dir'], 
                    options['force_update'], options['dry_run']
                )
            else:
                self._sync_existing_data(django_service, options['dry_run'])
            
            self.stdout.write(self.style.SUCCESS('RAG data synchronization completed!'))
            
        except Exception as e:
            logger.error(f"Failed to sync RAG data: {e}")
            raise CommandError(f'Failed to sync RAG data: {e}')
    
    def _get_rag_system(self):
        """Get RAG system from cache."""
        try:
            from django.core.cache import cache
            return cache.get('rag_system')
        except Exception:
            return None
    
    def _process_single_document(self, django_service: RAGDjangoService, 
                               document_path: str, force_update: bool, dry_run: bool):
        """Process a single document."""
        self.stdout.write(f'Processing document: {document_path}')
        
        if not os.path.exists(document_path):
            raise CommandError(f'Document not found: {document_path}')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN: Would process document'))
            return
        
        try:
            # Calculate file hash
            import hashlib
            with open(document_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            file_size = os.path.getsize(document_path)
            file_name = os.path.basename(document_path)
            
            # Check if document already exists
            existing_doc = RAGDocument.objects.filter(file_hash=file_hash).first()
            
            if existing_doc and not force_update:
                self.stdout.write(f'  Document already processed: {existing_doc.file_name}')
                return
            
            # Create or update document record
            if existing_doc:
                document = existing_doc
                document.status = 'pending'
                document.save()
            else:
                document = django_service.create_document_record(
                    file_path=document_path,
                    file_name=file_name,
                    file_hash=file_hash,
                    file_size=file_size
                )
            
            # Process document with RAG system
            django_service.update_document_processing_status(document.id, 'processing')
            
            try:
                faqs = django_service.rag_system.process_document(document_path, force_update)
                
                # Sync FAQs to Django
                django_faqs = django_service.sync_faqs_to_django(faqs, document.id)
                
                django_service.update_document_processing_status(
                    document.id, 'completed', len(django_faqs)
                )
                
                self.stdout.write(
                    self.style.SUCCESS(f'  Processed successfully: {len(django_faqs)} FAQs extracted')
                )
                
            except Exception as e:
                django_service.update_document_processing_status(
                    document.id, 'failed', 0, str(e)
                )
                self.stdout.write(self.style.ERROR(f'  Processing failed: {e}'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  Error processing document: {e}'))
            raise
    
    def _process_document_directory(self, django_service: RAGDjangoService, 
                                  directory_path: str, force_update: bool, dry_run: bool):
        """Process all documents in a directory."""
        self.stdout.write(f'Processing documents in directory: {directory_path}')
        
        if not os.path.exists(directory_path):
            raise CommandError(f'Directory not found: {directory_path}')
        
        # Find all DOCX files
        docx_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.docx'):
                    docx_files.append(os.path.join(root, file))
        
        self.stdout.write(f'Found {len(docx_files)} DOCX files')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN: Would process the following files:'))
            for file_path in docx_files:
                self.stdout.write(f'  - {file_path}')
            return
        
        # Process each file
        processed = 0
        failed = 0
        
        for file_path in docx_files:
            try:
                self._process_single_document(django_service, file_path, force_update, False)
                processed += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to process {file_path}: {e}'))
                failed += 1
        
        self.stdout.write(f'Directory processing complete: {processed} processed, {failed} failed')
    
    def _sync_existing_data(self, django_service: RAGDjangoService, dry_run: bool):
        """Sync existing RAG system data with Django models."""
        self.stdout.write('Syncing existing RAG system data...')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN: Would sync existing data'))
            return
        
        try:
            # Get system statistics
            stats = django_service.rag_system.get_system_stats()
            
            self.stdout.write('Current system statistics:')
            perf_metrics = stats.get('performance_metrics', {})
            self.stdout.write(f'  Queries processed: {perf_metrics.get("queries_processed", 0)}')
            self.stdout.write(f'  Documents processed: {perf_metrics.get("documents_processed", 0)}')
            
            # Get vector store statistics
            vector_stats = stats.get('vector_store_stats', {})
            if vector_stats and 'error' not in vector_stats:
                total_vectors = vector_stats.get('total_vectors', 0)
                self.stdout.write(f'  Total vectors in store: {total_vectors}')
                
                # Check if we have corresponding Django records
                django_faqs = RAGFAQEntry.objects.count()
                self.stdout.write(f'  Django FAQ entries: {django_faqs}')
                
                if total_vectors != django_faqs:
                    self.stdout.write(
                        self.style.WARNING(
                            f'  Mismatch detected: {total_vectors} vectors vs {django_faqs} Django records'
                        )
                    )
            
            # Cleanup expired sessions
            expired_count = django_service.cleanup_expired_sessions()
            if expired_count > 0:
                self.stdout.write(f'  Cleaned up {expired_count} expired sessions')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error syncing existing data: {e}'))
            raise
    
    def _cleanup_orphaned_records(self, django_service: RAGDjangoService, dry_run: bool):
        """Clean up orphaned records."""
        self.stdout.write('Cleaning up orphaned records...')
        
        try:
            # Find documents with no FAQs
            orphaned_docs = RAGDocument.objects.filter(rag_faqs__isnull=True, status='completed')
            orphaned_count = orphaned_docs.count()
            
            if orphaned_count > 0:
                self.stdout.write(f'Found {orphaned_count} documents with no FAQs')
                if not dry_run:
                    orphaned_docs.delete()
                    self.stdout.write(f'  Deleted {orphaned_count} orphaned documents')
                else:
                    self.stdout.write('  DRY RUN: Would delete orphaned documents')
            
            # Find FAQs with no embeddings (older than 1 day)
            from datetime import timedelta
            old_date = timezone.now() - timedelta(days=1)
            faqs_no_embeddings = RAGFAQEntry.objects.filter(
                question_embedding__isnull=True,
                created_at__lt=old_date
            )
            no_embedding_count = faqs_no_embeddings.count()
            
            if no_embedding_count > 0:
                self.stdout.write(f'Found {no_embedding_count} FAQs without embeddings')
                if not dry_run:
                    # In a real implementation, you might want to regenerate embeddings
                    # For now, we'll just report them
                    self.stdout.write('  Consider regenerating embeddings for these FAQs')
                else:
                    self.stdout.write('  DRY RUN: Would handle FAQs without embeddings')
            
            # Cleanup expired sessions
            expired_count = django_service.cleanup_expired_sessions()
            if expired_count > 0:
                self.stdout.write(f'Cleaned up {expired_count} expired sessions')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during cleanup: {e}'))
            raise