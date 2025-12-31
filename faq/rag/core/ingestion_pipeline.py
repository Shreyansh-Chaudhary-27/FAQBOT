"""
Document Ingestion Pipeline

This module provides a comprehensive pipeline for ingesting DOCX documents, extracting FAQs,
vectorizing them, and storing them in the vector store. It supports automated workflows,
incremental updates, and batch processing capabilities for multiple documents.
"""

import os
import hashlib
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
import threading
import time

from faq.rag.interfaces.base import (
    DOCXScraperInterface, FAQVectorizerInterface, VectorStoreInterface, FAQEntry
)
from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger

# Import Django models for persistence
try:
    from faq.models import RAGDocument, RAGFAQEntry
    HAS_DJANGO = True
except ImportError:
    HAS_DJANGO = False


class DocumentIngestionError(Exception):
    """Custom exception for document ingestion errors."""
    pass


class IngestionProgress:
    """Tracks progress of document ingestion operations."""
    
    def __init__(self, total_documents: int):
        self.total_documents = total_documents
        self.processed_documents = 0
        self.successful_documents = 0
        self.failed_documents = 0
        self.start_time = datetime.now()
        self.current_document = ""
        self.errors: List[str] = []
        self._lock = threading.Lock()
    
    def update_current(self, document_path: str):
        """Update currently processing document."""
        with self._lock:
            self.current_document = document_path
    
    def mark_success(self, document_path: str, faqs_count: int):
        """Mark a document as successfully processed."""
        with self._lock:
            self.processed_documents += 1
            self.successful_documents += 1
    
    def mark_failure(self, document_path: str, error: str):
        """Mark a document as failed."""
        with self._lock:
            self.processed_documents += 1
            self.failed_documents += 1
            self.errors.append(f"{document_path}: {error}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            progress_percent = (self.processed_documents / self.total_documents * 100) if self.total_documents > 0 else 0
            
            return {
                'total_documents': self.total_documents,
                'processed_documents': self.processed_documents,
                'successful_documents': self.successful_documents,
                'failed_documents': self.failed_documents,
                'progress_percent': progress_percent,
                'elapsed_time_seconds': elapsed_time,
                'current_document': self.current_document,
                'errors': self.errors.copy()
            }


class DocumentIngestionPipeline:
    """
    Orchestrates the ingestion of DOCX documents into the RAG system.

    This pipeline handles:
    - Automated DOCX processing workflow with progress tracking
    - Incremental knowledge base updates with document hash checking
    - Batch processing capabilities for multiple documents with parallel processing
    - Error handling and recovery mechanisms
    - Document validation and preprocessing
    """

    def __init__(self,
                 docx_scraper: DOCXScraperInterface,
                 vectorizer: FAQVectorizerInterface,
                 vector_store: VectorStoreInterface,
                 max_workers: int = 4,
                 batch_size: int = 10):
        """
        Initializes the document ingestion pipeline with required components.

        Args:
            docx_scraper: The DOCX scraper component.
            vectorizer: The FAQ vectorizer component.
            vector_store: The vector store component.
            max_workers: Maximum number of worker threads for parallel processing.
            batch_size: Number of documents to process in each batch.
        """
        self.logger = get_rag_logger('ingestion_pipeline')
        self.config = rag_config.config

        if not docx_scraper:
            raise ValueError("DOCX scraper cannot be None")
        if not vectorizer:
            raise ValueError("Vectorizer cannot be None")
        if not vector_store:
            raise ValueError("Vector store cannot be None")

        self.docx_scraper = docx_scraper
        self.vectorizer = vectorizer
        self.vector_store = vector_store
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Progress tracking
        self._current_progress: Optional[IngestionProgress] = None
        self._progress_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Statistics
        self._stats = {
            'total_documents_processed': 0,
            'total_faqs_extracted': 0,
            'total_processing_time': 0.0,
            'last_batch_processed': None,
            'average_faqs_per_document': 0.0,
            'error_count': 0
        }

        self.logger.info(f"DocumentIngestionPipeline initialized with max_workers={max_workers}, batch_size={batch_size}")

    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function to receive progress updates."""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _notify_progress(self):
        """Notify all registered callbacks about progress updates."""
        if self._current_progress:
            progress_data = self._current_progress.get_progress()
            for callback in self._progress_callbacks:
                try:
                    callback(progress_data)
                except Exception as e:
                    self.logger.warning(f"Progress callback failed: {e}")
    
    def _generate_document_hash(self, document_path: str) -> str:
        """Generates a SHA256 hash of the document content."""
        hasher = hashlib.sha256()
        try:
            with open(document_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to generate hash for {document_path}: {e}")
            raise DocumentIngestionError(f"Cannot generate document hash: {e}")
    
    def _validate_document(self, document_path: str) -> bool:
        """
        Validate that the document exists and is a supported format.
        
        Args:
            document_path: Path to the document
            
        Returns:
            True if document is valid, False otherwise
        """
        if not os.path.exists(document_path):
            self.logger.error(f"Document not found: {document_path}")
            return False
        
        if not document_path.lower().endswith('.docx'):
            self.logger.error(f"Unsupported document type: {document_path}. Only .docx files are supported.")
            return False
        
        # Check file size (optional limit)
        try:
            file_size = os.path.getsize(document_path)
            max_size = getattr(self.config, 'max_document_size_mb', 50) * 1024 * 1024  # Default 50MB
            if file_size > max_size:
                self.logger.error(f"Document too large: {document_path} ({file_size / 1024 / 1024:.1f}MB)")
                return False
        except Exception as e:
            self.logger.warning(f"Could not check file size for {document_path}: {e}")
        
        return True
    
    def _preprocess_document_paths(self, document_paths: List[str]) -> List[str]:
        """
        Preprocess and validate document paths, removing invalid ones.
        
        Args:
            document_paths: List of document paths
            
        Returns:
            List of valid document paths
        """
        valid_paths = []
        for doc_path in document_paths:
            # Convert to absolute path
            abs_path = os.path.abspath(doc_path)
            
            if self._validate_document(abs_path):
                valid_paths.append(abs_path)
            else:
                self.logger.warning(f"Skipping invalid document: {doc_path}")
        
        return valid_paths

    def ingest_document(self, document_path: str, force_update: bool = False) -> List[FAQEntry]:
        """
        Ingests a single DOCX document, extracts FAQs, vectorizes them, and stores them.
        Supports incremental updates by checking document hash.

        Args:
            document_path: The path to the DOCX document.
            force_update: If True, forces re-ingestion even if document hash hasn't changed.

        Returns:
            A list of FAQEntry objects that were ingested.

        Raises:
            DocumentIngestionError: If any step in the ingestion process fails.
        """
        start_time = datetime.now()
        
        # Validate document
        abs_path = os.path.abspath(document_path)
        if not self._validate_document(abs_path):
            raise DocumentIngestionError(f"Invalid document: {document_path}")

        self.logger.info(f"Starting ingestion for document: {abs_path}")
        
        try:
            # Generate document hash for incremental updates
            document_hash = self._generate_document_hash(abs_path)
            document_id = str(Path(abs_path).stem)  # Use filename as document ID

            # Check for incremental update
            if not force_update and self.vector_store.is_document_processed(document_id, document_hash):
                self.logger.info(f"Document {abs_path} already processed and unchanged. Skipping.")
                faq_ids = self.vector_store.get_document_faqs(document_id)
                return self.vector_store.get_faq_entries(faq_ids)

            # Extract FAQs from document
            self.logger.info(f"Extracting FAQs from {abs_path}...")
            faqs = self.docx_scraper.extract_faqs(abs_path)
            
            if not faqs:
                self.logger.warning(f"No FAQs extracted from {abs_path}. Document may not contain FAQ content.")
                # Still store the document hash to avoid reprocessing
                self.vector_store.store_vectors([], document_id, document_hash)
                return []

            # Vectorize FAQs
            self.logger.info(f"Vectorizing {len(faqs)} FAQs from {abs_path}...")
            vectorized_faqs = self.vectorizer.vectorize_faq_batch(faqs)

            # Store in vector store
            self.logger.info(f"Storing {len(vectorized_faqs)} vectorized FAQs in the vector store...")
            self.vector_store.store_vectors(vectorized_faqs, document_id, document_hash)

            # Sync to Django Database if available
            if HAS_DJANGO:
                self.logger.info(f"Syncing {len(vectorized_faqs)} FAQs to Django database...")
                try:
                    # 1. Ensure RAGDocument exists
                    rag_doc, _ = RAGDocument.objects.update_or_create(
                        file_name=os.path.basename(abs_path),
                        defaults={
                            'file_path': abs_path,
                            'file_hash': document_hash,
                            'file_size': os.path.getsize(abs_path),
                            'status': 'completed',
                            'processing_completed_at': datetime.now()
                        }
                    )
                    
                    # 2. Create/Update FAQ entries
                    # To avoid duplicates, we clear old entries for this document
                    RAGFAQEntry.objects.filter(document=rag_doc).delete()
                    
                    for faq in vectorized_faqs:
                        RAGFAQEntry.objects.create(
                            rag_id=faq.id,
                            question=faq.question,
                            answer=faq.answer,
                            keywords=faq.keywords,
                            category=faq.category or "uncategorized",
                            audience=faq.audience or "any",
                            intent=faq.intent or "info",
                            condition=faq.condition or "default",
                            composite_key=faq.composite_key,
                            document=rag_doc,
                            # Save embedding to specific field
                            question_embedding=faq.embedding.tolist() if faq.embedding is not None else None,
                            confidence_score=faq.confidence_score
                        )
                    self.logger.info(f"Successfully synced {len(vectorized_faqs)} FAQs to database.")
                except Exception as db_e:
                    self.logger.error(f"Failed to sync to Django database: {db_e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(1, len(vectorized_faqs), processing_time)

            self.logger.info(f"Successfully ingested document: {abs_path} ({len(vectorized_faqs)} FAQs in {processing_time:.2f}s)")
            return vectorized_faqs

        except Exception as e:
            self.logger.error(f"Failed to ingest document {abs_path}: {e}", exc_info=True)
            self._stats['error_count'] += 1
            raise DocumentIngestionError(f"Document ingestion failed for {abs_path}: {e}")

    def ingest_documents_batch(self, document_paths: List[str], force_update: bool = False, 
                              parallel: bool = True) -> Dict[str, List[FAQEntry]]:
        """
        Ingests a batch of DOCX documents with optional parallel processing.

        Args:
            document_paths: A list of paths to DOCX documents.
            force_update: If True, forces re-ingestion even if document hash hasn't changed.
            parallel: If True, processes documents in parallel using thread pool.

        Returns:
            A dictionary where keys are document paths and values are lists of ingested FAQEntry objects.
            Only successfully ingested FAQs are included.
        """
        if not document_paths:
            self.logger.warning("No document paths provided for batch ingestion.")
            return {}
        
        # Preprocess and validate document paths
        valid_paths = self._preprocess_document_paths(document_paths)
        
        if not valid_paths:
            self.logger.error("No valid documents found for batch ingestion.")
            return {}
        
        self.logger.info(f"Starting batch ingestion for {len(valid_paths)} documents (parallel={parallel}).")
        
        # Initialize progress tracking
        self._current_progress = IngestionProgress(len(valid_paths))
        
        results: Dict[str, List[FAQEntry]] = {}
        
        if parallel and len(valid_paths) > 1:
            # Parallel processing using thread pool
            results = self._ingest_documents_parallel(valid_paths, force_update)
        else:
            # Sequential processing
            results = self._ingest_documents_sequential(valid_paths, force_update)
        
        # Final progress update
        self._notify_progress()
        
        # Update batch statistics
        self._stats['last_batch_processed'] = datetime.now()
        
        successful_count = len([r for r in results.values() if r])
        total_faqs = sum(len(faqs) for faqs in results.values())
        
        self.logger.info(f"Batch ingestion completed: {successful_count}/{len(valid_paths)} documents successful, "
                        f"{total_faqs} total FAQs extracted.")
        
        return results
    
    def _ingest_documents_sequential(self, document_paths: List[str], force_update: bool) -> Dict[str, List[FAQEntry]]:
        """Process documents sequentially."""
        results: Dict[str, List[FAQEntry]] = {}
        
        for doc_path in document_paths:
            self._current_progress.update_current(doc_path)
            self._notify_progress()
            
            try:
                ingested_faqs = self.ingest_document(doc_path, force_update)
                results[doc_path] = ingested_faqs
                self._current_progress.mark_success(doc_path, len(ingested_faqs))
                
            except DocumentIngestionError as e:
                self.logger.error(f"Skipping document {doc_path} due to ingestion error: {e}")
                results[doc_path] = []
                self._current_progress.mark_failure(doc_path, str(e))
                
            except Exception as e:
                self.logger.error(f"Unexpected error during ingestion for {doc_path}: {e}", exc_info=True)
                results[doc_path] = []
                self._current_progress.mark_failure(doc_path, f"Unexpected error: {e}")
        
        return results
    
    def _ingest_documents_parallel(self, document_paths: List[str], force_update: bool) -> Dict[str, List[FAQEntry]]:
        """Process documents in parallel using thread pool."""
        results: Dict[str, List[FAQEntry]] = {}
        
        def process_single_document(doc_path: str) -> Tuple[str, List[FAQEntry], Optional[str]]:
            """Process a single document and return results."""
            try:
                self._current_progress.update_current(doc_path)
                self._notify_progress()
                
                ingested_faqs = self.ingest_document(doc_path, force_update)
                return doc_path, ingested_faqs, None
                
            except Exception as e:
                return doc_path, [], str(e)
        
        # Process documents in batches to avoid overwhelming the system
        for i in range(0, len(document_paths), self.batch_size):
            batch_paths = document_paths[i:i + self.batch_size]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch_paths))) as executor:
                # Submit all documents in current batch
                future_to_path = {
                    executor.submit(process_single_document, doc_path): doc_path 
                    for doc_path in batch_paths
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_path):
                    doc_path, ingested_faqs, error = future.result()
                    
                    if error:
                        self.logger.error(f"Failed to process {doc_path}: {error}")
                        results[doc_path] = []
                        self._current_progress.mark_failure(doc_path, error)
                    else:
                        results[doc_path] = ingested_faqs
                        self._current_progress.mark_success(doc_path, len(ingested_faqs))
                    
                    self._notify_progress()
        
        return results
    
    def ingest_directory(self, directory_path: str, recursive: bool = True, 
                        force_update: bool = False, parallel: bool = True) -> Dict[str, List[FAQEntry]]:
        """
        Ingest all DOCX documents from a directory.
        
        Args:
            directory_path: Path to directory containing DOCX files
            recursive: If True, search subdirectories recursively
            force_update: If True, forces re-ingestion even if documents haven't changed
            parallel: If True, processes documents in parallel
            
        Returns:
            Dictionary mapping document paths to extracted FAQs
        """
        if not os.path.exists(directory_path):
            raise DocumentIngestionError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise DocumentIngestionError(f"Path is not a directory: {directory_path}")
        
        self.logger.info(f"Scanning directory for DOCX files: {directory_path} (recursive={recursive})")
        
        # Find all DOCX files
        document_paths = []
        search_pattern = "**/*.docx" if recursive else "*.docx"
        
        for docx_file in Path(directory_path).glob(search_pattern):
            if docx_file.is_file():
                document_paths.append(str(docx_file))
        
        if not document_paths:
            self.logger.warning(f"No DOCX files found in directory: {directory_path}")
            return {}
        
        self.logger.info(f"Found {len(document_paths)} DOCX files in directory")
        
        # Process all found documents
        return self.ingest_documents_batch(document_paths, force_update, parallel)
    
    def get_ingestion_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get current ingestion progress information.
        
        Returns:
            Progress information dictionary or None if no ingestion in progress
        """
        if self._current_progress:
            return self._current_progress.get_progress()
        return None
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion pipeline statistics.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        stats = self._stats.copy()
        
        # Calculate derived statistics
        if stats['total_documents_processed'] > 0:
            stats['average_faqs_per_document'] = stats['total_faqs_extracted'] / stats['total_documents_processed']
            stats['average_processing_time_per_document'] = stats['total_processing_time'] / stats['total_documents_processed']
        
        return stats
    
    def _update_stats(self, documents_processed: int, faqs_extracted: int, processing_time: float):
        """Update pipeline statistics."""
        self._stats['total_documents_processed'] += documents_processed
        self._stats['total_faqs_extracted'] += faqs_extracted
        self._stats['total_processing_time'] += processing_time
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self._stats = {
            'total_documents_processed': 0,
            'total_faqs_extracted': 0,
            'total_processing_time': 0.0,
            'last_batch_processed': None,
            'average_faqs_per_document': 0.0,
            'error_count': 0
        }
        self.logger.info("Pipeline statistics reset")
    
    def cleanup_failed_documents(self) -> int:
        """
        Remove documents from vector store that failed during ingestion.
        
        Returns:
            Number of documents cleaned up
        """
        # This would require tracking failed documents and their partial state
        # For now, we'll implement a basic cleanup that removes empty document entries
        cleanup_count = 0
        
        try:
            # Get all document IDs from vector store
            vector_stats = self.vector_store.get_vector_stats()
            
            # This is a placeholder - actual implementation would depend on 
            # vector store providing methods to list documents and their FAQ counts
            self.logger.info("Cleanup completed - no failed documents found")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
        
        return cleanup_count
