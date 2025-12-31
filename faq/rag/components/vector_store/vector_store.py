"""
Vector Store Implementation

This module provides in-memory vector storage with persistence capabilities,
efficient indexing, and backup/recovery mechanisms for the RAG system.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from threading import Lock
import os

from faq.rag.interfaces.base import (
    VectorStoreInterface, 
    FAQEntry, 
    SimilarityMatch
)
from faq.rag.utils.ngram_utils import get_ngram_overlap


logger = logging.getLogger(__name__)


class VectorStore(VectorStoreInterface):
    """
    In-memory vector store with persistence and efficient similarity search.
    
    Features:
    - In-memory storage for fast access
    - Automatic persistence to disk
    - Vector indexing for efficient retrieval
    - Backup and recovery mechanisms
    - Thread-safe operations
    """
    
    def __init__(self, storage_path: str = "vector_store_data", backup_interval: int = 300):
        """
        Initialize the vector store.
        
        Args:
            storage_path: Directory path for persistent storage
            backup_interval: Backup interval in seconds (default: 5 minutes)
        """
        self.storage_path = Path(storage_path)
        self.backup_interval = backup_interval
        self._lock = Lock()
        
        # In-memory storage
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, FAQEntry] = {}
        self._index_map: Dict[str, int] = {}  # FAQ ID to index mapping
        self._vector_matrix: Optional[np.ndarray] = None  # Stacked vectors for batch operations
        self._needs_rebuild = False
        
        # Document processing tracking
        self._document_hashes: Dict[str, str] = {}  # document_id -> document_hash
        self._document_faqs: Dict[str, List[str]] = {}  # document_id -> list of FAQ IDs
        
        # Statistics
        self._stats = {
            'total_vectors': 0,
            'last_updated': None,
            'last_backup': None,
            'search_count': 0,
            'average_search_time': 0.0
        }
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_from_disk()
        
        logger.info(f"VectorStore initialized with {len(self._vectors)} vectors")
    
    def store_vectors(self, vectors: List[FAQEntry], document_id: Optional[str] = None, document_hash: Optional[str] = None) -> None:
        """
        Store FAQ vectors in the vector database.
        
        Args:
            vectors: List of FAQ entries with embeddings
            document_id: Optional document identifier for tracking
            document_hash: Optional document hash for incremental updates
        """
        with self._lock:
            stored_count = 0
            faq_ids_for_document = []
            
            # If document_id provided, remove existing FAQs from this document first
            if document_id and document_id in self._document_faqs:
                old_faq_ids = self._document_faqs[document_id]
                for old_faq_id in old_faq_ids:
                    if old_faq_id in self._vectors:
                        del self._vectors[old_faq_id]
                    if old_faq_id in self._metadata:
                        del self._metadata[old_faq_id]
                logger.info(f"Removed {len(old_faq_ids)} existing FAQs for document {document_id}")
            
            for faq_entry in vectors:
                if faq_entry.embedding is None:
                    logger.warning(f"FAQ entry {faq_entry.id} has no embedding, skipping")
                    continue
                
                # Store vector and metadata
                self._vectors[faq_entry.id] = faq_entry.embedding.copy()
                self._metadata[faq_entry.id] = faq_entry
                stored_count += 1
                
                if document_id:
                    faq_ids_for_document.append(faq_entry.id)
            
            # Update document tracking
            if document_id:
                self._document_hashes[document_id] = document_hash or ""
                self._document_faqs[document_id] = faq_ids_for_document
                logger.info(f"Tracked {len(faq_ids_for_document)} FAQs for document {document_id}")
            
            # Mark for index rebuild
            self._needs_rebuild = True
            self._stats['total_vectors'] = len(self._vectors)
            self._stats['last_updated'] = datetime.now()
            
            logger.info(f"Stored {stored_count} vectors, total: {len(self._vectors)}")
            
            # Auto-persist if significant changes
            if stored_count > 10:
                self._persist_to_disk()
    
    def search_similar(self, query_vector: np.ndarray, threshold: float = 0.7, top_k: int = 10) -> List[SimilarityMatch]:
        """
        Search for similar vectors above threshold.
        
        Args:
            query_vector: Query embedding vector
            threshold: Minimum similarity threshold (0.0 to 1.0)
            top_k: Maximum number of results to return
            
        Returns:
            List of similarity matches sorted by score (descending)
        """
        return self.batch_search_similar([query_vector], threshold, top_k)[0]
    
    def search_by_ngrams(self, query_ngrams: List[str], threshold: float = 0.9) -> List[SimilarityMatch]:
        """
        Search for FAQs based on N-gram keyword overlap.
        (Requirement: 90% overlap match)
        
        Args:
            query_ngrams: List of N-grams from the query
            threshold: Minimum overlap percentage (default: 0.9)
            
        Returns:
            List of similarity matches meeting the threshold
        """
        if not query_ngrams:
            return []
            
        query_ngram_set = set(query_ngrams)
        matches = []
        
        with self._lock:
            for faq_id, faq_entry in self._metadata.items():
                # FAQ keywords are stored as a list in our new implementation
                faq_keywords = faq_entry.keywords
                if not faq_keywords:
                    continue
                    
                faq_ngram_set = set(faq_keywords)
                # Calculate overlap: (Intersection / query_ngrams_count)
                overlap = get_ngram_overlap(faq_ngram_set, query_ngram_set)
                
                if overlap >= threshold:
                    match = SimilarityMatch(
                        faq_entry=faq_entry,
                        similarity_score=overlap,
                        match_type='keyword_ngram',
                        matched_components=['keywords']
                    )
                    matches.append(match)
        
        # Sort by overlap score descending
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        if matches:
            logger.info(f"N-Gram search found {len(matches)} matches (>= {threshold*100}%)")
            
        return matches
    
    def batch_search_similar(self, query_vectors: List[np.ndarray], threshold: float = 0.7, top_k: int = 10) -> List[List[SimilarityMatch]]:
        """
        Perform batch similarity search for multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            threshold: Minimum similarity threshold (0.0 to 1.0)
            top_k: Maximum number of results to return per query
            
        Returns:
            List of similarity match lists, one for each query vector
        """
        start_time = datetime.now()
        
        with self._lock:
            if not self._vectors or not query_vectors:
                return [[] for _ in query_vectors]
            
            # Rebuild index if needed
            if self._needs_rebuild:
                self._rebuild_index()
            
            # Normalize all query vectors
            normalized_queries = []
            for query_vector in query_vectors:
                query_norm = np.linalg.norm(query_vector)
                if query_norm == 0:
                    logger.warning("Query vector has zero norm")
                    normalized_queries.append(query_vector)
                else:
                    normalized_queries.append(query_vector / query_norm)
            
            # Stack query vectors for batch computation
            query_matrix = np.array(normalized_queries)
            
            # Compute similarities using batch matrix multiplication
            # Shape: (num_queries, num_faqs)
            similarities_matrix = np.dot(query_matrix, self._vector_matrix.T)
            
            # Process results for each query
            all_matches = []
            faq_ids = list(self._vectors.keys())
            
            for query_idx, similarities in enumerate(similarities_matrix):
                # Find matches above threshold
                valid_indices = np.where(similarities >= threshold)[0]
                
                if len(valid_indices) == 0:
                    all_matches.append([])
                    continue
                
                # Sort by similarity (descending) and take top_k
                sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
                top_indices = sorted_indices[:top_k]
                
                # Create similarity matches for this query
                matches = []
                for idx in top_indices:
                    faq_id = faq_ids[idx]
                    faq_entry = self._metadata[faq_id]
                    similarity_score = float(similarities[idx])
                    
                    match = SimilarityMatch(
                        faq_entry=faq_entry,
                        similarity_score=similarity_score,
                        match_type='semantic',
                        matched_components=['embedding']
                    )
                    matches.append(match)
                
                all_matches.append(matches)
            
            # Update statistics
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(search_time)
            
            logger.debug(f"Batch search completed for {len(query_vectors)} queries")
            return all_matches
    
    def search_with_filters(self, query_vector: np.ndarray, threshold: float = 0.7, top_k: int = 10,
                           category_filter: Optional[str] = None, 
                           audience_filter: Optional[str] = None,
                           intent_filter: Optional[str] = None,
                           condition_filter: Optional[str] = None,
                           confidence_filter: Optional[float] = None,
                           keyword_filter: Optional[List[str]] = None) -> List[SimilarityMatch]:
        """
        Search for similar vectors with additional filtering options including composite key components.
        
        Args:
            query_vector: Query embedding vector
            threshold: Minimum similarity threshold (0.0 to 1.0)
            top_k: Maximum number of results to return
            category_filter: Filter by category
            audience_filter: Filter by audience
            intent_filter: Filter by intent
            condition_filter: Filter by condition (supports '*' for any)
            confidence_filter: Minimum extraction confidence score
            keyword_filter: List of keywords that must be present
            
        Returns:
            List of similarity matches meeting all criteria
        """
        start_time = datetime.now()
        
        with self._lock:
            if not self._vectors:
                return []
            
            # Rebuild index if needed
            if self._needs_rebuild:
                self._rebuild_index()
            
            # Get initial similarity matches (get more for filtering)
            matches = self._compute_similarities(query_vector, threshold, top_k * 5)
            
            # Apply filters
            filtered_matches = []
            for match in matches:
                faq_entry = match.faq_entry
                
                # Audience filter
                if audience_filter and audience_filter != 'any':
                    if faq_entry.audience != audience_filter and faq_entry.audience != 'any':
                        continue
                
                # Category filter
                if category_filter and category_filter != 'general':
                    if faq_entry.category != category_filter:
                        continue
                
                # Intent filter
                if intent_filter and intent_filter not in ['information', 'any', 'all']:
                    if faq_entry.intent != intent_filter and faq_entry.intent not in ['information', 'any']:
                        continue
                
                # Condition filter (with wildcard support)
                if condition_filter and condition_filter != '*' and condition_filter != 'default':
                    if faq_entry.condition != condition_filter and faq_entry.condition != 'default':
                        continue
                
                # Confidence filter
                if confidence_filter and faq_entry.confidence_score < confidence_filter:
                    continue
                
                # Keyword filter
                if keyword_filter:
                    faq_keywords = [kw.lower() for kw in faq_entry.keywords]
                    filter_keywords = [kw.lower() for kw in keyword_filter]
                    if not any(kw in faq_keywords for kw in filter_keywords):
                        continue
                
                filtered_matches.append(match)
                
                # Stop when we have enough results
                if len(filtered_matches) >= top_k:
                    break
            
            # Update statistics
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(search_time)
            
            logger.debug(f"Found {len(filtered_matches)} filtered matches")
            return filtered_matches
    
    def search_with_ranking(self, query_vector: np.ndarray, threshold: float = 0.7,
                           top_k: int = 10, boost_recent: bool = False,
                           boost_high_confidence: bool = False) -> List[SimilarityMatch]:
        """
        Search with advanced ranking that considers multiple factors.
        
        Args:
            query_vector: Query embedding vector
            threshold: Minimum similarity threshold (0.0 to 1.0)
            top_k: Maximum number of results to return
            boost_recent: Boost recently updated FAQs
            boost_high_confidence: Boost high-confidence FAQs
            
        Returns:
            List of similarity matches with enhanced ranking
        """
        start_time = datetime.now()
        
        with self._lock:
            if not self._vectors:
                return []
            
            # Get initial similarity matches
            matches = self._compute_similarities(query_vector, threshold, top_k * 2)
            
            # Apply ranking boosts
            for match in matches:
                faq_entry = match.faq_entry
                boost_factor = 1.0
                
                # Recent update boost
                if boost_recent:
                    days_since_update = (datetime.now() - faq_entry.updated_at).days
                    if days_since_update < 30:  # Boost FAQs updated in last 30 days
                        boost_factor *= 1.1
                
                # High confidence boost
                if boost_high_confidence:
                    if faq_entry.confidence_score > 0.8:
                        boost_factor *= 1.05
                
                # Apply boost to similarity score
                match.similarity_score *= boost_factor
            
            # Re-sort by boosted scores and take top_k
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            final_matches = matches[:top_k]
            
            # Update statistics
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(search_time)
            
            logger.debug(f"Found {len(final_matches)} ranked matches")
            return final_matches
    
    def _compute_similarities(self, query_vector: np.ndarray, threshold: float, max_results: int) -> List[SimilarityMatch]:
        """
        Internal method to compute similarities without additional processing.
        
        Args:
            query_vector: Query embedding vector
            threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of similarity matches
        """
        # Rebuild index if needed
        if self._needs_rebuild:
            self._rebuild_index()
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            logger.warning("Query vector has zero norm")
            return []
        
        normalized_query = query_vector / query_norm
        
        # Compute similarities using vectorized operations
        similarities = np.dot(self._vector_matrix, normalized_query)
        
        # Find matches above threshold
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by similarity (descending) and take max_results
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        top_indices = sorted_indices[:max_results]
        
        # Create similarity matches
        matches = []
        faq_ids = list(self._vectors.keys())
        
        for idx in top_indices:
            faq_id = faq_ids[idx]
            faq_entry = self._metadata[faq_id]
            similarity_score = float(similarities[idx])
            
            match = SimilarityMatch(
                faq_entry=faq_entry,
                similarity_score=similarity_score,
                match_type='semantic',
                matched_components=['embedding']
            )
            matches.append(match)
        
        return matches
    
    def update_vector(self, faq_id: str, new_vector: np.ndarray) -> None:
        """
        Update a specific vector in the store.
        
        Args:
            faq_id: FAQ entry ID
            new_vector: New embedding vector
        """
        with self._lock:
            if faq_id not in self._vectors:
                logger.warning(f"FAQ ID {faq_id} not found in vector store")
                return
            
            # Update vector
            self._vectors[faq_id] = new_vector.copy()
            
            # Update metadata timestamp
            if faq_id in self._metadata:
                self._metadata[faq_id].updated_at = datetime.now()
            
            # Mark for index rebuild
            self._needs_rebuild = True
            self._stats['last_updated'] = datetime.now()
            
            logger.info(f"Updated vector for FAQ ID: {faq_id}")
    
    def delete_vector(self, faq_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            faq_id: FAQ entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if faq_id not in self._vectors:
                return False
            
            # Remove from storage
            del self._vectors[faq_id]
            if faq_id in self._metadata:
                del self._metadata[faq_id]
            
            # Remove from document tracking
            for doc_id, faq_list in self._document_faqs.items():
                if faq_id in faq_list:
                    faq_list.remove(faq_id)
                    break
            
            # Mark for index rebuild
            self._needs_rebuild = True
            self._stats['total_vectors'] = len(self._vectors)
            self._stats['last_updated'] = datetime.now()
            
            logger.info(f"Deleted vector for FAQ ID: {faq_id}")
            return True
    
    def clear_all(self) -> bool:
        """
        Clear all data from the vector store, including in-memory and persistent storage.
        
        Returns:
            True if cleared successfully
        """
        logger.info("Clearing all data from vector store")
        with self._lock:
            # Clear in-memory storage
            self._vectors = {}
            self._metadata = {}
            self._index_map = {}
            self._vector_matrix = None
            self._document_hashes = {}
            self._document_faqs = {}
            self._needs_rebuild = False
            
            # Reset statistics
            self._stats = {
                'total_vectors': 0,
                'last_updated': datetime.now(),
                'last_backup': self._stats.get('last_backup'),
                'search_count': 0,
                'average_search_time': 0.0
            }
            
            # Delete persistent files if they exist
            try:
                persistent_files = [
                    "vectors.pkl",
                    "metadata.pkl",
                    "document_hashes.pkl",
                    "document_faqs.pkl"
                ]
                for file_name in persistent_files:
                    file_path = self.storage_path / file_name
                    if file_path.exists():
                        file_path.unlink()
                logger.info("Successfully cleared persistent vector store files")
            except Exception as e:
                logger.error(f"Error deleting persistent files: {e}")
                return False
            
            return True
    
    def is_document_processed(self, document_id: str, document_hash: str) -> bool:
        """
        Check if a document has already been processed with the same hash.
        
        Args:
            document_id: Document identifier
            document_hash: Current document hash
            
        Returns:
            True if document is already processed with same hash, False otherwise
        """
        with self._lock:
            if document_id not in self._document_hashes:
                return False
            
            stored_hash = self._document_hashes[document_id]
            return stored_hash == document_hash
    
    def get_faq_entries(self, faq_ids: List[str]) -> List[FAQEntry]:
        """
        Get FAQEntry objects for the given IDs.
        """
        with self._lock:
            return [self._metadata[fid] for fid in faq_ids if fid in self._metadata]

    def get_document_faqs(self, document_id: str) -> List[str]:
        """
        Get FAQ IDs associated with a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of FAQ IDs for the document
        """
        with self._lock:
            return self._document_faqs.get(document_id, []).copy()
    
    def remove_document(self, document_id: str) -> int:
        """
        Remove all FAQs associated with a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of FAQs removed
        """
        with self._lock:
            if document_id not in self._document_faqs:
                return 0
            
            faq_ids = self._document_faqs[document_id]
            removed_count = 0
            
            for faq_id in faq_ids:
                if faq_id in self._vectors:
                    del self._vectors[faq_id]
                    removed_count += 1
                if faq_id in self._metadata:
                    del self._metadata[faq_id]
            
            # Remove document tracking
            del self._document_faqs[document_id]
            if document_id in self._document_hashes:
                del self._document_hashes[document_id]
            
            # Mark for index rebuild
            if removed_count > 0:
                self._needs_rebuild = True
                self._stats['total_vectors'] = len(self._vectors)
                self._stats['last_updated'] = datetime.now()
            
            logger.info(f"Removed {removed_count} FAQs for document {document_id}")
            return removed_count
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing store statistics
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Add current metrics
            stats.update({
                'storage_path': str(self.storage_path),
                'memory_usage_mb': self._estimate_memory_usage(),
                'index_status': 'needs_rebuild' if self._needs_rebuild else 'current',
                'vector_dimensions': self._get_vector_dimensions(),
            })
            
            return stats
    
    def backup_store(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the vector store.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the created backup
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.storage_path / f"backup_{timestamp}"
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save vectors
            vectors_file = backup_path / "vectors.pkl"
            with open(vectors_file, 'wb') as f:
                pickle.dump(self._vectors, f)
            
            # Save metadata
            metadata_file = backup_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
            
            # Save document tracking data
            doc_hashes_file = backup_path / "document_hashes.pkl"
            with open(doc_hashes_file, 'wb') as f:
                pickle.dump(self._document_hashes, f)
            
            doc_faqs_file = backup_path / "document_faqs.pkl"
            with open(doc_faqs_file, 'wb') as f:
                pickle.dump(self._document_faqs, f)
            
            # Save statistics
            stats_file = backup_path / "stats.json"
            with open(stats_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_stats = {}
                for key, value in self._stats.items():
                    if isinstance(value, datetime):
                        serializable_stats[key] = value.isoformat() if value else None
                    else:
                        serializable_stats[key] = value
                json.dump(serializable_stats, f, indent=2)
            
            self._stats['last_backup'] = datetime.now()
            
        logger.info(f"Backup created at: {backup_path}")
        return str(backup_path)
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore vector store from backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if restoration successful, False otherwise
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup path does not exist: {backup_path}")
            return False
        
        try:
            with self._lock:
                # Load vectors
                vectors_file = backup_path / "vectors.pkl"
                if vectors_file.exists():
                    with open(vectors_file, 'rb') as f:
                        self._vectors = pickle.load(f)
                
                # Load metadata
                metadata_file = backup_path / "metadata.pkl"
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        self._metadata = pickle.load(f)
                
                # Load document tracking data
                doc_hashes_file = backup_path / "document_hashes.pkl"
                if doc_hashes_file.exists():
                    with open(doc_hashes_file, 'rb') as f:
                        self._document_hashes = pickle.load(f)
                else:
                    self._document_hashes = {}
                
                doc_faqs_file = backup_path / "document_faqs.pkl"
                if doc_faqs_file.exists():
                    with open(doc_faqs_file, 'rb') as f:
                        self._document_faqs = pickle.load(f)
                else:
                    self._document_faqs = {}
                
                # Load statistics
                stats_file = backup_path / "stats.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        loaded_stats = json.load(f)
                        # Convert datetime strings back to datetime objects
                        for key, value in loaded_stats.items():
                            if key in ['last_updated', 'last_backup'] and value:
                                try:
                                    self._stats[key] = datetime.fromisoformat(value)
                                except (ValueError, TypeError):
                                    self._stats[key] = None
                            else:
                                self._stats[key] = value
                
                # Mark for index rebuild
                self._needs_rebuild = True
                self._stats['total_vectors'] = len(self._vectors)
                
            logger.info(f"Restored {len(self._vectors)} vectors from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _rebuild_index(self) -> None:
        """Rebuild the vector index for efficient similarity search."""
        if not self._vectors:
            self._vector_matrix = None
            self._index_map = {}
            self._needs_rebuild = False
            return
        
        # Stack all vectors into a matrix
        faq_ids = list(self._vectors.keys())
        vectors = [self._vectors[faq_id] for faq_id in faq_ids]
        
        # Normalize vectors for cosine similarity
        normalized_vectors = []
        for vector in vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vectors.append(vector / norm)
            else:
                normalized_vectors.append(vector)
        
        self._vector_matrix = np.array(normalized_vectors)
        
        # Update index mapping
        self._index_map = {faq_id: idx for idx, faq_id in enumerate(faq_ids)}
        
        self._needs_rebuild = False
        logger.debug(f"Rebuilt vector index with {len(vectors)} vectors")
    
    def _persist_to_disk(self) -> None:
        """Persist current state to disk."""
        try:
            # Save vectors
            vectors_file = self.storage_path / "vectors.pkl"
            with open(vectors_file, 'wb') as f:
                pickle.dump(self._vectors, f)
            
            # Save metadata
            metadata_file = self.storage_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
            
            # Save document tracking data
            doc_hashes_file = self.storage_path / "document_hashes.pkl"
            with open(doc_hashes_file, 'wb') as f:
                pickle.dump(self._document_hashes, f)
            
            doc_faqs_file = self.storage_path / "document_faqs.pkl"
            with open(doc_faqs_file, 'wb') as f:
                pickle.dump(self._document_faqs, f)
            
            logger.debug("Persisted vector store to disk")
            
        except Exception as e:
            logger.error(f"Failed to persist to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load existing data from disk."""
        try:
            # Load vectors
            vectors_file = self.storage_path / "vectors.pkl"
            if vectors_file.exists():
                with open(vectors_file, 'rb') as f:
                    self._vectors = pickle.load(f)
            
            # Load metadata
            metadata_file = self.storage_path / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self._metadata = pickle.load(f)
            
            # Load document tracking data
            doc_hashes_file = self.storage_path / "document_hashes.pkl"
            if doc_hashes_file.exists():
                with open(doc_hashes_file, 'rb') as f:
                    self._document_hashes = pickle.load(f)
            else:
                self._document_hashes = {}
            
            doc_faqs_file = self.storage_path / "document_faqs.pkl"
            if doc_faqs_file.exists():
                with open(doc_faqs_file, 'rb') as f:
                    self._document_faqs = pickle.load(f)
            else:
                self._document_faqs = {}
            
            # Update statistics
            self._stats['total_vectors'] = len(self._vectors)
            if self._vectors:
                self._needs_rebuild = True
            
            logger.debug(f"Loaded {len(self._vectors)} vectors from disk")
            
        except Exception as e:
            logger.warning(f"Failed to load from disk: {e}")
            # Initialize empty storage
            self._vectors = {}
            self._metadata = {}
            self._document_hashes = {}
            self._document_faqs = {}
    
    def _update_search_stats(self, search_time: float) -> None:
        """Update search performance statistics."""
        self._stats['search_count'] += 1
        
        # Update average search time using exponential moving average
        alpha = 0.1  # Smoothing factor
        if self._stats['average_search_time'] == 0.0:
            self._stats['average_search_time'] = search_time
        else:
            self._stats['average_search_time'] = (
                alpha * search_time + 
                (1 - alpha) * self._stats['average_search_time']
            )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = 0
        
        # Estimate vector storage
        for vector in self._vectors.values():
            total_bytes += vector.nbytes
        
        # Estimate metadata (rough approximation)
        total_bytes += len(self._metadata) * 1024  # ~1KB per FAQ entry
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_vector_dimensions(self) -> Optional[int]:
        """Get the dimensions of stored vectors."""
        if not self._vectors:
            return None
        
        # Get dimensions from first vector
        first_vector = next(iter(self._vectors.values()))
        return first_vector.shape[0] if first_vector.ndim == 1 else first_vector.shape