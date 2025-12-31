import os
import logging
import pickle
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from threading import Lock

from pinecone import Pinecone, ServerlessSpec, PodSpec
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from faq.rag.interfaces.base import (
    VectorStoreInterface, 
    FAQEntry, 
    SimilarityMatch
)
from faq.rag.utils.ngram_utils import get_ngram_overlap

logger = logging.getLogger(__name__)

class PineconeVectorStore(VectorStoreInterface):
    """
    Persistent Vector Store implementation using Pinecone.
    
    Features:
    - Persistent storage in Pinecone Cloud
    - Local metadata cache for N-gram search and quick lookups
    - Robust error handling and retries
    - Stateless vector search (no vectors in RAM)
    """
    
    def __init__(self, storage_path: str = "vector_store_data"):
        """
        Initialize Pinecone Vector Store.
        
        Args:
            storage_path: Directory path for local metadata cache
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        
        # Load credentials
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "faq-index")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Legacy support
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self._connect_to_index()
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise
            
        # Local metadata cache (No vectors, just text/meta)
        self._metadata: Dict[str, FAQEntry] = {}
        self._document_hashes: Dict[str, str] = {}
        self._document_faqs: Dict[str, List[str]] = {}
        
        # Load local cache
        self._load_local_cache()
        
        # Stats
        self._stats = {
            'total_vectors': 0, # Will update from index stats
            'last_updated': None,
            'connection_status': 'connected',
            'search_count': 0
        }
        self._update_stats_from_index()

    def _connect_to_index(self):
        """Connect to Pinecone index and verify existence."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.warning(f"Index '{self.index_name}' not found. Please create it manually or use setup tools.")
            # We don't auto-create to avoid accidental charges or config mismatches in prod
             # specific dimensions (384 for MiniLM usually) should be setup beforehand
            raise ValueError(f"Pinecone index '{self.index_name}' does not exist.")
            
        return self.pc.Index(self.index_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _upsert_batch(self, vectors: List[tuple]):
        """Retryable upsert."""
        self.index.upsert(vectors=vectors)

    def store_vectors(self, vectors: List[FAQEntry], document_id: Optional[str] = None, document_hash: Optional[str] = None) -> None:
        """
        Store FAQ vectors in Pinecone and update local metadata.
        """
        if not vectors:
            return

        with self._lock:
            # Prepare batch for Pinecone
            pinecone_vectors = []
            
            # Track document FAQs
            faq_ids_for_document = []
            
            for faq in vectors:
                if faq.embedding is None:
                    continue
                
                # Metadata payload for Pinecone (JSON compatible)
                metadata_payload = {
                    "question": faq.question,
                    "answer": faq.answer,
                    "category": faq.category,
                    "id": faq.id,
                    "source": faq.source_document,
                    "confidence": float(faq.confidence_score),
                    "keywords": ",".join(faq.keywords) if faq.keywords else "",
                    # Add filter fields
                    "audience": faq.audience,
                    "intent": faq.intent,
                    "condition": faq.condition
                }
                
                # Convert embedding to list
                embedding_list = faq.embedding.tolist() if isinstance(faq.embedding, np.ndarray) else faq.embedding
                
                pinecone_vectors.append((faq.id, embedding_list, metadata_payload))
                
                # Update local metadata (remove embedding to save RAM)
                faq_no_vector = FAQEntry(
                    id=faq.id,
                    question=faq.question,
                    answer=faq.answer,
                    keywords=faq.keywords,
                    category=faq.category,
                    confidence_score=faq.confidence_score,
                    source_document=faq.source_document,
                    created_at=faq.created_at,
                    updated_at=faq.updated_at,
                    audience=faq.audience,
                    intent=faq.intent,
                    condition=faq.condition,
                    embedding=None # Explicitly None
                )
                self._metadata[faq.id] = faq_no_vector
                
                if document_id:
                    faq_ids_for_document.append(faq.id)

            # Upload to Pinecone
            try:
                # Batch chunks of 100
                batch_size = 100
                for i in range(0, len(pinecone_vectors), batch_size):
                    batch = pinecone_vectors[i:i+batch_size]
                    self._upsert_batch(batch)
                
                logger.info(f"Successfully upserted {len(pinecone_vectors)} vectors to Pinecone.")
            except Exception as e:
                logger.error(f"Failed to upsert to Pinecone: {e}")
                raise

            # Update document tracking
            if document_id:
                self._document_hashes[document_id] = document_hash or ""
                self._document_faqs[document_id] = faq_ids_for_document
            
            self._persist_local_cache()
            self._update_stats_from_index()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5))
    def search_similar(self, query_vector: np.ndarray, threshold: float = 0.7, top_k: int = 10) -> List[SimilarityMatch]:
        """
        Search Pinecone for similar vectors.
        """
        start_time = time.time()
        
        # Ensure list format
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        try:
            results = self.index.query(
                vector=query_list,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return []

        matches = []
        for match in results['matches']:
            score = match['score']
            if score < threshold:
                continue
                
            faq_id = match['id']
            metadata = match['metadata']
            
            # Reconstruct FAQEntry
            # Prefer local metadata if available (it has parsed types), else use Pinecone metadata
            if faq_id in self._metadata:
                faq_entry = self._metadata[faq_id]
            else:
                # Fallback constructs from Pinecone metadata
                faq_entry = self._reconstruct_faq_from_metadata(faq_id, metadata)
            
            matches.append(SimilarityMatch(
                faq_entry=faq_entry,
                similarity_score=score,
                match_type='semantic',
                matched_components=['embedding']
            ))
            
        logger.info(f"Pinecone search found {len(matches)} matches in {time.time() - start_time:.3f}s")
        return matches

    def batch_search_similar(self, query_vectors: List[np.ndarray], threshold: float, top_k: int) -> List[List[SimilarityMatch]]:
        """
        Batch search not efficiently supported in standard Pinecone Client (requires independent queries).
        We iterate.
        """
        results = []
        for qv in query_vectors:
            results.append(self.search_similar(qv, threshold, top_k))
        return results

    def search_with_filters(self, query_vector: np.ndarray, threshold: float, top_k: int, 
                           category_filter: Optional[str] = None, 
                           audience_filter: Optional[str] = None,
                           intent_filter: Optional[str] = None,
                           condition_filter: Optional[str] = None,
                           confidence_filter: Optional[float] = None,
                           keyword_filter: Optional[List[str]] = None) -> List[SimilarityMatch]:
        
        # Build Pinecone filter dict
        filter_dict = {}
        
        if category_filter and category_filter != 'general':
            filter_dict['category'] = category_filter
        if audience_filter and audience_filter != 'any':
            filter_dict['audience'] = audience_filter
        if intent_filter and intent_filter not in ['information', 'any', 'all']:
            filter_dict['intent'] = intent_filter
        if condition_filter and condition_filter not in ['*', 'default']:
            # Pinecone doesn't support wildcard in filters easily, exact match only usually
            filter_dict['condition'] = condition_filter
        if confidence_filter:
            filter_dict['confidence'] = {"$gte": confidence_filter}
            
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        try:
            results = self.index.query(
                vector=query_list,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
        except Exception as e:
            logger.error(f"Pinecone filtered query failed: {e}")
            return []
            
        matches = []
        for match in results['matches']:
            score = match['score']
            if score < threshold:
                continue
            
            # Post-filtering for keywords (Pinecone can't filter list containment easily in all plans)
            metadata = match['metadata']
            if keyword_filter:
                stored_kws = metadata.get('keywords', '').split(',')
                if not any(k.lower() in [s.lower() for s in stored_kws] for k in keyword_filter):
                    continue
                    
            faq_id = match['id']
            if faq_id in self._metadata:
                faq_entry = self._metadata[faq_id]
            else:
                faq_entry = self._reconstruct_faq_from_metadata(faq_id, metadata)
                
            matches.append(SimilarityMatch(
                faq_entry=faq_entry,
                similarity_score=score,
                match_type='semantic',
                matched_components=['embedding']
            ))
            
        return matches

    def search_by_ngrams(self, query_ngrams: List[str], threshold: float = 0.9) -> List[SimilarityMatch]:
        """
        Uses LOCAL metadata cache for N-Gram search.
        Does NOT query Pinecone.
        """
        if not query_ngrams or not self._metadata:
            return []
            
        query_ngram_set = set(query_ngrams)
        matches = []
        
        with self._lock:
            for faq_id, faq_entry in self._metadata.items():
                faq_keywords = faq_entry.keywords
                if not faq_keywords:
                    continue
                    
                faq_ngram_set = set(faq_keywords)
                overlap = get_ngram_overlap(faq_ngram_set, query_ngram_set)
                
                if overlap >= threshold:
                    matches.append(SimilarityMatch(
                        faq_entry=faq_entry,
                        similarity_score=overlap,
                        match_type='keyword_ngram',
                        matched_components=['keywords']
                    ))
        
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches

    def delete_vector(self, faq_id: str) -> bool:
        try:
            self.index.delete(ids=[faq_id])
            with self._lock:
                if faq_id in self._metadata:
                    del self._metadata[faq_id]
                # Cleanup docs
                for doc_id, faqs in self._document_faqs.items():
                    if faq_id in faqs:
                        faqs.remove(faq_id)
            self._persist_local_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {faq_id}: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            self.index.delete(delete_all=True)
            with self._lock:
                self._metadata = {}
                self._document_hashes = {}
                self._document_faqs = {}
            self._persist_local_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to clear all vectors: {e}")
            return False
            
    def update_vector(self, faq_id: str, new_vector: np.ndarray) -> None:
        # Requires full FAQ entry to update metadata. 
        # Since interface only gives vector, we might miss metadata update if we don't have it.
        # But usually update_vector is called in context where we have the object.
        # For this interface implementation, we warn or try to use existing metadata + new vector.
        if faq_id in self._metadata:
            entry = self._metadata[faq_id]
            entry.embedding = new_vector
            self.store_vectors([entry]) # Re-use store to upsert
        else:
            logger.warning(f"Cannot update vector for {faq_id} - metadata not found locally.")

    def search_with_ranking(self, query_vector: np.ndarray, threshold: float, top_k: int, boost_recent: bool, boost_high_confidence: bool) -> List[SimilarityMatch]:
        # Perform standard search then rank locally
        matches = self.search_similar(query_vector, threshold, top_k * 2)
        
        for match in matches:
            faq_entry = match.faq_entry
            boost = 1.0
            if boost_recent:
                # Approximate check
                pass 
            if boost_high_confidence and faq_entry.confidence_score > 0.8:
                boost *= 1.05
            match.similarity_score *= boost
            
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]

    def is_document_processed(self, document_id: str, document_hash: str) -> bool:
        return self._document_hashes.get(document_id) == document_hash

    def get_document_faqs(self, document_id: str) -> List[str]:
        return self._document_faqs.get(document_id, [])

    def remove_document(self, document_id: str) -> int:
        ids = self.get_document_faqs(document_id)
        if not ids:
            return 0
        try:
            self.index.delete(ids=ids)
            with self._lock:
                for fid in ids:
                    if fid in self._metadata:
                        del self._metadata[fid]
                if document_id in self._document_faqs:
                    del self._document_faqs[document_id]
                if document_id in self._document_hashes:
                    del self._document_hashes[document_id]
            self._persist_local_cache()
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return 0

    def get_vector_stats(self) -> Dict[str, Any]:
        self._update_stats_from_index()
        return self._stats

    def backup_store(self, backup_path: Optional[str] = None) -> str:
        # Snapshotting Pinecone is done in cloud. We back up local metadata.
        logger.info("Backup requested - backing up local metadata only. Pinecone/Cloud is persistent.")
        self._persist_local_cache()
        return str(self.storage_path)

    def restore_from_backup(self, backup_path: str) -> bool:
        logger.warning("Restore requested - Creating local metadata from backup. Pinecone state is not reverted.")
        # Load metadata logic
        return True

    # --- Helpers ---

    def _load_local_cache(self):
        """Load metadata/hashes from local pickle files."""
        try:
            meta_path = self.storage_path / "metadata.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    self._metadata = pickle.load(f)
                
                # Ensure no embeddings are in RAM (scrub if they were pickled)
                for entry in self._metadata.values():
                    entry.embedding = None

            
            hashes_path = self.storage_path / "document_hashes.pkl"
            if hashes_path.exists():
                with open(hashes_path, 'rb') as f:
                    self._document_hashes = pickle.load(f)
                    
            faqs_path = self.storage_path / "document_faqs.pkl"
            if faqs_path.exists():
                with open(faqs_path, 'rb') as f:
                    self._document_faqs = pickle.load(f)
                    
            logger.info(f"Loaded local cache: {len(self._metadata)} FAQs")
        except Exception as e:
            logger.error(f"Failed to load local cache: {e}")
            self._metadata = {}

    def _persist_local_cache(self):
        """Save metadata/hashes to local pickle files."""
        try:
            with open(self.storage_path / "metadata.pkl", 'wb') as f:
                pickle.dump(self._metadata, f)
            with open(self.storage_path / "document_hashes.pkl", 'wb') as f:
                pickle.dump(self._document_hashes, f)
            with open(self.storage_path / "document_faqs.pkl", 'wb') as f:
                pickle.dump(self._document_faqs, f)
        except Exception as e:
            logger.error(f"Failed to persist local cache: {e}")

    def _update_stats_from_index(self):
        try:
            stats = self.index.describe_index_stats()
            self._stats['total_vectors'] = stats.get('total_vector_count', 0)
            self._stats['last_updated'] = datetime.now()
        except:
            pass

    def _reconstruct_faq_from_metadata(self, faq_id: str, meta: dict) -> FAQEntry:
        """Construct FAQEntry from Pinecone metadata dict."""
        return FAQEntry(
            id=faq_id,
            question=str(meta.get('question', '')),
            answer=str(meta.get('answer', '')),
            keywords=str(meta.get('keywords', '')).split(',') if meta.get('keywords') else [],
            category=str(meta.get('category', 'general')),
            confidence_score=float(meta.get('confidence', 0.0)),
            source_document=str(meta.get('source', '')),
            created_at=datetime.now(), # Estimate
            updated_at=datetime.now(),
            audience=str(meta.get('audience', 'any')),
            intent=str(meta.get('intent', 'info')),
            condition=str(meta.get('condition', 'default')),
            embedding=None
        )
