"""
Local Embedding Service Implementation

This module provides local embedding generation using sentence-transformers,
avoiding API calls and rate limits for vectorization.
"""

import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger


logger = get_rag_logger(__name__)


class LocalEmbeddingService:
    """
    Service for generating embeddings locally using Sentence-Transformers.
    """
    
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the local embedding service."""
        if self._model is not None:
            return
            
        self.config = rag_config.config
        self.model_name = self.config.local_embedding_model
        
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Local embedding model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            raise RuntimeError(f"Local embedding model load failed: {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """
        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension())
            
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate local embedding: {e}")
            return np.zeros(self.get_embedding_dimension())

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.
        """
        if not texts:
            return []
            
        try:
            logger.info(f"Generating local embeddings for {len(texts)} texts...")
            embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            return [e.astype(np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate local embeddings batch: {e}")
            return [np.zeros(self.get_embedding_dimension()) for _ in texts]

    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._model.get_sentence_embedding_dimension()

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding is properly formatted.
        """
        if embedding is None:
            return False
        if not isinstance(embedding, np.ndarray):
            return False
        if embedding.shape != (self.get_embedding_dimension(),):
            return False
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False
        return True

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self._model else "unhealthy",
            "model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "device": str(self._model.device) if self._model else "N/A"
        }
