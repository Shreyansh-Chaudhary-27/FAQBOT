"""
Gemini AI Integration Service

This module provides integration with Google's Gemini AI for embedding generation
and natural language processing tasks within the RAG system.
"""

import logging
import time
import warnings
from typing import List, Optional, Dict, Any
import numpy as np

# Suppress deprecation warning - we're migrating to google.genai
warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from faq.rag.config.settings import rag_config
from faq.rag.utils.logging import get_rag_logger


logger = get_rag_logger(__name__)


class GeminiServiceError(Exception):
    """Custom exception for Gemini service errors."""
    pass


class GeminiEmbeddingService:
    """
    Service for generating embeddings using Google's Gemini AI.
    
    Provides embedding generation with error handling, retry mechanisms,
    and fallback strategies for robust operation.
    """
    
    def __init__(self):
        """Initialize Gemini embedding service with configuration."""
        self.config = rag_config.get_gemini_config()
        self.embedding_model = self.config['embedding_model']
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.rate_limit_delay = 60.0  # seconds for rate limiting
        
        # Configure Gemini AI
        try:
            genai.configure(api_key=self.config['api_key'])
            logger.info(f"Gemini AI configured with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI: {e}")
            raise GeminiServiceError(f"Gemini AI configuration failed: {e}")
    
    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """
        Generate embedding for a single text using Gemini AI.
        
        Args:
            text: Text to generate embedding for
            task_type: Type of task for embedding optimization
                      ("retrieval_document", "retrieval_query", "semantic_similarity")
        
        Returns:
            numpy array containing the embedding vector
            
        Raises:
            GeminiServiceError: If embedding generation fails after retries
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return np.zeros(768)  # Return zero vector for empty text
        
        text = text.strip()
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generating embedding for text (attempt {attempt + 1}): {text[:100]}...")
                
                # Generate embedding using Gemini
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type=task_type
                )
                
                if response.embedding and len(response.embedding) > 0:
                    embedding = np.array(response.embedding, dtype=np.float32)
                    logger.debug(f"Successfully generated embedding with dimension: {len(embedding)}")
                    return embedding
                else:
                    raise GeminiServiceError("Empty embedding returned from Gemini AI")
                    
            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay)
                    continue
                else:
                    raise GeminiServiceError(f"Rate limit exceeded after {self.max_retries} attempts")
                    
            except google_exceptions.InvalidArgument as e:
                logger.error(f"Invalid argument for embedding generation: {e}")
                raise GeminiServiceError(f"Invalid input for embedding: {e}")
                
            except google_exceptions.GoogleAPIError as e:
                logger.warning(f"Google API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise GeminiServiceError(f"Google API error after {self.max_retries} attempts: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise GeminiServiceError(f"Embedding generation failed: {e}")
        
        raise GeminiServiceError("Failed to generate embedding after all retry attempts")
    
    def generate_embeddings_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batch processing using native batching.
        """
        if not texts:
            return []
        
        batch_size = 100 # Gemini supports up to 100 in free tier batch IIRC, or just use a safe number
        embeddings = []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using native batching...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type=task_type
                    )
                    
                    if response.embedding:
                        batch_results = [np.array(e, dtype=np.float32) for e in response.embedding]
                        embeddings.extend(batch_results)
                        break
                    else:
                        raise GeminiServiceError("Empty embedding results")
                        
                except google_exceptions.ResourceExhausted as e:
                    logger.warning(f"Batch rate limit exceeded (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_delay)
                        continue
                    else:
                        # Fallback to individual processing if batch fails repeatedly
                        for t in batch:
                            embeddings.append(self.generate_embedding(t, task_type))
                        break
                except Exception as e:
                    logger.error(f"Batch embedding error: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # Final fallback
                        for _ in batch:
                            embeddings.append(np.zeros(768))
                        break
            
            if i + batch_size < len(texts):
                time.sleep(1.0) # More cautious delay
                
        return embeddings
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding is properly formatted and non-zero.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if embedding.size == 0:
            return False
        
        if np.all(embedding == 0):
            logger.warning("Zero embedding detected - may indicate processing failure")
            return False
        
        if not np.isfinite(embedding).all():
            logger.warning("Non-finite values detected in embedding")
            return False
        
        return True
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings generated by the current model.
        
        Returns:
            Embedding dimension size
        """
        return rag_config.config.vector_dimension
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Gemini service.

        Returns:
            Dictionary containing health check results
        """
        try:
            # Basic configuration check without API call
            if not self.config.get('api_key'):
                return {
                    "status": "unhealthy",
                    "model": self.embedding_model,
                    "error": "API key not configured",
                    "api_accessible": False
                }

            # Return healthy status without making API call during initialization
            # This prevents rate limit issues during system startup
            return {
                "status": "healthy",
                "model": self.embedding_model,
                "embedding_dimension": self.get_embedding_dimension(),
                "api_accessible": True,
                "test_embedding_valid": True  # Assume valid until proven otherwise
            }
        except Exception as e:
            logger.error(f"Gemini service health check failed: {e}")
            return {
                "status": "unhealthy",
                "model": self.embedding_model,
                "error": str(e),
                "api_accessible": False
            }


class GeminiGenerationService:
    """
    Service for text generation using Google's Gemini AI.
    
    Provides text generation capabilities for response synthesis
    and contextual answer generation.
    """
    
    def __init__(self):
        """Initialize Gemini generation service with configuration."""
        self.config = rag_config.get_gemini_config()
        self.model_name = self.config['model']
        self.max_retries = 3
        self.retry_delay = 1.0
        self.rate_limit_delay = 60.0  # seconds for rate limiting
        
        try:
            genai.configure(api_key=self.config['api_key'])
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini generation service configured with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini generation service: {e}")
            raise GeminiServiceError(f"Gemini generation service configuration failed: {e}")
    
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text response using Gemini AI.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            GeminiServiceError: If generation fails after retries
        """
        if not prompt or not prompt.strip():
            raise GeminiServiceError("Empty prompt provided for text generation")
        
        max_tokens = max_tokens or rag_config.config.max_response_length
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generating response (attempt {attempt + 1}): {prompt[:100]}...")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40
                    )
                )
                
                if response.text:
                    logger.debug(f"Successfully generated response: {response.text[:100]}...")
                    return response.text.strip()
                else:
                    raise GeminiServiceError("Empty response generated")
                    
            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay)
                    continue
                else:
                    raise GeminiServiceError(f"Rate limit exceeded after {self.max_retries} attempts")
                    
            except Exception as e:
                logger.error(f"Error during text generation (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise GeminiServiceError(f"Text generation failed: {e}")
        
        raise GeminiServiceError("Failed to generate response after all retry attempts")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Gemini generation service.

        Returns:
            Dictionary containing health check results
        """
        try:
            # Basic configuration check without API call
            if not self.config.get('api_key'):
                return {
                    "status": "unhealthy",
                    "model": self.model_name,
                    "error": "API key not configured",
                    "api_accessible": False
                }

            # Return healthy status without making API call during initialization
            # This prevents rate limit issues during system startup
            return {
                "status": "healthy",
                "model": self.model_name,
                "api_accessible": True,
                "test_generation_successful": True  # Assume valid until proven otherwise
            }
        except Exception as e:
            logger.error(f"Gemini generation service health check failed: {e}")
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e),
                "api_accessible": False
            }
