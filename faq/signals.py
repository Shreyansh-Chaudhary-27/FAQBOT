"""
Django signals for automatic FAQ synchronization with RAG system.

This module handles automatic vectorization and synchronization of FAQs
when they are created or updated through the admin panel.
"""

import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings

from .models import RAGFAQEntry
from .rag.interfaces.base import FAQEntry as RAGFAQInterface
from datetime import datetime

logger = logging.getLogger(__name__)


@receiver(post_save, sender=RAGFAQEntry)
def sync_faq_to_rag_system(sender, instance, created, **kwargs):
    """
    Signal handler to automatically sync FAQ entries to the RAG system
    when they are created or updated.
    
    This ensures that newly added FAQs are immediately vectorized and
    available for semantic search with proper confidence scores.
    
    Args:
        sender: The model class (RAGFAQEntry)
        instance: The actual instance being saved
        created: Boolean indicating if this is a new instance
        **kwargs: Additional keyword arguments
    """
    try:
        # Import here to avoid circular imports
        from .rag.core import initialize_rag_system
        
        # Log the sync attempt
        action = "created" if created else "updated"
        logger.info(f"FAQ {action}: {instance.id} - '{instance.question[:50]}...'")
        logger.info(f"Initiating automatic RAG system sync for FAQ {instance.id}")
        
        # Initialize RAG system
        try:
            rag_system = initialize_rag_system(
                validate_config=False,
                perform_health_check=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG system for FAQ sync: {e}")
            return
        
        # Convert Django FAQ to RAG FAQ interface
        rag_faq = RAGFAQInterface(
            id=instance.rag_id,
            question=instance.question,
            answer=instance.answer,
            keywords=instance.keywords.split(',') if instance.keywords else [],
            category=instance.category or "Manual Entry",
            confidence_score=1.0,  # Set high confidence for manually added FAQs
            source_document="manual_entry",
            created_at=instance.created_at,
            updated_at=instance.updated_at,
            # Add composite key components
            audience=instance.audience or "any",
            intent=instance.intent or "info",
            condition=instance.condition or "default"
        )
        
        # Vectorize and add to RAG system
        logger.info(f"Vectorizing FAQ {instance.id}...")
        try:
            # Update knowledge base with the single FAQ
            # This will generate embeddings and store in vector store
            rag_system.update_knowledge_base([rag_faq])
            
            # Update the Django model with the generated embedding
            if rag_faq.embedding is not None:
                instance.set_question_embedding_array(rag_faq.embedding)
                instance.embedding_model = rag_system.vectorizer.embedding_generator.service.__class__.__name__
                instance.embedding_version = "1.0"
                # Use update to avoid triggering the signal again
                RAGFAQEntry.objects.filter(pk=instance.pk).update(
                    question_embedding=instance.question_embedding,
                    embedding_model=instance.embedding_model,
                    embedding_version=instance.embedding_version
                )
            
            logger.info(f"✓ Successfully synced FAQ {instance.id} to RAG system")
            logger.info(f"  - Question: {instance.question[:100]}...")
            logger.info(f"  - Embedding generated: {rag_faq.embedding is not None}")
            logger.info(f"  - Confidence score: {rag_faq.confidence_score}")
            
        except Exception as e:
            logger.error(f"Failed to vectorize and sync FAQ {instance.id}: {e}")
            logger.exception("Full traceback:")
            # Don't raise the exception - we don't want to prevent the FAQ from being saved
            
    except Exception as e:
        logger.error(f"Error in FAQ sync signal handler: {e}")
        logger.exception("Full traceback:")
        # Don't raise - we don't want to prevent the FAQ from being saved
