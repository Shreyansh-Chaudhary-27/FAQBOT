"""
Base Interfaces for RAG System Components

This module defines abstract base classes and interfaces for all RAG system components,
ensuring consistent APIs and enabling dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class FAQEntry:
    """Data model for FAQ entries."""
    id: str
    question: str
    answer: str
    keywords: List[str]
    category: str
    confidence_score: float
    source_document: str
    created_at: datetime
    updated_at: datetime
    audience: str = "any"
    intent: str = "info"
    condition: str = "default"
    embedding: Optional[np.ndarray] = None
    
    @property
    def composite_key(self) -> str:
        """Generate a composite key for this FAQ entry."""
        aud = (self.audience or "any").lower().strip()
        cat = (self.category or "general").lower().strip()
        intnt = (self.intent or "info").lower().strip()
        cond = (self.condition or "default").lower().strip()
        return f"{aud}::{cat}::{intnt}::{cond}"


@dataclass
class ProcessedQuery:
    """Data model for processed user queries."""
    original_query: str
    corrected_query: str
    intent: str
    expanded_queries: List[str]
    language: str
    confidence: float
    embedding: Optional[np.ndarray] = None
    components: Optional[Dict[str, str]] = None  # Composite key components: audience, category, etc.
    ngram_keywords: Optional[List[str]] = None  # N-gram keywords for precision matching


@dataclass
class Response:
    """Data model for generated responses."""
    text: str
    confidence: float
    source_faqs: List[FAQEntry]
    context_used: bool
    generation_method: str  # 'rag', 'direct_match', 'synthesized'
    query_id: str
    processed_query: ProcessedQuery
    metadata: Dict[str, Any]
    processing_time: Optional[float] = None


@dataclass
class ConversationContext:
    """Data model for conversation context."""
    session_id: str
    history: List[Dict[str, Any]]
    current_topic: Optional[str]
    user_preferences: Dict[str, Any]
    last_activity: datetime
    context_embeddings: List[np.ndarray]
    consecutive_no_match_count: int = 0


@dataclass
class DocumentStructure:
    """Data model for document structure analysis."""
    document_type: str
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    lists: List[Dict[str, Any]]
    paragraphs: List[str]


@dataclass
class ValidationResult:
    """Data model for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class SimilarityMatch:
    """Data model for similarity matching results."""
    faq_entry: FAQEntry
    similarity_score: float
    match_type: str  # 'exact', 'semantic', 'keyword'
    matched_components: List[str]  # which parts matched (question, answer, keywords)


class DOCXScraperInterface(ABC):
    """Abstract interface for DOCX document scraping."""
    
    @abstractmethod
    def extract_faqs(self, docx_path: str) -> List[FAQEntry]:
        """Extract FAQ entries from a DOCX document."""
        pass
    
    @abstractmethod
    def parse_document_structure(self, document_path: str) -> DocumentStructure:
        """Parse and analyze document structure."""
        pass
    
    @abstractmethod
    def identify_faq_patterns(self, content: List[str]) -> List[Dict[str, Any]]:
        """Identify FAQ patterns in document content."""
        pass
    
    @abstractmethod
    def validate_extraction(self, faqs: List[FAQEntry]) -> ValidationResult:
        """Validate extracted FAQ entries."""
        pass


class QueryProcessorInterface(ABC):
    """Abstract interface for query processing."""
    
    @abstractmethod
    def correct_typos(self, query: str) -> str:
        """Correct typos and spelling errors in query."""
        pass
    
    @abstractmethod
    def extract_intent(self, query: str) -> str:
        """Extract intent from user query."""
        pass
    
    @abstractmethod
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better matching."""
        pass
    
    @abstractmethod
    def detect_language(self, query: str) -> str:
        """Detect the language of the query."""
        pass
    
    @abstractmethod
    def preprocess_query(self, query: str) -> ProcessedQuery:
        """Complete query preprocessing pipeline."""
        pass
    
    @abstractmethod
    def process_with_context(self, query: str, context: Optional['ConversationContext'] = None) -> ProcessedQuery:
        """Process query with conversation context for improved understanding."""
        pass
    
    @abstractmethod
    def detect_ambiguity(self, query: str, context: Optional['ConversationContext'] = None) -> Dict[str, Any]:
        """Detect ambiguous queries that need clarification."""
        pass
    
    @abstractmethod
    def handle_follow_up_question(self, query: str, context: 'ConversationContext') -> ProcessedQuery:
        """Handle follow-up questions by leveraging conversation context."""
        pass


class FAQVectorizerInterface(ABC):
    """Abstract interface for FAQ vectorization."""
    
    @abstractmethod
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text using AI model."""
        pass
    
    @abstractmethod
    def vectorize_faq_entry(self, faq: FAQEntry) -> FAQEntry:
        """Generate embeddings for FAQ entry components."""
        pass
    
    @abstractmethod
    def update_vector_index(self, vectors: List[FAQEntry]) -> None:
        """Update the vector index with new embeddings."""
        pass
    
    @abstractmethod
    def find_similar_vectors(self, query_vector: np.ndarray, top_k: int) -> List[SimilarityMatch]:
        """Find similar vectors using cosine similarity."""
        pass


class ResponseGeneratorInterface(ABC):
    """Abstract interface for response generation."""
    
    @abstractmethod
    def generate_response(self, query: str, retrieved_faqs: List[FAQEntry]) -> Response:
        """Generate contextual response from retrieved FAQs."""
        pass
    
    @abstractmethod
    def synthesize_multiple_sources(self, faqs: List[FAQEntry]) -> str:
        """Synthesize information from multiple FAQ sources."""
        pass
    
    @abstractmethod
    def maintain_context(self, conversation_history: List[Dict[str, Any]]) -> ConversationContext:
        """Maintain conversation context across interactions."""
        pass
    
    @abstractmethod
    def calculate_confidence(self, response: Response) -> float:
        """Calculate confidence score for generated response."""
        pass


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage and retrieval."""
    
    @abstractmethod
    def store_vectors(self, vectors: List[FAQEntry], document_id: Optional[str] = None, document_hash: Optional[str] = None) -> None:
        """Store FAQ vectors in the vector database with optional document tracking."""
        pass
    
    @abstractmethod
    def search_similar(self, query_vector: np.ndarray, threshold: float, top_k: int = 10) -> List[SimilarityMatch]:
        """Search for similar vectors above threshold."""
        pass
    
    @abstractmethod
    def search_by_ngrams(self, query_ngrams: List[str], threshold: float = 0.9) -> List[SimilarityMatch]:
        """Search for FAQs based on N-gram keyword overlap."""
        pass
    
    @abstractmethod
    def batch_search_similar(self, query_vectors: List[np.ndarray], threshold: float, top_k: int) -> List[List[SimilarityMatch]]:
        """Perform batch similarity search for multiple query vectors."""
        pass
    
    @abstractmethod
    def search_with_filters(self, query_vector: np.ndarray, threshold: float, top_k: int, 
                           category_filter: Optional[str] = None, 
                           audience_filter: Optional[str] = None,
                           intent_filter: Optional[str] = None,
                           condition_filter: Optional[str] = None,
                           confidence_filter: Optional[float] = None,
                           keyword_filter: Optional[List[str]] = None) -> List[SimilarityMatch]:
        """Search for similar vectors with additional filtering options."""
        pass
    
    @abstractmethod
    def search_with_ranking(self, query_vector: np.ndarray, threshold: float, top_k: int,
                           boost_recent: bool, boost_high_confidence: bool) -> List[SimilarityMatch]:
        """Search with advanced ranking that considers multiple factors."""
        pass
    
    @abstractmethod
    def update_vector(self, faq_id: str, new_vector: np.ndarray) -> None:
        """Update a specific vector in the store."""
        pass
    
    @abstractmethod
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass
    
    @abstractmethod
    def delete_vector(self, faq_id: str) -> bool:
        """Delete a vector from the store."""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all data from the vector store."""
        pass
    
    @abstractmethod
    def is_document_processed(self, document_id: str, document_hash: str) -> bool:
        """Check if a document has already been processed with the same hash."""
        pass
    
    @abstractmethod
    def get_document_faqs(self, document_id: str) -> List[str]:
        """Get FAQ IDs associated with a document."""
        pass
    
    @abstractmethod
    def remove_document(self, document_id: str) -> int:
        """Remove all FAQs associated with a document."""
        pass
    
    @abstractmethod
    def backup_store(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the vector store."""
        pass
    
    @abstractmethod
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore vector store from backup."""
        pass


class RAGSystemInterface(ABC):
    """Abstract interface for the main RAG system orchestrator."""
    
    @abstractmethod
    def process_document(self, document_path: str) -> List[FAQEntry]:
        """Process a document and extract FAQs."""
        pass
    
    @abstractmethod
    def answer_query(self, query: str, session_id: Optional[str] = None) -> Response:
        """Answer a user query using RAG pipeline."""
        pass
    
    @abstractmethod
    def update_knowledge_base(self, faqs: List[FAQEntry]) -> None:
        """Update the knowledge base with new FAQs."""
        pass
    
    @abstractmethod
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        pass


class ConversationManagerInterface(ABC):
    """Abstract interface for conversation context management."""
    
    @abstractmethod
    def create_session(self, session_id: str) -> ConversationContext:
        """Create a new conversation session."""
        pass
    
    @abstractmethod
    def update_context(self, session_id: str, interaction: Dict[str, Any]) -> None:
        """Update conversation context with new interaction."""
        pass
    
    @abstractmethod
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for session."""
        pass
    
    @abstractmethod
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired conversation sessions."""
        pass


class AnalyticsManagerInterface(ABC):
    """Abstract interface for analytics and logging."""

    @abstractmethod
    def log_query(self, query_id: str, query_text: str, processed_query: ProcessedQuery, response: Response, timestamp: datetime) -> None:
        """Log a processed query and its response."""
        pass

    @abstractmethod
    def log_document_ingestion(self, document_path: str, faqs_ingested: int, timestamp: datetime, status: str, error: Optional[str] = None) -> None:
        """Log document ingestion events."""
        pass

    @abstractmethod
    def log_system_event(self, event_type: str, details: Dict[str, Any], timestamp: datetime) -> None:
        """Log general system events (e.g., component health changes, configuration updates)."""
        pass

    @abstractmethod
    def get_query_patterns(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retrieve common query patterns and trends."""
        pass

    @abstractmethod
    def get_performance_metrics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Retrieve system performance metrics, including total queries, successful queries, success rate, and average confidence."""
        pass


class FeedbackManagerInterface(ABC):
    """Abstract interface for user feedback management."""

    @abstractmethod
    def submit_feedback(self, query_id: str, user_id: str, rating: int, comments: Optional[str] = None) -> None:
        """Submit user feedback for a specific query."""
        pass

    @abstractmethod
    def get_feedback(self, query_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve user feedback."""
        pass

    @abstractmethod
    def analyze_feedback(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze collected feedback to identify areas for improvement."""
        pass