"""
RAG System Factory

Factory class for creating and configuring RAG system components
with proper dependency injection.
"""

from typing import Optional
from ..interfaces.base import (
    DOCXScraperInterface, QueryProcessorInterface,
    FAQVectorizerInterface, VectorStoreInterface,
    ResponseGeneratorInterface, ConversationManagerInterface,
    AnalyticsManagerInterface, FeedbackManagerInterface
)
from ..config.settings import rag_config
from ..utils.logging import get_rag_logger
from .rag_system import RAGSystem
from .ingestion_pipeline import DocumentIngestionPipeline
from .analytics_manager import AnalyticsManager
from .feedback_manager import FeedbackManager
from .system_improvement import SystemImprovementManager
from typing import Any
from faq.rag.components.query_processor import ProcessedQuery

class RAGSystemFactory:
    """Factory for creating RAG system instances."""
    
    def __init__(self):
        """Initialize factory."""
        self.logger = get_rag_logger('factory')
        self.config = rag_config.config
    
    def create_rag_system(self,
                         docx_scraper: Optional[DOCXScraperInterface] = None,
                         query_processor: Optional[QueryProcessorInterface] = None,
                         vectorizer: Optional[FAQVectorizerInterface] = None,
                         vector_store: Optional[VectorStoreInterface] = None,
                         response_generator: Optional[ResponseGeneratorInterface] = None,
                         conversation_manager: Optional[ConversationManagerInterface] = None,
                         document_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None,
                         analytics_manager: Optional[AnalyticsManagerInterface] = None,
                         feedback_manager: Optional[FeedbackManagerInterface] = None,
                         system_improvement_manager: Optional[SystemImprovementManager] = None) -> RAGSystem:
        """Create a RAG system with provided or default components."""
        
        self.logger.info("Creating RAG system instance")
        
        # Use provided components or create defaults
        # Note: Default implementations will be created in subsequent tasks
        if docx_scraper is None:
            self.logger.warning("No DOCX scraper provided - will need implementation")
        
        if query_processor is None:
            self.logger.warning("No query processor provided - will need implementation")
        
        if vectorizer is None:
            self.logger.warning("No vectorizer provided - will need implementation")
        
        if vector_store is None:
            self.logger.warning("No vector store provided - will need implementation")
        
        if response_generator is None:
            response_generator = self._create_default_response_generator()
        
        if conversation_manager is None:
            self.logger.warning("No conversation manager provided - will need implementation")
        
        # Create RAG system with components
        rag_system = RAGSystem(
            docx_scraper=docx_scraper,
            query_processor=query_processor,
            vectorizer=vectorizer,
            vector_store=vector_store,
            response_generator=response_generator,
            conversation_manager=conversation_manager,
            document_ingestion_pipeline=document_ingestion_pipeline,
            analytics_manager=analytics_manager,
            feedback_manager=feedback_manager,
            system_improvement_manager=system_improvement_manager
        )
        
        self.logger.info("RAG system created successfully")
        return rag_system
    
    def create_default_system(self) -> RAGSystem:
        """
        Create RAG system with default component implementations.
        
        Returns:
            RAGSystem: Fully configured RAG system with available components
        """
        self.logger.info("Creating RAG system with default components")
        
        # Create components with error handling
        components = {}
        
        # Create DOCX scraper
        try:
            from ..components.docx_scraper.scraper import DOCXScraper
            components['docx_scraper'] = DOCXScraper()
            self.logger.info("DOCX scraper created successfully")
        except Exception as e:
            # Lightweight fallback so initialization never fails
            self.logger.warning(f"Failed to create DOCX scraper, using fallback: {e}")
            from ..interfaces.base import DOCXScraperInterface, DocumentStructure, FAQEntry, ValidationResult
            from dataclasses import dataclass
            from datetime import datetime

            @dataclass
            class _FallbackValidation(ValidationResult):
                is_valid: bool = True
                errors: list = None
                warnings: list = None
                metadata: dict = None

            class _FallbackDocxScraper(DOCXScraperInterface):
                """Minimal extractor that returns no FAQs but keeps system alive."""
                def extract_faqs(self, docx_path: str):
                    return []

                def parse_document_structure(self, document_path: str) -> DocumentStructure:
                    return DocumentStructure(document_type="unknown", sections=[], tables=[], lists=[], paragraphs=[])

                def identify_faq_patterns(self, content):
                    return []

                def validate_extraction(self, faqs):
                    return _FallbackValidation(
                        is_valid=True,
                        errors=[],
                        warnings=["Fallback scraper used"],
                        metadata={"timestamp": datetime.utcnow().isoformat()}
                    )

            components['docx_scraper'] = _FallbackDocxScraper()
        
        # Create query processor
        try:
            from ..components.query_processor.query_processor import QueryProcessor
            components['query_processor'] = QueryProcessor()
            self.logger.info("Query processor created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create query processor, using fallback: {e}")
            from ..interfaces.base import QueryProcessorInterface, ProcessedQuery, ConversationContext
            from datetime import datetime
            import numpy as np

            class _FallbackQueryProcessor(QueryProcessorInterface):
                """Simple query processor that passes text through."""
                def correct_typos(self, query: str) -> str:
                    return query

                def extract_intent(self, query: str) -> str:
                    return "information"

                def expand_query(self, query: str):
                    return [query]

                def detect_language(self, query: str) -> str:
                    return "en"

                def preprocess_query(self, query: str) -> ProcessedQuery:
                    return ProcessedQuery(
                        original_query=query,
                        corrected_query=query,
                        intent="information",
                        expanded_queries=[query],
                        language="en",
                        confidence=0.5,
                        embedding=np.zeros(self._embedding_dim, dtype=float),
                    )

                def process_with_context(self, query: str, context: ConversationContext = None) -> ProcessedQuery:
                    return self.preprocess_query(query)

                def detect_ambiguity(self, query: str, context: ConversationContext = None):
                    return {"is_ambiguous": False, "reasons": []}

                def handle_follow_up_question(self, query: str, context: ConversationContext) -> ProcessedQuery:
                    return self.preprocess_query(query)

                @property
                def _embedding_dim(self):
                    try:
                        return rag_config.config.vector_dimension
                    except Exception:
                        return 384

            components['query_processor'] = _FallbackQueryProcessor()
        
        # Create vectorizer
        try:
            from ..components.vectorizer.vectorizer import FAQVectorizer
            components['vectorizer'] = FAQVectorizer(use_advanced_matching=True)
            self.logger.info("FAQ vectorizer created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create vectorizer, using fallback: {e}")
            from ..interfaces.base import FAQVectorizerInterface, FAQEntry, SimilarityMatch
            import numpy as np

            class _FallbackVectorizer(FAQVectorizerInterface):
                """Lightweight vectorizer that returns zero embeddings."""
                def __init__(self):
                    try:
                        self.dim = rag_config.config.vector_dimension
                    except Exception:
                        self.dim = 384

                def generate_embeddings(self, text: str) -> np.ndarray:
                    return np.zeros(self.dim, dtype=float)

                def vectorize_faq_entry(self, faq: FAQEntry) -> FAQEntry:
                    faq.embedding = self.generate_embeddings(faq.question)
                    return faq

                def update_vector_index(self, vectors):
                    return None

                def find_similar_vectors(self, query_vector: np.ndarray, top_k: int):
                    return []

            components['vectorizer'] = _FallbackVectorizer()
        
        # Create vector store
        try:
            import os
            if os.getenv('PINECONE_API_KEY'):
                from ..components.vector_store.pinecone_store import PineconeVectorStore
                self.logger.info("Initializing Pinecone Vector Store (Persistent)")
                components['vector_store'] = PineconeVectorStore(storage_path=self.config.vector_store_path)
            else:
                from ..components.vector_store.vector_store import VectorStore
                self.logger.info("Initializing Local Vector Store")
                components['vector_store'] = VectorStore(storage_path=self.config.vector_store_path)
            
            self.logger.info("Vector store created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create vector store: {e}")
            components['vector_store'] = None
        
        # Create response generator (with fallback)
        components['response_generator'] = self._create_default_response_generator()
        
        # Create conversation manager
        try:
            from ..components.conversation_manager.conversation_manager import ConversationManager
            components['conversation_manager'] = ConversationManager(
                session_timeout_minutes=self.config.session_timeout_minutes,
                max_history_length=self.config.max_conversation_history
            )
            self.logger.info("Conversation manager created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create conversation manager: {e}")
            components['conversation_manager'] = None

        # Create Document Ingestion Pipeline (New)
        if components.get('docx_scraper') and components.get('vectorizer') and components.get('vector_store'):
            try:
                components['document_ingestion_pipeline'] = DocumentIngestionPipeline(
                    docx_scraper=components['docx_scraper'],
                    vectorizer=components['vectorizer'],
                    vector_store=components['vector_store']
                )
                self.logger.info("Document Ingestion Pipeline created successfully")
            except Exception as e:
                self.logger.warning(f"Failed to create Document Ingestion Pipeline: {e}")
                components['document_ingestion_pipeline'] = None
        else:
            self.logger.warning("Skipping Document Ingestion Pipeline creation due to missing dependencies.")
            # Provide minimal pipeline so system API stays available
            try:
                from datetime import datetime

                class _FallbackIngestionPipeline:
                    def __init__(self, vector_store):
                        self.vector_store = vector_store

                    def ingest_document(self, document_path, force_update=False):
                        return []

                    def ingest_documents_batch(self, document_paths, force_update=False, parallel=True):
                        return {p: [] for p in document_paths}

                components['document_ingestion_pipeline'] = _FallbackIngestionPipeline(components.get('vector_store'))
            except Exception as e:
                self.logger.warning(f"Failed to create fallback ingestion pipeline: {e}")
                components['document_ingestion_pipeline'] = None
        
        # Create Analytics Manager (Task 10.1)
        try:
            components['analytics_manager'] = AnalyticsManager()
            self.logger.info("Analytics manager created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create analytics manager: {e}")
            components['analytics_manager'] = None
        
        # Create Feedback Manager (Task 10.1)
        try:
            components['feedback_manager'] = FeedbackManager()
            self.logger.info("Feedback manager created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create feedback manager: {e}")
            components['feedback_manager'] = None
        
        # Create System Improvement Manager (Task 10.4)
        try:
            components['system_improvement_manager'] = SystemImprovementManager(
                analytics_manager=components.get('analytics_manager'),
                feedback_manager=components.get('feedback_manager'),
                vectorizer=components.get('vectorizer'),
                vector_store=components.get('vector_store')
            )
            self.logger.info("System improvement manager created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create system improvement manager: {e}")
            components['system_improvement_manager'] = None
        
        # Create RAG system with all components
        rag_system = RAGSystem(**components)
        
        self.logger.info("Default RAG system created successfully")
        return rag_system
    
    def _create_default_response_generator(self) -> ResponseGeneratorInterface:
        """Create default response generator (Gemini-powered with fallback)."""
        try:
            from ..components.response_generator.gemini_response_generator import GeminiResponseGenerator
            self.logger.info("Creating Gemini AI response generator")
            return GeminiResponseGenerator()
        except Exception as e:
            self.logger.warning(f"Failed to create Gemini response generator, using basic: {e}")
            from ..components.response_generator.response_generator import BasicResponseGenerator
            return BasicResponseGenerator()
    
    def create_response_generator(self, use_ai: bool = True) -> ResponseGeneratorInterface:
        """
        Create response generator with optional AI enhancement.
        
        Args:
            use_ai: Whether to use AI-powered generation (default: True)
            
        Returns:
            ResponseGeneratorInterface implementation
        """
        if use_ai:
            try:
                from ..components.response_generator.gemini_response_generator import GeminiResponseGenerator
                self.logger.info("Creating Gemini AI response generator")
                return GeminiResponseGenerator()
            except Exception as e:
                self.logger.warning(f"Failed to create Gemini response generator: {e}")
                # Fall through to basic generator
        
        self.logger.info("Creating basic response generator")
        from ..components.response_generator.response_generator import BasicResponseGenerator
        return BasicResponseGenerator()


# Global factory instance
rag_factory = RAGSystemFactory()