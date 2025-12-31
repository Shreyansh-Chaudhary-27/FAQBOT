"""
RAG System Core Orchestrator

Main system that coordinates all RAG components to provide end-to-end
FAQ processing and query answering capabilities with comprehensive error handling,
configuration management, and system initialization.
"""

import traceback
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from ..interfaces.base import (
    RAGSystemInterface, FAQEntry, Response, ProcessedQuery,
    DOCXScraperInterface, QueryProcessorInterface,
    FAQVectorizerInterface, VectorStoreInterface,
    ResponseGeneratorInterface, ConversationManagerInterface,
    AnalyticsManagerInterface, FeedbackManagerInterface
)
from .ingestion_pipeline import DocumentIngestionPipeline
from .performance_monitor import PerformanceMonitor, AlertSeverity
from ..config.settings import rag_config
from ..utils.logging import get_rag_logger
from ..utils.debug_logger import get_debug_logger
from .system_improvement import SystemImprovementManager


class RAGSystemError(Exception):
    """Custom exception for RAG system errors."""
    pass


class RAGSystem(RAGSystemInterface):
    """
    Main RAG system orchestrator that coordinates all components to provide
    end-to-end query processing pipeline with comprehensive error handling
    and system monitoring.
    
    This orchestrator implements:
    - End-to-end query processing pipeline
    - Component coordination and error handling  
    - System configuration and initialization
    - Performance monitoring and statistics
    - Graceful degradation on component failures
    """
    
    def __init__(self,
                 docx_scraper: Optional[DOCXScraperInterface] = None,
                 query_processor: Optional[QueryProcessorInterface] = None,
                 vectorizer: Optional[FAQVectorizerInterface] = None,
                 vector_store: Optional[VectorStoreInterface] = None,
                 response_generator: Optional[ResponseGeneratorInterface] = None,
                 conversation_manager: Optional[ConversationManagerInterface] = None,
                 document_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None,
                 analytics_manager: Optional[AnalyticsManagerInterface] = None,
                 feedback_manager: Optional[FeedbackManagerInterface] = None,
                 system_improvement_manager: Optional['SystemImprovementManager'] = None,
                 enable_performance_monitoring: bool = True):
        """
        Initialize RAG system with component dependencies.
        
        Args:
            docx_scraper: Document scraping component (optional)
            query_processor: Query processing component (optional)
            vectorizer: FAQ vectorization component (optional)
            vector_store: Vector storage component (optional)
            response_generator: Response generation component (optional)
            conversation_manager: Conversation management component (optional)
            document_ingestion_pipeline: Document ingestion pipeline (optional)
            analytics_manager: Analytics manager (optional)
            feedback_manager: Feedback manager (optional)
            system_improvement_manager: System improvement manager (optional)
            enable_performance_monitoring: Enable advanced performance monitoring (default: True)
        """
        # Initialize logging and configuration
        self.logger = get_rag_logger('rag_system')
        self.debug_logger = get_debug_logger('rag_system')
        self.config = rag_config.config
        
        # Component initialization with validation
        self.docx_scraper = docx_scraper
        self.query_processor = query_processor
        self.vectorizer = vectorizer
        self.vector_store = vector_store
        self.response_generator = response_generator
        self.conversation_manager = conversation_manager
        self.document_ingestion_pipeline = document_ingestion_pipeline
        self.analytics_manager = analytics_manager
        self.feedback_manager = feedback_manager
        self.system_improvement_manager = system_improvement_manager
        
        # Initialize performance monitoring
        self.performance_monitor = None
        if enable_performance_monitoring:
            try:
                self.performance_monitor = PerformanceMonitor(
                    analytics_manager=analytics_manager,
                    alert_callbacks=[self._handle_performance_alert]
                )
                self.logger.info("Performance monitoring enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize performance monitor: {e}")
        
        # System state
        self._initialized = False
        self._component_status: Dict[str, bool] = {}
        self._system_stats = {
            'queries_processed': 0,
            'documents_processed': 0,
            'errors_encountered': 0,
            'last_error': None,
            'initialization_time': datetime.now(),
            'last_activity': None
        }
        
        # Initialize system
        self._initialize_system()
        self.logger.info("RAG System orchestrator initialized successfully")

    def process_feedback_for_improvement(self) -> None:
        """
        Processes accumulated feedback to drive system improvements.
        This method should be called periodically or based on specific triggers.
        """
        if self.system_improvement_manager:
            self.logger.info("Triggering system improvement based on feedback...")
            try:
                self.system_improvement_manager.analyze_and_adapt()
                self.logger.info("System improvement process completed.")
            except Exception as e:
                self.logger.error(f"Error during system improvement process: {e}")
        else:
            self.logger.warning("SystemImprovementManager not available. Cannot process feedback for improvement.")
    
    def _initialize_system(self) -> None:
        """Initialize the RAG system and validate component availability."""
        try:
            self.logger.info("Initializing RAG system components...")
            
            # Validate and initialize each component
            self._validate_components()
            
            # Perform component health checks
            self._perform_health_checks()
            
            # Initialize system configuration
            self._initialize_configuration()
            
            self._initialized = True
            self.logger.info("RAG system initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"RAG system initialization failed: {e}")
            self._initialized = False
            raise RAGSystemError(f"System initialization failed: {e}")
    
    def _validate_components(self) -> None:
        """Validate that required components are available."""
        component_status = {
            'docx_scraper': self.docx_scraper is not None,
            'query_processor': self.query_processor is not None,
            'vectorizer': self.vectorizer is not None,
            'vector_store': self.vector_store is not None,
            'response_generator': self.response_generator is not None,
            'conversation_manager': self.conversation_manager is not None,
            'document_ingestion_pipeline': self.document_ingestion_pipeline is not None,
            'analytics_manager': self.analytics_manager is not None,
            'feedback_manager': self.feedback_manager is not None
        }
        
        self._component_status = component_status
        
        # Log component availability
        for component, available in component_status.items():
            status = "available" if available else "missing"
            self.logger.info(f"Component {component}: {status}")
        
        # Check for critical components
        critical_components = ['query_processor', 'response_generator']
        missing_critical = [comp for comp in critical_components if not component_status[comp]]
        
        if missing_critical:
            raise RAGSystemError(f"Critical components missing: {missing_critical}")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on available components."""
        health_results = {}
        
        # Check vectorizer health if available
        if self.vectorizer:
            try:
                vectorizer_health = self.vectorizer.health_check()
                health_results['vectorizer'] = vectorizer_health
                self.logger.debug(f"Vectorizer health: {vectorizer_health.get('status', 'unknown')}")
            except Exception as e:
                health_results['vectorizer'] = {'status': 'unhealthy', 'error': str(e)}
                self.logger.warning(f"Vectorizer health check failed (non-blocking): {e}")
        
        # Check vector store health if available
        if self.vector_store:
            try:
                store_stats = self.vector_store.get_vector_stats()
                health_results['vector_store'] = {'status': 'healthy', 'stats': store_stats}
                self.logger.debug(f"Vector store health: healthy ({store_stats.get('total_vectors', 0)} vectors)")
            except Exception as e:
                health_results['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
                self.logger.warning(f"Vector store health check failed (non-blocking): {e}")
        
        # Check response generator health if available
        if self.response_generator:
            try:
                generator_stats = self.response_generator.get_generator_stats()
                health_results['response_generator'] = {'status': 'healthy', 'stats': generator_stats}
                self.logger.debug("Response generator health: healthy")
            except Exception as e:
                health_results['response_generator'] = {'status': 'unhealthy', 'error': str(e)}
                self.logger.warning(f"Response generator health check failed (non-blocking): {e}")
        
        self._component_status.update({f"{k}_health": v for k, v in health_results.items()})
        
        # Log summary but don't fail initialization
        healthy_count = sum(1 for v in health_results.values() if v.get('status') == 'healthy')
        total_count = len(health_results)
        self.logger.info(f"Health checks completed: {healthy_count}/{total_count} components healthy")
    
    def _initialize_configuration(self) -> None:
        """Initialize system configuration and validate settings."""
        try:
            # Validate configuration
            required_config = ['similarity_threshold', 'max_results', 'vector_dimension']
            for setting in required_config:
                if not hasattr(self.config, setting):
                    self.logger.warning(f"Configuration setting '{setting}' not found, using defaults")
            
            # Log configuration summary
            self.logger.info(f"Configuration loaded - Similarity threshold: {self.config.similarity_threshold}, "
                           f"Max results: {self.config.max_results}, Vector dimension: {self.config.vector_dimension}")
            
        except Exception as e:
            self.logger.error(f"Configuration initialization failed: {e}")
            raise RAGSystemError(f"Configuration initialization failed: {e}")
    
    def process_document(self, document_path: str, force_update: bool = False) -> List[FAQEntry]:
        """
        Process a document and extract FAQs with comprehensive error handling using the ingestion pipeline.

        Args:
            document_path: Path to the document to process
            force_update: If True, forces re-ingestion even if document is already processed.

        Returns:
            List of processed FAQ entries with embeddings

        Raises:
            RAGSystemError: If document processing fails
        """
        if not self._initialized:
            raise RAGSystemError("RAG system not properly initialized")

        if not self.document_ingestion_pipeline:
            raise RAGSystemError("Document ingestion pipeline component not available")

        self.logger.info(f"Processing document: {document_path} (force_update={force_update})")

        try:
            self._system_stats['last_activity'] = datetime.now()
            vectorized_faqs = self.document_ingestion_pipeline.ingest_document(document_path, force_update)
            self._system_stats['documents_processed'] += 1
            self.logger.info(f"Successfully processed document with {len(vectorized_faqs)} FAQs using ingestion pipeline")
            if self.analytics_manager:
                self.analytics_manager.log_document_ingestion(document_path, len(vectorized_faqs), datetime.now(), "success")
            return vectorized_faqs

        except Exception as e:
            self._system_stats['errors_encountered'] += 1
            self._system_stats['last_error'] = {
                'timestamp': datetime.now(),
                'operation': 'process_document',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.logger.error(f"Error processing document {document_path} with ingestion pipeline: {str(e)}")
            if self.analytics_manager:
                self.analytics_manager.log_document_ingestion(document_path, 0, datetime.now(), "failed", str(e))
            raise RAGSystemError(f"Document processing failed: {e}")


    

    
    def get_ingestion_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get current document ingestion progress.
        
        Returns:
            Progress information dictionary or None if no ingestion in progress
        """
        if not self.document_ingestion_pipeline:
            return None
        
        return self.document_ingestion_pipeline.get_ingestion_progress()
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get document ingestion pipeline statistics.
        
        Returns:
            Dictionary containing ingestion pipeline statistics
        """
        if not self.document_ingestion_pipeline:
            return {}
        
        return self.document_ingestion_pipeline.get_ingestion_stats()

    def answer_query(self, query: str, session_id: Optional[str] = None, query_id: Optional[str] = None) -> Response:
        """
        Answer a user query using the complete RAG pipeline with error handling and fallbacks.
        
        Args:
            query: User query string
            session_id: Optional session ID for conversation context
            
        Returns:
            Response object with generated answer and metadata
            
        Raises:
            RAGSystemError: If query processing fails completely
        """
        if not self._initialized:
            raise RAGSystemError("RAG system not properly initialized")
        
        if not self.response_generator:
            raise RAGSystemError("Response generator component not available")
        
        self.logger.info(f"Processing query: {query[:100]}...")
        
        # DEBUG: Log initial query
        self.debug_logger.log_query_processing(query, step="initial_query")
        
        # Start performance timing
        start_time = time.time()
        
        try:
            # Update activity timestamp
            self._system_stats['last_activity'] = datetime.now()
            
            # Step 1: Process the query with error handling
            processed_query = self._process_query_with_fallback(query, session_id)
            
            # DEBUG: Log processed query
            self.debug_logger.log_query_processing(query, processed_query, step="query_processed")
            
            # Step 2: Get conversation context if session provided
            context = self._get_conversation_context(session_id)
            
            # Step 3: Retrieve relevant FAQs with error handling
            retrieved_faqs = self._retrieve_relevant_faqs(processed_query, context)
            
            # DEBUG: Log retrieved FAQs
            self.debug_logger.log_query_processing(query, processed_query, retrieved_faqs, step="faqs_retrieved")
            
            # Step 4: Generate response with error handling
            response = self._generate_response_with_fallback(processed_query, retrieved_faqs, context, query_id, original_processed_query=processed_query)
            
            # DEBUG: Log final response
            self.debug_logger.log_query_processing(query, processed_query, retrieved_faqs, response, step="response_generated")
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add response time to metadata
            if 'response_time' not in response.metadata:
                response.metadata['response_time'] = response_time
            
            # Step 5: Update conversation context if session provided
            self._update_conversation_context(session_id, query, processed_query, response)
            
            # Update statistics
            self._system_stats['queries_processed'] += 1

            # Performance monitoring
            if self.performance_monitor:
                try:
                    quality_metrics = self.performance_monitor.measure_response_quality(
                        query, processed_query, response, response_time
                    )
                    self.logger.debug(f"Response quality measured: {quality_metrics}")
                except Exception as e:
                    self.logger.warning(f"Performance monitoring failed: {e}")

            # Log query and response if analytics manager is available
            if self.analytics_manager:
                query_id = f"query_{datetime.now().timestamp()}"
                self.analytics_manager.log_query(query_id, query, processed_query, response, datetime.now())

            self.logger.info(f"Generated response with confidence {response.confidence:.2f} in {response_time:.2f}s")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._system_stats['errors_encountered'] += 1
            self._system_stats['last_error'] = {
                'timestamp': datetime.now(),
                'operation': 'answer_query',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.logger.error(f"Error answering query: {str(e)}")
            
            # Monitor system health on error
            if self.performance_monitor:
                try:
                    self.performance_monitor.monitor_system_health(
                        component_name='rag_system',
                        status='degraded',
                        response_time=response_time,
                        error_rate=1.0  # This query failed
                    )
                except Exception as monitor_error:
                    self.logger.warning(f"Health monitoring failed: {monitor_error}")
            
            # Try to generate a fallback response
            try:
                fallback_response = self._generate_fallback_response(query, str(e), query_id, processed_query)
                self.logger.info("Generated fallback response due to error")
                return fallback_response
            except Exception as fallback_error:
                self.logger.error(f"Fallback response generation failed: {fallback_error}")
                raise RAGSystemError(f"Query processing failed: {e}")

    def submit_user_feedback(self, query_id: str, user_id: str, rating: int, comments: Optional[str] = None) -> None:
        """
        Submits user feedback for a given query.

        Args:
            query_id: The ID of the query the feedback is for.
            user_id: The ID of the user submitting the feedback.
            rating: The rating given by the user (e.g., 1-5).
            comments: Optional comments from the user.
        """
        if not self.feedback_manager:
            self.logger.warning("Feedback manager not available, feedback will not be submitted.")
            return

        try:
            self.feedback_manager.submit_feedback(query_id, user_id, rating, comments)
            self.logger.info(f"User feedback submitted for query {query_id}.")
            
            # Trigger system improvement process after feedback submission
            if self.system_improvement_manager:
                self.logger.info("Triggering system improvement based on new feedback...")
                self.system_improvement_manager.analyze_and_adapt()
            else:
                self.logger.warning("SystemImprovementManager not available. Cannot trigger improvement process.")
        except Exception as e:
            self.logger.error(f"Failed to submit user feedback for query {query_id}: {e}")
            if self.analytics_manager:
                self.analytics_manager.log_system_event("feedback_submission_failed", {"query_id": query_id, "error": str(e)}, datetime.now())
    
    def _process_query_with_fallback(self, query: str, session_id: Optional[str]) -> ProcessedQuery:
        """Process query with fallback handling."""
        if not self.query_processor:
            self.logger.warning("Query processor not available, using basic processing")
            return ProcessedQuery(
                original_query=query,
                corrected_query=query.strip(),
                intent="unknown",
                expanded_queries=[query.strip()],
                language="en",
                confidence=0.5,
                embedding=None
            )
        
        try:
            # Get conversation context for enhanced processing
            context = self._get_conversation_context(session_id)
            
            if context:
                return self.query_processor.process_with_context(query, context)
            else:
                return self.query_processor.preprocess_query(query)
                
        except Exception as e:
            self.logger.warning(f"Query processing failed, using basic fallback: {e}")
            return ProcessedQuery(
                original_query=query,
                corrected_query=query.strip(),
                intent="unknown",
                expanded_queries=[query.strip()],
                language="en",
                confidence=0.3,
                embedding=None
            )
    
    def _get_conversation_context(self, session_id: Optional[str]):
        """Get conversation context with error handling."""
        if not session_id or not self.conversation_manager:
            return None
        
        try:
            return self.conversation_manager.get_context(session_id)
        except Exception as e:
            self.logger.warning(f"Failed to get conversation context: {e}")
            return None
    
    def _retrieve_relevant_faqs(self, processed_query: ProcessedQuery, context) -> List[FAQEntry]:
        """Retrieve relevant FAQs with N-gram matching, semantic validation, and vector fallback."""
        retrieved_faqs = []
        
        # Strategy 1: N-Gram Keyword Search (Requirement: 90% overlap)
        if self.vector_store and processed_query.ngram_keywords:
            try:
                self.logger.info(f"Attempting N-Gram keyword search (90% threshold)...")
                ngram_matches = self.vector_store.search_by_ngrams(
                    processed_query.ngram_keywords, 
                    threshold=0.9
                )
                
                if ngram_matches:
                    self.logger.info(f"N-Gram search found {len(ngram_matches)} candidates. Starting AI semantic validation...")
                    validated_faqs = []
                    
                    # Validate top candidates with Gemini (as the Final Judge)
                    # Limit to top 3 to keep it fast
                    for match in ngram_matches[:3]:
                        if self.response_generator.validate_candidate_relevance(processed_query.corrected_query, match.faq_entry):
                            validated_faqs.append(match.faq_entry)
                        else:
                            self.logger.info(f"FAQ '{match.faq_entry.question}' failed AI semantic validation.")
                    
                    if validated_faqs:
                        self.logger.info(f"AI validated {len(validated_faqs)} correct matches.")
                        return validated_faqs
                
                self.logger.info("No 90% N-Gram matches validated by AI. Proceeding to semantic fallback.")
            except Exception as e:
                self.logger.warning(f"N-Gram search or validation failed: {e}")

        # Strategy 2: Filtered vector similarity search (Fallback)
        if self.vectorizer and self.vector_store:
            try:
                self.logger.debug("Attempting filtered vector similarity search fallback...")
                
                # Generate query embedding
                query_embedding = self.vectorizer.generate_embeddings(processed_query.corrected_query)
                
                # Extract components for filtering
                components = processed_query.components or {}
                
                # Search with composite filters
                similar_matches = self.vector_store.search_with_filters(
                    query_embedding, 
                    threshold=self.config.similarity_threshold,
                    top_k=self.config.max_results,
                    audience_filter=components.get('audience'),
                    category_filter=components.get('category'),
                    intent_filter=components.get('intent'),
                    condition_filter=components.get('condition')
                )
                
                # Extract FAQ entries from similarity matches
                retrieved_faqs = [match.faq_entry for match in similar_matches]
                
                if not retrieved_faqs:
                    self.logger.info(f"Filtered search found 0 relevant FAQs. Falling back to unfiltered semantic search.")
                    similar_matches = self.vector_store.search_similar(
                        query_embedding, 
                        threshold=self.config.similarity_threshold,
                        top_k=self.config.max_results
                    )
                    retrieved_faqs = [match.faq_entry for match in similar_matches]
                else:
                    self.logger.info(f"Filtered search found {len(retrieved_faqs)} relevant FAQs")
                
            except Exception as e:
                self.logger.warning(f"Filtered vector similarity search failed: {e}")
                # Fallback to unfiltered search if filtered failed
                try:
                    query_embedding = self.vectorizer.generate_embeddings(processed_query.corrected_query)
                    similar_matches = self.vector_store.search_similar(
                        query_embedding, 
                        self.config.similarity_threshold,
                        self.config.max_results
                    )
                    retrieved_faqs = [match.faq_entry for match in similar_matches]
                except Exception as inner_e:
                    self.logger.error(f"Unfiltered fallback search also failed: {inner_e}")
        
        # Strategy 2: Fallback to keyword-based search (if vector search failed)
        if not retrieved_faqs and self.vector_store:
            try:
                self.logger.debug("Attempting keyword-based fallback search...")
                # This would require implementing keyword search in vector store
                # For now, we'll skip this fallback
                pass
            except Exception as e:
                self.logger.warning(f"Keyword search fallback failed: {e}")
        
        # Strategy 3: Use expanded queries if main query failed
        if not retrieved_faqs and processed_query.expanded_queries and self.vectorizer and self.vector_store:
            for expanded_query in processed_query.expanded_queries[:3]:  # Try top 3 expansions
                try:
                    self.logger.debug(f"Trying expanded query: {expanded_query}")
                    
                    expanded_embedding = self.vectorizer.generate_embeddings(expanded_query)
                    expanded_matches = self.vector_store.search_similar(
                        expanded_embedding,
                        self.config.similarity_threshold * 0.8,  # Lower threshold for expansions
                        self.config.max_results
                    )
                    
                    if expanded_matches:
                        retrieved_faqs = [match.faq_entry for match in expanded_matches]
                        self.logger.info(f"Expanded query search found {len(retrieved_faqs)} FAQs")
                        break
                        
                except Exception as e:
                    self.logger.debug(f"Expanded query search failed: {e}")
                    continue
        
        return retrieved_faqs
    
    def _generate_response_with_fallback(self, processed_query: ProcessedQuery, 
                                       retrieved_faqs: List[FAQEntry], context, query_id: Optional[str], original_processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate response with fallback handling."""
        try:
            # Primary response generation
            response = self.response_generator.generate_response(
                processed_query.corrected_query, 
                retrieved_faqs,
                query_id=query_id,
                processed_query=processed_query
            )
            
            # Enhance response with context if available
            if context and hasattr(self.response_generator, 'format_response_with_context'):
                try:
                    enhanced_text = self.response_generator.format_response_with_context(response, context)
                    response.text = enhanced_text
                    response.context_used = True
                except Exception as e:
                    self.logger.warning(f"Context enhancement failed: {e}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            # Generate basic fallback response
            return self._generate_fallback_response(processed_query.original_query, str(e), query_id)
    
    def _generate_fallback_response(self, query: str, error_msg: str, query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate a basic fallback response when all else fails."""
        fallback_text = (
            f"I apologize, but I encountered an issue while processing your question about '{query}'. "
            "Please try rephrasing your question or contact support for assistance."
        )
        
        return Response(
            text=fallback_text,
            confidence=0.1,
            source_faqs=[],
            context_used=False,
            query_id=query_id,
            processed_query=processed_query,
            processing_time=0.0,
            generation_method='fallback',
            metadata={
                'error': error_msg,
                'fallback_reason': 'system_error',
                'original_query': query
            }
        )
    
    def _handle_performance_alert(self, alert) -> None:
        """Handle performance alerts from the performance monitor."""
        try:
            # Log the alert
            self.logger.warning(f"Performance Alert: [{alert.severity.value.upper()}] "
                              f"{alert.component}.{alert.metric}: {alert.message}")
            
            # Log to analytics manager if available
            if self.analytics_manager:
                self.analytics_manager.log_system_event(
                    event_type=f"performance_alert_{alert.severity.value}",
                    details={
                        'component': alert.component,
                        'metric': alert.metric,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold,
                        'message': alert.message
                    },
                    timestamp=alert.timestamp
                )
            
            # Take action based on alert severity
            if alert.severity == AlertSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ALERT: {alert.message}")
                # Could implement automatic recovery actions here
            
        except Exception as e:
            self.logger.error(f"Failed to handle performance alert: {e}")

    def _update_conversation_context(self, session_id: Optional[str], query: str, 
                                   processed_query: ProcessedQuery, response: Response) -> None:
        """Update conversation context with error handling."""
        if not session_id or not self.conversation_manager:
            return
        
        try:
            interaction = {
                'query': query,
                'processed_query': processed_query.corrected_query,
                'response': response.text,
                'confidence': response.confidence,
                'context_used': response.context_used,
                'metadata': {
                    'intent': processed_query.intent,
                    'language': processed_query.language,
                    'generation_method': response.generation_method
                }
            }
            self.conversation_manager.update_context(session_id, interaction)
            
        except Exception as e:
            self.logger.warning(f"Failed to update conversation context: {e}")
    
    def update_knowledge_base(self, faqs: List[FAQEntry]) -> None:
        """
        Update the knowledge base with new FAQs with comprehensive error handling.
        
        Args:
            faqs: List of FAQ entries to add to the knowledge base
            
        Raises:
            RAGSystemError: If knowledge base update fails
        """
        if not self._initialized:
            raise RAGSystemError("RAG system not properly initialized")
        
        if not faqs:
            self.logger.warning("No FAQs provided for knowledge base update")
            return
        
        self.logger.info(f"Updating knowledge base with {len(faqs)} FAQs")
        
        try:
            # Update activity timestamp
            self._system_stats['last_activity'] = datetime.now()
            
            # Generate embeddings for new FAQs if vectorizer available
            vectorized_faqs = []
            if self.vectorizer:
                try:
                    self.logger.info("Generating embeddings for new FAQs...")
                    vectorized_faqs = self.vectorizer.vectorize_faq_batch(faqs)
                    self.logger.info(f"Generated embeddings for {len(vectorized_faqs)} FAQs")
                except Exception as e:
                    self.logger.error(f"Embedding generation failed: {e}")
                    # Continue with FAQs without embeddings
                    vectorized_faqs = faqs
            else:
                self.logger.warning("Vectorizer not available, FAQs will not have embeddings")
                vectorized_faqs = faqs
            
            # Store in vector store if available
            if self.vector_store:
                try:
                    self.logger.info("Storing vectors in vector store...")
                    self.vector_store.store_vectors(vectorized_faqs)
                    self.logger.info("Vectors stored successfully")
                except Exception as e:
                    self.logger.error(f"Vector storage failed: {e}")
                    raise RAGSystemError(f"Failed to store vectors: {e}")
            else:
                self.logger.warning("Vector store not available, FAQs cannot be stored for similarity search")
            
            self.logger.info("Knowledge base updated successfully")
            
        except Exception as e:
            self._system_stats['errors_encountered'] += 1
            self._system_stats['last_error'] = {
                'timestamp': datetime.now(),
                'operation': 'update_knowledge_base',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.logger.error(f"Error updating knowledge base: {str(e)}")
            raise RAGSystemError(f"Knowledge base update failed: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system performance statistics and health information.
        
        Returns:
            Dictionary containing system statistics, component status, and performance metrics
        """
        try:
            stats = {
                'system_info': {
                    'initialized': self._initialized,
                    'initialization_time': self._system_stats['initialization_time'].isoformat(),
                    'last_activity': self._system_stats['last_activity'].isoformat() if self._system_stats['last_activity'] else None,
                    'uptime_seconds': (datetime.now() - self._system_stats['initialization_time']).total_seconds()
                },
                'performance_metrics': {
                    'queries_processed': self._system_stats['queries_processed'],
                    'documents_processed': self._system_stats['documents_processed'],
                    'errors_encountered': self._system_stats['errors_encountered'],
                    'error_rate': (self._system_stats['errors_encountered'] / max(1, self._system_stats['queries_processed'])) * 100
                },
                'component_status': self._component_status.copy(),
                'configuration': {
                    'similarity_threshold': self.config.similarity_threshold,
                    'max_results': self.config.max_results,
                    'vector_dimension': self.config.vector_dimension,
                    'session_timeout_minutes': self.config.session_timeout_minutes,
                    'max_conversation_history': self.config.max_conversation_history
                }
            }
            
            # Add component-specific statistics
            if self.vector_store:
                try:
                    vector_stats = self.vector_store.get_vector_stats()
                    stats['vector_store_stats'] = vector_stats
                except Exception as e:
                    stats['vector_store_stats'] = {'error': str(e)}
            
            if self.vectorizer:
                try:
                    vectorizer_stats = self.vectorizer.get_vectorizer_stats()
                    stats['vectorizer_stats'] = vectorizer_stats
                except Exception as e:
                    stats['vectorizer_stats'] = {'error': str(e)}
            
            if self.response_generator:
                try:
                    generator_stats = self.response_generator.get_generator_stats()
                    stats['response_generator_stats'] = generator_stats
                except Exception as e:
                    stats['response_generator_stats'] = {'error': str(e)}
            
            if self.conversation_manager:
                try:
                    conversation_stats = self.conversation_manager.get_session_stats()
                    stats['conversation_stats'] = conversation_stats
                except Exception as e:
                    stats['conversation_stats'] = {'error': str(e)}
            
            # Add last error information if available
            if self._system_stats['last_error']:
                stats['last_error'] = {
                    'timestamp': self._system_stats['last_error']['timestamp'].isoformat(),
                    'operation': self._system_stats['last_error']['operation'],
                    'error': self._system_stats['last_error']['error']
                    # Exclude traceback from stats for brevity
                }
            
            # Determine overall system status
            if not self._initialized:
                stats['system_status'] = 'not_initialized'
            elif self._system_stats['errors_encountered'] > 0:
                error_rate = stats['performance_metrics']['error_rate']
                if error_rate > 50:
                    stats['system_status'] = 'degraded'
                elif error_rate > 10:
                    stats['system_status'] = 'warning'
                else:
                    stats['system_status'] = 'operational'
            else:
                stats['system_status'] = 'operational'
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {str(e)}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary containing health check results for all components
        """
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'performance_monitoring': {}
        }
        
        try:
            # Check system initialization
            if not self._initialized:
                health_results['overall_status'] = 'unhealthy'
                health_results['issues'].append('System not properly initialized')
            
            # Check each component
            components_to_check = [
                ('docx_scraper', self.docx_scraper),
                ('query_processor', self.query_processor),
                ('vectorizer', self.vectorizer),
                ('vector_store', self.vector_store),
                ('response_generator', self.response_generator),
                ('conversation_manager', self.conversation_manager)
            ]
            
            for component_name, component in components_to_check:
                if component is None:
                    health_results['components'][component_name] = {
                        'status': 'not_available',
                        'message': 'Component not initialized'
                    }
                    continue
                
                try:
                    # Perform component-specific health checks
                    start_time = time.time()
                    
                    if hasattr(component, 'health_check'):
                        component_health = component.health_check()
                        health_results['components'][component_name] = component_health
                        
                        if component_health.get('status') != 'healthy':
                            health_results['issues'].append(f"{component_name}: {component_health.get('status', 'unknown')}")
                    else:
                        # Basic availability check
                        health_results['components'][component_name] = {
                            'status': 'available',
                            'message': 'Component available (no health check method)'
                        }
                    
                    # Monitor component health performance
                    response_time = time.time() - start_time
                    component_status = health_results['components'][component_name].get('status', 'unknown')
                    
                    if self.performance_monitor:
                        try:
                            # Map component status to health monitoring status
                            monitor_status = 'healthy'
                            if component_status in ['degraded', 'warning']:
                                monitor_status = 'degraded'
                            elif component_status in ['unhealthy', 'error', 'not_available']:
                                monitor_status = 'unhealthy'
                            
                            self.performance_monitor.monitor_system_health(
                                component_name=component_name,
                                status=monitor_status,
                                response_time=response_time,
                                error_rate=0.0 if component_status == 'healthy' else 0.5
                            )
                        except Exception as e:
                            self.logger.warning(f"Performance monitoring failed for {component_name}: {e}")
                        
                except Exception as e:
                    health_results['components'][component_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    health_results['issues'].append(f"{component_name}: {str(e)}")
                    
                    # Monitor error condition
                    if self.performance_monitor:
                        try:
                            self.performance_monitor.monitor_system_health(
                                component_name=component_name,
                                status='unhealthy',
                                response_time=1.0,
                                error_rate=1.0
                            )
                        except Exception as monitor_error:
                            self.logger.warning(f"Performance monitoring failed: {monitor_error}")
            
            # Add performance monitoring information
            if self.performance_monitor:
                try:
                    active_alerts = self.performance_monitor.get_active_alerts()
                    health_results['performance_monitoring'] = {
                        'enabled': True,
                        'active_alerts': len(active_alerts),
                        'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                        'high_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH])
                    }
                    
                    # Add critical alerts to issues
                    for alert in active_alerts:
                        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                            health_results['issues'].append(f"Performance Alert: {alert.message}")
                            
                except Exception as e:
                    health_results['performance_monitoring'] = {'enabled': True, 'error': str(e)}
            else:
                health_results['performance_monitoring'] = {'enabled': False}
            
            # Determine overall health status
            if health_results['issues']:
                if len(health_results['issues']) > 3:
                    health_results['overall_status'] = 'unhealthy'
                else:
                    health_results['overall_status'] = 'degraded'
            
            return health_results
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
    
    def reset_system(self) -> bool:
        """
        Reset the RAG system to initial state.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self.logger.info("Resetting RAG system...")
            
            # Reset statistics
            self._system_stats = {
                'queries_processed': 0,
                'documents_processed': 0,
                'errors_encountered': 0,
                'last_error': None,
                'initialization_time': datetime.now(),
                'last_activity': None
            }
            
            # Clear conversation sessions if manager available
            if self.conversation_manager:
                try:
                    # This would require implementing a clear_all_sessions method
                    # For now, we'll just clean up expired sessions
                    self.conversation_manager.cleanup_expired_sessions()
                except Exception as e:
                    self.logger.warning(f"Failed to clear conversation sessions: {e}")
            
            # Clear vector store if available
            if self.vector_store:
                try:
                    # This would require implementing a clear method
                    # For now, we'll just log the intent
                    self.logger.info("Vector store reset would be performed here")
                except Exception as e:
                    self.logger.warning(f"Failed to clear vector store: {e}")
            
            self.logger.info("RAG system reset completed")
            return True
            
        except Exception as e:
            self.logger.error(f"System reset failed: {e}")
            return False
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the RAG system.
        """
        try:
            self.logger.info("Shutting down RAG system...")
            
            # Stop performance monitoring
            if self.performance_monitor:
                try:
                    self.performance_monitor.stop_monitoring()
                    self.logger.info("Performance monitoring stopped")
                except Exception as e:
                    self.logger.warning(f"Performance monitoring shutdown failed: {e}")
            
            # Cleanup conversation sessions
            if self.conversation_manager:
                try:
                    self.conversation_manager.cleanup_expired_sessions()
                except Exception as e:
                    self.logger.warning(f"Conversation cleanup failed during shutdown: {e}")
            
            # Backup vector store if available
            if self.vector_store:
                try:
                    backup_path = self.vector_store.backup_store()
                    self.logger.info(f"Vector store backed up to: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Vector store backup failed during shutdown: {e}")
            
            self._initialized = False
            self.logger.info("RAG system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"System shutdown failed: {e}")
    
    def is_ready(self) -> bool:
        """
        Check if the RAG system is ready to process queries.
        
        Returns:
            True if system is ready, False otherwise
        """
        if not self._initialized:
            return False
        
        # Check that critical components are available
        critical_components = [self.response_generator]
        return all(component is not None for component in critical_components)
    
    def get_component_status(self) -> Dict[str, bool]:
        """
        Get the availability status of all components.
        
        Returns:
            Dictionary mapping component names to availability status
        """
        return {
            'docx_scraper': self.docx_scraper is not None,
            'query_processor': self.query_processor is not None,
            'vectorizer': self.vectorizer is not None,
            'vector_store': self.vector_store is not None,
            'response_generator': self.response_generator is not None,
            'conversation_manager': self.conversation_manager is not None,
            'performance_monitor': self.performance_monitor is not None,
            'system_improvement_manager': self.system_improvement_manager is not None,
            'system_initialized': self._initialized
        }

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Dictionary containing performance report or error message
        """
        if not self.performance_monitor:
            return {'error': 'Performance monitoring not enabled'}
        
        try:
            return self.performance_monitor.get_performance_report(hours)
        except Exception as e:
            self.logger.error(f"Failed to get performance report: {e}")
            return {'error': str(e)}

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get currently active performance alerts.
        
        Returns:
            List of active alerts or empty list if monitoring disabled
        """
        if not self.performance_monitor:
            return []
        
        try:
            alerts = self.performance_monitor.get_active_alerts()
            return [
                {
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'message': alert.message
                }
                for alert in alerts
            ]
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return []

    def track_confidence_scores(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze confidence score patterns and trends.
        
        Args:
            window_minutes: Time window for analysis in minutes
            
        Returns:
            Dictionary containing confidence score analysis
        """
        if not self.performance_monitor:
            return {'error': 'Performance monitoring not enabled'}
        
        try:
            return self.performance_monitor.track_confidence_scores(window_minutes)
        except Exception as e:
            self.logger.error(f"Failed to track confidence scores: {e}")
            return {'error': str(e)}

    def update_performance_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]) -> bool:
        """
        Update performance monitoring thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
            
        Returns:
            True if successful, False otherwise
        """
        if not self.performance_monitor:
            self.logger.warning("Performance monitoring not enabled")
            return False
        
        try:
            self.performance_monitor.update_thresholds(new_thresholds)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update performance thresholds: {e}")
            return False

    def resolve_performance_alert(self, component: str, metric: str) -> bool:
        """
        Resolve a performance alert.
        
        Args:
            component: Component name
            metric: Metric name
            
        Returns:
            True if alert was resolved, False otherwise
        """
        if not self.performance_monitor:
            return False
        
        try:
            return self.performance_monitor.resolve_alert(component, metric)
        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {e}")
            return False

    def create_embedding_version(self, model_name: str, faqs: List[FAQEntry]) -> Optional[str]:
        """
        Create a new embedding version with the system improvement manager.
        
        Args:
            model_name: Name of the embedding model
            faqs: List of FAQ entries to create embeddings for
            
        Returns:
            Version ID if successful, None otherwise
        """
        if not self.system_improvement_manager:
            self.logger.warning("System improvement manager not available")
            return None
        
        try:
            # Get current performance metrics for the version
            performance_metrics = {}
            if self.analytics_manager:
                recent_metrics = self.analytics_manager.get_performance_metrics()
                performance_metrics = {
                    'confidence_score': recent_metrics.get('response_quality', {}).get('average_confidence', 0.0),
                    'response_time': recent_metrics.get('performance_timing', {}).get('average_response_time', 0.0)
                }
            
            version_id = self.system_improvement_manager.create_embedding_version(
                model_name=model_name,
                faqs=faqs,
                performance_metrics=performance_metrics
            )
            
            self.logger.info(f"Created embedding version {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding version: {e}")
            return None

    def rollback_embedding_version(self, version_id: str) -> bool:
        """
        Rollback to a previous embedding version.
        
        Args:
            version_id: ID of the version to rollback to
            
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.system_improvement_manager:
            self.logger.warning("System improvement manager not available")
            return False
        
        try:
            success = self.system_improvement_manager.rollback_embedding_version(version_id)
            if success:
                self.logger.info(f"Successfully rolled back to embedding version {version_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rollback embedding version: {e}")
            return False

    def create_ab_test(self, name: str, description: str, strategy: str, 
                      control_config: Dict[str, Any], treatment_config: Dict[str, Any],
                      traffic_split: float = 0.5, duration_days: int = 7) -> Optional[str]:
        """
        Create an A/B test for system improvements.
        
        Args:
            name: Name of the A/B test
            description: Description of what is being tested
            strategy: Improvement strategy being tested
            control_config: Configuration for control group
            treatment_config: Configuration for treatment group
            traffic_split: Percentage of traffic for treatment (0.0 to 1.0)
            duration_days: Duration of the test in days
            
        Returns:
            Test ID if successful, None otherwise
        """
        if not self.system_improvement_manager:
            self.logger.warning("System improvement manager not available")
            return None
        
        try:
            # Import the enum here to avoid circular imports
            from .system_improvement import ImprovementStrategy
            
            # Convert string strategy to enum
            strategy_enum = ImprovementStrategy(strategy)
            
            test_id = self.system_improvement_manager.create_ab_test(
                name=name,
                description=description,
                strategy=strategy_enum,
                control_config=control_config,
                treatment_config=treatment_config,
                traffic_split=traffic_split,
                duration_days=duration_days
            )
            
            self.logger.info(f"Created A/B test {test_id}: {name}")
            return test_id
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            return None

    def get_user_ab_test_assignment(self, user_id: str, test_id: str) -> str:
        """
        Get A/B test assignment for a user.
        
        Args:
            user_id: ID of the user
            test_id: ID of the A/B test
            
        Returns:
            Group assignment ('control' or 'treatment')
        """
        if not self.system_improvement_manager:
            return 'control'
        
        try:
            return self.system_improvement_manager.assign_user_to_ab_test(user_id, test_id)
        except Exception as e:
            self.logger.error(f"Failed to get A/B test assignment: {e}")
            return 'control'

    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get system improvement recommendations.
        
        Returns:
            List of improvement recommendations
        """
        if not self.system_improvement_manager:
            return []
        
        try:
            return self.system_improvement_manager.get_improvement_recommendations()
        except Exception as e:
            self.logger.error(f"Failed to get improvement recommendations: {e}")
            return []

    def get_system_improvement_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system improvement report.
        
        Returns:
            Dictionary containing system improvement status and recommendations
        """
        if not self.system_improvement_manager:
            return {'error': 'System improvement manager not available'}
        
        try:
            return self.system_improvement_manager.get_system_improvement_report()
        except Exception as e:
            self.logger.error(f"Failed to get system improvement report: {e}")
            return {'error': str(e)}