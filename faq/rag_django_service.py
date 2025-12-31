"""
Django Integration Service for RAG System

This service provides integration between the RAG system components and Django models,
enabling seamless data synchronization and management through Django's ORM.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from faq.models import (
    RAGDocument, RAGFAQEntry, RAGConversationSession,
    RAGQueryLog, RAGUserFeedback, RAGSystemMetrics, RAGPerformanceAlert,
    EndUser
)
from faq.rag.config.settings import rag_config
from .rag.interfaces.base import (
    FAQEntry, ProcessedQuery, Response, ConversationContext,
    AnalyticsManagerInterface, FeedbackManagerInterface
)
from .rag.core.rag_system import RAGSystem


logger = logging.getLogger(__name__)


class RAGDjangoService:
    """
    Service class that integrates RAG system with Django models.
    Provides methods for synchronizing data between RAG components and Django ORM.
    """
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        """
        Initialize the Django integration service.
        
        Args:
            rag_system: Optional RAG system instance
        """
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
    
    # Document Management
    
    def create_document_record(self, file_path: str, file_name: str, 
                             file_hash: str, file_size: int) -> RAGDocument:
        """
        Create a new document record in Django.
        
        Args:
            file_path: Path to the document file
            file_name: Original filename
            file_hash: SHA-256 hash of document content
            file_size: File size in bytes
            
        Returns:
            Created RAGDocument instance
        """
        try:
            document = RAGDocument.objects.create(
                file_path=file_path,
                file_name=file_name,
                file_hash=file_hash,
                file_size=file_size,
                status='pending'
            )
            self.logger.info(f"Created document record: {document.file_name}")
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to create document record: {e}")
            raise
    
    def update_document_processing_status(self, document_id: int, status: str, 
                                        faqs_extracted: int = 0, 
                                        error_message: str = "") -> None:
        """
        Update document processing status.
        
        Args:
            document_id: Document ID
            status: New status
            faqs_extracted: Number of FAQs extracted
            error_message: Error message if processing failed
        """
        try:
            document = RAGDocument.objects.get(id=document_id)
            document.status = status
            document.faqs_extracted = faqs_extracted
            document.error_message = error_message
            
            if status == 'processing' and not document.processing_started_at:
                document.processing_started_at = timezone.now()
            elif status in ['completed', 'failed']:
                document.processing_completed_at = timezone.now()
            
            document.save()
            self.logger.info(f"Updated document {document_id} status to {status}")
            
        except ObjectDoesNotExist:
            self.logger.error(f"Document {document_id} not found")
            raise
        except Exception as e:
            self.logger.error(f"Failed to update document status: {e}")
            raise
    
    def sync_faqs_to_django(self, faqs: List[FAQEntry], document_id: int) -> List[RAGFAQEntry]:
        """
        Synchronize FAQ entries from RAG system to Django models.
        
        Args:
            faqs: List of FAQ entries from RAG system
            document_id: Associated document ID
            
        Returns:
            List of created RAGFAQEntry instances
        """
        try:
            document = RAGDocument.objects.get(id=document_id)
            django_faqs = []
            
            with transaction.atomic():
                for faq in faqs:
                    # Convert RAG FAQ to Django model
                    django_faq, created = RAGFAQEntry.objects.update_or_create(
                        rag_id=faq.id,
                        defaults={
                            'document': document,
                            'question': faq.question,
                            'answer': faq.answer,
                            'keywords': ', '.join(faq.keywords) if faq.keywords else '',
                            'category': faq.category,
                            'confidence_score': faq.confidence_score,
                            'source_section': faq.source_document,
                            'embedding_model': rag_config.config.embedding_type,
                            'embedding_version': '1.0'
                        }
                    )
                    
                    # Store embeddings if available
                    if faq.embedding is not None:
                        django_faq.set_question_embedding_array(faq.embedding)
                        django_faq.save()
                    
                    django_faqs.append(django_faq)
            
            self.logger.info(f"Synchronized {len(django_faqs)} FAQs to Django for document {document_id}")
            return django_faqs
            
        except Exception as e:
            self.logger.error(f"Failed to sync FAQs to Django: {e}")
            raise
    
    # Session Management
    
    def get_or_create_session(self, session_id: str, user_email: str = None) -> RAGConversationSession:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Session identifier
            user_email: Optional user email
            
        Returns:
            RAGConversationSession instance
        """
        try:
            # Try to get existing session
            session = RAGConversationSession.objects.get(session_id=session_id)
            
            # Update last activity
            session.last_activity = timezone.now()
            session.save(update_fields=['last_activity'])
            
            return session
            
        except ObjectDoesNotExist:
            # Create new session
            user = None
            if user_email:
                try:
                    user = EndUser.objects.get(email=user_email)
                except ObjectDoesNotExist:
                    # Create user if doesn't exist
                    user = EndUser.objects.create(
                        name=user_email.split('@')[0],
                        email=user_email,
                        session_id=session_id
                    )
            
            # Set expiration time (24 hours from now)
            expires_at = timezone.now() + timedelta(hours=24)
            
            session = RAGConversationSession.objects.create(
                session_id=session_id,
                user=user,
                expires_at=expires_at
            )
            
            self.logger.info(f"Created new session: {session_id}")
            return session
    
    def update_session_stats(self, session_id: str, response_confidence: float, 
                           successful: bool = True) -> None:
        """
        Update session statistics.
        
        Args:
            session_id: Session identifier
            response_confidence: Response confidence score
            successful: Whether the response was successful
        """
        try:
            session = RAGConversationSession.objects.get(session_id=session_id)
            
            session.total_interactions += 1
            if successful:
                session.successful_responses += 1
            
            # Update average confidence (running average)
            if session.total_interactions == 1:
                session.average_confidence = response_confidence
            else:
                session.average_confidence = (
                    (session.average_confidence * (session.total_interactions - 1) + response_confidence) 
                    / session.total_interactions
                )
            
            session.save()
            
        except ObjectDoesNotExist:
            self.logger.warning(f"Session {session_id} not found for stats update")
        except Exception as e:
            self.logger.error(f"Failed to update session stats: {e}")
    
    def query_rag_system(self, query: str, session_id: str, user_email: str):
        """
        Queries the RAG system with the given input.
        """
        if not self.rag_system:
            raise ValueError("RAGSystem is not initialized in RAGDjangoService.")
        return self.rag_system.answer_query(query=query, session_id=session_id)
    
    # Query Logging
    
    def log_query_and_response(self, query_id: str, session_id: str, 
                             processed_query: ProcessedQuery, response: Response,
                             processing_time: float, source_faq_ids: List[str] = None) -> RAGQueryLog:
        """
        Log a query and its response to Django.
        
        Args:
            query_id: Unique query identifier
            session_id: Session identifier
            processed_query: Processed query object
            response: Response object
            processing_time: Processing time in seconds
            source_faq_ids: List of source FAQ IDs
            
        Returns:
            Created RAGQueryLog instance
        """
        try:
            # Get session
            session = None
            if session_id:
                try:
                    session = RAGConversationSession.objects.get(session_id=session_id)
                except ObjectDoesNotExist:
                    self.logger.warning(f"Session {session_id} not found for query logging")
            
            # Create query log
            query_log = RAGQueryLog.objects.create(
                query_id=query_id,
                session=session,
                original_query=processed_query.original_query,
                corrected_query=processed_query.corrected_query,
                intent=processed_query.intent,
                language=processed_query.language,
                response_text=response.text,
                confidence_score=response.confidence,
                generation_method=response.generation_method,
                context_used=response.context_used,
                processing_time=processing_time,
                matched_faqs_count=len(response.source_faqs),
                query_metadata=response.metadata
            )
            
            # Link source FAQs if provided
            if source_faq_ids:
                source_faqs = RAGFAQEntry.objects.filter(rag_id__in=source_faq_ids)
                query_log.source_faqs.set(source_faqs)
                
                # Update FAQ match counts
                for faq in source_faqs:
                    faq.increment_match_count()
            
            self.logger.info(f"Logged query: {query_id}")
            return query_log
            
        except Exception as e:
            self.logger.error(f"Failed to log query: {e}")
            raise
    
    # Feedback Management
    
    def submit_feedback(self, query_id: str, user_email: str, rating: int, 
                       comments: str = "", accuracy_rating: int = None,
                       relevance_rating: int = None, helpfulness_rating: int = None) -> RAGUserFeedback:
        """
        Submit user feedback for a query.
        
        Args:
            query_id: Query identifier
            user_email: User email
            rating: Overall rating (1-5)
            comments: Optional comments
            accuracy_rating: Accuracy rating (1-5)
            relevance_rating: Relevance rating (1-5)
            helpfulness_rating: Helpfulness rating (1-5)
            
        Returns:
            Created RAGUserFeedback instance
        """
        try:
            # Get or create query log
            query_log, created = RAGQueryLog.objects.get_or_create(
                query_id=query_id,
                defaults={
                    'original_query': f"Feedback for unlogged query {query_id}",
                    'response_text': "N/A",
                    'confidence_score': 0.0,
                    'generation_method': 'feedback_only',
                    'processing_time': 0.0
                }
            )
            
            # Get or create user
            user = None
            if user_email:
                user, created = EndUser.objects.get_or_create(
                    email=user_email,
                    defaults={
                        'name': user_email.split('@')[0],
                        'session_id': f"feedback_{timezone.now().timestamp()}"
                    }
                )
            
            # Create feedback
            feedback = RAGUserFeedback.objects.create(
                query_log=query_log,
                user=user,
                rating=rating,
                comments=comments,
                accuracy_rating=accuracy_rating,
                relevance_rating=relevance_rating,
                helpfulness_rating=helpfulness_rating
            )
            
            self.logger.info(f"Submitted feedback for query: {query_id}")
            return feedback
            
        except ObjectDoesNotExist:
            self.logger.error(f"Query {query_id} not found for feedback")
            raise
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {e}")
            raise
    
    # Metrics and Monitoring
    
    def record_system_metric(self, metric_type: str, value: float, 
                           component_name: str = "", threshold: float = None,
                           window_start: datetime = None, window_end: datetime = None,
                           metadata: Dict[str, Any] = None) -> RAGSystemMetrics:
        """
        Record a system performance metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            component_name: Component name
            threshold: Alert threshold
            window_start: Metric window start
            window_end: Metric window end
            metadata: Additional metadata
            
        Returns:
            Created RAGSystemMetrics instance
        """
        try:
            if not window_start:
                window_start = timezone.now() - timedelta(minutes=5)
            if not window_end:
                window_end = timezone.now()
            
            metric = RAGSystemMetrics.objects.create(
                metric_type=metric_type,
                component_name=component_name,
                value=value,
                threshold=threshold,
                window_start=window_start,
                window_end=window_end,
                metadata=metadata or {}
            )
            
            return metric
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
            raise
    
    def create_performance_alert(self, alert_id: str, component_name: str, 
                               metric_name: str, severity: str, message: str,
                               current_value: float, threshold_value: float) -> RAGPerformanceAlert:
        """
        Create a performance alert.
        
        Args:
            alert_id: Unique alert identifier
            component_name: Component that triggered the alert
            metric_name: Metric that triggered the alert
            severity: Alert severity
            message: Alert message
            current_value: Current metric value
            threshold_value: Threshold that was exceeded
            
        Returns:
            Created RAGPerformanceAlert instance
        """
        try:
            alert = RAGPerformanceAlert.objects.create(
                alert_id=alert_id,
                component_name=component_name,
                metric_name=metric_name,
                severity=severity,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value
            )
            
            self.logger.warning(f"Created performance alert: {alert_id}")
            return alert
            
        except Exception as e:
            self.logger.error(f"Failed to create performance alert: {e}")
            raise
    
    # Analytics and Reporting
    
    def get_system_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get system analytics for the specified number of days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing analytics data
        """
        try:
            start_date = timezone.now() - timedelta(days=days)
            
            from django.db import models
            
            # Query statistics
            total_queries = RAGQueryLog.objects.filter(created_at__gte=start_date).count()
            avg_confidence = RAGQueryLog.objects.filter(
                created_at__gte=start_date
            ).aggregate(avg_confidence=models.Avg('confidence_score'))['avg_confidence'] or 0
            
            avg_processing_time = RAGQueryLog.objects.filter(
                created_at__gte=start_date
            ).aggregate(avg_time=models.Avg('processing_time'))['avg_time'] or 0
            
            # Feedback statistics
            feedback_count = RAGUserFeedback.objects.filter(
                created_at__gte=start_date
            ).count()
            
            avg_rating = RAGUserFeedback.objects.filter(
                created_at__gte=start_date
            ).aggregate(avg_rating=models.Avg('rating'))['avg_rating'] or 0
            
            # Document statistics
            documents_processed = RAGDocument.objects.filter(
                created_at__gte=start_date,
                status='completed'
            ).count()
            
            total_faqs = RAGFAQEntry.objects.filter(
                created_at__gte=start_date
            ).count()
            
            # Active sessions
            active_sessions = RAGConversationSession.objects.filter(
                last_activity__gte=start_date
            ).count()
            
            # Performance alerts
            active_alerts = RAGPerformanceAlert.objects.filter(
                status='active',
                created_at__gte=start_date
            ).count()
            
            return {
                'period_days': days,
                'query_statistics': {
                    'total_queries': total_queries,
                    'average_confidence': round(avg_confidence, 3),
                    'average_processing_time': round(avg_processing_time, 3)
                },
                'feedback_statistics': {
                    'feedback_count': feedback_count,
                    'average_rating': round(avg_rating, 2)
                },
                'document_statistics': {
                    'documents_processed': documents_processed,
                    'total_faqs_extracted': total_faqs
                },
                'session_statistics': {
                    'active_sessions': active_sessions
                },
                'system_health': {
                    'active_alerts': active_alerts
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system analytics: {e}")
            return {}
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired conversation sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            expired_sessions = RAGConversationSession.objects.filter(
                expires_at__lt=timezone.now()
            )
            count = expired_sessions.count()
            expired_sessions.delete()
            
            self.logger.info(f"Cleaned up {count} expired sessions")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_top_faqs(self, limit: int = 10) -> List[RAGFAQEntry]:
        """
        Get the most frequently matched FAQs.
        
        Args:
            limit: Maximum number of FAQs to return
            
        Returns:
            List of top FAQ entries
        """
        try:
            return list(RAGFAQEntry.objects.filter(
                query_matches__gt=0
            ).order_by('-query_matches')[:limit])
            
        except Exception as e:
            self.logger.error(f"Failed to get top FAQs: {e}")
            return []
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system health status.
        
        Returns:
            Dictionary containing health summary
        """
        try:
            # Count active alerts by severity
            critical_alerts = RAGPerformanceAlert.objects.filter(
                status='active', severity='critical'
            ).count()
            
            high_alerts = RAGPerformanceAlert.objects.filter(
                status='active', severity='high'
            ).count()
            
            medium_alerts = RAGPerformanceAlert.objects.filter(
                status='active', severity='medium'
            ).count()
            
            low_alerts = RAGPerformanceAlert.objects.filter(
                status='active', severity='low'
            ).count()
            
            # Recent query success rate
            recent_queries = RAGQueryLog.objects.filter(
                created_at__gte=timezone.now() - timedelta(hours=1)
            )
            
            total_recent = recent_queries.count()
            successful_recent = recent_queries.filter(confidence_score__gte=0.5).count()
            success_rate = (successful_recent / total_recent * 100) if total_recent > 0 else 0
            
            # Determine overall health status
            if critical_alerts > 0:
                health_status = 'critical'
            elif high_alerts > 0:
                health_status = 'degraded'
            elif medium_alerts > 0 or success_rate < 80:
                health_status = 'warning'
            else:
                health_status = 'healthy'
            
            return {
                'overall_status': health_status,
                'alerts': {
                    'critical': critical_alerts,
                    'high': high_alerts,
                    'medium': medium_alerts,
                    'low': low_alerts,
                    'total_active': critical_alerts + high_alerts + medium_alerts + low_alerts
                },
                'performance': {
                    'recent_success_rate': round(success_rate, 1),
                    'recent_query_count': total_recent
                },
                'timestamp': timezone.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health summary: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': timezone.now().isoformat()
            }


# Django-specific Analytics and Feedback Managers

class DjangoAnalyticsManager(AnalyticsManagerInterface):
    """
    Django-based implementation of AnalyticsManagerInterface.
    Stores analytics data in Django models.
    """
    
    def __init__(self):
        self.service = RAGDjangoService()
        self.logger = logging.getLogger(__name__)
    
    def log_query(self, query_id: str, query_text: str, processed_query: ProcessedQuery, 
                 response: Response, timestamp: datetime) -> None:
        """Log a processed query and its response."""
        try:
            processing_time = response.metadata.get('response_time', 0.0)
            source_faq_ids = [faq.id for faq in response.source_faqs]
            
            self.service.log_query_and_response(
                query_id=query_id,
                session_id=response.metadata.get('session_id'),
                processed_query=processed_query,
                response=response,
                processing_time=processing_time,
                source_faq_ids=source_faq_ids
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log query in Django: {e}")
    
    def log_document_ingestion(self, document_path: str, faqs_ingested: int, 
                             timestamp: datetime, status: str, error: Optional[str] = None) -> None:
        """Log document ingestion events."""
        try:
            # This would typically be handled by the document processing workflow
            # For now, we'll record it as a system metric
            self.service.record_system_metric(
                metric_type='document_ingestion',
                value=faqs_ingested,
                component_name='document_processor',
                metadata={
                    'document_path': document_path,
                    'status': status,
                    'error': error
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log document ingestion: {e}")
    
    def log_system_event(self, event_type: str, details: Dict[str, Any], timestamp: datetime) -> None:
        """Log general system events."""
        try:
            self.service.record_system_metric(
                metric_type='system_event',
                value=1.0,  # Event occurrence
                component_name=details.get('component', 'system'),
                metadata={
                    'event_type': event_type,
                    'details': details
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
    
    def get_query_patterns(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retrieve common query patterns and trends."""
        try:
            if not start_date:
                start_date = timezone.now() - timedelta(days=7)
            if not end_date:
                end_date = timezone.now()
            
            # This is a simplified implementation
            # In a real system, you'd want more sophisticated pattern analysis
            from django.db import models
            
            patterns = RAGQueryLog.objects.filter(
                created_at__range=[start_date, end_date]
            ).values('intent', 'language').annotate(
                count=models.Count('id'),
                avg_confidence=models.Avg('confidence_score')
            ).order_by('-count')
            
            return list(patterns)
            
        except Exception as e:
            self.logger.error(f"Failed to get query patterns: {e}")
            return []
    
    def get_performance_metrics(self, start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Retrieve system performance metrics."""
        try:
            if not start_date:
                start_date = timezone.now() - timedelta(days=1)
            if not end_date:
                end_date = timezone.now()
            
            from django.db import models
            from django.db.models import Q
            
            metrics = RAGQueryLog.objects.filter(
                created_at__range=[start_date, end_date]
            ).aggregate(
                total_queries=models.Count('id'),
                avg_confidence=models.Avg('confidence_score'),
                avg_processing_time=models.Avg('processing_time'),
                successful_queries=models.Count('id', filter=Q(confidence_score__gte=0.5))
            )
            
            success_rate = 0
            if metrics['total_queries'] > 0:
                success_rate = (metrics['successful_queries'] / metrics['total_queries']) * 100
            
            return {
                'total_queries': metrics['total_queries'] or 0,
                'successful_queries': metrics['successful_queries'] or 0,
                'success_rate': round(success_rate, 2),
                'average_confidence': round(metrics['avg_confidence'] or 0, 3),
                'response_quality': {
                    'average_confidence': round(metrics['avg_confidence'] or 0, 3)
                },
                'performance_timing': {
                    'average_response_time': round(metrics['avg_processing_time'] or 0, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}


class DjangoFeedbackManager(FeedbackManagerInterface):
    """
    Django-based implementation of FeedbackManagerInterface.
    Stores feedback data in Django models.
    """
    
    def __init__(self):
        self.service = RAGDjangoService()
        self.logger = logging.getLogger(__name__)
    
    def submit_feedback(self, query_id: str, user_id: str, rating: int, 
                       comments: Optional[str] = None) -> None:
        """Submit user feedback for a specific query."""
        try:
            self.service.submit_feedback(
                query_id=query_id,
                user_email=user_id,  # Assuming user_id is email
                rating=rating,
                comments=comments or ""
            )
            
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {e}")
    
    def get_feedback(self, query_id: Optional[str] = None, 
                    user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve user feedback."""
        try:
            queryset = RAGUserFeedback.objects.all()
            
            if query_id:
                queryset = queryset.filter(query_log__query_id=query_id)
            if user_id:
                queryset = queryset.filter(user__email=user_id)
            
            feedback_list = []
            for feedback in queryset:
                feedback_list.append({
                    'query_id': feedback.query_log.query_id,
                    'user_id': feedback.user.email if feedback.user else None,
                    'rating': feedback.rating,
                    'comments': feedback.comments,
                    'accuracy_rating': feedback.accuracy_rating,
                    'relevance_rating': feedback.relevance_rating,
                    'helpfulness_rating': feedback.helpfulness_rating,
                    'created_at': feedback.created_at.isoformat()
                })
            
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Failed to get feedback: {e}")
            return []
    
    def analyze_feedback(self, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze collected feedback to identify areas for improvement."""
        try:
            if not start_date:
                start_date = timezone.now() - timedelta(days=30)
            if not end_date:
                end_date = timezone.now()
            
            from django.db import models
            
            feedback_stats = RAGUserFeedback.objects.filter(
                created_at__range=[start_date, end_date]
            ).aggregate(
                total_feedback=models.Count('id'),
                avg_rating=models.Avg('rating'),
                avg_accuracy=models.Avg('accuracy_rating'),
                avg_relevance=models.Avg('relevance_rating'),
                avg_helpfulness=models.Avg('helpfulness_rating')
            )
            
            # Rating distribution
            rating_distribution = RAGUserFeedback.objects.filter(
                created_at__range=[start_date, end_date]
            ).values('rating').annotate(count=models.Count('id')).order_by('rating')
            
            return {
                'total_feedback': feedback_stats['total_feedback'] or 0,
                'average_rating': round(feedback_stats['avg_rating'] or 0, 2),
                'average_accuracy': round(feedback_stats['avg_accuracy'] or 0, 2),
                'average_relevance': round(feedback_stats['avg_relevance'] or 0, 2),
                'average_helpfulness': round(feedback_stats['avg_helpfulness'] or 0, 2),
                'rating_distribution': list(rating_distribution),
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feedback: {e}")
            return {}