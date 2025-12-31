from django.db import models
from django.contrib.auth.models import User
import json
import numpy as np
from datetime import datetime
from django.utils import timezone
import uuid

ist_time = timezone.localtime(timezone.now())


class EndUser(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    session_id = models.CharField(max_length=255, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} <{self.email}>"


class UserRequest(models.Model):
    user = models.ForeignKey(EndUser, on_delete=models.CASCADE, related_name="requests")
    session_id = models.CharField(max_length=255, db_index=True)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Request #{self.id} by {self.user.email}"


class BotResponse(models.Model):
    request = models.OneToOneField(UserRequest, on_delete=models.CASCADE, related_name="response")
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Response to Request #{self.request_id}"


class FAQEntry(models.Model):
    question = models.TextField()
    answer = models.TextField()
    keywords = models.CharField(max_length=512, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.question[:60]


class ChatFeedback(models.Model):
    """Store user feedback on chatbot responses for learning."""
    RATING_POSITIVE = 1
    RATING_NEGATIVE = -1
    RATING_CHOICES = [
        (RATING_POSITIVE, 'Helpful'),
        (RATING_NEGATIVE, 'Not Helpful'),
    ]
    
    message_id = models.CharField(max_length=100, db_index=True)
    faq = models.ForeignKey(FAQEntry, on_delete=models.SET_NULL, null=True, blank=True, related_name='feedback')
    rating = models.SmallIntegerField(choices=RATING_CHOICES)
    query = models.TextField(help_text="Original user query")
    response = models.TextField(help_text="Bot response that was rated")
    session_id = models.CharField(max_length=255, blank=True)
    user_email = models.EmailField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        rating_str = "👍" if self.rating > 0 else "👎"
        return f"{rating_str} {self.query[:30]}..."


class AuditLog(models.Model):
    """
    Model for tracking admin actions and security events.
    
    Implements Requirements 3.5:
    - Track admin actions and data access
    - Maintain security audit trail
    - Log encryption/decryption operations
    """
    
    # Event types for categorizing audit log entries
    EVENT_TYPES = [
        ('LOGIN', 'Admin Login'),
        ('LOGOUT', 'Admin Logout'),
        ('DATA_ACCESS', 'Data Access'),
        ('ENCRYPTION', 'Encryption Operation'),
        ('DECRYPTION', 'Decryption Operation'),
        ('SECURITY_VIOLATION', 'Security Violation'),
        ('SESSION_TIMEOUT', 'Session Timeout'),
        ('UNAUTHORIZED_ACCESS', 'Unauthorized Access Attempt'),
    ]
    
    # Severity levels for security events
    SEVERITY_LEVELS = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    # Admin user who performed the action (nullable for anonymous attempts)
    admin_user = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        help_text="Admin user who performed the action"
    )
    
    # Event type and severity
    event_type = models.CharField(
        max_length=20, 
        choices=EVENT_TYPES,
        help_text="Type of event being logged"
    )
    severity = models.CharField(
        max_length=10, 
        choices=SEVERITY_LEVELS, 
        default='LOW',
        help_text="Severity level of the event"
    )
    
    # Event details
    description = models.TextField(
        help_text="Detailed description of the event"
    )
    ip_address = models.GenericIPAddressField(
        null=True, 
        blank=True,
        help_text="IP address from which the action was performed"
    )
    user_agent = models.TextField(
        blank=True,
        help_text="User agent string from the request"
    )
    
    # Request details
    request_path = models.CharField(
        max_length=255, 
        blank=True,
        help_text="URL path that was accessed"
    )
    request_method = models.CharField(
        max_length=10, 
        blank=True,
        help_text="HTTP method used (GET, POST, etc.)"
    )
    
    # Additional context data (JSON field for flexible data storage)
    context_data = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Additional context data for the event"
    )
    
    # Timestamp
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the event occurred"
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event_type', 'created_at']),
            models.Index(fields=['admin_user', 'created_at']),
            models.Index(fields=['severity', 'created_at']),
            models.Index(fields=['ip_address', 'created_at']),
        ]
    
    def __str__(self):
        user_info = f"{self.admin_user.username}" if self.admin_user else "Anonymous"
        return f"{self.event_type} by {user_info} at {self.created_at}"
    
    @classmethod
    def log_admin_action(cls, event_type, description, admin_user=None, request=None, 
                        severity='LOW', context_data=None):
        """
        Convenience method for logging admin actions.
        
        Args:
            event_type: Type of event (must be one of EVENT_TYPES)
            description: Detailed description of the event
            admin_user: User object of the admin performing the action
            request: Django request object for extracting IP, user agent, etc.
            severity: Severity level of the event
            context_data: Additional context data as a dictionary
        """
        audit_entry = cls(
            admin_user=admin_user,
            event_type=event_type,
            severity=severity,
            description=description,
            context_data=context_data or {}
        )
        
        if request:
            # Extract request information
            audit_entry.ip_address = cls._get_client_ip(request)
            audit_entry.user_agent = request.META.get('HTTP_USER_AGENT', '')
            audit_entry.request_path = request.path
            audit_entry.request_method = request.method
        
        audit_entry.save()
        return audit_entry
    
    @staticmethod
    def _get_client_ip(request):
        """Extract client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


# RAG System Models

class RAGDocument(models.Model):
    """
    Model for tracking processed documents in the RAG system.
    Integrates with the RAG system's document ingestion pipeline.
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending Processing'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('updated', 'Updated'),
    ]
    
    # Document identification
    file_path = models.CharField(max_length=500, help_text="Path to the source document")
    file_name = models.CharField(max_length=255, help_text="Original filename")
    file_hash = models.CharField(max_length=64, db_index=True, help_text="SHA-256 hash of document content")
    file_size = models.BigIntegerField(help_text="File size in bytes")
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)
    
    # Processing results
    faqs_extracted = models.IntegerField(default=0, help_text="Number of FAQs extracted from document")
    error_message = models.TextField(blank=True, help_text="Error message if processing failed")
    
    # Metadata
    document_metadata = models.JSONField(default=dict, blank=True, help_text="Additional document metadata")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['file_hash', 'status']),
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.file_name} ({self.status})"
    
    @property
    def processing_duration(self):
        """Calculate processing duration if completed"""
        if self.processing_started_at and self.processing_completed_at:
            return self.processing_completed_at - self.processing_started_at
        return None


class RAGFAQEntry(models.Model):
    """
    Enhanced FAQ model for RAG system with embeddings and semantic search capabilities.
    Extends the basic FAQEntry with RAG-specific features.
    """
    
    # Link to source document
    document = models.ForeignKey(RAGDocument, on_delete=models.CASCADE, related_name='rag_faqs')
    
    # FAQ content (enhanced from basic FAQEntry)
    question = models.TextField(help_text="FAQ question text")
    answer = models.TextField(help_text="FAQ answer text")
    keywords = models.CharField(max_length=512, blank=True, db_index=True, help_text="Extracted keywords")
    category = models.CharField(max_length=100, blank=True, db_index=True, help_text="FAQ category")
    
    # Composite key components
    audience = models.CharField(max_length=100, blank=True, db_index=True, help_text="Target audience for the FAQ")
    intent = models.CharField(max_length=100, blank=True, db_index=True, help_text="User intent")
    condition = models.CharField(max_length=200, blank=True, db_index=True, help_text="Specific condition or context")
    
    # Unique composite key for precise filtering
    # Format: audience::category::intent::condition
    composite_key = models.CharField(max_length=500, db_index=True, null=True, blank=True, help_text="Composite identifier for filtering")
    
    # RAG-specific fields
    rag_id = models.CharField(max_length=100, unique=True, db_index=True, help_text="Unique RAG system identifier")
    confidence_score = models.FloatField(default=0.0, help_text="Extraction confidence score")
    source_section = models.CharField(max_length=200, blank=True, help_text="Document section where FAQ was found")
    
    # Embedding data (stored as JSON for compatibility)
    question_embedding = models.JSONField(null=True, blank=True, help_text="Question embedding vector")
    answer_embedding = models.JSONField(null=True, blank=True, help_text="Answer embedding vector")
    keywords_embedding = models.JSONField(null=True, blank=True, help_text="Keywords embedding vector")
    
    # Semantic search metadata
    embedding_model = models.CharField(max_length=100, blank=True, help_text="Model used for embeddings")
    embedding_version = models.CharField(max_length=50, blank=True, help_text="Embedding model version")
    
    # Usage statistics
    query_matches = models.IntegerField(default=0, help_text="Number of times matched in queries")
    last_matched = models.DateTimeField(null=True, blank=True, help_text="Last time this FAQ was matched")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['rag_id']),
            models.Index(fields=['composite_key']),
            models.Index(fields=['audience', 'category', 'intent']),
            models.Index(fields=['category', 'confidence_score']),
            models.Index(fields=['document', 'created_at']),
            models.Index(fields=['query_matches', 'last_matched']),
        ]
        verbose_name = "RAG FAQ Entry"
        verbose_name_plural = "RAG FAQ Entries"
    
    def save(self, *args, **kwargs):
        """Override save to automatically generate composite key and rag_id if not provided."""
        if not self.rag_id:
            self.rag_id = f"manual_{uuid.uuid4().hex[:8]}"
            
        if not self.composite_key:
            # Normalize components
            aud = (self.audience or "any").lower().strip()
            cat = (self.category or "general").lower().strip()
            intnt = (self.intent or "info").lower().strip()
            cond = (self.condition or "default").lower().strip()
            
            self.composite_key = f"{aud}::{cat}::{intnt}::{cond}"
            
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.question[:60]}... (Score: {self.confidence_score:.2f})"
    
    def get_question_embedding_array(self):
        """Convert question embedding JSON to numpy array"""
        if self.question_embedding:
            return np.array(self.question_embedding)
        return None
    
    def set_question_embedding_array(self, embedding_array):
        """Set question embedding from numpy array"""
        if embedding_array is not None:
            self.question_embedding = embedding_array.tolist()
        else:
            self.question_embedding = None
    
    def get_answer_embedding_array(self):
        """Convert answer embedding JSON to numpy array"""
        if self.answer_embedding:
            return np.array(self.answer_embedding)
        return None
    
    def set_answer_embedding_array(self, embedding_array):
        """Set answer embedding from numpy array"""
        if embedding_array is not None:
            self.answer_embedding = embedding_array.tolist()
        else:
            self.answer_embedding = None
    
    def increment_match_count(self):
        """Increment query match count and update last matched timestamp"""
        self.query_matches += 1
        self.last_matched = timezone.now()
        self.save(update_fields=['query_matches', 'last_matched'])


class RAGConversationSession(models.Model):
    """
    Model for tracking conversation sessions in the RAG system.
    Integrates with the conversation manager component.
    """
    
    # Session identification
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    user = models.ForeignKey(EndUser, on_delete=models.CASCADE, related_name='rag_sessions', null=True, blank=True)
    
    # Session metadata
    current_topic = models.CharField(max_length=200, blank=True, help_text="Current conversation topic")
    language = models.CharField(max_length=10, default='en', help_text="Session language")
    user_preferences = models.JSONField(default=dict, blank=True, help_text="User preferences for this session")
    
    # Session statistics
    total_interactions = models.IntegerField(default=0, help_text="Total number of interactions")
    successful_responses = models.IntegerField(default=0, help_text="Number of successful responses")
    average_confidence = models.FloatField(default=0.0, help_text="Average response confidence")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True, help_text="Session expiration time")
    
    class Meta:
        ordering = ['-last_activity']
        indexes = [
            models.Index(fields=['session_id', 'last_activity']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['expires_at']),
        ]
    
    def __str__(self):
        return f"Session {self.session_id} ({self.total_interactions} interactions)"
    
    @property
    def is_expired(self):
        """Check if session is expired"""
        if self.expires_at:
            return timezone.now() > self.expires_at
        return False
    
    @property
    def success_rate(self):
        """Calculate success rate percentage"""
        if self.total_interactions > 0:
            return (self.successful_responses / self.total_interactions) * 100
        return 0.0


class RAGQueryLog(models.Model):
    """
    Model for logging RAG system queries and responses.
    Integrates with the analytics manager component.
    """
    
    GENERATION_METHODS = [
        ('rag', 'RAG Generation'),
        ('direct_match', 'Direct Match'),
        ('synthesized', 'Multi-source Synthesis'),
        ('fallback', 'Fallback Response'),
    ]
    
    # Query identification
    query_id = models.CharField(max_length=100, unique=True, db_index=True)
    session = models.ForeignKey(RAGConversationSession, on_delete=models.CASCADE, related_name='queries', null=True, blank=True)
    
    # Query content
    original_query = models.TextField(help_text="Original user query")
    corrected_query = models.TextField(blank=True, help_text="Typo-corrected query")
    intent = models.CharField(max_length=100, blank=True, help_text="Extracted query intent")
    language = models.CharField(max_length=10, default='en', help_text="Detected language")
    
    # Response content
    response_text = models.TextField(help_text="Generated response text")
    confidence_score = models.FloatField(help_text="Response confidence score")
    generation_method = models.CharField(max_length=20, choices=GENERATION_METHODS, help_text="Response generation method")
    context_used = models.BooleanField(default=False, help_text="Whether conversation context was used")
    
    # Performance metrics
    processing_time = models.FloatField(help_text="Query processing time in seconds")
    matched_faqs_count = models.IntegerField(default=0, help_text="Number of FAQs matched")
    
    # Source FAQs (many-to-many relationship)
    source_faqs = models.ManyToManyField(RAGFAQEntry, blank=True, related_name='query_logs')
    
    # Additional metadata
    query_metadata = models.JSONField(default=dict, blank=True, help_text="Additional query metadata")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['query_id']),
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['confidence_score', 'created_at']),
            models.Index(fields=['generation_method', 'created_at']),
            models.Index(fields=['language', 'created_at']),
        ]
    
    def __str__(self):
        return f"Query: {self.original_query[:50]}... (Confidence: {self.confidence_score:.2f})"


class RAGUserFeedback(models.Model):
    """
    Model for storing user feedback on RAG system responses.
    Integrates with the feedback manager component.
    """
    
    RATING_CHOICES = [
        (1, 'Very Poor'),
        (2, 'Poor'),
        (3, 'Average'),
        (4, 'Good'),
        (5, 'Excellent'),
    ]
    
    # Feedback identification
    query_log = models.OneToOneField(RAGQueryLog, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(EndUser, on_delete=models.CASCADE, related_name='rag_feedback', null=True, blank=True)
    
    # Feedback content
    rating = models.IntegerField(choices=RATING_CHOICES, help_text="User rating (1-5)")
    comments = models.TextField(blank=True, help_text="Optional user comments")
    
    # Feedback categories
    accuracy_rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True, help_text="Response accuracy rating")
    relevance_rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True, help_text="Response relevance rating")
    helpfulness_rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True, help_text="Response helpfulness rating")
    
    # Feedback metadata
    feedback_metadata = models.JSONField(default=dict, blank=True, help_text="Additional feedback metadata")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['rating', 'created_at']),
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"Feedback for Query {self.query_log.query_id}: {self.rating}/5"


class RAGSystemMetrics(models.Model):
    """
    Model for storing RAG system performance metrics.
    Integrates with the performance monitor component.
    """
    
    METRIC_TYPES = [
        ('response_time', 'Response Time'),
        ('confidence_score', 'Confidence Score'),
        ('success_rate', 'Success Rate'),
        ('error_rate', 'Error Rate'),
        ('query_volume', 'Query Volume'),
        ('system_health', 'System Health'),
    ]
    
    # Metric identification
    metric_type = models.CharField(max_length=50, choices=METRIC_TYPES, db_index=True)
    component_name = models.CharField(max_length=100, blank=True, help_text="Component name (if applicable)")
    
    # Metric values
    value = models.FloatField(help_text="Metric value")
    threshold = models.FloatField(null=True, blank=True, help_text="Alert threshold (if applicable)")
    
    # Time window
    window_start = models.DateTimeField(help_text="Metric window start time")
    window_end = models.DateTimeField(help_text="Metric window end time")
    
    # Additional data
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional metric metadata")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['metric_type', 'created_at']),
            models.Index(fields=['component_name', 'metric_type']),
            models.Index(fields=['window_start', 'window_end']),
        ]
    
    def __str__(self):
        component = f" ({self.component_name})" if self.component_name else ""
        return f"{self.metric_type}{component}: {self.value}"


class RAGPerformanceAlert(models.Model):
    """
    Model for storing RAG system performance alerts.
    Integrates with the performance monitor component.
    """
    
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('resolved', 'Resolved'),
        ('acknowledged', 'Acknowledged'),
    ]
    
    # Alert identification
    alert_id = models.CharField(max_length=100, unique=True, db_index=True)
    component_name = models.CharField(max_length=100, help_text="Component that triggered the alert")
    metric_name = models.CharField(max_length=100, help_text="Metric that triggered the alert")
    
    # Alert details
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', db_index=True)
    message = models.TextField(help_text="Alert message")
    
    # Metric values
    current_value = models.FloatField(help_text="Current metric value")
    threshold_value = models.FloatField(help_text="Threshold that was exceeded")
    
    # Resolution
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.CharField(max_length=100, blank=True, help_text="Who/what resolved the alert")
    resolution_notes = models.TextField(blank=True, help_text="Resolution notes")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['alert_id']),
            models.Index(fields=['severity', 'status']),
            models.Index(fields=['component_name', 'status']),
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.severity.upper()} Alert: {self.component_name}.{self.metric_name}"
    
    def resolve(self, resolved_by=None, notes=None):
        """Mark alert as resolved"""
        self.status = 'resolved'
        self.resolved_at = timezone.now()
        if resolved_by:
            self.resolved_by = resolved_by
        if notes:
            self.resolution_notes = notes
        self.save()
