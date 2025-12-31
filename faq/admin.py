from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.utils import timezone

from .models import (
    EndUser, UserRequest, BotResponse, FAQEntry,
    RAGDocument, RAGFAQEntry, RAGConversationSession, 
    RAGQueryLog, RAGUserFeedback, RAGSystemMetrics, RAGPerformanceAlert
)


@admin.register(EndUser)
class EndUserAdmin(admin.ModelAdmin):
    list_display = ("name", "email", "session_id", "created_at")
    search_fields = ("name", "email", "session_id")


@admin.register(UserRequest)
class UserRequestAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "session_id", "created_at")
    search_fields = ("user__email", "session_id", "text")
    list_select_related = ("user",)


@admin.register(BotResponse)
class BotResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "request", "created_at")
    search_fields = ("text", "request__user__email")
    list_select_related = ("request",)


@admin.register(FAQEntry)
class FAQEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "question", "keywords", "updated_at")
    search_fields = ("question", "answer", "keywords")


# RAG System Admin Classes

@admin.register(RAGDocument)
class RAGDocumentAdmin(admin.ModelAdmin):
    list_display = ("file_name", "status", "faqs_extracted", "file_size_mb", "created_at", "processing_duration_display")
    list_filter = ("status", "created_at")
    search_fields = ("file_name", "file_path")
    readonly_fields = ("file_hash", "processing_duration_display", "created_at", "updated_at")
    
    fieldsets = (
        ("Document Information", {
            "fields": ("file_path", "file_name", "file_hash", "file_size")
        }),
        ("Processing Status", {
            "fields": ("status", "processing_started_at", "processing_completed_at", "processing_duration_display")
        }),
        ("Results", {
            "fields": ("faqs_extracted", "error_message")
        }),
        ("Metadata", {
            "fields": ("document_metadata", "created_at", "updated_at"),
            "classes": ("collapse",)
        }),
    )
    
    def file_size_mb(self, obj):
        """Display file size in MB"""
        if obj.file_size:
            return f"{obj.file_size / (1024 * 1024):.2f} MB"
        return "Unknown"
    file_size_mb.short_description = "File Size"
    
    def processing_duration_display(self, obj):
        """Display processing duration"""
        duration = obj.processing_duration
        if duration:
            return str(duration)
        return "N/A"
    processing_duration_display.short_description = "Processing Duration"


@admin.register(RAGFAQEntry)
class RAGFAQEntryAdmin(admin.ModelAdmin):
    list_display = ("question_preview", "category", "confidence_score", "query_matches", "document_link", "created_at")
    list_filter = ("category", "confidence_score", "embedding_model", "document__status", "created_at")
    search_fields = ("question", "answer", "keywords", "rag_id")
    readonly_fields = ("rag_id", "query_matches", "last_matched", "created_at", "updated_at")
    
    fieldsets = (
        ("FAQ Content", {
            "fields": ("question", "answer", "keywords", "category")
        }),
        ("RAG System Data", {
            "fields": ("rag_id", "document", "confidence_score", "source_section")
        }),
        ("Embeddings", {
            "fields": ("embedding_model", "embedding_version", "question_embedding", "answer_embedding", "keywords_embedding"),
            "classes": ("collapse",)
        }),
        ("Usage Statistics", {
            "fields": ("query_matches", "last_matched")
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",)
        }),
    )
    
    def question_preview(self, obj):
        """Display truncated question"""
        return obj.question[:80] + "..." if len(obj.question) > 80 else obj.question
    question_preview.short_description = "Question"
    
    def document_link(self, obj):
        """Link to related document"""
        if obj.document:
            url = reverse("admin:faq_ragdocument_change", args=[obj.document.pk])
            return format_html('<a href="{}">{}</a>', url, obj.document.file_name)
        return "No document"
    document_link.short_description = "Document"


@admin.register(RAGConversationSession)
class RAGConversationSessionAdmin(admin.ModelAdmin):
    list_display = ("session_id", "user_link", "total_interactions", "success_rate_display", "current_topic", "last_activity", "is_expired_display")
    list_filter = ("language", "created_at", "last_activity")
    search_fields = ("session_id", "user__email", "current_topic")
    readonly_fields = ("success_rate_display", "is_expired_display", "created_at", "last_activity")
    
    fieldsets = (
        ("Session Information", {
            "fields": ("session_id", "user", "current_topic", "language")
        }),
        ("Statistics", {
            "fields": ("total_interactions", "successful_responses", "average_confidence", "success_rate_display")
        }),
        ("Preferences", {
            "fields": ("user_preferences",),
            "classes": ("collapse",)
        }),
        ("Timestamps", {
            "fields": ("created_at", "last_activity", "expires_at", "is_expired_display")
        }),
    )
    
    def user_link(self, obj):
        """Link to related user"""
        if obj.user:
            url = reverse("admin:faq_enduser_change", args=[obj.user.pk])
            return format_html('<a href="{}">{}</a>', url, obj.user.email)
        return "Anonymous"
    user_link.short_description = "User"
    
    def success_rate_display(self, obj):
        """Display success rate as percentage"""
        return f"{obj.success_rate:.1f}%"
    success_rate_display.short_description = "Success Rate"
    
    def is_expired_display(self, obj):
        """Display expiration status"""
        return obj.is_expired
    is_expired_display.short_description = "Expired"
    is_expired_display.boolean = True


@admin.register(RAGQueryLog)
class RAGQueryLogAdmin(admin.ModelAdmin):
    list_display = ("query_preview", "confidence_score", "generation_method", "processing_time", "matched_faqs_count", "session_link", "created_at")
    list_filter = ("generation_method", "confidence_score", "language", "context_used", "created_at")
    search_fields = ("query_id", "original_query", "corrected_query", "intent")
    readonly_fields = ("query_id", "created_at")
    filter_horizontal = ("source_faqs",)
    
    fieldsets = (
        ("Query Information", {
            "fields": ("query_id", "session", "original_query", "corrected_query", "intent", "language")
        }),
        ("Response", {
            "fields": ("response_text", "confidence_score", "generation_method", "context_used")
        }),
        ("Performance", {
            "fields": ("processing_time", "matched_faqs_count", "source_faqs")
        }),
        ("Metadata", {
            "fields": ("query_metadata", "created_at"),
            "classes": ("collapse",)
        }),
    )
    
    def query_preview(self, obj):
        """Display truncated query"""
        return obj.original_query[:60] + "..." if len(obj.original_query) > 60 else obj.original_query
    query_preview.short_description = "Query"
    
    def session_link(self, obj):
        """Link to related session"""
        if obj.session:
            url = reverse("admin:faq_ragconversationsession_change", args=[obj.session.pk])
            return format_html('<a href="{}">{}</a>', url, obj.session.session_id)
        return "No session"
    session_link.short_description = "Session"


@admin.register(RAGUserFeedback)
class RAGUserFeedbackAdmin(admin.ModelAdmin):
    list_display = ("query_link", "rating", "accuracy_rating", "relevance_rating", "helpfulness_rating", "user_link", "created_at")
    list_filter = ("rating", "accuracy_rating", "relevance_rating", "helpfulness_rating", "created_at")
    search_fields = ("query_log__query_id", "query_log__original_query", "comments", "user__email")
    readonly_fields = ("created_at",)
    
    fieldsets = (
        ("Feedback Information", {
            "fields": ("query_log", "user", "rating", "comments")
        }),
        ("Detailed Ratings", {
            "fields": ("accuracy_rating", "relevance_rating", "helpfulness_rating")
        }),
        ("Metadata", {
            "fields": ("feedback_metadata", "created_at"),
            "classes": ("collapse",)
        }),
    )
    
    def query_link(self, obj):
        """Link to related query"""
        if obj.query_log:
            url = reverse("admin:faq_ragquerylog_change", args=[obj.query_log.pk])
            return format_html('<a href="{}">{}</a>', url, obj.query_log.query_id)
        return "No query"
    query_link.short_description = "Query"
    
    def user_link(self, obj):
        """Link to related user"""
        if obj.user:
            url = reverse("admin:faq_enduser_change", args=[obj.user.pk])
            return format_html('<a href="{}">{}</a>', url, obj.user.email)
        return "Anonymous"
    user_link.short_description = "User"


@admin.register(RAGSystemMetrics)
class RAGSystemMetricsAdmin(admin.ModelAdmin):
    list_display = ("metric_type", "component_name", "value", "threshold", "window_display", "created_at")
    list_filter = ("metric_type", "component_name", "created_at")
    search_fields = ("metric_type", "component_name")
    readonly_fields = ("created_at",)
    
    fieldsets = (
        ("Metric Information", {
            "fields": ("metric_type", "component_name", "value", "threshold")
        }),
        ("Time Window", {
            "fields": ("window_start", "window_end")
        }),
        ("Metadata", {
            "fields": ("metadata", "created_at"),
            "classes": ("collapse",)
        }),
    )
    
    def window_display(self, obj):
        """Display time window"""
        return f"{obj.window_start.strftime('%H:%M')} - {obj.window_end.strftime('%H:%M')}"
    window_display.short_description = "Time Window"


@admin.register(RAGPerformanceAlert)
class RAGPerformanceAlertAdmin(admin.ModelAdmin):
    list_display = ("alert_id", "severity_display", "status", "component_name", "metric_name", "current_value", "threshold_value", "created_at")
    list_filter = ("severity", "status", "component_name", "created_at")
    search_fields = ("alert_id", "component_name", "metric_name", "message")
    readonly_fields = ("alert_id", "created_at", "updated_at")
    
    fieldsets = (
        ("Alert Information", {
            "fields": ("alert_id", "component_name", "metric_name", "severity", "status", "message")
        }),
        ("Metric Values", {
            "fields": ("current_value", "threshold_value")
        }),
        ("Resolution", {
            "fields": ("resolved_at", "resolved_by", "resolution_notes")
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",)
        }),
    )
    
    actions = ["mark_resolved", "mark_acknowledged"]
    
    def severity_display(self, obj):
        """Display severity with color coding"""
        colors = {
            'low': 'green',
            'medium': 'orange', 
            'high': 'red',
            'critical': 'darkred'
        }
        color = colors.get(obj.severity, 'black')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.severity.upper()
        )
    severity_display.short_description = "Severity"
    
    def mark_resolved(self, request, queryset):
        """Mark selected alerts as resolved"""
        updated = queryset.filter(status='active').update(
            status='resolved',
            resolved_at=timezone.now(),
            resolved_by=request.user.username
        )
        self.message_user(request, f"{updated} alerts marked as resolved.")
    mark_resolved.short_description = "Mark selected alerts as resolved"
    
    def mark_acknowledged(self, request, queryset):
        """Mark selected alerts as acknowledged"""
        updated = queryset.filter(status='active').update(status='acknowledged')
        self.message_user(request, f"{updated} alerts marked as acknowledged.")
    mark_acknowledged.short_description = "Mark selected alerts as acknowledged"
