"""
RAG System API URL Configuration

URL patterns for the RAG-based FAQ system REST API endpoints.
"""

from django.urls import path
from . import rag_api_views, views

app_name = 'rag_api'

urlpatterns = [
    # Document Processing Endpoints
    path('documents/upload/', rag_api_views.upload_document, name='upload_document'),
    path('documents/process/', rag_api_views.process_document_path, name='process_document_path'),
    path('documents/batch/', rag_api_views.process_documents_batch, name='process_documents_batch'),
    path('documents/directory/', rag_api_views.process_directory, name='process_directory'),
    
    # Query Processing Endpoints
    path('query/', rag_api_views.answer_query, name='answer_query'),
    path('query/process/', rag_api_views.process_query_only, name='process_query_only'),
    
    # Conversation Management Endpoints
    path('conversations/', rag_api_views.create_conversation_session, name='create_conversation_session'),
    path('conversations/<str:session_id>/', rag_api_views.get_conversation_context, name='get_conversation_context'),
    path('conversations/<str:session_id>/delete/', rag_api_views.delete_conversation_session, name='delete_conversation_session'),
    
    # System Management Endpoints
    path('system/status/', rag_api_views.system_status, name='system_status'),
    path('system/health/', rag_api_views.system_health, name='system_health'),
    path('system/components/', rag_api_views.component_status, name='component_status'),
    
    # Feedback and Analytics Endpoints
    path('feedback/', rag_api_views.submit_feedback, name='submit_feedback'),
    
    # Ingestion Monitoring Endpoints
    path('ingestion/progress/', rag_api_views.ingestion_progress, name='ingestion_progress'),
    path('ingestion/stats/', rag_api_views.ingestion_stats, name='ingestion_stats'),
    
    # Django Integration Endpoints
    path('django/query/', views.rag_query, name='django_query'),
    path('django/feedback/', views.rag_feedback, name='django_feedback'),
    path('django/analytics/', views.rag_analytics, name='django_analytics'),
    path('django/health/', views.rag_health, name='django_health'),
    path('django/documents/', views.rag_documents, name='django_documents'),
    path('django/faqs/', views.rag_faqs, name='django_faqs'),
    path('django/top-faqs/', views.rag_top_faqs, name='django_top_faqs'),
    path('django/process-document/', views.rag_process_document_django, name='django_process_document'),
]