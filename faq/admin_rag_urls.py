"""
Admin URLs for RAG system management.
"""

from django.urls import path
from . import admin_rag_views

app_name = 'rag_admin'

urlpatterns = [
    path('dashboard/', admin_rag_views.rag_dashboard, name='dashboard'),
    path('process-document/', admin_rag_views.process_document_admin, name='process_document'),
    path('resolve-alert/', admin_rag_views.resolve_alert_admin, name='resolve_alert'),
    path('health/', admin_rag_views.system_health_admin, name='health'),
    path('analytics/', admin_rag_views.analytics_admin, name='analytics'),
    path('cleanup-sessions/', admin_rag_views.cleanup_expired_sessions_admin, name='cleanup_sessions'),
    path('api/stats/', admin_rag_views.system_stats_api, name='stats_api'),
]