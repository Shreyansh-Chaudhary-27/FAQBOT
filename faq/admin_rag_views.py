"""
Custom admin views for RAG system management.
"""

from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from .models import RAGDocument, RAGFAQEntry, RAGConversationSession, RAGPerformanceAlert
from .rag_django_service import RAGDjangoService


@staff_member_required
def rag_dashboard(request):
    """
    RAG system dashboard for administrators.
    """
    try:
        # Initialize service
        django_service = RAGDjangoService()
        
        # Get system analytics
        analytics = django_service.get_system_analytics(days=7)
        health = django_service.get_system_health_summary()
        
        # Get recent documents
        recent_documents = RAGDocument.objects.order_by('-created_at')[:10]
        
        # Get top FAQs
        top_faqs = django_service.get_top_faqs(limit=10)
        
        # Get active alerts
        active_alerts = RAGPerformanceAlert.objects.filter(status='active').order_by('-created_at')[:5]
        
        context = {
            'analytics': analytics,
            'health': health,
            'recent_documents': recent_documents,
            'top_faqs': top_faqs,
            'active_alerts': active_alerts,
            'title': 'RAG System Dashboard'
        }
        
        return render(request, 'admin/rag_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading dashboard: {e}")
        return redirect('admin:index')


@staff_member_required
@require_http_methods(["POST"])
def process_document_admin(request):
    """
    Process a document from the admin interface.
    """
    try:
        document_id = request.POST.get('document_id')
        force_update = request.POST.get('force_update') == 'on'
        
        if not document_id:
            messages.error(request, "Document ID is required")
            return redirect('admin:faq_ragdocument_changelist')
        
        # Get document
        try:
            document = RAGDocument.objects.get(id=document_id)
        except RAGDocument.DoesNotExist:
            messages.error(request, "Document not found")
            return redirect('admin:faq_ragdocument_changelist')
        
        # Initialize service
        from django.core.cache import cache
        rag_system = cache.get('rag_system')
        if not rag_system:
            messages.error(request, "RAG system not initialized. Run init_rag_system management command first.")
            return redirect('admin:faq_ragdocument_changelist')
        
        django_service = RAGDjangoService(rag_system)
        
        # Update status to processing
        django_service.update_document_processing_status(document.id, 'processing')
        
        try:
            # Process document
            faqs = rag_system.process_document(document.file_path, force_update)
            
            # Sync FAQs to Django
            django_faqs = django_service.sync_faqs_to_django(faqs, document.id)
            
            # Update status
            django_service.update_document_processing_status(
                document.id, 'completed', len(django_faqs)
            )
            
            messages.success(request, f"Successfully processed document and extracted {len(django_faqs)} FAQs")
            
        except Exception as e:
            # Update status with error
            django_service.update_document_processing_status(
                document.id, 'failed', 0, str(e)
            )
            messages.error(request, f"Document processing failed: {e}")
        
        return redirect('admin:faq_ragdocument_change', document.id)
        
    except Exception as e:
        messages.error(request, f"Error processing document: {e}")
        return redirect('admin:faq_ragdocument_changelist')


@staff_member_required
@csrf_exempt
@require_http_methods(["POST"])
def resolve_alert_admin(request):
    """
    Resolve a performance alert from the admin interface.
    """
    try:
        data = json.loads(request.body)
        alert_id = data.get('alert_id')
        resolution_notes = data.get('resolution_notes', '')
        
        if not alert_id:
            return JsonResponse({"error": "Alert ID is required"}, status=400)
        
        # Get alert
        try:
            alert = RAGPerformanceAlert.objects.get(id=alert_id)
        except RAGPerformanceAlert.DoesNotExist:
            return JsonResponse({"error": "Alert not found"}, status=404)
        
        # Resolve alert
        alert.resolve(resolved_by=request.user.username, notes=resolution_notes)
        
        return JsonResponse({
            "success": True,
            "message": "Alert resolved successfully"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@staff_member_required
def system_health_admin(request):
    """
    System health check for administrators.
    """
    try:
        # Initialize service
        django_service = RAGDjangoService()
        
        # Get health summary
        health = django_service.get_system_health_summary()
        
        # Get RAG system health if available
        rag_health = None
        try:
            from django.core.cache import cache
            rag_system = cache.get('rag_system')
            if rag_system:
                rag_health = rag_system.health_check()
        except Exception as e:
            rag_health = {"error": str(e)}
        
        context = {
            'django_health': health,
            'rag_health': rag_health,
            'title': 'RAG System Health Check'
        }
        
        return render(request, 'admin/rag_health.html', context)
        
    except Exception as e:
        messages.error(request, f"Error checking system health: {e}")
        return redirect('admin:index')


@staff_member_required
def analytics_admin(request):
    """
    Analytics dashboard for administrators.
    """
    try:
        days = int(request.GET.get('days', '7'))
        
        # Initialize service
        django_service = RAGDjangoService()
        
        # Get analytics
        analytics = django_service.get_system_analytics(days)
        
        # Get additional statistics
        from .models import RAGQueryLog, RAGUserFeedback
        from django.db.models import Avg, Count
        from datetime import timedelta
        from django.utils import timezone
        
        start_date = timezone.now() - timedelta(days=days)
        
        # Query patterns
        query_patterns = RAGQueryLog.objects.filter(
            created_at__gte=start_date
        ).values('intent', 'language').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence_score')
        ).order_by('-count')[:10]
        
        # Feedback analysis
        feedback_analysis = RAGUserFeedback.objects.filter(
            created_at__gte=start_date
        ).aggregate(
            total_feedback=Count('id'),
            avg_rating=Avg('rating'),
            avg_accuracy=Avg('accuracy_rating'),
            avg_relevance=Avg('relevance_rating'),
            avg_helpfulness=Avg('helpfulness_rating')
        )
        
        context = {
            'analytics': analytics,
            'query_patterns': query_patterns,
            'feedback_analysis': feedback_analysis,
            'days': days,
            'title': f'RAG Analytics ({days} days)'
        }
        
        return render(request, 'admin/rag_analytics.html', context)
        
    except ValueError:
        messages.error(request, "Invalid days parameter")
        return redirect('admin:index')
    except Exception as e:
        messages.error(request, f"Error loading analytics: {e}")
        return redirect('admin:index')


@staff_member_required
@require_http_methods(["POST"])
def cleanup_expired_sessions_admin(request):
    """
    Clean up expired conversation sessions from admin interface.
    """
    try:
        # Initialize service
        django_service = RAGDjangoService()
        
        # Cleanup expired sessions
        count = django_service.cleanup_expired_sessions()
        
        messages.success(request, f"Cleaned up {count} expired sessions")
        return redirect('admin:faq_ragconversationsession_changelist')
        
    except Exception as e:
        messages.error(request, f"Error cleaning up sessions: {e}")
        return redirect('admin:faq_ragconversationsession_changelist')


@staff_member_required
@csrf_exempt
@require_http_methods(["GET"])
def system_stats_api(request):
    """
    API endpoint for system statistics (for AJAX requests from admin).
    """
    try:
        # Get RAG system stats if available
        stats = {}
        try:
            from django.core.cache import cache
            rag_system = cache.get('rag_system')
            if rag_system:
                stats = rag_system.get_system_stats()
        except Exception as e:
            stats = {"error": str(e)}
        
        # Get Django model counts
        django_stats = {
            'documents': RAGDocument.objects.count(),
            'faqs': RAGFAQEntry.objects.count(),
            'sessions': RAGConversationSession.objects.count(),
            'active_alerts': RAGPerformanceAlert.objects.filter(status='active').count()
        }
        
        return JsonResponse({
            'rag_system_stats': stats,
            'django_stats': django_stats
        })
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)