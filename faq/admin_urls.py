"""
URL configuration for admin dashboard within the faq app.

Implements Requirements 5.1, 5.3:
- Set up admin dashboard URL patterns at `/admin-dashboard/` path
- Ensure separation from existing chatbot URLs
- Implement read-only access controls for data viewing

URL Structure:
- /admin-dashboard/login/ - Admin login with CAPTCHA
- /admin-dashboard/logout/ - Admin logout
- /admin-dashboard/ - Main dashboard (read-only)
- /admin-dashboard/users/ - User history view (read-only)
- /admin-dashboard/users/<id>/ - Conversation detail view (read-only)
- /admin-dashboard/faq/ - FAQ management view (read-only)

All admin dashboard URLs are completely separate from existing chatbot URLs:
- Chatbot API: /api/* (users/, requests/, responses/, faqs/, health/)
- Chatbot Home: / (home page)
- Admin Dashboard: /admin-dashboard/* (completely separate namespace)
"""
from django.urls import path, include
from django.views.generic import TemplateView
from django.contrib.auth.views import LogoutView
from . import admin_views

# Namespace for admin dashboard URLs to ensure separation from chatbot URLs
app_name = 'admin_dashboard'

urlpatterns = [
    # Authentication URLs (public access within admin dashboard)
    path('login/', admin_views.AdminLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('refresh-captcha/', admin_views.refresh_captcha, name='refresh_captcha'),
    
    # Protected admin dashboard URLs (require authentication + read-only access for user data)
    path('', admin_views.AdminDashboardView.as_view(), name='dashboard'),
    path('users/', admin_views.UserHistoryView.as_view(), name='user_history'),
    path('users/<int:user_id>/', admin_views.ConversationDetailView.as_view(), name='conversation_detail'),
    
    # Analytics Page (Connected via TemplateView as requested)
    path('analytics/', TemplateView.as_view(template_name='faq/admin/analytics.html'), name='analytics'),
    
    # FAQ Management URLs (allow modification for knowledge base management)
    path('faq/', admin_views.FAQManagementView.as_view(), name='faq_management'),
    path('faq/create/', admin_views.FAQCreateView.as_view(), name='faq_create'),
    path('faq/<int:pk>/edit/', admin_views.FAQUpdateView.as_view(), name='faq_edit'),
    path('faq/<int:pk>/delete/', admin_views.FAQDeleteView.as_view(), name='faq_delete'),
    path('category/', admin_views.CategoryManagementView.as_view(), name='category_management'),
    path('tag/', admin_views.TagManagementView.as_view(), name='tag_management'),
    path('feedback/', admin_views.FeedbackListView.as_view(), name='feedback_list'),
    path('settings/', admin_views.SettingsView.as_view(), name='settings'),
    
    # RAG System Management URLs
    path('rag/', include('faq.admin_rag_urls')),
]