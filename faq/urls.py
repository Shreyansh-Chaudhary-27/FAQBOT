from django.urls import path, include

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("users/", views.users, name="users"),
    path("requests/", views.requests_, name="requests"),
    path("responses/", views.responses, name="responses"),
    path("faqs/", views.faqs, name="faqs"),
    
    # Smart FAQ search endpoints
    path("smart-search/", views.smart_faq_search, name="smart_faq_search"),
    path("explain-match/", views.explain_faq_match, name="explain_faq_match"),
    path("add-synonyms/", views.add_synonyms, name="add_synonyms"),
    path("search-demo/", views.faq_search_demo, name="faq_search_demo"),
    
    # AI-enhanced endpoints
    # path("ai-search/", views.ai_faq_search, name="ai_faq_search"), # Removed
    path("chatbot/", views.chatbot_response, name="chatbot_response"),
    path("feedback/", views.submit_feedback, name="submit_feedback"),
    path("ai-demo/", views.ai_chatbot_demo, name="ai_chatbot_demo"),
    
    # Document ingestion endpoints
    path("ingest-document/", views.ingest_document, name="ingest_document"),
    path("ingest-batch/", views.ingest_documents_batch, name="ingest_documents_batch"),
    path("ingest-directory/", views.ingest_directory, name="ingest_directory"),
    path("ingestion-progress/", views.ingestion_progress, name="ingestion_progress"),
    path("ingestion-stats/", views.ingestion_stats, name="ingestion_stats"),
    
    # RAG System API endpoints
    path("rag/", include('faq.rag_urls')),
]
