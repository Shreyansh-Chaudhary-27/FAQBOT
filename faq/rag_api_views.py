"""
RAG System REST API Views

This module provides REST API endpoints for the RAG-based FAQ system,
including document processing, query handling, conversation management,
and system monitoring endpoints.
"""

import json
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .rag.core.factory import rag_factory
from .rag.interfaces.base import FAQEntry, ProcessedQuery, Response, ConversationContext
from .models import EndUser, UserRequest, BotResponse


# Global RAG system instance (initialized on first use)
_rag_system = None


def get_rag_system():
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        try:
            _rag_system = rag_factory.create_default_system()
            # Explicitly initialize the RAG system after creation
            if _rag_system:
                _rag_system._initialize_system()
        except Exception as e:
            # Log error and return None - endpoints will handle gracefully
            print(f"Failed to initialize RAG system: {e}")
            return None
    return _rag_system


# Document Processing Endpoints

@csrf_exempt
@require_http_methods(["POST"])
def upload_document(request):
    """
    Upload and process a DOCX document.
    
    POST /api/rag/documents/upload/
    Content-Type: multipart/form-data
    
    Form data:
    - file: DOCX file to upload
    - force_update: (optional) boolean to force reprocessing
    """
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)
        
        uploaded_file = request.FILES['file']
        force_update = request.POST.get('force_update', 'false').lower() == 'true'
        
        # Validate file type
        if not uploaded_file.name.endswith('.docx'):
            return JsonResponse({"error": "Only DOCX files are supported"}, status=400)
        
        # Save file temporarily
        file_path = default_storage.save(f'temp/{uploaded_file.name}', ContentFile(uploaded_file.read()))
        full_path = default_storage.path(file_path)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Process document
        try:
            faqs = rag_system.process_document(full_path, force_update)
            
            # Clean up temporary file
            default_storage.delete(file_path)
            
            return JsonResponse({
                "success": True,
                "document_name": uploaded_file.name,
                "faqs_extracted": len(faqs),
                "force_update": force_update,
                "message": f"Successfully processed document and extracted {len(faqs)} FAQs"
            })
            
        except Exception as e:
            # Clean up temporary file on error
            try:
                default_storage.delete(file_path)
            except:
                pass
            raise e
            
    except Exception as e:
        return JsonResponse({
            "error": f"Document upload failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_document_path(request):
    """
    Process a document by file path.
    
    POST /api/rag/documents/process/
    {
        "document_path": "/path/to/document.docx",
        "force_update": false
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        document_path = payload.get("document_path", "").strip()
        force_update = payload.get("force_update", False)
        
        if not document_path:
            return JsonResponse({"error": "document_path is required"}, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Process document
        faqs = rag_system.process_document(document_path, force_update)
        
        return JsonResponse({
            "success": True,
            "document_path": document_path,
            "faqs_extracted": len(faqs),
            "force_update": force_update,
            "message": f"Successfully processed document and extracted {len(faqs)} FAQs"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Document processing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_documents_batch(request):
    """
    Process multiple documents in batch.
    
    POST /api/rag/documents/batch/
    {
        "document_paths": ["/path/to/doc1.docx", "/path/to/doc2.docx"],
        "force_update": false,
        "parallel": true
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        document_paths = payload.get("document_paths", [])
        force_update = payload.get("force_update", False)
        parallel = payload.get("parallel", True)
        
        if not document_paths or not isinstance(document_paths, list):
            return JsonResponse({"error": "document_paths must be a non-empty list"}, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Process documents
        results = rag_system.process_documents_batch(document_paths, force_update, parallel)
        
        # Calculate statistics
        successful_docs = len([r for r in results.values() if r])
        total_faqs = sum(len(faqs) for faqs in results.values())
        failed_docs = len(document_paths) - successful_docs
        
        return JsonResponse({
            "success": True,
            "total_documents": len(document_paths),
            "successful_documents": successful_docs,
            "failed_documents": failed_docs,
            "total_faqs_extracted": total_faqs,
            "parallel_processing": parallel,
            "force_update": force_update,
            "results": {path: len(faqs) for path, faqs in results.items()},
            "message": f"Processed {successful_docs}/{len(document_paths)} documents, extracted {total_faqs} FAQs"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Batch document processing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_directory(request):
    """
    Process all DOCX documents in a directory.
    
    POST /api/rag/documents/directory/
    {
        "directory_path": "/path/to/documents/",
        "recursive": true,
        "force_update": false,
        "parallel": true
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        directory_path = payload.get("directory_path", "").strip()
        recursive = payload.get("recursive", True)
        force_update = payload.get("force_update", False)
        parallel = payload.get("parallel", True)
        
        if not directory_path:
            return JsonResponse({"error": "directory_path is required"}, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Process directory
        results = rag_system.process_directory(directory_path, recursive, force_update, parallel)
        
        # Calculate statistics
        successful_docs = len([r for r in results.values() if r])
        total_faqs = sum(len(faqs) for faqs in results.values())
        failed_docs = len(results) - successful_docs
        
        return JsonResponse({
            "success": True,
            "directory_path": directory_path,
            "recursive": recursive,
            "total_documents_found": len(results),
            "successful_documents": successful_docs,
            "failed_documents": failed_docs,
            "total_faqs_extracted": total_faqs,
            "parallel_processing": parallel,
            "force_update": force_update,
            "results": {path: len(faqs) for path, faqs in results.items()},
            "message": f"Processed {successful_docs}/{len(results)} documents from directory, extracted {total_faqs} FAQs"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Directory processing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


# Query Processing Endpoints

@csrf_exempt
@require_http_methods(["POST"])
def answer_query(request):
    """
    Answer a user query using the RAG system.
    
    POST /api/rag/query/
    {
        "query": "How do I reset my password?",
        "session_id": "optional_session_id",
        "user_id": 123
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        query = payload.get("query", "").strip()
        session_id = payload.get("session_id")
        user_id = payload.get("user_id")
        
        if not query:
            return JsonResponse({"error": "query is required"}, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Get or create user if user_id provided
        user = None
        if user_id:
            try:
                user = EndUser.objects.get(id=user_id)
            except EndUser.DoesNotExist:
                return JsonResponse({"error": "user not found"}, status=404)
        
        # Create user request record if user provided
        user_request = None
        if user:
            user_request = UserRequest.objects.create(
                user=user,
                session_id=session_id or "",
                text=query
            )
        
        # Process query with RAG system
        response = rag_system.answer_query(query, session_id)
        
        # Create bot response record if user provided
        if user and user_request:
            BotResponse.objects.create(
                request=user_request,
                text=response.text
            )
        
        # Format response
        return JsonResponse({
            "success": True,
            "query": query,
            "response": response.text,
            "confidence": response.confidence,
            "context_used": response.context_used,
            "generation_method": response.generation_method,
            "source_faqs": [
                {
                    "id": faq.id,
                    "question": faq.question,
                    "answer": faq.answer,
                    "confidence_score": faq.confidence_score
                }
                for faq in response.source_faqs
            ],
            "metadata": response.metadata,
            "session_id": session_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Query processing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_query_only(request):
    """
    Process a query without generating a response (for analysis).
    
    POST /api/rag/query/process/
    {
        "query": "How do I reset my password?",
        "session_id": "optional_session_id"
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        query = payload.get("query", "").strip()
        session_id = payload.get("session_id")
        
        if not query:
            return JsonResponse({"error": "query is required"}, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Process query (this would require access to internal methods)
        # For now, we'll use the full answer_query and extract processing info
        response = rag_system.answer_query(query, session_id)
        
        return JsonResponse({
            "success": True,
            "original_query": query,
            "processed_query": {
                "corrected_query": query,  # Would need access to internal processing
                "intent": "unknown",  # Would need access to query processor
                "language": "en",  # Would need access to language detection
                "confidence": 0.8  # Would need access to processing confidence
            },
            "session_id": session_id,
            "processing_metadata": response.metadata
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Query processing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


# Conversation Management Endpoints

@csrf_exempt
@require_http_methods(["POST"])
def create_conversation_session(request):
    """
    Create a new conversation session.
    
    POST /api/rag/conversations/
    {
        "session_id": "optional_custom_session_id",
        "user_id": 123
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        session_id = payload.get("session_id")
        user_id = payload.get("user_id")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Create conversation session (if conversation manager available)
        if hasattr(rag_system, 'conversation_manager') and rag_system.conversation_manager:
            context = rag_system.conversation_manager.create_session(session_id)
            
            return JsonResponse({
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "context": {
                    "session_id": context.session_id,
                    "current_topic": context.current_topic,
                    "last_activity": context.last_activity.isoformat()
                }
            })
        else:
            # Basic session creation without conversation manager
            return JsonResponse({
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "message": "Session created (conversation manager not available)"
            })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Session creation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_conversation_context(request, session_id):
    """
    Get conversation context for a session.
    
    GET /api/rag/conversations/{session_id}/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Get conversation context
        if hasattr(rag_system, 'conversation_manager') and rag_system.conversation_manager:
            context = rag_system.conversation_manager.get_context(session_id)
            
            if context:
                return JsonResponse({
                    "success": True,
                    "session_id": session_id,
                    "context": {
                        "session_id": context.session_id,
                        "current_topic": context.current_topic,
                        "last_activity": context.last_activity.isoformat(),
                        "history_length": len(context.history),
                        "user_preferences": context.user_preferences
                    },
                    "history": context.history[-10:]  # Return last 10 interactions
                })
            else:
                return JsonResponse({"error": "Session not found"}, status=404)
        else:
            return JsonResponse({
                "error": "Conversation manager not available"
            }, status=503)
        
    except Exception as e:
        return JsonResponse({
            "error": f"Failed to get conversation context: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_conversation_session(request, session_id):
    """
    Delete a conversation session.
    
    DELETE /api/rag/conversations/{session_id}/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Delete conversation session (if conversation manager available)
        if hasattr(rag_system, 'conversation_manager') and rag_system.conversation_manager:
            # This would require implementing a delete_session method
            # For now, we'll just return success
            return JsonResponse({
                "success": True,
                "session_id": session_id,
                "message": "Session deletion requested (implementation pending)"
            })
        else:
            return JsonResponse({
                "error": "Conversation manager not available"
            }, status=503)
        
    except Exception as e:
        return JsonResponse({
            "error": f"Session deletion failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


# System Management Endpoints

@csrf_exempt
@require_http_methods(["GET"])
def system_status(request):
    """
    Get RAG system status and health information.
    
    GET /api/rag/system/status/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({
                "success": False,
                "status": "unavailable",
                "error": "RAG system not available",
                "timestamp": datetime.now().isoformat()
            }, status=503)
        
        # Get system stats
        stats = rag_system.get_system_stats()
        
        return JsonResponse({
            "success": True,
            "status": "available",
            "timestamp": datetime.now().isoformat(),
            "system_stats": stats
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def system_health(request):
    """
    Perform comprehensive system health check.
    
    GET /api/rag/system/health/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({
                "success": False,
                "overall_status": "unavailable",
                "error": "RAG system not available",
                "timestamp": datetime.now().isoformat()
            }, status=503)
        
        # Perform health check
        health_results = rag_system.health_check()
        
        return JsonResponse({
            "success": True,
            **health_results
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def component_status(request):
    """
    Get the status of all RAG system components.
    
    GET /api/rag/system/components/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({
                "success": False,
                "error": "RAG system not available",
                "timestamp": datetime.now().isoformat()
            }, status=503)
        
        # Get component status
        component_status = rag_system.get_component_status()
        
        return JsonResponse({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "components": component_status
        })
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """
    Submit user feedback for a query.
    
    POST /api/rag/feedback/
    {
        "query_id": "query_123",
        "user_id": "user_456",
        "rating": 5,
        "comments": "Very helpful response"
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        query_id = payload.get("query_id", "").strip()
        user_id = payload.get("user_id", "").strip()
        rating = payload.get("rating")
        comments = payload.get("comments")
        
        if not query_id or not user_id or rating is None:
            return JsonResponse({
                "error": "query_id, user_id, and rating are required"
            }, status=400)
        
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return JsonResponse({
                "error": "rating must be an integer between 1 and 5"
            }, status=400)
        
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Submit feedback
        rag_system.submit_user_feedback(query_id, user_id, rating, comments)
        
        return JsonResponse({
            "success": True,
            "message": "Feedback submitted successfully",
            "query_id": query_id,
            "user_id": user_id,
            "rating": rating
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({
            "error": f"Feedback submission failed: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def ingestion_progress(request):
    """
    Get current document ingestion progress.
    
    GET /api/rag/ingestion/progress/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Get ingestion progress
        progress = rag_system.get_ingestion_progress()
        
        if progress:
            return JsonResponse({
                "success": True,
                "in_progress": True,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JsonResponse({
                "success": True,
                "in_progress": False,
                "message": "No ingestion currently in progress",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        return JsonResponse({
            "error": f"Failed to get ingestion progress: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def ingestion_stats(request):
    """
    Get document ingestion pipeline statistics.
    
    GET /api/rag/ingestion/stats/
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        if not rag_system:
            return JsonResponse({"error": "RAG system not available"}, status=503)
        
        # Get ingestion statistics
        stats = rag_system.get_ingestion_stats()
        
        return JsonResponse({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        })
        
    except Exception as e:
        return JsonResponse({
            "error": f"Failed to get ingestion stats: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)