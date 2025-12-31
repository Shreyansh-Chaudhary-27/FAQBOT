import json
import uuid
import logging

from django.http import JsonResponse
from django.db.models import Q
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import EndUser, UserRequest, BotResponse, FAQEntry, RAGConversationSession


from .rag_django_service import RAGDjangoService
from .rag_api_views import get_rag_system

logger = logging.getLogger(__name__)

from django.utils import timezone

ist_time = timezone.localtime(timezone.now())

@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok"})


def home(request):
    return render(request, "faq/home.html")


@csrf_exempt
@require_http_methods(["GET", "POST"])
def users(request):
    if request.method == "GET":
        # Return list of users for integration testing
        users = EndUser.objects.all().order_by('id')
        data = [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "session_id": u.session_id
            }
            for u in users
        ]
        return JsonResponse(data, safe=False)

    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    name = payload.get("name")
    email = payload.get("email")
    session_id = payload.get("session_id")
    if not email:
        return JsonResponse({"error": "email required"}, status=400)
    user, created = EndUser.objects.get_or_create(email=email)
    if name is not None:
        user.name = name
    if session_id is not None:
        user.session_id = session_id
    user.save()
    return JsonResponse({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "session_id": user.session_id,
        "is_new": created
    })


@csrf_exempt
@require_http_methods(["GET", "POST"])
def requests_(request):
    if request.method == "GET":
        # Return list of requests for integration testing
        requests = UserRequest.objects.all().order_by('id')
        data = [
            {
                "id": r.id,
                "user_id": r.user.id,
                "session_id": r.session_id,
                "text": r.text
            }
            for r in requests
        ]
        return JsonResponse(data, safe=False)

    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    user_id = payload.get("user_id")
    email = payload.get("email")
    name = payload.get("name")
    session_id = payload.get("session_id")
    text = payload.get("text")
    if text is None:
        return JsonResponse({"error": "text required"}, status=400)
    user = None
    if user_id:
        try:
            user = EndUser.objects.get(id=user_id)
        except EndUser.DoesNotExist:
            return JsonResponse({"error": "user not found"}, status=404)
    elif email:
        user, _ = EndUser.objects.get_or_create(email=email)
        if name is not None:
            user.name = name
        if session_id is not None:
            user.session_id = session_id
        user.save()
    else:
        return JsonResponse({"error": "email or user_id required"}, status=400)
    if session_id is None:
        session_id = user.session_id
    ur = UserRequest.objects.create(
        user=user, session_id=session_id or "", text=text)
    return JsonResponse({"id": ur.id, "user_id": user.id, "session_id": ur.session_id, "text": ur.text})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def responses(request):
    if request.method == "GET":
        # Return list of responses for integration testing
        responses = BotResponse.objects.all().order_by('id')
        data = [{"id": r.id, "request_id": r.request.id, "text": r.text}
                for r in responses]
        return JsonResponse(data, safe=False)

    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    request_id = payload.get("request_id")
    text = payload.get("text")
    if not request_id or text is None:
        return JsonResponse({"error": "request_id and text required"}, status=400)
    try:
        req = UserRequest.objects.get(id=request_id)
    except UserRequest.DoesNotExist:
        return JsonResponse({"error": "request not found"}, status=404)
    if hasattr(req, "response"):
        return JsonResponse({"error": "response already exists"}, status=409)
    br = BotResponse.objects.create(request=req, text=text)
    return JsonResponse({"id": br.id, "request_id": req.id, "text": br.text})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def faqs(request):
    if request.method == "GET":
        keyword = request.GET.get("keyword")
        q = request.GET.get("q")
        smart_search = request.GET.get("smart", "false").lower() == "true"
        max_results = int(request.GET.get("max_results", "10"))
        min_confidence = float(request.GET.get("min_confidence", "0.3"))

        if smart_search and q:
            # Use intelligent FAQ matching
            from .faq_matcher import faq_matcher
            matches = faq_matcher.find_best_matches(
                query=q,
                max_results=max_results,
                min_confidence=min_confidence
            )

            data = []
            for match in matches:
                data.append({
                    "id": match['id'],
                    "question": match['question'],
                    "answer": match['answer'],
                    "keywords": match['keywords'],
                    "confidence": round(match['confidence'], 3),
                    "match_details": {
                        "keyword_match": round(match['match_details']['keyword_match'], 3),
                        "question_similarity": round(match['match_details']['question_similarity'], 3),
                        "semantic_similarity": round(match['match_details']['semantic_similarity'], 3),
                        "fuzzy_match": round(match['match_details']['fuzzy_match'], 3)
                    }
                })

            return JsonResponse({
                "results": data,
                "search_type": "smart",
                "query": q,
                "total_matches": len(data)
            })
        else:
            # Use traditional search
            qs = FAQEntry.objects.all()
            if keyword:
                qs = qs.filter(keywords__icontains=keyword)
            if q:
                qs = qs.filter(Q(question__icontains=q) |
                               Q(answer__icontains=q))
            data = [{"id": f.id, "question": f.question, "answer": f.answer,
                     "keywords": f.keywords} for f in qs.order_by("id")]
            return JsonResponse({
                "results": data,
                "search_type": "traditional",
                "query": q or keyword,
                "total_matches": len(data)
            })

    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    question = payload.get("question")
    answer = payload.get("answer")
    keywords = payload.get("keywords", "")
    if not question or answer is None:
        return JsonResponse({"error": "question and answer required"}, status=400)
    f = FAQEntry.objects.create(
        question=question, answer=answer, keywords=keywords)
    return JsonResponse({"id": f.id, "question": f.question, "answer": f.answer, "keywords": f.keywords}, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def smart_faq_search(request):
    """
    Enhanced FAQ search endpoint that uses intelligent matching
    to find relevant answers based on semantic similarity and synonyms.

    POST /api/smart-search/
    {
        "query": "How do I reset my password?",
        "max_results": 5,
        "min_confidence": 0.3
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        query = payload.get("query", "").strip()
        max_results = payload.get("max_results", 5)
        min_confidence = payload.get("min_confidence", 0.3)

        if not query:
            return JsonResponse({"error": "query is required"}, status=400)

        # Use intelligent FAQ matching
        from .faq_matcher import faq_matcher
        matches = faq_matcher.find_best_matches(
            query=query,
            max_results=max_results,
            min_confidence=min_confidence
        )

        # Format response
        results = []
        for match in matches:
            results.append({
                "id": match['id'],
                "question": match['question'],
                "answer": match['answer'],
                "keywords": match['keywords'],
                "confidence": round(match['confidence'], 3),
                "match_explanation": {
                    "keyword_match": round(match['match_details']['keyword_match'], 3),
                    "question_similarity": round(match['match_details']['question_similarity'], 3),
                    "semantic_similarity": round(match['match_details']['semantic_similarity'], 3),
                    "fuzzy_match": round(match['match_details']['fuzzy_match'], 3)
                }
            })

        return JsonResponse({
            "success": True,
            "query": query,
            "results": results,
            "total_matches": len(results),
            "search_parameters": {
                "max_results": max_results,
                "min_confidence": min_confidence
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Search failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def explain_faq_match(request):
    """
    Explain why a specific FAQ matched a query.

    POST /api/explain-match/
    {
        "query": "How do I reset my password?",
        "faq_id": 123
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        query = payload.get("query", "").strip()
        faq_id = payload.get("faq_id")

        if not query or not faq_id:
            return JsonResponse({"error": "query and faq_id are required"}, status=400)

        # Get match explanation
        from .faq_matcher import faq_matcher
        explanation = faq_matcher.get_match_explanation(query, faq_id)

        if "error" in explanation:
            return JsonResponse(explanation, status=404)

        return JsonResponse({
            "success": True,
            "explanation": explanation
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Explanation failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_synonyms(request):
    """
    Add new synonyms to improve matching.

    POST /api/add-synonyms/
    {
        "word": "login",
        "synonyms": ["sign in", "log in", "access"]
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        word = payload.get("word", "").strip().lower()
        synonyms = payload.get("synonyms", [])

        if not word or not synonyms:
            return JsonResponse({"error": "word and synonyms are required"}, status=400)

        # Add synonyms
        from .faq_matcher import faq_matcher
        faq_matcher.add_synonym(word, [s.lower().strip() for s in synonyms])

        return JsonResponse({
            "success": True,
            "message": f"Added {len(synonyms)} synonyms for '{word}'"
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Adding synonyms failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def chatbot_response(request):
    """
    Generate chatbot responses using RAG system with semantic understanding.
    Flow:
    1. User Query -> RAG System (Gemini embeddings for semantic understanding)
    2. Vector Similarity Search -> Find semantically similar FAQs
    3. Gemini Response Generation -> Contextual answer based on retrieved FAQs
    4. No Match -> Polite fallback response
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        message = payload.get("message", "").strip()
        user_id = payload.get("user_id")
        session_id = payload.get("session_id", "")

        if not message:
            return JsonResponse({"error": "message is required"}, status=400)

        # Ensure session_id exists
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create or get user based on user_id or session_id
        user = None
        if user_id:
            try:
                user = EndUser.objects.get(id=user_id)
            except EndUser.DoesNotExist:
                logger.warning(f"User with ID {user_id} not found. Proceeding with session_id.")
        
        if not user:
            # If no user_id or user_id was invalid, try to get/create an anonymous user for this session
            # We use email as a unique identifier for EndUser. For anonymous sessions, we can use a dummy email.
            anonymous_email = f"anonymous_{session_id}@example.com"
            user, created = EndUser.objects.get_or_create(
                email=anonymous_email,
                defaults={'name': f"Anonymous User {session_id}", 'session_id': session_id}
            )
            if not created:
                # Update session_id if user already existed (e.g., returning anonymous user)
                user.session_id = session_id
                user.save()

        # Get or create RAGConversationSession
        rag_session, created = RAGConversationSession.objects.get_or_create(
            session_id=session_id,
            defaults={'user': user}
        )
        # If the session already existed, ensure its user is updated if a new user is identified
        if not created and rag_session.user != user:
            rag_session.user = user
            rag_session.save()

        # ===== USE RAG SYSTEM FOR SEMANTIC UNDERSTANDING =====
        # Get RAG system instance
        rag_system = get_rag_system()
        
        # Generate unique query ID for tracking
        query_id = str(uuid.uuid4())
        
        logger.info(f"Processing query with RAG system: {message[:100]}...")
        
        # Use RAG system to answer query with semantic understanding
        # This will:
        # 1. Generate Gemini embeddings for the query
        # 2. Perform vector similarity search to find semantically similar FAQs
        # 3. Generate contextual response using Gemini
        rag_response = rag_system.answer_query(
            query=message,
            session_id=session_id,
            query_id=query_id
        )
        
        # Extract response details
        response_text = rag_response.text
        confidence = rag_response.confidence
        
        # Get source FAQ IDs from the response
        source_faq_ids = [faq.id for faq in rag_response.source_faqs] if rag_response.source_faqs else []
        match_id = source_faq_ids[0] if source_faq_ids else None
        
        # Determine match type based on confidence and source
        if confidence >= 0.7 and source_faq_ids:
            match_type = "semantic_match"
        elif confidence >= 0.3 and source_faq_ids:
            match_type = "partial_match"
        else:
            match_type = "fallback"
        
        # Get corrected/processed query if available
        corrected_query = message
        if hasattr(rag_response, 'processed_query') and rag_response.processed_query:
            corrected_query = rag_response.processed_query.corrected_query
        
        logger.info(f"RAG response generated - Confidence: {confidence:.2f}, Match type: {match_type}, Source FAQs: {len(source_faq_ids)}")

        # Log Interaction to database
        user_request = UserRequest.objects.create(
            user=user,
            session_id=session_id,
            text=message  # Store original text
        )
        BotResponse.objects.create(
            request=user_request,
            text=response_text
        )

        return JsonResponse({
            "success": True,
            "response": response_text,
            "confidence": confidence,
            "source_faq_id": match_id,
            "source_faq_ids": source_faq_ids,  # Return all source FAQ IDs
            "corrected_query": corrected_query,
            "match_type": match_type,
            "query_id": query_id  # Include query ID for feedback tracking
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}", exc_info=True)
        return JsonResponse({"error": f"Chatbot response failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def faq_search_demo(request):
    """
    Demo page for testing the smart FAQ search functionality.
    """
    return render(request, "faq/search_demo.html")


@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """
    Store user feedback on chatbot responses for learning.

    POST /api/feedback/
    {
        "message_id": "msg_123456789",
        "faq_id": 5,
        "rating": 1 or -1,
        "query": "user's original question",
        "response": "bot's response",
        "session_id": "session_123",
        "user_email": "user@example.com"
    }
    """
    try:
        from .models import ChatFeedback, FAQEntry

        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}

        message_id = payload.get("message_id", "")
        faq_id = payload.get("faq_id")
        rating = payload.get("rating", 0)
        query = payload.get("query", "")
        response_text = payload.get("response", "")
        session_id = payload.get("session_id", "")
        user_email = payload.get("user_email", "")

        if not message_id or not rating:
            return JsonResponse({"error": "message_id and rating are required"}, status=400)

        # Get FAQ if provided
        faq = None
        if faq_id:
            try:
                faq = FAQEntry.objects.get(id=faq_id)
            except FAQEntry.DoesNotExist:
                pass

        # Create feedback record
        feedback = ChatFeedback.objects.create(
            message_id=message_id,
            faq=faq,
            rating=rating,
            query=query,
            response=response_text,
            session_id=session_id,
            user_email=user_email or None
        )

        # If positive feedback and we have a FAQ, potentially add query keywords as synonyms
        # If positive feedback and we have a FAQ, potentially add query keywords as synonyms
        if rating > 0 and faq and query:
            logger.info(
                f"Positive feedback for FAQ {faq.id}: '{query}' -> '{faq.question[:50]}'")

            # Active Learning: Extract new keywords from the successful query
            # This helps the bot understand user vocabulary better over time
            try:
                from .faq_matcher import faq_matcher
                query_keywords = faq_matcher._extract_keywords(query)
                current_keywords = set(faq.keywords.lower().split(
                    ', ')) if faq.keywords else set()

                new_keywords = []
                for kw in query_keywords:
                    if kw not in current_keywords:
                        new_keywords.append(kw)
                        current_keywords.add(kw)

                if new_keywords:
                    # Append new keywords to the database entry
                    updated_keywords = faq.keywords + ", " + \
                        ", ".join(new_keywords) if faq.keywords else ", ".join(
                            new_keywords)
                    # Clean up commas
                    updated_keywords = ", ".join(
                        [k.strip() for k in updated_keywords.split(',') if k.strip()])

                    faq.keywords = updated_keywords
                    faq.save()
                    logger.info(
                        f"🧠 Learned new keywords for FAQ #{faq.id}: {new_keywords}")
            except Exception as e:
                logger.error(f"Error active learning from feedback: {e}")

        # If negative feedback, log for review
        if rating < 0:
            logger.warning(
                f"Negative feedback: Query='{query}' Response='{response_text[:100]}'")

        return JsonResponse({
            "success": True,
            "feedback_id": feedback.id,
            "message": "Thank you for your feedback!"
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def ai_chatbot_demo(request):
    """
    Demo page for testing the AI-enhanced chatbot functionality.
    """
    return render(request, "faq/ai_demo.html")


@csrf_exempt
@require_http_methods(["POST"])
def ingest_document(request):
    """
    Ingest a single DOCX document into the RAG system.

    POST /api/ingest-document/
    {
        "document_path": "/path/to/document.docx",
        "force_update": false
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        document_path = payload.get("document_path", "").strip()
        force_update = payload.get("force_update", False)

        if not document_path:
            return JsonResponse({"error": "document_path is required"}, status=400)

        # Initialize RAG system if needed
        from .rag.core.factory import rag_factory
        rag_system = rag_factory.create_default_system()

        # Process the document
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
        return JsonResponse({"error": f"Document ingestion failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def ingest_documents_batch(request):
    """
    Ingest multiple DOCX documents into the RAG system.

    POST /api/ingest-batch/
    {
        "document_paths": ["/path/to/doc1.docx", "/path/to/doc2.docx"],
        "force_update": false,
        "parallel": true
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        document_paths = payload.get("document_paths", [])
        force_update = payload.get("force_update", False)
        parallel = payload.get("parallel", True)

        if not document_paths or not isinstance(document_paths, list):
            return JsonResponse({"error": "document_paths must be a non-empty list"}, status=400)

        # Initialize RAG system if needed
        from .rag.core.factory import rag_factory
        rag_system = rag_factory.create_default_system()

        # Process the documents
        results = rag_system.process_documents_batch(
            document_paths, force_update, parallel)

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
        return JsonResponse({"error": f"Batch document ingestion failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def ingest_directory(request):
    """
    Ingest all DOCX documents from a directory into the RAG system.

    POST /api/ingest-directory/
    {
        "directory_path": "/path/to/documents/",
        "recursive": true,
        "force_update": false,
        "parallel": true
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        directory_path = payload.get("directory_path", "").strip()
        recursive = payload.get("recursive", True)
        force_update = payload.get("force_update", False)
        parallel = payload.get("parallel", True)

        if not directory_path:
            return JsonResponse({"error": "directory_path is required"}, status=400)

        # Initialize RAG system if needed
        from .rag.core.factory import rag_factory
        rag_system = rag_factory.create_default_system()

        # Process the directory
        results = rag_system.process_directory(
            directory_path, recursive, force_update, parallel)

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
        return JsonResponse({"error": f"Directory ingestion failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def ingestion_progress(request):
    """
    Get current document ingestion progress.

    GET /api/ingestion-progress/
    """
    try:
        # Initialize RAG system if needed
        from .rag.core.factory import rag_factory
        rag_system = rag_factory.create_default_system()

        # Get progress information
        progress = rag_system.get_ingestion_progress()

        if progress:
            return JsonResponse({
                "success": True,
                "in_progress": True,
                "progress": progress
            })
        else:
            return JsonResponse({
                "success": True,
                "in_progress": False,
                "message": "No ingestion currently in progress"
            })

    except Exception as e:
        return JsonResponse({"error": f"Failed to get ingestion progress: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def ingestion_stats(request):
    """
    Get document ingestion pipeline statistics.

    GET /api/ingestion-stats/
    """
    try:
        # Initialize RAG system if needed
        from .rag.core.factory import rag_factory
        rag_system = rag_factory.create_default_system()

        # Get ingestion statistics
        stats = rag_system.get_ingestion_stats()

        return JsonResponse({
            "success": True,
            "stats": stats
        })

    except Exception as e:
        return JsonResponse({"error": f"Failed to get ingestion stats: {str(e)}"}, status=500)


# RAG System Django Integration Views

@csrf_exempt
@require_http_methods(["POST"])
def rag_query(request):
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        query = payload.get("query", "").strip()
        session_id = payload.get("session_id")
        user_email = payload.get("user_email")

        if not query:
            logger.warning("Bad Request: Query is missing in RAG request.")
            return JsonResponse({"error": "query is required"}, status=400)

        rag_system = get_rag_system()
        django_service = RAGDjangoService(rag_system)

        if session_id:
            session = django_service.get_or_create_session(
                session_id, user_email)
        else:
            session_id = str(uuid.uuid4())
            session = django_service.get_or_create_session(
                session_id, user_email)

        query_id = str(uuid.uuid4())
        rag_response = rag_system.answer_query(
            query, session_id=session_id, query_id=query_id)

        # Log the query and response
        django_service.log_query_and_response(
            query_id=rag_response.query_id,
            session_id=session.session_id,
            processed_query=rag_response.processed_query,
            response=rag_response,
            processing_time=rag_response.processing_time,
            source_faq_ids=[
                faq.id for faq in rag_response.response.source_faqs]
        )

        return JsonResponse({
            "query_id": rag_response.query_id,
            "response": rag_response.response.text,
            "confidence": rag_response.response.confidence,
            "source_faqs": [
                {"id": faq.id, "question": faq.question, "answer": faq.answer}
                for faq in rag_response.response.source_faqs
            ],
            "session_id": session.session_id,
            "processing_time": rag_response.processing_time,
            "message": "RAG query successful"
        })

    except json.JSONDecodeError:
        logger.error(
            "Invalid JSON in request body for RAG query.", exc_info=True)
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred during RAG query processing: {e}")
        return JsonResponse({"error": f"Internal Server Error: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def rag_feedback(request):
    """
    Submit feedback for a RAG system response.

    POST /api/rag/feedback/
    {
        "query_id": "uuid",
        "user_email": "user@example.com",
        "rating": 4,
        "comments": "Very helpful response",
        "accuracy_rating": 5,
        "relevance_rating": 4,
        "helpfulness_rating": 4
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        query_id = payload.get("query_id")
        user_email = payload.get("user_email")
        rating = payload.get("rating")
        comments = payload.get("comments", "")
        accuracy_rating = payload.get("accuracy_rating")
        relevance_rating = payload.get("relevance_rating")
        helpfulness_rating = payload.get("helpfulness_rating")

        if not query_id or not rating:
            return JsonResponse({"error": "query_id and rating are required"}, status=400)

        if rating < 1 or rating > 5:
            return JsonResponse({"error": "rating must be between 1 and 5"}, status=400)

        # Initialize RAG Django service
        from .rag_django_service import RAGDjangoService
        django_service = RAGDjangoService()

        # Submit feedback
        feedback = django_service.submit_feedback(
            query_id=query_id,
            user_email=user_email,
            rating=rating,
            comments=comments,
            accuracy_rating=accuracy_rating,
            relevance_rating=relevance_rating,
            helpfulness_rating=helpfulness_rating
        )

        return JsonResponse({
            "success": True,
            "feedback_id": feedback.id,
            "message": "Feedback submitted successfully"
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Feedback submission failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def rag_analytics(request):
    """
    Get RAG system analytics from Django models.

    GET /api/rag/analytics/?days=7
    """
    try:
        days = int(request.GET.get("days", "7"))

        # Initialize RAG Django service
        from .rag_django_service import RAGDjangoService
        django_service = RAGDjangoService()

        # Get analytics
        analytics = django_service.get_system_analytics(days)

        return JsonResponse({
            "success": True,
            "analytics": analytics
        })

    except ValueError:
        return JsonResponse({"error": "days parameter must be an integer"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Analytics retrieval failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def rag_health(request):
    """
    Get RAG system health status from Django integration.

    GET /api/rag/health/
    """
    try:
        # Initialize RAG Django service
        from .rag_django_service import RAGDjangoService
        django_service = RAGDjangoService()

        # Get health summary
        health = django_service.get_system_health_summary()

        return JsonResponse({
            "success": True,
            "health": health
        })

    except Exception as e:
        return JsonResponse({"error": f"Health check failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def rag_documents(request):
    """
    Get RAG document processing status from Django models.

    GET /api/rag/documents/?status=completed&limit=10
    """
    try:
        from .models import RAGDocument

        status = request.GET.get("status")
        limit = int(request.GET.get("limit", "10"))

        queryset = RAGDocument.objects.all()

        if status:
            queryset = queryset.filter(status=status)

        documents = queryset.order_by('-created_at')[:limit]

        data = []
        for doc in documents:
            data.append({
                "id": doc.id,
                "file_name": doc.file_name,
                "file_path": doc.file_path,
                "status": doc.status,
                "faqs_extracted": doc.faqs_extracted,
                "file_size": doc.file_size,
                "created_at": doc.created_at.isoformat(),
                "processing_started_at": doc.processing_started_at.isoformat() if doc.processing_started_at else None,
                "processing_completed_at": doc.processing_completed_at.isoformat() if doc.processing_completed_at else None,
                "error_message": doc.error_message
            })

        return JsonResponse({
            "success": True,
            "documents": data,
            "total_count": queryset.count()
        })

    except ValueError:
        return JsonResponse({"error": "limit parameter must be an integer"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Document retrieval failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def rag_faqs(request):
    """
    Get RAG FAQ entries from Django models.

    GET /api/rag/faqs/?category=general&limit=10&search=password
    """
    try:
        from .models import RAGFAQEntry
        from django.db.models import Q

        category = request.GET.get("category")
        limit = int(request.GET.get("limit", "10"))
        search = request.GET.get("search")

        queryset = RAGFAQEntry.objects.all()

        if category:
            queryset = queryset.filter(category=category)

        if search:
            queryset = queryset.filter(
                (Q(question__icontains=search)
                 | Q(answer__icontains=search)
                 | Q(keywords__icontains=search))
            )

        faqs = queryset.order_by('-query_matches', '-created_at')[:limit]

        data = []
        for faq in faqs:
            data.append({
                "id": faq.id,
                "rag_id": faq.rag_id,
                "question": faq.question,
                "answer": faq.answer,
                "keywords": faq.keywords,
                "category": faq.category,
                "confidence_score": faq.confidence_score,
                "query_matches": faq.query_matches,
                "last_matched": faq.last_matched.isoformat() if faq.last_matched else None,
                "document_id": faq.document.id,
                "document_name": faq.document.file_name,
                "created_at": faq.created_at.isoformat()
            })

        return JsonResponse({
            "success": True,
            "faqs": data,
            "total_count": queryset.count()
        })

    except ValueError:
        return JsonResponse({"error": "limit parameter must be an integer"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"FAQ retrieval failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def rag_top_faqs(request):
    """
    Get the most frequently matched FAQs.

    GET /api/rag/top-faqs/?limit=10
    """
    try:
        limit = int(request.GET.get("limit", "10"))

        # Initialize RAG Django service
        from .rag_django_service import RAGDjangoService
        django_service = RAGDjangoService()

        # Get top FAQs
        top_faqs = django_service.get_top_faqs(limit)

        data = []
        for faq in top_faqs:
            data.append({
                "id": faq.id,
                "rag_id": faq.rag_id,
                "question": faq.question,
                "answer": faq.answer,
                "category": faq.category,
                "query_matches": faq.query_matches,
                "last_matched": faq.last_matched.isoformat() if faq.last_matched else None,
                "confidence_score": faq.confidence_score
            })

        return JsonResponse({
            "success": True,
            "top_faqs": data
        })

    except ValueError:
        return JsonResponse({"error": "limit parameter must be an integer"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Top FAQs retrieval failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def rag_process_document_django(request):
    """
    Process a document using RAG system with full Django integration.

    POST /api/rag/process-document/
    {
        "document_path": "/path/to/document.docx",
        "force_update": false
    }
    """
    try:
        payload = json.loads(request.body.decode("utf-8")
                             ) if request.body else {}
        document_path = payload.get("document_path", "").strip()
        force_update = payload.get("force_update", False)

        if not document_path:
            return JsonResponse({"error": "document_path is required"}, status=400)

        import os
        if not os.path.exists(document_path):
            return JsonResponse({"error": "document file not found"}, status=404)

        # Initialize RAG Django service
        from .rag_django_service import RAGDjangoService
        from django.core.cache import cache

        # Get RAG system from cache or initialize
        rag_system = cache.get('rag_system')
        if not rag_system:
            from .rag.core.factory import rag_factory
            rag_system = rag_factory.create_default_system()
            cache.set('rag_system', rag_system, timeout=None)

        django_service = RAGDjangoService(rag_system)

        # Calculate file hash and metadata
        import hashlib
        with open(document_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        file_size = os.path.getsize(document_path)
        file_name = os.path.basename(document_path)

        # Create document record
        document = django_service.create_document_record(
            file_path=document_path,
            file_name=file_name,
            file_hash=file_hash,
            file_size=file_size
        )

        # Process document
        django_service.update_document_processing_status(
            document.id, 'processing')

        try:
            # Process with RAG system
            faqs = rag_system.process_document(document_path, force_update)

            # Sync FAQs to Django
            django_faqs = django_service.sync_faqs_to_django(faqs, document.id)

            # Update document status
            django_service.update_document_processing_status(
                document.id, 'completed', len(django_faqs)
            )

            return JsonResponse({
                "success": True,
                "document_id": document.id,
                "document_path": document_path,
                "faqs_extracted": len(django_faqs),
                "force_update": force_update,
                "message": f"Successfully processed document and extracted {len(django_faqs)} FAQs"
            })

        except Exception as e:
            # Update document status with error
            django_service.update_document_processing_status(
                document.id, 'failed', 0, str(e)
            )
            raise

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Document processing failed: {str(e)}"}, status=500)
