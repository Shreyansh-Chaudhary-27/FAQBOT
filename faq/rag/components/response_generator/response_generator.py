import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

# Ensure django.conf.settings is imported if needed for logging path,
# though Pathlib handles it relative to this file.
# from django.conf import settings

from faq.gemini_service import generate_text_response
from faq.rag.config.settings import rag_config

# --- Logging Setup ---
logger = logging.getLogger("faq.rag.response_generator")
if not logger.handlers:
    try:
        # Resolve <repo_root>/backend/debug_logs/rag_debug_response_generator.log
        # The file is located at backend/faq/rag/components/response_generator/response_generator.py
        # parents[4] points to <repo_root>/backend
        log_path = Path(__file__).resolve().parents[4] / "debug_logs" / "rag_debug_response_generator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    except Exception as e:
        logger.error("Failed to set up file logger for RAG response generator: %s", e)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
# --- End Logging Setup ---


from faq.rag.interfaces.base import FAQEntry, ProcessedQuery, Response, ResponseGeneratorInterface


class ResponseGeneratorError(Exception):
    """Custom exception for response generator errors."""
    pass
# --- END PLACEHOLDER TYPES ---


class BasicResponseGenerator(ResponseGeneratorInterface):
    """
    Basic response generator that uses Gemini for contextual response generation
    with retrieved FAQ content.
    """
    def __init__(self):
        self.formatting_rules = self._load_formatting_rules()
        self.generation_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.average_response_time = 0.0

    def _load_formatting_rules(self):
        """Load formatting rules for response presentation."""
        return {
            'max_answer_length': 500,     # Maximum length for single answer
            'max_total_length': 1000,     # Maximum total response length
            'preserve_formatting': True   # Preserve original FAQ formatting
        }

    def generate_response(self, query: str, retrieved_faqs: List[FAQEntry], query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """
        Generate contextual response from retrieved FAQs using Gemini.
        """
        context_chunks = [faq.answer for faq in retrieved_faqs if faq.answer]
        context_text = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk and chunk.strip())

        # Use the processed_query if available, otherwise fall back to original query
        actual_query = processed_query.corrected_query if processed_query else query

        prompt = (
            "You are a helpful assistant that answers using the provided context.\n\n"
            f"Question:\n{actual_query}\n\n"
            f"Relevant Context:\n{context_text if context_text else '[No explicit context provided]'}\n\n"
            "Instructions:\n"
            "- Use only the relevant context above when possible.\n"
            "- If the context is insufficient or missing, explicitly say you do not know.\n"
            "- Be concise and precise.\n"
        )

        logger.info("Generating response for query_id=%s | question_len=%d | context_len=%d",
                    query_id, len(actual_query or ""), len(context_chunks))

        import time
        start_time = time.time()

        try:
            self.generation_count += 1
            answer_text = generate_text_response(prompt, model_name=rag_config.config.gemini_model)

            # Track response time
            response_time = time.time() - start_time
            self.total_response_time += response_time
            self.average_response_time = self.total_response_time / self.generation_count

            logger.info("Response generated | length=%d | time=%.2fs", len(answer_text or ""), response_time)

            # Create a Response object with all required fields
            return Response(
                text=answer_text,
                confidence=1.0 if retrieved_faqs else 0.5,
                source_faqs=retrieved_faqs,
                context_used=len(retrieved_faqs) > 0,
                generation_method='rag' if retrieved_faqs else 'direct',
                query_id=query_id or "unknown",
                processed_query=processed_query or ProcessedQuery(query, query, "unknown", [], "en", 1.0),
                metadata={
                    "ai_generated": True,
                    "response_time": response_time,
                    "num_sources": len(retrieved_faqs)
                }
            )
        except Exception as exc:
            logger.error("RAG response generation failed for query_id=%s: %s", query_id, exc)
            self.error_count += 1
            raise ResponseGeneratorError(f"Failed to generate response: {exc}") from exc

    def synthesize_multiple_sources(self, faqs: List[FAQEntry]) -> str:
        """Synthesize information from multiple FAQ sources."""
        if not faqs:
            return ""
        return "\n\n".join([f"Source {i+1}: {faq.answer}" for i, faq in enumerate(faqs)])

    def maintain_context(self, conversation_history: List[Dict[str, Any]]) -> Any:
        """Maintain conversation context (placeholder)."""
        # Returning None or a simple placeholder as this is a basic generator
        return None

    def calculate_confidence(self, response: Response) -> float:
        """Calculate confidence score for generated response."""
        if not response.source_faqs:
            return 0.1
        return 0.9

    def validate_candidate_relevance(self, query: str, faq: FAQEntry) -> bool:
        """Basic implementation of candidate relevance validation (always True for basic generator)."""
        return True

    def get_generator_stats(self) -> dict:
        """
        Get statistics about response generation performance.

        Returns:
            Dictionary containing generation statistics
        """
        return {
            "total_generations": self.generation_count,
            "error_count": self.error_count,
            "success_rate": (self.generation_count - self.error_count) / max(self.generation_count, 1),
            "average_response_time": self.average_response_time,
            "generator_type": "basic_gemini"
        }
