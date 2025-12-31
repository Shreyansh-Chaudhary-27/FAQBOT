from typing import Optional
from django.conf import settings
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

def configure_gemini():
    """
    Configures the Google Gemini API with the API key from Django settings.
    Raises RuntimeError if the API key is not found.
    """
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured in Django settings.")
        raise RuntimeError("Google API key not configured.")
    genai.configure(api_key=api_key)
    logger.debug("Gemini API configured successfully.")

def generate_text_response(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Generates a text response from Gemini given a prompt.
    Ensures the Gemini API is configured before making the call.
    Returns an empty string if no text is produced or on error.
    """
    if model_name is None:
        model_name = getattr(settings, "GEMINI_MODEL", "gemini-1.5-flash")
    try:
        genai.configure(api_key=getattr(settings, "GEMINI_API_KEY", None))
        model = genai.GenerativeModel(model_name)
        logger.info("Sending prompt to Gemini model '%s': %s", model_name, prompt[:200] + "...")
        response = model.generate_content(prompt)

        # Extract text from the response, handling potential safety blocks or empty responses
        if hasattr(response, "text") and response.text:
            logger.info("Received text response from Gemini (length: %d)", len(response.text))
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            # If 'text' attribute is missing, try to get text from candidates
            texts = [getattr(c, "text", "") for c in response.candidates if hasattr(c, "text")]
            if texts:
                full_text = "\n\n".join([t for t in texts if t])
                logger.info("Received candidate text response from Gemini (length: %d)", len(full_text))
                return full_text
        
        # If no text or candidates, check for safety ratings
        if hasattr(response, "prompt_feedback") and response.prompt_feedback.safety_ratings:
            safety_ratings = response.prompt_feedback.safety_ratings
            logger.warning("Gemini response blocked due to safety settings. Ratings: %s", safety_ratings)
            # You might want to return a specific message here, or raise an exception
            return "The response was blocked due to safety concerns."

        logger.warning("Gemini returned no text content or candidates for the prompt.")
        return "" # Return empty string if no valid text response

    except RuntimeError as e:
        # Re-raise if it's our own configuration error
        raise e
    except Exception as exc:
        logger.exception("Gemini generation failed for prompt: %s", prompt[:200] + "...", exc_info=True)
        # Depending on desired behavior, you might want to return a user-friendly error message
        return "An error occurred while generating the response."
