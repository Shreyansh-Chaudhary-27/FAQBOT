from typing import Optional
from django.conf import settings
from google import genai
import logging

logger = logging.getLogger(__name__)

def configure_gemini():
    """
    Verifies the Google Gemini API key from Django settings.
    Raises RuntimeError if the API key is not found.
    """
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured in Django settings.")
        raise RuntimeError("Google API key not configured.")
    
    # Optional: logic to verify client init works, though it's lightweight
    try:
        genai.Client(api_key=api_key)
        logger.debug("Gemini API key verification successful.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        raise RuntimeError(f"Gemini Client initialization failed: {e}")

def generate_text_response(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Generates a text response from Gemini given a prompt.
    Instantiates a Client for each call (stateless).
    Returns an empty string if no text is produced or on error.
    """
    if model_name is None:
        model_name = getattr(settings, "GEMINI_MODEL", "gemini-1.5-flash")
        
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        logger.error("GEMINI_API_KEY not found during generation.")
        return ""

    try:
        client = genai.Client(api_key=api_key)
        logger.info("Sending prompt to Gemini model '%s': %s", model_name, prompt[:200] + "...")
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        # Extract text from the response
        if response.text:
            logger.info("Received text response from Gemini (length: %d)", len(response.text))
            return response.text.strip()
            
        # Check for candidates if main text is empty (structure varies by SDK version but .text usually handles it)
        if response.candidates:
             # Basic concatenation if multiple parts
             texts = []
             for cand in response.candidates:
                 if cand.content and cand.content.parts:
                     for part in cand.content.parts:
                         if part.text:
                             texts.append(part.text)
             if texts:
                 full_text = "\n\n".join(texts)
                 return full_text
        
        logger.warning("Gemini returned no text content.")
        return ""

    except Exception as exc:
        logger.exception("Gemini generation failed for prompt: %s", prompt[:200] + "...", exc_info=True)
        return "An error occurred while generating the response."
