# DOCX Scraper Component
# Document parsing and FAQ extraction from DOCX files

from .scraper import DOCXScraper
from .document_reader import DOCXDocumentReader
from .pattern_recognizer import FAQPatternRecognizer, FAQPattern
from .validator import FAQValidator, DuplicateMatch

__all__ = [
    'DOCXScraper',
    'DOCXDocumentReader', 
    'FAQPatternRecognizer',
    'FAQPattern',
    'FAQValidator',
    'DuplicateMatch'
]