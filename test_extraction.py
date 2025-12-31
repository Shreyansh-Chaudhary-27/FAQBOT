import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')

import django
django.setup()

from faq.rag.components.docx_scraper.scraper import DOCXScraper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_extraction():
    print("=" * 50)
    print("TESTING FAQ EXTRACTION FROM DOCX")
    print("=" * 50)

    scraper = DOCXScraper()
    doc_path = os.path.join(project_root, "AI chatbot.docx")
    
    if not os.path.exists(doc_path):
        print("File not found.")
        return

    print(f"Processing {doc_path}...")
    faqs = scraper.extract_faqs(doc_path)
    
    print(f"Extracted {len(faqs)} FAQs.")
    for i, faq in enumerate(faqs[:10]):
        print(f"\nFAQ {i+1}:")
        print(f" Q: {faq.question}")
        print(f" A: {faq.answer}")
        print(f" Keywords: {faq.keywords}")

    if len(faqs) > 10:
        print(f"\n... and {len(faqs) - 10} more.")

if __name__ == "__main__":
    test_extraction()
