
import os
import sys
import django

# Setup Django
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.components.docx_scraper.scraper import DOCXScraper

def verify_extraction():
    scraper = DOCXScraper()
    docx_path = 'c:/final correction/backend/FAQ.docx'
    
    print(f"Extracting FAQs from {docx_path}...")
    faqs = scraper.extract_faqs(docx_path)
    
    print(f"Extracted {len(faqs)} FAQs.\n")
    
    # Show the first few as samples
    for i, faq in enumerate(faqs[:10]):
        print(f"--- FAQ {i+1} ---")
        print(f"Q: {faq.question}")
        # print(f"A: {faq.answer[:100]}...")
        print(f"Category: {faq.category}")
        print(f"Audience: {faq.audience}")
        print(f"Intent:   {faq.intent}")
        print(f"Condition: {faq.condition}")
        print(f"Keywords:  {faq.keywords}")
        print()

    # Summary of counts
    from collections import Counter
    print("--- Distribution ---")
    print(f"Categories: {Counter(f.category for f in faqs)}")
    print(f"Audiences:  {Counter(f.audience for f in faqs)}")
    print(f"Intents:    {Counter(f.intent for f in faqs)}")

if __name__ == "__main__":
    verify_extraction()
