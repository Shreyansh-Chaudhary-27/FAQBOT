import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.components.docx_scraper.pattern_recognizer import FAQPatternRecognizer

recognizer = FAQPatternRecognizer()
text = "Q: What are the professional conduct expectations?\n A: Maintain discipline, respect all team members, meet deadlines, and communicate professionally. Disrespect or unprofessional behaviour may lead to disciplinary measures."

patterns = recognizer.recognize_paragraph_patterns([text])
print(f"Text: {text[:50]}...")
print(f"Patterns found: {len(patterns)}")
for p in patterns:
    print(f" - Q: {p.question}")
    print(f" - A: {p.answer}")
    print(f" - Confidence: {p.confidence}")

# Test is_question_like
q_text = "Q: What are the professional conduct expectations?"
print(f"\nis_question_like('{q_text}'): {recognizer.is_question_like(q_text)}")

# Test is_answer_like
a_text = "A: Maintain discipline, respect all team members, meet deadlines, and communicate professionally. Disrespect or unprofessional behaviour may lead to disciplinary measures."
print(f"is_answer_like('{a_text}'): {recognizer.is_answer_like(a_text)}")
