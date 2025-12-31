import re
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Import FAQ entries from a DOCX file. Detects Q/A pairs and saves to FAQEntry."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str, help="Path to DOCX file")
        parser.add_argument("--keywords", type=str, default="auto", help="Keyword mode: 'auto' or empty string")

    def handle(self, path, keywords, **options):
        try:
            from docx import Document  # type: ignore
        except Exception as e:
            raise CommandError("python-docx is required. Install with 'pip install python-docx'.")

        from faq.models import FAQEntry

        try:
            doc = Document(path)
        except Exception as e:
            raise CommandError(f"Unable to open DOCX: {e}")

        entries = []
        question = None
        answer_lines = []

        def flush():
            nonlocal question, answer_lines
            if question and answer_lines:
                answer = "\n".join(answer_lines).strip()
                if answer:
                    entries.append((question.strip(), answer))
            question = None
            answer_lines = []

        q_pattern = re.compile(r"^(?:Q\s*[:\-]\s*)?(.*\?)(\s*)$", re.IGNORECASE)
        a_prefix = re.compile(r"^(?:A\s*[:\-]\s*)", re.IGNORECASE)

        for p in doc.paragraphs:
            text = (p.text or "").strip()
            if not text:
                continue
            m = q_pattern.match(text)
            if m:
                flush()
                question = m.group(1)
                continue
            if question is None:
                # Try headings as questions
                if p.style and getattr(p.style, "name", "").lower().startswith("heading"):
                    flush()
                    question = text
                    continue
            # Accumulate answer lines (strip optional 'A:' prefix)
            answer_lines.append(a_prefix.sub("", text))

        flush()

        created = 0
        for q, a in entries:
            kw = ""
            if keywords == "auto":
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", q.lower())
                stop = {"the","and","or","for","to","of","in","on","at","a","an","is","are","how","what","where","when","who","why"}
                tokens = [t for t in tokens if t not in stop][:12]
                kw = ",".join(dict.fromkeys(tokens))
            FAQEntry.objects.create(question=q, answer=a, keywords=kw)
            created += 1

        self.stdout.write(self.style.SUCCESS(f"Imported {created} FAQ entries from {path}"))
