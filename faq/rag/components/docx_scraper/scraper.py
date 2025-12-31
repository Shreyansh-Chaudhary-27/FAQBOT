"""
Main DOCX Scraper Component

This module integrates document reading, pattern recognition, and validation
to provide a complete DOCX FAQ extraction system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import re

from faq.rag.interfaces.base import DOCXScraperInterface, FAQEntry, DocumentStructure, ValidationResult
from faq.rag.utils.ngram_utils import generate_ngrams
from .document_reader import DOCXDocumentReader
from .pattern_recognizer import FAQPatternRecognizer, FAQPattern
from .validator import FAQValidator

logger = logging.getLogger(__name__)


class DOCXScraper(DOCXScraperInterface):
    """Complete DOCX scraper implementation with pattern recognition and validation."""
    
    def __init__(self):
        """Initialize the scraper with all components."""
        self.document_reader = DOCXDocumentReader()
        self.pattern_recognizer = FAQPatternRecognizer()
        self.validator = FAQValidator()
    
    def extract_faqs(self, docx_path: str) -> List[FAQEntry]:
        """
        Extract FAQ entries from a DOCX document.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            List of extracted FAQ entries
        """
        try:
            logger.info(f"Starting FAQ extraction from: {docx_path}")
            
            # Load the document
            document = self.document_reader.load_document(docx_path)
            if not document:
                logger.error(f"Failed to load document: {docx_path}")
                return []
            
            # Analyze document structure
            structure = self.document_reader.analyze_document_structure(document)
            
            # Identify FAQ patterns
            patterns = self.pattern_recognizer.identify_faq_patterns(structure)
            
            if not patterns:
                logger.warning(f"No FAQ patterns found in document: {docx_path}")
                return []
            
            # Convert patterns to FAQ entries with rule-based metadata extraction
            faqs = self._convert_patterns_to_faqs(patterns, docx_path, structure)
            
            # Validate and clean up
            faqs = self._validate_and_clean_faqs(faqs)
            
            # Categorize FAQs
            categorized = self.validator.categorize_faqs(faqs)
            
            # Update FAQ categories
            for category, faq_list in categorized.items():
                for faq in faq_list:
                    if not faq.category or faq.category == 'general':
                        faq.category = category
            
            logger.info(f"Successfully extracted {len(faqs)} FAQ entries from {docx_path}")
            return faqs
            
        except Exception as e:
            logger.error(f"Error extracting FAQs from {docx_path}: {str(e)}")
            return []
    
    def parse_document_structure(self, document_path: str) -> DocumentStructure:
        """
        Parse and analyze document structure.
        
        Args:
            document_path: Path to the document
            
        Returns:
            DocumentStructure object
        """
        try:
            document = self.document_reader.load_document(document_path)
            if not document:
                return DocumentStructure(
                    document_type="docx",
                    sections=[],
                    tables=[],
                    lists=[],
                    paragraphs=[]
                )
            
            return self.document_reader.analyze_document_structure(document)
            
        except Exception as e:
            logger.error(f"Error parsing document structure: {str(e)}")
            return DocumentStructure(
                document_type="docx",
                sections=[],
                tables=[],
                lists=[],
                paragraphs=[]
            )
    
    def identify_faq_patterns(self, content: List[str]) -> List[Dict[str, Any]]:
        """
        Identify FAQ patterns in document content.
        
        Args:
            content: List of content strings
            
        Returns:
            List of FAQ pattern dictionaries
        """
        try:
            # Create a simple document structure from content
            structure = DocumentStructure(
                document_type="text",
                sections=[],
                tables=[],
                lists=[],
                paragraphs=content
            )
            
            patterns = self.pattern_recognizer.identify_faq_patterns(structure)
            
            # Convert patterns to dictionaries
            pattern_dicts = []
            for pattern in patterns:
                pattern_dicts.append({
                    "pattern_type": pattern.pattern_type,
                    "question": pattern.question,
                    "answer": pattern.answer,
                    "confidence": pattern.confidence,
                    "source_location": pattern.source_location,
                    "keywords": pattern.keywords,
                    "metadata": pattern.metadata
                })
            
            return pattern_dicts
            
        except Exception as e:
            logger.error(f"Error identifying FAQ patterns: {str(e)}")
            return []
    
    def validate_extraction(self, faqs: List[FAQEntry]) -> ValidationResult:
        """
        Validate extracted FAQ entries.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            ValidationResult with validation status
        """
        return self.validator.validate_extraction(faqs)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing emojis, numbering, and common prefixes."""
        if not text:
            return ""
            
        # 1. First Pass: Strip common textual prefixes
        prefixes = [
            r'^q\d*[:\.\-\s]+', r'^question\d*[:\.\-\s]+', r'^query\d*[:\.\-\s]+',
            r'^a\d*[:\.\-\s]+', r'^answer\d*[:\.\-\s]+', r'^response\d*[:\.\-\s]+',
            r'^📌\s*'
        ]
        
        cleaned = text.strip()
        for prefix in prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
            
        # 2. Second Pass: Strip all leading non-alphanumeric characters and emoji numbers
        # This handles things like "1️⃣9️⃣ ", "📌 ", "1. ", "(a) "
        
        # Remove combining characters (like keycap enclosures)
        cleaned = re.sub(r'[\uFE0F\u20E3]', '', cleaned).strip()
        
        # Strip leading numbers followed by punctuation/spaces or leading non-letters/non-digits
        cleaned = re.sub(r'^[^\w\s]+', '', cleaned).strip()
        cleaned = re.sub(r'^\d+[\.\)\s\-]*', '', cleaned).strip()
        
        # Strip again to catch nested cases (like emoji numbers that left digits behind)
        cleaned = re.sub(r'^[^\w\s]+', '', cleaned).strip()
        
        return cleaned

    def _extract_metadata_from_context(self, question: str, answer: str, section_title: str) -> Dict[str, str]:
        """
        Extract audience, category, intent, and condition using rule-based heuristics.
        """
        metadata = {
            "audience": "any",
            "category": "general",
            "intent": "information",
            "condition": "default"
        }
        
        context_text = f"{section_title} {question} {answer}".lower()
        
        # 1. Determine Category from Section Title
        category_map = {
            "onboarding": ["onboarding", "apply", "join"],
            "leave": ["leave", "holiday", "absent", "vacation"],
            "payroll": ["salary", "payment", "payroll", "stipend"],
            "policy": ["policy", "rule", "protocol", "notice"],
            "hierarchy": ["hierarchy", "report", "manager", "senior"],
            "sales": ["sales", "lead", "client"],
            "technical": ["domain", "portal", "system", "error", "bot"]
        }
        
        for cat, keywords in category_map.items():
            if any(kw in section_title.lower() for kw in keywords):
                metadata["category"] = cat
                break
        
        # 2. Determine Audience
        if any(kw in context_text for kw in ["intern", "internship", "trainee"]):
            metadata["audience"] = "intern"
        elif any(kw in context_text for kw in ["sales", "lead", "deal"]):
            metadata["audience"] = "sales_internal"
        elif any(kw in context_text for kw in ["apply", "applicant", "vacancy"]):
            metadata["audience"] = "applicant"
        elif any(kw in context_text for kw in ["full-time", "employee", "staff"]):
            metadata["audience"] = "full_time"

        # 3. Determine Intent
        if any(kw in question.lower() for kw in ["how", "process", "steps", "procedure"]):
            metadata["intent"] = "process"
        elif any(kw in question.lower() for kw in ["why", "reason"]):
            metadata["intent"] = "reason"
        elif any(kw in context_text for kw in ["rule", "policy", "must", "required"]):
            metadata["intent"] = "policy"
        elif any(kw in question.lower() for kw in ["who", "whom", "contact"]):
            metadata["intent"] = "reporting"

        # 4. Determine Condition
        if any(kw in context_text for kw in ["urgent", "emergency", "immediate"]):
            metadata["condition"] = "urgent"
        elif any(kw in context_text for kw in ["3 month", "three month"]):
            metadata["condition"] = "3_months_plus"
        elif any(kw in context_text for kw in ["unauthorized", "violation"]):
            metadata["condition"] = "violation"

        return metadata

    def _convert_patterns_to_faqs(self, patterns: List[FAQPattern], source_document: str, structure: Optional[DocumentStructure] = None) -> List[FAQEntry]:
        """
        Convert FAQ patterns to FAQ entries with rule-based composite keys.
        """
        faqs = []
        current_time = datetime.now()
        
        for pattern in patterns:
            try:
                # 1. Find the current section title for this pattern
                section_title = "General"
                if structure and structure.sections:
                    # If pattern has source_location info with paragraph/row index
                    p_idx = pattern.metadata.get("paragraph_index") or pattern.metadata.get("question_paragraph")
                    if p_idx is not None:
                        # Find the section that contains this paragraph index
                        # Basic reader puts paragraphs in current_section["paragraphs"]
                        curr_p_count = 0
                        for section in structure.sections:
                            p_in_section = len(section["paragraphs"])
                            if curr_p_count <= p_idx < curr_p_count + p_in_section:
                                section_title = section["title"]
                                break
                            curr_p_count += p_in_section
                
                # 2. Clean Question and Answer
                question = self._clean_text(pattern.question)
                answer = self._clean_text(pattern.answer)
                
                if not question or not answer:
                    continue
                
                # 3. Rule-Based Metadata Extraction
                meta_rules = self._extract_metadata_from_context(question, answer, section_title)
                
                # 4. Generate N-Gram Keywords from Question (Requirement: 90% overlap match)
                ngram_keywords = generate_ngrams(question)
                
                # Combine with pattern's own keywords
                final_keywords_set = set(ngram_keywords)
                if pattern.keywords:
                    if isinstance(pattern.keywords, list):
                        for kw in pattern.keywords:
                            final_keywords_set.add(kw.strip())
                    else:
                        for kw in pattern.keywords.split(','):
                            final_keywords_set.add(kw.strip())
                
                faq = FAQEntry(
                    id=str(uuid.uuid4()),
                    question=question,
                    answer=answer,
                    keywords=list(final_keywords_set),
                    category=meta_rules["category"],
                    audience=meta_rules["audience"],
                    intent=meta_rules["intent"],
                    condition=meta_rules["condition"],
                    confidence_score=pattern.confidence,
                    source_document=source_document,
                    created_at=current_time,
                    updated_at=current_time,
                    embedding=None
                )
                
                faqs.append(faq)
                
            except Exception as e:
                logger.error(f"Error converting pattern to FAQ: {str(e)}")
                continue
                
        return faqs
    
    def _validate_and_clean_faqs(self, faqs: List[FAQEntry]) -> List[FAQEntry]:
        """
        Validate and clean FAQ entries, removing invalid ones.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            List of valid FAQ entries
        """
        valid_faqs = []
        
        for faq in faqs:
            validation_result = self.validator.validate_faq_entry(faq)
            
            if validation_result.is_valid:
                valid_faqs.append(faq)
            else:
                logger.warning(f"Removing invalid FAQ {faq.id}: {validation_result.errors}")
        
        # Check for duplicates and handle them
        if len(valid_faqs) > 1:
            duplicates = self.validator.detect_duplicates(valid_faqs)
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} potential duplicate pairs")
                
                # For now, just log duplicates. In a production system,
                # you might want to merge or remove duplicates based on business rules
                for dup in duplicates:
                    if dup.match_type == 'exact':
                        logger.warning(f"Exact duplicate found: {dup.faq1_id} <-> {dup.faq2_id}")
                    else:
                        logger.info(f"Similar FAQ found: {dup.faq1_id} <-> {dup.faq2_id} "
                                  f"(similarity: {dup.similarity_score:.2f})")
        
        logger.info(f"Validated {len(valid_faqs)} out of {len(faqs)} FAQ entries")
        return valid_faqs