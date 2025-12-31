"""
FAQ Pattern Recognition

This module provides intelligent pattern recognition for identifying FAQ structures
in DOCX documents, including table-based, list-based, and paragraph-based patterns.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FAQPattern:
    """Represents a detected FAQ pattern in the document."""
    pattern_type: str  # 'table', 'list', 'paragraph'
    question: str
    answer: str
    confidence: float  # 0.0 to 1.0
    source_location: str  # Description of where it was found
    keywords: List[str]
    metadata: Dict[str, Any]


class FAQPatternRecognizer:
    """Intelligent FAQ pattern recognition for various document structures."""
    
    def __init__(self):
        """Initialize the pattern recognizer with common patterns."""
        self.question_indicators = [
            r'\?$',  # Ends with question mark
            r'^(what|how|when|where|why|who|which|can|could|would|should|is|are|do|does|did)',
            r'^q\d*[:\.]',  # Q1:, Q:, etc.
            r'^question\s*\d*[:\.]',
            r'^faq\s*\d*[:\.]',
            r'^query\s*\d*[:\.]',
            r'^\d+[️⃣⃣]*[:\.\-\s]+', # Emoji numbers like 1️⃣
        ]
        
        self.answer_indicators = [
            r'^a\d*[:\.]',  # A1:, A:, etc.
            r'^answer\s*\d*[:\.]',
            r'^response\s*\d*[:\.]',
            r'^solution\s*\d*[:\.]',
        ]
        
        # Common FAQ section headers
        self.faq_section_headers = [
            r'frequently\s+asked\s+questions',
            r'faq',
            r'questions?\s+and\s+answers?',
            r'q\s*&\s*a',
            r'help\s+center',
            r'support\s+questions?',
            r'common\s+questions?',
        ]
        
        # Minimum confidence threshold for FAQ detection
        self.min_confidence = 0.3
    
    def is_question_like(self, text: str) -> float:
        """
        Determine if text looks like a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        text_lower = text.lower().strip()
        confidence = 0.0
        
        # Check for question indicators
        for pattern in self.question_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                confidence += 0.3
        
        # Boost confidence for question marks
        if text.strip().endswith('?'):
            confidence += 0.4
        
        # Boost for interrogative words at the beginning
        interrogatives = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'could', 'would', 'should']
        first_word = text_lower.split()[0] if text_lower.split() else ''
        if first_word in interrogatives:
            confidence += 0.3
        
        # Penalize very long texts (likely not questions)
        if len(text) > 200:
            confidence *= 0.7
        
        # Penalize very short texts
        if len(text) < 10:
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def is_answer_like(self, text: str) -> float:
        """
        Determine if text looks like an answer.
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        text_lower = text.lower().strip()
        confidence = 0.0
        
        # Check for answer indicators
        for pattern in self.answer_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                confidence += 0.4
        
        # Boost for declarative sentences
        if not text.strip().endswith('?'):
            confidence += 0.2
        
        # Boost for longer texts (answers tend to be longer)
        if len(text) > 20:
            confidence += 0.2
        
        if len(text) > 50:
            confidence += 0.1
        
        # Boost for sentences that start with common answer patterns
        answer_starters = ['yes', 'no', 'you can', 'you should', 'to', 'the', 'this', 'that', 'it']
        first_words = ' '.join(text_lower.split()[:2])
        for starter in answer_starters:
            if first_words.startswith(starter):
                confidence += 0.1
                break
        
        return min(confidence, 1.0)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for categorization.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - can be enhanced with NLP libraries
        text_lower = text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        keywords = [word for word in words if word not in stop_words]
        
        # Return unique keywords, limited to top 10
        return list(dict.fromkeys(keywords))[:10]
    
    def recognize_table_patterns(self, tables_data: List[Dict[str, Any]]) -> List[FAQPattern]:
        """
        Recognize FAQ patterns in table structures.
        
        Args:
            tables_data: List of table data from document
            
        Returns:
            List of detected FAQ patterns
        """
        patterns = []
        
        for table_idx, table in enumerate(tables_data):
            if table["row_count"] < 2:  # Need at least header + 1 data row
                continue
            
            # Check if this looks like a FAQ table
            header_row = table["rows"][0] if table["rows"] else None
            if not header_row:
                continue
            
            header_cells = [cell["text"].lower() for cell in header_row["cells"]]
            
            # Look for question/answer column patterns
            question_col = None
            answer_col = None
            
            for col_idx, header in enumerate(header_cells):
                if any(word in header for word in ['question', 'q', 'query', 'ask']):
                    question_col = col_idx
                elif any(word in header for word in ['answer', 'a', 'response', 'solution']):
                    answer_col = col_idx
            
            # If we found both columns, extract FAQ pairs
            if question_col is not None and answer_col is not None:
                for row_idx, row in enumerate(table["rows"][1:], 1):  # Skip header
                    if len(row["cells"]) > max(question_col, answer_col):
                        question_text = row["cells"][question_col]["text"].strip()
                        answer_text = row["cells"][answer_col]["text"].strip()
                        
                        if question_text and answer_text:
                            question_confidence = self.is_question_like(question_text)
                            answer_confidence = self.is_answer_like(answer_text)
                            overall_confidence = (question_confidence + answer_confidence) / 2
                            
                            if overall_confidence >= self.min_confidence:
                                keywords = self.extract_keywords(question_text + " " + answer_text)
                                
                                pattern = FAQPattern(
                                    pattern_type="table",
                                    question=question_text,
                                    answer=answer_text,
                                    confidence=overall_confidence,
                                    source_location=f"Table {table_idx + 1}, Row {row_idx + 1}",
                                    keywords=keywords,
                                    metadata={
                                        "table_index": table_idx,
                                        "row_index": row_idx,
                                        "question_col": question_col,
                                        "answer_col": answer_col
                                    }
                                )
                                patterns.append(pattern)
            
            # Alternative: Try to detect Q&A patterns without explicit headers
            elif table["col_count"] == 2:
                for row_idx, row in enumerate(table["rows"]):
                    if len(row["cells"]) >= 2:
                        cell1_text = row["cells"][0]["text"].strip()
                        cell2_text = row["cells"][1]["text"].strip()
                        
                        if cell1_text and cell2_text:
                            # Try both orientations
                            q1_conf = self.is_question_like(cell1_text)
                            a1_conf = self.is_answer_like(cell2_text)
                            
                            q2_conf = self.is_question_like(cell2_text)
                            a2_conf = self.is_answer_like(cell1_text)
                            
                            if q1_conf + a1_conf > q2_conf + a2_conf and (q1_conf + a1_conf) / 2 >= self.min_confidence:
                                keywords = self.extract_keywords(cell1_text + " " + cell2_text)
                                
                                pattern = FAQPattern(
                                    pattern_type="table",
                                    question=cell1_text,
                                    answer=cell2_text,
                                    confidence=(q1_conf + a1_conf) / 2,
                                    source_location=f"Table {table_idx + 1}, Row {row_idx + 1}",
                                    keywords=keywords,
                                    metadata={
                                        "table_index": table_idx,
                                        "row_index": row_idx,
                                        "inferred_structure": True
                                    }
                                )
                                patterns.append(pattern)
        
        logger.info(f"Recognized {len(patterns)} FAQ patterns from tables")
        return patterns
    
    def recognize_list_patterns(self, lists_data: List[Dict[str, Any]]) -> List[FAQPattern]:
        """
        Recognize FAQ patterns in list structures.
        
        Args:
            lists_data: List of list items from document
            
        Returns:
            List of detected FAQ patterns
        """
        patterns = []
        
        # Group consecutive list items that might form Q&A pairs
        i = 0
        while i < len(lists_data) - 1:
            current_item = lists_data[i]
            next_item = lists_data[i + 1] if i + 1 < len(lists_data) else None
            
            current_text = current_item["text"].strip()
            next_text = next_item["text"].strip() if next_item else ""
            
            if current_text and next_text:
                # Check if current looks like question and next looks like answer
                q_conf = self.is_question_like(current_text)
                a_conf = self.is_answer_like(next_text)
                
                overall_confidence = (q_conf + a_conf) / 2
                
                if overall_confidence >= self.min_confidence:
                    keywords = self.extract_keywords(current_text + " " + next_text)
                    
                    pattern = FAQPattern(
                        pattern_type="list",
                        question=current_text,
                        answer=next_text,
                        confidence=overall_confidence,
                        source_location=f"List items {i + 1}-{i + 2}",
                        keywords=keywords,
                        metadata={
                            "question_index": i,
                            "answer_index": i + 1,
                            "list_type": current_item["type"]
                        }
                    )
                    patterns.append(pattern)
                    i += 2  # Skip the next item since we used it as an answer
                    continue
            
            i += 1
        
        logger.info(f"Recognized {len(patterns)} FAQ patterns from lists")
        return patterns
    
    def recognize_paragraph_patterns(self, paragraphs: List[str]) -> List[FAQPattern]:
        """
        Recognize FAQ patterns in paragraph structures.
        
        Args:
            paragraphs: List of paragraph texts
            
        Returns:
            List of detected FAQ patterns
        """
        patterns = []
        
        # Look for consecutive paragraphs that form Q&A pairs
        i = 0
        while i < len(paragraphs):
            current_para = paragraphs[i].strip()
            
            # 1. Try consecutive paragraphs if there's a next one
            if i < len(paragraphs) - 1:
                next_para = paragraphs[i + 1].strip()
                
                if current_para and next_para:
                    # Check if current looks like question and next looks like answer
                    q_conf = self.is_question_like(current_para)
                    a_conf = self.is_answer_like(next_para)
                    
                    overall_confidence = (q_conf + a_conf) / 2
                    
                    if overall_confidence >= self.min_confidence:
                        keywords = self.extract_keywords(current_para + " " + next_para)
                        
                        pattern = FAQPattern(
                            pattern_type="paragraph",
                            question=current_para,
                            answer=next_para,
                            confidence=overall_confidence,
                            source_location=f"Paragraphs {i + 1}-{i + 2}",
                            keywords=keywords,
                            metadata={
                                "question_paragraph": i,
                                "answer_paragraph": i + 1
                            }
                        )
                        patterns.append(pattern)
                        i += 2  # Skip the next paragraph since we used it as an answer
                        continue
            
            # 2. Also look for single paragraphs that contain both Q&A
            if current_para:
                # Look for patterns like "Q: ... A: ..." within a single paragraph
                qa_match = re.search(r'(.*?\?)\s*(.*)', current_para, re.DOTALL)
                if qa_match:
                    potential_question = qa_match.group(1).strip()
                    potential_answer = qa_match.group(2).strip()
                    
                    if potential_question and potential_answer:
                        q_conf = self.is_question_like(potential_question)
                        a_conf = self.is_answer_like(potential_answer)
                        overall_confidence = (q_conf + a_conf) / 2
                        
                        if overall_confidence >= self.min_confidence:
                            keywords = self.extract_keywords(current_para)
                            
                            pattern = FAQPattern(
                                pattern_type="paragraph",
                                question=potential_question,
                                answer=potential_answer,
                                confidence=overall_confidence * 0.8,  # Slightly lower confidence for single paragraph
                                source_location=f"Paragraph {i + 1} (inline Q&A)",
                                keywords=keywords,
                                metadata={
                                    "paragraph_index": i,
                                    "inline_qa": True
                                }
                            )
                            patterns.append(pattern)
            
            i += 1
        
        logger.info(f"Recognized {len(patterns)} FAQ patterns from paragraphs")
        return patterns
    
    def identify_faq_patterns(self, document_structure) -> List[FAQPattern]:
        """
        Identify all FAQ patterns in the document structure.
        
        Args:
            document_structure: DocumentStructure object
            
        Returns:
            List of all detected FAQ patterns
        """
        all_patterns = []
        
        try:
            # Recognize patterns from tables
            if document_structure.tables:
                table_patterns = self.recognize_table_patterns(document_structure.tables)
                all_patterns.extend(table_patterns)
            
            # Recognize patterns from lists
            if document_structure.lists:
                list_patterns = self.recognize_list_patterns(document_structure.lists)
                all_patterns.extend(list_patterns)
            
            # Recognize patterns from paragraphs
            if document_structure.paragraphs:
                paragraph_patterns = self.recognize_paragraph_patterns(document_structure.paragraphs)
                all_patterns.extend(paragraph_patterns)
            
            # Sort patterns by confidence (highest first)
            all_patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            logger.info(f"Total FAQ patterns identified: {len(all_patterns)}")
            return all_patterns
            
        except Exception as e:
            logger.error(f"Error identifying FAQ patterns: {str(e)}")
            return []