"""
FAQ Validation and Duplicate Detection

This module provides validation for extracted FAQ entries and duplicate detection
using text similarity algorithms.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib

from faq.rag.interfaces.base import FAQEntry, ValidationResult
from .pattern_recognizer import FAQPattern

logger = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    """Represents a potential duplicate FAQ match."""
    faq1_id: str
    faq2_id: str
    similarity_score: float
    match_type: str  # 'exact', 'near_exact', 'similar'
    matched_components: List[str]  # 'question', 'answer', 'both'
    confidence: float


class FAQValidator:
    """Validates FAQ entries and detects duplicates."""
    
    def __init__(self):
        """Initialize the validator with configuration."""
        self.min_question_length = 5
        self.min_answer_length = 10
        self.max_question_length = 500
        self.max_answer_length = 2000
        
        # Similarity thresholds for duplicate detection
        self.exact_match_threshold = 0.95
        self.near_exact_threshold = 0.85
        self.similar_threshold = 0.70
        
        # Common validation patterns
        self.question_patterns = [
            r'\?$',  # Should end with question mark
            r'^(what|how|when|where|why|who|which|can|could|would|should|is|are|do|does|did)',
        ]
        
        # Words that indicate incomplete or placeholder content
        self.placeholder_words = {
            'todo', 'tbd', 'placeholder', 'example', 'sample', 'test', 'dummy',
            'lorem', 'ipsum', 'xxx', 'yyy', 'zzz', '[placeholder]', '[todo]'
        }
    
    def validate_faq_entry(self, faq: FAQEntry) -> ValidationResult:
        """
        Validate a single FAQ entry for completeness and quality.
        
        Args:
            faq: FAQ entry to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        try:
            # Validate required fields
            if not faq.id or not faq.id.strip():
                errors.append("FAQ ID is required")
            
            if not faq.question or not faq.question.strip():
                errors.append("Question is required")
            else:
                # Validate question content
                question = faq.question.strip()
                
                if len(question) < self.min_question_length:
                    errors.append(f"Question too short (minimum {self.min_question_length} characters)")
                
                if len(question) > self.max_question_length:
                    warnings.append(f"Question very long ({len(question)} characters)")
                
                # Check if question looks like a question
                if not any(re.search(pattern, question.lower()) for pattern in self.question_patterns):
                    warnings.append("Question doesn't appear to be a proper question")
                
                # Check for placeholder content
                if any(word in question.lower() for word in self.placeholder_words):
                    warnings.append("Question appears to contain placeholder content")
            
            if not faq.answer or not faq.answer.strip():
                errors.append("Answer is required")
            else:
                # Validate answer content
                answer = faq.answer.strip()
                
                if len(answer) < self.min_answer_length:
                    errors.append(f"Answer too short (minimum {self.min_answer_length} characters)")
                
                if len(answer) > self.max_answer_length:
                    warnings.append(f"Answer very long ({len(answer)} characters)")
                
                # Check for placeholder content
                if any(word in answer.lower() for word in self.placeholder_words):
                    warnings.append("Answer appears to contain placeholder content")
            
            # Validate other fields
            if not faq.category or not faq.category.strip():
                warnings.append("Category not specified")
            
            if not faq.keywords:
                warnings.append("No keywords specified")
            elif len(faq.keywords) > 20:
                warnings.append(f"Too many keywords ({len(faq.keywords)})")
            
            if faq.confidence_score < 0.0 or faq.confidence_score > 1.0:
                errors.append("Confidence score must be between 0.0 and 1.0")
            
            if not faq.source_document or not faq.source_document.strip():
                warnings.append("Source document not specified")
            
            # Validate timestamps
            if faq.created_at > datetime.now():
                warnings.append("Created timestamp is in the future")
            
            if faq.updated_at > datetime.now():
                warnings.append("Updated timestamp is in the future")
            
            if faq.updated_at < faq.created_at:
                errors.append("Updated timestamp is before created timestamp")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={
                    "faq_id": faq.id,
                    "question_length": len(faq.question) if faq.question else 0,
                    "answer_length": len(faq.answer) if faq.answer else 0,
                    "keyword_count": len(faq.keywords) if faq.keywords else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating FAQ entry {faq.id}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                metadata={"faq_id": faq.id if hasattr(faq, 'id') else 'unknown'}
            )
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_norm = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Exact match
        if text1_norm == text2_norm:
            return 1.0
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        return similarity
    
    def generate_content_hash(self, text: str) -> str:
        """
        Generate a hash for text content to help with duplicate detection.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        # Normalize text before hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def detect_duplicates(self, faqs: List[FAQEntry]) -> List[DuplicateMatch]:
        """
        Detect duplicate FAQ entries using text similarity.
        
        Args:
            faqs: List of FAQ entries to check
            
        Returns:
            List of potential duplicate matches
        """
        duplicates = []
        
        try:
            # Create hash maps for exact matches
            question_hashes = {}
            answer_hashes = {}
            
            for faq in faqs:
                if faq.question:
                    q_hash = self.generate_content_hash(faq.question)
                    if q_hash in question_hashes:
                        question_hashes[q_hash].append(faq)
                    else:
                        question_hashes[q_hash] = [faq]
                
                if faq.answer:
                    a_hash = self.generate_content_hash(faq.answer)
                    if a_hash in answer_hashes:
                        answer_hashes[a_hash].append(faq)
                    else:
                        answer_hashes[a_hash] = [faq]
            
            # Find exact duplicates by hash
            for q_hash, faq_list in question_hashes.items():
                if len(faq_list) > 1:
                    for i in range(len(faq_list)):
                        for j in range(i + 1, len(faq_list)):
                            duplicates.append(DuplicateMatch(
                                faq1_id=faq_list[i].id,
                                faq2_id=faq_list[j].id,
                                similarity_score=1.0,
                                match_type='exact',
                                matched_components=['question'],
                                confidence=1.0
                            ))
            
            for a_hash, faq_list in answer_hashes.items():
                if len(faq_list) > 1:
                    for i in range(len(faq_list)):
                        for j in range(i + 1, len(faq_list)):
                            # Check if this pair is already marked as duplicate
                            existing = any(
                                (d.faq1_id == faq_list[i].id and d.faq2_id == faq_list[j].id) or
                                (d.faq1_id == faq_list[j].id and d.faq2_id == faq_list[i].id)
                                for d in duplicates
                            )
                            
                            if existing:
                                # Update existing match to include answer
                                for d in duplicates:
                                    if ((d.faq1_id == faq_list[i].id and d.faq2_id == faq_list[j].id) or
                                        (d.faq1_id == faq_list[j].id and d.faq2_id == faq_list[i].id)):
                                        if 'answer' not in d.matched_components:
                                            d.matched_components.append('answer')
                                        d.match_type = 'exact'
                                        break
                            else:
                                duplicates.append(DuplicateMatch(
                                    faq1_id=faq_list[i].id,
                                    faq2_id=faq_list[j].id,
                                    similarity_score=1.0,
                                    match_type='exact',
                                    matched_components=['answer'],
                                    confidence=1.0
                                ))
            
            # Find near-exact and similar duplicates
            processed_pairs = set()
            
            for i in range(len(faqs)):
                for j in range(i + 1, len(faqs)):
                    faq1, faq2 = faqs[i], faqs[j]
                    pair_key = tuple(sorted([faq1.id, faq2.id]))
                    
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    
                    # Skip if already found as exact duplicate
                    if any((d.faq1_id == faq1.id and d.faq2_id == faq2.id) or
                           (d.faq1_id == faq2.id and d.faq2_id == faq1.id) for d in duplicates):
                        continue
                    
                    # Calculate similarities
                    q_similarity = self.calculate_text_similarity(faq1.question, faq2.question)
                    a_similarity = self.calculate_text_similarity(faq1.answer, faq2.answer)
                    
                    # Determine if this is a duplicate
                    matched_components = []
                    max_similarity = 0.0
                    
                    if q_similarity >= self.similar_threshold:
                        matched_components.append('question')
                        max_similarity = max(max_similarity, q_similarity)
                    
                    if a_similarity >= self.similar_threshold:
                        matched_components.append('answer')
                        max_similarity = max(max_similarity, a_similarity)
                    
                    if matched_components:
                        # Determine match type
                        if max_similarity >= self.exact_match_threshold:
                            match_type = 'exact'
                        elif max_similarity >= self.near_exact_threshold:
                            match_type = 'near_exact'
                        else:
                            match_type = 'similar'
                        
                        # Calculate overall confidence
                        confidence = (q_similarity + a_similarity) / 2
                        
                        duplicates.append(DuplicateMatch(
                            faq1_id=faq1.id,
                            faq2_id=faq2.id,
                            similarity_score=max_similarity,
                            match_type=match_type,
                            matched_components=matched_components,
                            confidence=confidence
                        ))
            
            # Sort by similarity score (highest first)
            duplicates.sort(key=lambda d: d.similarity_score, reverse=True)
            
            logger.info(f"Found {len(duplicates)} potential duplicate pairs")
            return duplicates
            
        except Exception as e:
            logger.error(f"Error detecting duplicates: {str(e)}")
            return []
    
    def categorize_faqs(self, faqs: List[FAQEntry]) -> Dict[str, List[FAQEntry]]:
        """
        Automatically categorize FAQ entries based on keywords and content.
        
        Args:
            faqs: List of FAQ entries to categorize
            
        Returns:
            Dictionary mapping categories to FAQ lists
        """
        categories = {}
        
        try:
            # Define category keywords
            category_keywords = {
                'installation': ['install', 'setup', 'download', 'configure', 'installation'],
                'usage': ['how', 'use', 'using', 'usage', 'tutorial', 'guide'],
                'troubleshooting': ['error', 'problem', 'issue', 'fix', 'troubleshoot', 'debug'],
                'configuration': ['config', 'setting', 'configure', 'configuration', 'customize'],
                'features': ['feature', 'capability', 'function', 'functionality', 'what'],
                'api': ['api', 'endpoint', 'request', 'response', 'integration'],
                'security': ['security', 'authentication', 'authorization', 'permission', 'access'],
                'performance': ['performance', 'speed', 'optimization', 'slow', 'fast'],
                'general': []  # Default category
            }
            
            for faq in faqs:
                # If FAQ already has a category, use it
                if faq.category and faq.category.strip() and faq.category.lower() != 'general':
                    category = faq.category.lower()
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(faq)
                    continue
                
                # Auto-categorize based on keywords
                text_to_analyze = (faq.question + " " + faq.answer + " " + " ".join(faq.keywords)).lower()
                
                best_category = 'general'
                best_score = 0
                
                for category, keywords in category_keywords.items():
                    if category == 'general':
                        continue
                    
                    score = sum(1 for keyword in keywords if keyword in text_to_analyze)
                    if score > best_score:
                        best_score = score
                        best_category = category
                
                # Update FAQ category
                faq.category = best_category
                
                if best_category not in categories:
                    categories[best_category] = []
                categories[best_category].append(faq)
            
            logger.info(f"Categorized FAQs into {len(categories)} categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error categorizing FAQs: {str(e)}")
            return {'general': faqs}
    
    def validate_extraction(self, faqs: List[FAQEntry]) -> ValidationResult:
        """
        Validate a complete set of extracted FAQ entries.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            ValidationResult with overall validation status
        """
        errors = []
        warnings = []
        
        try:
            if not faqs:
                errors.append("No FAQ entries found")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metadata={"total_faqs": 0}
                )
            
            # Validate individual entries
            valid_faqs = 0
            invalid_faqs = 0
            
            for faq in faqs:
                result = self.validate_faq_entry(faq)
                if result.is_valid:
                    valid_faqs += 1
                else:
                    invalid_faqs += 1
                    errors.extend([f"FAQ {faq.id}: {error}" for error in result.errors])
                
                warnings.extend([f"FAQ {faq.id}: {warning}" for warning in result.warnings])
            
            # Check for duplicates
            duplicates = self.detect_duplicates(faqs)
            if duplicates:
                warnings.append(f"Found {len(duplicates)} potential duplicate pairs")
                for dup in duplicates[:5]:  # Show first 5 duplicates
                    warnings.append(f"Potential duplicate: {dup.faq1_id} <-> {dup.faq2_id} "
                                  f"(similarity: {dup.similarity_score:.2f})")
            
            # Overall statistics
            total_faqs = len(faqs)
            success_rate = valid_faqs / total_faqs if total_faqs > 0 else 0
            
            if success_rate < 0.5:
                errors.append(f"Low success rate: {success_rate:.1%} of FAQs are valid")
            elif success_rate < 0.8:
                warnings.append(f"Moderate success rate: {success_rate:.1%} of FAQs are valid")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={
                    "total_faqs": total_faqs,
                    "valid_faqs": valid_faqs,
                    "invalid_faqs": invalid_faqs,
                    "success_rate": success_rate,
                    "duplicate_pairs": len(duplicates)
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating FAQ extraction: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                metadata={"total_faqs": len(faqs) if faqs else 0}
            )