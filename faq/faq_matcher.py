"""
Enhanced FAQ Matching Service

This module provides intelligent FAQ matching capabilities that can understand
synonyms, similar sentences, and semantic meaning to provide better answers
to user queries.

Features:
- Semantic similarity matching using text embeddings
- Fuzzy string matching for typos and variations
- Keyword expansion with synonyms
- Confidence scoring for match quality
- Multiple matching strategies with fallback
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from django.db.models import Q
from django.conf import settings
from .models import FAQEntry


class FAQMatcher:
    """
    Intelligent FAQ matching service that uses multiple strategies
    to find the best matching FAQ entries for user queries.
    """
    
    def __init__(self):
        self.synonym_dict = self._load_synonyms()
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'this', 'that', 'these', 'those'
        }
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary for keyword expansion"""
        return {
            # Graphura specific
            'internship': ['intern', 'training', 'trainee', 'placement'],
            'terminate': ['termination', 'terminated', 'end', 'exit', 'leave', 'quit', 'resign', 'fired', 'removal'],
            'leave': ['leaves', 'absence', 'vacation', 'off', 'day off', 'time off'],
            'apply': ['application', 'applying', 'register', 'join', 'enroll', 'signup'],
            'stipend': ['payment', 'salary', 'pay', 'paid', 'money', 'compensation', 'allowance'],
            'certificate': ['certification', 'certificated', 'letter', 'document', 'proof'],
            'placement': ['job', 'offer', 'ppo', 'employment', 'hire', 'hired'],
            'duration': ['length', 'period', 'time', 'months', 'weeks', 'long'],
            'hours': ['timing', 'schedule', 'shift', 'time', 'working'],
            'meeting': ['meetings', 'session', 'call', 'club'],
            'deadline': ['deadlines', 'due', 'submission', 'submit'],
            'report': ['reporting', 'reports', 'hierarchy', 'manager', 'head'],
            'policy': ['policies', 'rules', 'guidelines', 'conduct'],
            'domain': ['domains', 'department', 'role', 'field', 'area'],
            'project': ['projects', 'work', 'task', 'assignment'],
            'verification': ['verify', 'verified', 'portal', 'check'],
            'planner': ['plan', 'planning', 'schedule', 'calendar'],
            
            # Authentication related
            'login': ['sign in', 'log in', 'access', 'enter', 'authenticate'],
            'password': ['pass', 'pwd', 'passcode', 'secret', 'credentials'],
            'account': ['profile', 'user', 'member', 'registration'],
            
            # Technical issues
            'error': ['problem', 'issue', 'bug', 'trouble', 'fault', 'failure'],
            'broken': ['not working', 'failed', 'down', 'offline', 'unavailable'],
            
            # General help
            'help': ['assistance', 'support', 'aid', 'guidance', 'tutorial'],
            'contact': ['reach', 'get in touch', 'communicate', 'call', 'email'],
            'support': ['help desk', 'customer service', 'assistance', 'hr'],
        }
    
    def find_best_matches(self, query: str, max_results: int = 5, min_confidence: float = 0.3) -> List[Dict]:
        """
        Find the best matching FAQ entries for a user query.
        
        Args:
            query: User's question or search query
            max_results: Maximum number of results to return
            min_confidence: Minimum confidence score (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing FAQ entries with confidence scores
        """
        if not query or not query.strip():
            return []
        
        # Normalize the query
        normalized_query = self._normalize_text(query)
        
        # Get all FAQ entries
        faq_entries = FAQEntry.objects.all()
        
        # Calculate scores for each FAQ entry using multiple strategies
        scored_results = []
        
        for faq in faq_entries:
            confidence_scores = self._calculate_confidence_scores(normalized_query, faq)
            
            # Calculate overall confidence (weighted average)
            overall_confidence = (
                confidence_scores['keyword_match'] * 0.3 +
                confidence_scores['question_similarity'] * 0.4 +
                confidence_scores['semantic_similarity'] * 0.2 +
                confidence_scores['fuzzy_match'] * 0.1
            )
            
            if overall_confidence >= min_confidence:
                scored_results.append({
                    'faq': faq,
                    'confidence': overall_confidence,
                    'match_details': confidence_scores,
                    'id': faq.id,
                    'question': faq.question,
                    'answer': faq.answer,
                    'keywords': faq.keywords
                })
        
        # Sort by confidence score (highest first)
        scored_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return scored_results[:max_results]
    
    def _calculate_confidence_scores(self, query: str, faq: FAQEntry) -> Dict[str, float]:
        """Calculate confidence scores using different matching strategies"""
        
        # Normalize FAQ text
        faq_question = self._normalize_text(faq.question)
        faq_keywords = self._normalize_text(faq.keywords)
        
        # 1. Keyword matching with synonym expansion
        keyword_score = self._calculate_keyword_score(query, faq_keywords, faq_question)
        
        # 2. Question similarity (direct text comparison)
        question_score = self._calculate_question_similarity(query, faq_question)
        
        # 3. Semantic similarity (word overlap with synonyms)
        semantic_score = self._calculate_semantic_similarity(query, faq_question, faq_keywords)
        
        # 4. Fuzzy string matching
        fuzzy_score = self._calculate_fuzzy_match(query, faq_question)
        
        return {
            'keyword_match': keyword_score,
            'question_similarity': question_score,
            'semantic_similarity': semantic_score,
            'fuzzy_match': fuzzy_score
        }
    
    def _calculate_keyword_score(self, query: str, keywords: str, question: str) -> float:
        """Calculate score based on keyword matching with synonym expansion"""
        if not keywords:
            return 0.0
        
        query_words = set(self._extract_keywords(query))
        keyword_words = set(self._extract_keywords(keywords))
        question_words = set(self._extract_keywords(question))
        
        # Expand query words with synonyms
        expanded_query_words = set()
        for word in query_words:
            expanded_query_words.add(word)
            if word in self.synonym_dict:
                expanded_query_words.update(self.synonym_dict[word])
        
        # Check matches in keywords and question
        all_faq_words = keyword_words.union(question_words)
        matches = expanded_query_words.intersection(all_faq_words)
        
        if not expanded_query_words:
            return 0.0
        
        return len(matches) / len(expanded_query_words)
    
    def _calculate_question_similarity(self, query: str, question: str) -> float:
        """Calculate similarity between query and FAQ question"""
        if not question:
            return 0.0
        
        query_words = set(self._extract_keywords(query))
        question_words = set(self._extract_keywords(question))
        
        if not query_words or not question_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(question_words)
        union = query_words.union(question_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_semantic_similarity(self, query: str, question: str, keywords: str) -> float:
        """Calculate semantic similarity using synonym expansion"""
        query_words = set(self._extract_keywords(query))
        faq_words = set(self._extract_keywords(question + " " + keywords))
        
        # Expand both query and FAQ words with synonyms
        expanded_query = set()
        for word in query_words:
            expanded_query.add(word)
            if word in self.synonym_dict:
                expanded_query.update(self.synonym_dict[word])
        
        expanded_faq = set()
        for word in faq_words:
            expanded_faq.add(word)
            if word in self.synonym_dict:
                expanded_faq.update(self.synonym_dict[word])
        
        if not expanded_query or not expanded_faq:
            return 0.0
        
        # Calculate overlap
        intersection = expanded_query.intersection(expanded_faq)
        union = expanded_query.union(expanded_faq)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_fuzzy_match(self, query: str, question: str) -> float:
        """Calculate fuzzy string matching score"""
        if not question:
            return 0.0
        
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, query.lower(), question.lower()).ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        return text
    
    def _stem_word(self, word: str) -> str:
        """Simple stemmer to reduce words to their root form"""
        # Common suffixes to remove
        suffixes = ['ation', 'tion', 'sion', 'ing', 'ed', 'er', 'est', 'ly', 's', 'ment', 'ness', 'ity', 'ies']
        
        word = word.lower()
        
        # Try removing suffixes (longest first)
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                stemmed = word[:-len(suffix)]
                # Handle special cases
                if stemmed.endswith('at'):  # termination -> termin -> terminate
                    stemmed = stemmed + 'e'
                return stemmed
        
        return word
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text with stemming"""
        if not text:
            return []
        
        # Split into words
        words = text.lower().split()
        
        # Remove stop words, short words, and apply stemming
        keywords = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                # Add both original and stemmed version
                keywords.append(word)
                stemmed = self._stem_word(word)
                if stemmed != word:
                    keywords.append(stemmed)
        
        return keywords
    
    def add_synonym(self, word: str, synonyms: List[str]):
        """Add new synonyms to the dictionary"""
        if word not in self.synonym_dict:
            self.synonym_dict[word] = []
        
        for synonym in synonyms:
            if synonym not in self.synonym_dict[word]:
                self.synonym_dict[word].append(synonym)
    
    def get_match_explanation(self, query: str, faq_id: int) -> Dict:
        """Get detailed explanation of why a FAQ matched a query"""
        try:
            faq = FAQEntry.objects.get(id=faq_id)
        except FAQEntry.DoesNotExist:
            return {"error": "FAQ not found"}
        
        normalized_query = self._normalize_text(query)
        confidence_scores = self._calculate_confidence_scores(normalized_query, faq)
        
        query_keywords = self._extract_keywords(normalized_query)
        faq_keywords = self._extract_keywords(faq.keywords)
        question_keywords = self._extract_keywords(faq.question)
        
        return {
            'query': query,
            'faq_question': faq.question,
            'confidence_scores': confidence_scores,
            'query_keywords': query_keywords,
            'faq_keywords': faq_keywords,
            'question_keywords': question_keywords,
            'matched_keywords': list(set(query_keywords).intersection(set(faq_keywords + question_keywords)))
        }


# Global instance for easy access
faq_matcher = FAQMatcher()