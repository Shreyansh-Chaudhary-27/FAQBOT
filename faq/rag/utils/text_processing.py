"""
Text Processing Utilities for RAG System

Common text processing functions used across RAG components.
"""

import re
import string
from typing import List, Set, Dict, Any
from collections import Counter


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)]', '', text)
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    if not text:
        return []
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = text.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Filter words
    filtered_words = [
        word for word in words 
        if len(word) > 2 and word not in stop_words
    ]
    
    # Count frequency and return top keywords
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(max_keywords)]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap."""
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def detect_question_patterns(text: str) -> bool:
    """Detect if text contains question patterns."""
    if not text:
        return False
    
    # Check for question words
    question_words = {
        'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose',
        'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
        'is', 'are', 'was', 'were'
    }
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check if starts with question word
    if words and words[0] in question_words:
        return True
    
    # Check for question mark
    if '?' in text:
        return True
    
    return False


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    
    # Simple sentence splitting on periods, exclamation marks, and question marks
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    return text.strip()


def extract_text_features(text: str) -> Dict[str, Any]:
    """Extract various features from text for analysis."""
    if not text:
        return {
            'length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'has_question_mark': False,
            'is_question': False,
            'keywords': []
        }
    
    words = text.split()
    sentences = split_into_sentences(text)
    
    return {
        'length': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0.0,
        'has_question_mark': '?' in text,
        'is_question': detect_question_patterns(text),
        'keywords': extract_keywords(text)
    }