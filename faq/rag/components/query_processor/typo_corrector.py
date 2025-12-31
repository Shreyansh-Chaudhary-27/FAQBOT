"""
Typo Correction and Spell Checking Module

This module provides typo correction and spell checking functionality for user queries.
It uses multiple approaches including dictionary-based correction, phonetic matching,
and common typo patterns to improve query understanding.
"""

import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict
import logging

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class TypoCorrector:
    """
    Handles typo correction and spell checking for user queries.
    
    Uses multiple correction strategies:
    1. Dictionary-based correction using edit distance
    2. Common typo pattern matching
    3. Phonetic similarity matching
    4. Context-aware corrections
    """
    
    def __init__(self):
        self.dictionary = self._load_dictionary()
        self.common_typos = self._load_common_typos()
        self.confidence_threshold = 0.7
        
        # Initialize external spell checker if available
        if SPELLCHECKER_AVAILABLE:
            self.spell_checker = SpellChecker()
            # Add custom words to the spell checker
            self.spell_checker.word_frequency.load_words(['faq', 'faqs', 'login', 'logout', 'signup', 
                                                         'username', 'password', 'email', 'website',
                                                         'app', 'application', 'software', 'system',
                                                         'database', 'server', 'api', 'url', 'http',
                                                         'https', 'wifi', 'internet', 'online', 'offline'])
        else:
            self.spell_checker = None
            logger.warning("PySpellChecker not available, using basic dictionary only")
        
    def _load_dictionary(self) -> set:
        """Load a basic English dictionary for spell checking."""
        # Basic dictionary with common words
        basic_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'she', 'or', 'an',
            'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
            'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
            'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
            'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were',
            'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about',
            'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into',
            'him', 'time', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over',
            'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being',
            'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'go', 'well', 'were',
            'been', 'have', 'had', 'has', 'say', 'says', 'said', 'come', 'came', 'going', 'want', 'did',
            'got', 'may', 'find', 'give', 'man', 'day', 'too', 'any', 'my', 'say', 'does', 'set', 'each',
            'why', 'ask', 'men', 'read', 'need', 'land', 'different', 'home', 'us', 'move', 'try', 'kind',
            'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air', 'away', 'animal', 'house',
            'point', 'page', 'letter', 'mother', 'answer', 'found', 'study', 'still', 'learn', 'should',
            'america', 'world', 'high', 'every', 'near', 'add', 'food', 'between', 'own', 'below', 'country',
            'plant', 'last', 'school', 'father', 'keep', 'tree', 'never', 'start', 'city', 'earth', 'eye',
            'light', 'thought', 'head', 'under', 'story', 'saw', 'left', 'dont', 'few', 'while', 'along',
            'might', 'close', 'something', 'seem', 'next', 'hard', 'open', 'example', 'begin', 'life',
            'always', 'those', 'both', 'paper', 'together', 'got', 'group', 'often', 'run', 'important',
            'until', 'children', 'side', 'feet', 'car', 'mile', 'night', 'walk', 'white', 'sea', 'began',
            'grow', 'took', 'river', 'four', 'carry', 'state', 'once', 'book', 'hear', 'stop', 'without',
            'second', 'later', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far', 'indian', 'really',
            'almost', 'let', 'above', 'girl', 'sometimes', 'mountain', 'cut', 'young', 'talk', 'soon',
            'list', 'song', 'being', 'leave', 'family', 'its', 'question', 'answer', 'help', 'problem',
            'issue', 'support', 'service', 'customer', 'account', 'password', 'login', 'email', 'phone',
            'address', 'order', 'payment', 'shipping', 'delivery', 'return', 'refund', 'cancel', 'change',
            'update', 'information', 'contact', 'website', 'online', 'internet', 'computer', 'software',
            'application', 'program', 'system', 'error', 'bug', 'fix', 'repair', 'install', 'download',
            'upload', 'file', 'document', 'report', 'data', 'database', 'search', 'find', 'locate'
        }
        return basic_words
    
    def _load_common_typos(self) -> Dict[str, str]:
        """Load common typo patterns and their corrections."""
        return {
            # Common letter swaps
            'teh': 'the',
            'adn': 'and',
            'nad': 'and',
            'hte': 'the',
            'taht': 'that',
            'thta': 'that',
            'waht': 'what',
            'hwat': 'what',
            'wnat': 'want',
            'whne': 'when',
            'wehn': 'when',
            'whre': 'where',
            'wher': 'where',
            'whihc': 'which',
            'whcih': 'which',
            'wich': 'which',
            'recieve': 'receive',
            'recive': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred',
            'occurance': 'occurrence',
            'accomodate': 'accommodate',
            'begining': 'beginning',
            'beleive': 'believe',
            'calender': 'calendar',
            'cemetary': 'cemetery',
            'changable': 'changeable',
            'collegue': 'colleague',
            'comming': 'coming',
            'commited': 'committed',
            'concious': 'conscious',
            'definite': 'definite',
            'dilemna': 'dilemma',
            'embarass': 'embarrass',
            'enviroment': 'environment',
            'existance': 'existence',
            'experiance': 'experience',
            'familar': 'familiar',
            'finaly': 'finally',
            'foriegn': 'foreign',
            'freind': 'friend',
            'goverment': 'government',
            'grammer': 'grammar',
            'harrass': 'harass',
            'independant': 'independent',
            'intrest': 'interest',
            'knowlege': 'knowledge',
            'lenght': 'length',
            'liason': 'liaison',
            'libary': 'library',
            'maintainance': 'maintenance',
            'neccessary': 'necessary',
            'occassion': 'occasion',
            'perseverence': 'perseverance',
            'posession': 'possession',
            'prefered': 'preferred',
            'priviledge': 'privilege',
            'publically': 'publicly',
            'reccomend': 'recommend',
            'refered': 'referred',
            'relevent': 'relevant',
            'resistence': 'resistance',
            'seperation': 'separation',
            'succesful': 'successful',
            'supress': 'suppress',
            'tommorow': 'tomorrow',
            'truely': 'truly',
            'untill': 'until',
            'usefull': 'useful',
            'wierd': 'weird',
            # Common contractions and informal spellings (high priority)
            'dont': "don't",
            'doesnt': "doesn't", 
            'cant': "can't",
            'wont': "won't",
            'isnt': "isn't",
            'arent': "aren't",
            'wasnt': "wasn't",
            'werent': "weren't",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hadnt': "hadn't",
            'shouldnt': "shouldn't",
            'wouldnt': "wouldn't",
            'couldnt': "couldn't",
            'mustnt': "mustn't",
            'neednt': "needn't",
            'darent': "daren't",
            'u': 'you',
            'ur': 'your',
            'youre': "you're",
            'theres': "there's",
            'theyre': "they're",
            'its': "it's",
            'hes': "he's",
            'shes': "she's",
            'weve': "we've",
            'youve': "you've",
            'theyve': "they've",
            'ive': "I've",
            'im': "I'm",
            'id': "I'd",
            'ill': "I'll",
            'wed': "we'd",
            'youd': "you'd",
            'theyd': "they'd",
            'shed': "she'd",
            'hed': "he'd"
        }
    
    def correct_typos(self, query: str) -> Tuple[str, float]:
        """
        Correct typos in the given query.
        
        Args:
            query: The input query string
            
        Returns:
            Tuple of (corrected_query, confidence_score)
        """
        if not query or not query.strip():
            return query, 1.0
        
        original_query = query.strip()
        words = self._tokenize(original_query)
        corrected_words = []
        total_corrections = 0
        
        for word in words:
            corrected_word, was_corrected = self._correct_word(word)
            corrected_words.append(corrected_word)
            if was_corrected:
                total_corrections += 1
        
        corrected_query = self._reconstruct_query(corrected_words, original_query)
        
        # Calculate confidence based on correction ratio
        if len(words) == 0:
            confidence = 1.0
        else:
            correction_ratio = total_corrections / len(words)
            # Higher confidence when fewer corrections needed
            confidence = max(0.1, 1.0 - (correction_ratio * 0.5))
        
        logger.info(f"Typo correction: '{original_query}' -> '{corrected_query}' (confidence: {confidence:.2f})")
        
        return corrected_query, confidence
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, preserving punctuation context."""
        # Simple tokenization that preserves word boundaries
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _correct_word(self, word: str) -> Tuple[str, bool]:
        """
        Correct a single word using various strategies.
        
        Args:
            word: The word to correct
            
        Returns:
            Tuple of (corrected_word, was_corrected)
        """
        if not word:
            return word, False
        
        word_lower = word.lower()
        
        # Strategy 1: Check common typos dictionary FIRST (prioritize contractions)
        if word_lower in self.common_typos:
            return self.common_typos[word_lower], True
        
        # Strategy 2: Check if word is already correct
        if word_lower in self.dictionary:
            return word, False
        
        # Strategy 3: Use external spell checker if available
        if self.spell_checker and word_lower not in self.spell_checker:
            # Get the most likely correction
            candidates = self.spell_checker.candidates(word_lower)
            if candidates:
                # Get the most likely candidate (first one is usually best)
                best_candidate = next(iter(candidates))
                if self._calculate_similarity(word_lower, best_candidate) > 0.6:
                    return best_candidate, True
        
        # Strategy 4: Find closest dictionary match using edit distance
        best_match = self._find_closest_match(word_lower)
        if best_match and self._calculate_similarity(word_lower, best_match) > self.confidence_threshold:
            return best_match, True
        
        # Strategy 5: Handle common patterns (doubled letters, missing letters, etc.)
        pattern_correction = self._apply_pattern_corrections(word_lower)
        if pattern_correction != word_lower and pattern_correction in self.dictionary:
            return pattern_correction, True
        
        # If no correction found, return original word
        return word, False
    
    def _find_closest_match(self, word: str) -> Optional[str]:
        """Find the closest matching word in the dictionary using edit distance."""
        if len(word) < 2:
            return None
        
        best_match = None
        best_similarity = 0
        
        # Only check words with similar length to avoid false positives
        length_range = range(max(1, len(word) - 2), len(word) + 3)
        
        for dict_word in self.dictionary:
            if len(dict_word) in length_range:
                similarity = self._calculate_similarity(word, dict_word)
                if similarity > best_similarity and similarity > self.confidence_threshold:
                    best_similarity = similarity
                    best_match = dict_word
        
        return best_match
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using sequence matching."""
        return SequenceMatcher(None, word1, word2).ratio()
    
    def _apply_pattern_corrections(self, word: str) -> str:
        """Apply common typo pattern corrections."""
        corrected = word
        
        # Remove doubled letters (except for common double letters)
        common_doubles = {'ll', 'ss', 'tt', 'ff', 'mm', 'nn', 'pp', 'rr', 'cc', 'dd', 'gg', 'bb'}
        
        # Check for tripled letters and reduce to double
        for i in range(len(corrected) - 2):
            if corrected[i] == corrected[i+1] == corrected[i+2]:
                corrected = corrected[:i+2] + corrected[i+3:]
                break
        
        # Try removing one doubled letter if it's not a common double
        for i in range(len(corrected) - 1):
            if corrected[i] == corrected[i+1]:
                double = corrected[i:i+2]
                if double not in common_doubles:
                    # Try removing one of the doubled letters
                    test_word = corrected[:i] + corrected[i+1:]
                    if test_word in self.dictionary:
                        return test_word
        
        return corrected
    
    def _reconstruct_query(self, corrected_words: List[str], original_query: str) -> str:
        """Reconstruct the query maintaining original capitalization and punctuation."""
        if not corrected_words:
            return original_query
        
        # Simple reconstruction - join words with spaces
        # In a more sophisticated version, we would preserve original spacing and punctuation
        result = ' '.join(corrected_words)
        
        # Preserve original capitalization for first word if it was capitalized
        if original_query and original_query[0].isupper() and result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def get_correction_confidence(self, original: str, corrected: str) -> float:
        """
        Calculate confidence score for a correction.
        
        Args:
            original: Original query
            corrected: Corrected query
            
        Returns:
            Confidence score between 0 and 1
        """
        if original == corrected:
            return 1.0
        
        # Calculate based on similarity and number of changes
        similarity = self._calculate_similarity(original.lower(), corrected.lower())
        
        # Penalize based on the number of word changes
        original_words = self._tokenize(original)
        corrected_words = self._tokenize(corrected)
        
        if len(original_words) != len(corrected_words):
            # Different number of words - lower confidence
            return max(0.1, similarity * 0.7)
        
        word_changes = sum(1 for orig, corr in zip(original_words, corrected_words) if orig != corr)
        change_ratio = word_changes / max(1, len(original_words))
        
        # Higher confidence when fewer words changed
        confidence = similarity * (1.0 - change_ratio * 0.3)
        
        return max(0.1, min(1.0, confidence))