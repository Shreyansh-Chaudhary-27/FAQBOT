"""
Multi-Language Support and Language Detection Module

This module provides language detection capabilities and language-specific
processing rules for user queries. It handles mixed-language queries and
provides appropriate processing based on detected languages.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Enumeration of supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Container for language detection results."""
    primary_language: SupportedLanguage
    confidence: float
    detected_languages: Dict[SupportedLanguage, float]
    is_mixed_language: bool
    language_segments: List[Tuple[str, SupportedLanguage, float]]


@dataclass
class LanguageProcessingRules:
    """Container for language-specific processing rules."""
    language: SupportedLanguage
    stop_words: Set[str]
    common_words: Set[str]
    question_words: Set[str]
    punctuation_rules: Dict[str, str]
    normalization_rules: Dict[str, str]


class LanguageDetector:
    """
    Detects languages in user queries and provides language-specific processing.
    
    Uses multiple detection strategies:
    1. Character-based detection (scripts, special characters)
    2. Word-based detection (common words, stop words)
    3. Pattern-based detection (language-specific patterns)
    4. Statistical analysis for mixed languages
    """
    
    def __init__(self):
        self.language_patterns = self._load_language_patterns()
        self.language_rules = self._load_language_rules()
        self.character_scripts = self._load_character_scripts()
        self.confidence_threshold = 0.6
        
    def _load_language_patterns(self) -> Dict[SupportedLanguage, Dict[str, List[str]]]:
        """Load language-specific patterns for detection."""
        return {
            SupportedLanguage.ENGLISH: {
                'common_words': [
                    'the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
                    'what', 'how', 'where', 'when', 'who', 'why', 'which', 'that',
                    'this', 'these', 'those', 'with', 'for', 'from', 'about'
                ],
                'question_words': ['what', 'how', 'where', 'when', 'who', 'why', 'which'],
                'patterns': [r'\b(ing|ed|ly|tion|sion)\b', r'\b(a|an|the)\s+\w+']
            },
            SupportedLanguage.SPANISH: {
                'common_words': [
                    'el', 'la', 'los', 'las', 'un', 'una', 'y', 'es', 'son', 'está',
                    'están', 'ser', 'estar', 'tener', 'hacer', 'poder', 'deber',
                    'qué', 'cómo', 'dónde', 'cuándo', 'quién', 'por qué', 'cuál',
                    'que', 'con', 'para', 'por', 'de', 'en', 'sobre'
                ],
                'question_words': ['qué', 'cómo', 'dónde', 'cuándo', 'quién', 'por qué', 'cuál'],
                'patterns': [r'\b(ción|sión|mente|ando|iendo)\b', r'\b(el|la|los|las)\s+\w+']
            },
            SupportedLanguage.FRENCH: {
                'common_words': [
                    'le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'sont', 'être',
                    'avoir', 'faire', 'pouvoir', 'devoir', 'aller', 'venir',
                    'que', 'quoi', 'comment', 'où', 'quand', 'qui', 'pourquoi',
                    'avec', 'pour', 'par', 'de', 'dans', 'sur', 'sous'
                ],
                'question_words': ['que', 'quoi', 'comment', 'où', 'quand', 'qui', 'pourquoi'],
                'patterns': [r'\b(tion|sion|ment|ant|ent)\b', r'\b(le|la|les)\s+\w+']
            },
            SupportedLanguage.GERMAN: {
                'common_words': [
                    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen',
                    'und', 'ist', 'sind', 'war', 'waren', 'haben', 'hat', 'hatte',
                    'was', 'wie', 'wo', 'wann', 'wer', 'warum', 'welche', 'welcher',
                    'mit', 'für', 'von', 'zu', 'in', 'auf', 'über', 'unter'
                ],
                'question_words': ['was', 'wie', 'wo', 'wann', 'wer', 'warum', 'welche'],
                'patterns': [r'\b(ung|keit|heit|lich|isch)\b', r'\b(der|die|das)\s+\w+']
            },
            SupportedLanguage.ITALIAN: {
                'common_words': [
                    'il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'e', 'è', 'sono',
                    'essere', 'avere', 'fare', 'potere', 'dovere', 'andare',
                    'che', 'cosa', 'come', 'dove', 'quando', 'chi', 'perché',
                    'con', 'per', 'da', 'di', 'in', 'su', 'sotto'
                ],
                'question_words': ['che', 'cosa', 'come', 'dove', 'quando', 'chi', 'perché'],
                'patterns': [r'\b(zione|sione|mente|ando|endo)\b', r'\b(il|la|lo)\s+\w+']
            },
            SupportedLanguage.PORTUGUESE: {
                'common_words': [
                    'o', 'a', 'os', 'as', 'um', 'uma', 'e', 'é', 'são', 'está',
                    'estão', 'ser', 'estar', 'ter', 'fazer', 'poder', 'dever',
                    'que', 'o que', 'como', 'onde', 'quando', 'quem', 'por que',
                    'com', 'para', 'por', 'de', 'em', 'sobre', 'sob'
                ],
                'question_words': ['que', 'o que', 'como', 'onde', 'quando', 'quem', 'por que'],
                'patterns': [r'\b(ção|são|mente|ando|endo)\b', r'\b(o|a|os|as)\s+\w+']
            },
            SupportedLanguage.RUSSIAN: {
                'common_words': [
                    'и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а',
                    'по', 'это', 'она', 'к', 'но', 'они', 'мы', 'как', 'из',
                    'что', 'как', 'где', 'когда', 'кто', 'почему', 'какой'
                ],
                'question_words': ['что', 'как', 'где', 'когда', 'кто', 'почему', 'какой'],
                'patterns': [r'[а-яё]+', r'\b(ость|ение|ание)\b']
            },
            SupportedLanguage.CHINESE: {
                'common_words': [
                    '的', '是', '在', '有', '我', '你', '他', '她', '它', '们',
                    '这', '那', '什么', '怎么', '哪里', '什么时候', '谁', '为什么'
                ],
                'question_words': ['什么', '怎么', '哪里', '什么时候', '谁', '为什么'],
                'patterns': [r'[\u4e00-\u9fff]+']
            },
            SupportedLanguage.JAPANESE: {
                'common_words': [
                    'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し',
                    'れ', 'さ', 'ある', 'いる', 'する', 'です', 'ます',
                    '何', 'どう', 'どこ', 'いつ', '誰', 'なぜ', 'どの'
                ],
                'question_words': ['何', 'どう', 'どこ', 'いつ', '誰', 'なぜ', 'どの'],
                'patterns': [r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+']
            },
            SupportedLanguage.KOREAN: {
                'common_words': [
                    '이', '그', '저', '의', '를', '을', '가', '이', '에', '에서',
                    '와', '과', '하다', '있다', '없다', '되다', '오다', '가다',
                    '무엇', '어떻게', '어디', '언제', '누구', '왜', '어느'
                ],
                'question_words': ['무엇', '어떻게', '어디', '언제', '누구', '왜', '어느'],
                'patterns': [r'[\uac00-\ud7af]+']
            },
            SupportedLanguage.ARABIC: {
                'common_words': [
                    'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'ذلك', 'تلك',
                    'ما', 'كيف', 'أين', 'متى', 'من', 'لماذا', 'أي'
                ],
                'question_words': ['ما', 'كيف', 'أين', 'متى', 'من', 'لماذا', 'أي'],
                'patterns': [r'[\u0600-\u06ff]+']
            }
        }
    
    def _load_language_rules(self) -> Dict[SupportedLanguage, LanguageProcessingRules]:
        """Load language-specific processing rules."""
        rules = {}
        
        for language, patterns in self.language_patterns.items():
            rules[language] = LanguageProcessingRules(
                language=language,
                stop_words=set(patterns.get('common_words', [])),
                common_words=set(patterns.get('common_words', [])),
                question_words=set(patterns.get('question_words', [])),
                punctuation_rules={},  # Can be expanded for language-specific punctuation
                normalization_rules={}  # Can be expanded for language-specific normalization
            )
        
        return rules
    
    def _load_character_scripts(self) -> Dict[str, SupportedLanguage]:
        """Load character script mappings for language detection."""
        return {
            'latin': SupportedLanguage.ENGLISH,  # Default for Latin script
            'cyrillic': SupportedLanguage.RUSSIAN,
            'chinese': SupportedLanguage.CHINESE,
            'japanese': SupportedLanguage.JAPANESE,
            'korean': SupportedLanguage.KOREAN,
            'arabic': SupportedLanguage.ARABIC
        }
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language(s) in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageDetectionResult with detection information
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                primary_language=SupportedLanguage.UNKNOWN,
                confidence=0.0,
                detected_languages={},
                is_mixed_language=False,
                language_segments=[]
            )
        
        text = text.strip().lower()
        
        # Detect character scripts first
        script_scores = self._detect_character_scripts(text)
        
        # Detect based on word patterns
        word_scores = self._detect_word_patterns(text)
        
        # Combine scores
        combined_scores = self._combine_detection_scores(script_scores, word_scores)
        
        # Determine primary language and confidence
        primary_language, confidence = self._determine_primary_language(combined_scores)
        
        # Check for mixed languages
        is_mixed, segments = self._detect_mixed_languages(text, combined_scores)
        
        result = LanguageDetectionResult(
            primary_language=primary_language,
            confidence=confidence,
            detected_languages=combined_scores,
            is_mixed_language=is_mixed,
            language_segments=segments
        )
        
        logger.info(f"Language detection: '{text[:50]}...' -> {primary_language.value} (confidence: {confidence:.2f})")
        
        return result
    
    def _detect_character_scripts(self, text: str) -> Dict[SupportedLanguage, float]:
        """Detect languages based on character scripts."""
        scores = {}
        total_chars = len(text)
        
        if total_chars == 0:
            return scores
        
        # Count characters by script
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        cyrillic_chars = len(re.findall(r'[а-яё]', text, re.IGNORECASE))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        
        # Calculate scores based on character frequency
        if cyrillic_chars > 0:
            scores[SupportedLanguage.RUSSIAN] = cyrillic_chars / total_chars
        
        if chinese_chars > 0:
            scores[SupportedLanguage.CHINESE] = chinese_chars / total_chars
        
        if japanese_chars > 0:
            scores[SupportedLanguage.JAPANESE] = japanese_chars / total_chars
        
        if korean_chars > 0:
            scores[SupportedLanguage.KOREAN] = korean_chars / total_chars
        
        if arabic_chars > 0:
            scores[SupportedLanguage.ARABIC] = arabic_chars / total_chars
        
        # For Latin script, we need word-based detection
        if latin_chars > 0:
            # Will be determined by word patterns
            pass
        
        return scores
    
    def _detect_word_patterns(self, text: str) -> Dict[SupportedLanguage, float]:
        """Detect languages based on word patterns and common words."""
        scores = {}
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return scores
        
        total_words = len(words)
        
        for language, patterns in self.language_patterns.items():
            common_words = patterns.get('common_words', [])
            question_words = patterns.get('question_words', [])
            
            # Count matches with common words
            common_matches = sum(1 for word in words if word in common_words)
            question_matches = sum(1 for word in words if word in question_words)
            
            # Calculate score
            score = 0
            if total_words > 0:
                score += (common_matches / total_words) * 0.7
                score += (question_matches / total_words) * 0.3
            
            # Check for language-specific patterns
            pattern_matches = 0
            for pattern in patterns.get('patterns', []):
                pattern_matches += len(re.findall(pattern, text, re.IGNORECASE))
            
            if pattern_matches > 0:
                score += min(pattern_matches / total_words, 0.3)
            
            if score > 0:
                scores[language] = min(score, 1.0)
        
        return scores
    
    def _combine_detection_scores(self, script_scores: Dict[SupportedLanguage, float], 
                                word_scores: Dict[SupportedLanguage, float]) -> Dict[SupportedLanguage, float]:
        """Combine character script and word pattern scores."""
        combined = {}
        
        # Start with script scores (they're more reliable)
        for language, score in script_scores.items():
            combined[language] = score * 0.8  # High weight for script detection
        
        # Add word pattern scores
        for language, score in word_scores.items():
            if language in combined:
                # Combine with existing script score
                combined[language] = min(combined[language] + score * 0.2, 1.0)
            else:
                # Use word score only
                combined[language] = score * 0.6  # Lower weight when no script match
        
        return combined
    
    def _determine_primary_language(self, scores: Dict[SupportedLanguage, float]) -> Tuple[SupportedLanguage, float]:
        """Determine the primary language from detection scores."""
        if not scores:
            return SupportedLanguage.ENGLISH, 0.3  # Default to English with low confidence
        
        # Find the highest scoring language
        primary_language = max(scores.items(), key=lambda x: x[1])
        
        # If confidence is too low, default to English
        if primary_language[1] < 0.2:
            return SupportedLanguage.ENGLISH, 0.3
        
        return primary_language[0], primary_language[1]
    
    def _detect_mixed_languages(self, text: str, scores: Dict[SupportedLanguage, float]) -> Tuple[bool, List[Tuple[str, SupportedLanguage, float]]]:
        """Detect if the text contains mixed languages and segment them."""
        # Simple mixed language detection
        # Count languages with significant scores
        significant_languages = [lang for lang, score in scores.items() if score > 0.2]
        
        is_mixed = len(significant_languages) > 1
        
        # For now, return simple segmentation
        # In a more sophisticated implementation, we would segment the text
        segments = []
        if is_mixed:
            for language in significant_languages:
                segments.append((text, language, scores[language]))
        else:
            primary_lang = max(scores.items(), key=lambda x: x[1])[0] if scores else SupportedLanguage.ENGLISH
            segments.append((text, primary_lang, scores.get(primary_lang, 0.3)))
        
        return is_mixed, segments
    
    def get_processing_rules(self, language: SupportedLanguage) -> LanguageProcessingRules:
        """Get processing rules for a specific language."""
        return self.language_rules.get(language, self.language_rules[SupportedLanguage.ENGLISH])
    
    def normalize_mixed_language_query(self, text: str, detection_result: LanguageDetectionResult) -> str:
        """
        Normalize a mixed-language query for better processing.
        
        Args:
            text: Original text
            detection_result: Language detection result
            
        Returns:
            Normalized text
        """
        if not detection_result.is_mixed_language:
            return text
        
        # For mixed languages, we'll apply basic normalization
        # In a more sophisticated implementation, we would handle each segment differently
        
        normalized = text
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle common mixed-language patterns
        # For example, English questions with non-English keywords
        if SupportedLanguage.ENGLISH in detection_result.detected_languages:
            # Ensure English question words are properly formatted
            english_rules = self.get_processing_rules(SupportedLanguage.ENGLISH)
            for question_word in english_rules.question_words:
                pattern = rf'\b{re.escape(question_word)}\b'
                normalized = re.sub(pattern, question_word, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def translate_to_english(self, text: str, source_language: SupportedLanguage) -> str:
        """
        Basic translation to English for non-English queries.
        
        Note: This is a placeholder implementation. In a production system,
        you would integrate with a proper translation service like Google Translate.
        
        Args:
            text: Text to translate
            source_language: Source language
            
        Returns:
            Translated text (or original if already English)
        """
        if source_language == SupportedLanguage.ENGLISH:
            return text
        
        # Placeholder implementation - in reality, you would use a translation API
        logger.warning(f"Translation requested from {source_language.value} to English, but no translation service configured")
        
        # For now, return original text with a note
        return f"[{source_language.value}] {text}"
    
    def is_supported_language(self, language: SupportedLanguage) -> bool:
        """Check if a language is supported for processing."""
        return language in self.language_rules and language != SupportedLanguage.UNKNOWN