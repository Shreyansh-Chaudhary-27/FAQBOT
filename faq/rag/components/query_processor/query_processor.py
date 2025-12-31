"""
Main Query Processor Implementation

This module provides the main QueryProcessor class that integrates typo correction,
intent extraction, and language detection to provide comprehensive query processing
capabilities for the RAG system.
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import re

from faq.rag.interfaces.base import QueryProcessorInterface, ProcessedQuery, ConversationContext
from faq.rag.utils.ngram_utils import generate_ngrams
from .typo_corrector import TypoCorrector
from .intent_extractor import IntentExtractor, QueryIntent, ExtractedIntent
from .language_detector import LanguageDetector, SupportedLanguage, LanguageDetectionResult

logger = logging.getLogger(__name__)


class QueryProcessor(QueryProcessorInterface):
    """
    Main query processor that integrates all query processing capabilities.
    
    This class orchestrates typo correction, intent extraction, language detection,
    and natural language understanding to provide comprehensive query processing
    for the RAG system.
    """
    
    def __init__(self):
        """Initialize the query processor with all components."""
        self.typo_corrector = TypoCorrector()
        self.intent_extractor = IntentExtractor()
        self.language_detector = LanguageDetector()
        
        # Natural language understanding patterns
        self.informal_patterns = self._load_informal_patterns()
        self.expansion_templates = self._load_expansion_templates()
        
    def _load_informal_patterns(self) -> Dict[str, str]:
        """Load patterns for handling informal grammar and incomplete sentences."""
        return {
            # Missing articles
            r'\b(need|want|looking for|find)\s+(\w+)': r'\1 a \2',
            r'\b(have|got)\s+(problem|issue|question)\s+with\s+(\w+)': r'\1 a \2 with \3',
            
            # Missing auxiliary verbs
            r'\b(you|i|we|they)\s+(going|coming|working|trying)': r'\1 are \2',
            r'\b(it|this|that)\s+(working|broken|good|bad)': r'\1 is \2',
            
            # Incomplete questions
            r'^(what|how|where|when|who|why)\s+about\s+(.+)': r'\1 is \2',
            r'^(tell me|show me)\s+about\s+(.+)': r'what is \2',
            
            # Missing "to be" verbs
            r'\b(this|that|it)\s+(good|bad|working|broken|available|possible)': r'\1 is \2',
            r'\b(you|we|they)\s+(sure|certain|ready|available)': r'\1 are \2',
            
            # Colloquial expressions
            r'\bhow come\b': 'why',
            r'\bwanna\b': 'want to',
            r'\bgonna\b': 'going to',
            r'\bkinda\b': 'kind of',
            r'\bsorta\b': 'sort of',
            
            # Missing prepositions
            r'\b(help)\s+(me)\s+(\w+)': r'\1 \2 with \3',
            r'\b(tell)\s+(me)\s+(\w+)': r'\1 \2 about \3',
        }
    
    def _load_expansion_templates(self) -> Dict[str, List[str]]:
        """Load templates for query expansion based on intent."""
        return {
            'instruction': [
                'how to {query}',
                'steps to {query}',
                'instructions for {query}',
                'guide to {query}',
                'tutorial on {query}',
                'way to {query}'
            ],
            'definition': [
                'what is {query}',
                'define {query}',
                'meaning of {query}',
                'explanation of {query}',
                '{query} definition',
                'what does {query} mean'
            ],
            'problem': [
                'fix {query}',
                'solve {query}',
                'troubleshoot {query}',
                'resolve {query}',
                'repair {query}',
                '{query} not working'
            ],
            'information': [
                'information about {query}',
                'details about {query}',
                'learn about {query}',
                'find out about {query}',
                '{query} information',
                'tell me about {query}'
            ],
            'help': [
                'help with {query}',
                'assistance with {query}',
                'support for {query}',
                'guidance on {query}',
                'need help with {query}'
            ],
            'comparison': [
                'compare {query}',
                'difference between {query}',
                '{query} vs',
                '{query} comparison',
                'which is better {query}'
            ]
        }
    
    def correct_typos(self, query: str) -> str:
        """
        Correct typos and spelling errors in query.
        
        Args:
            query: Original query string
            
        Returns:
            Query with corrected typos
        """
        corrected, confidence = self.typo_corrector.correct_typos(query)
        return corrected
    
    def extract_intent(self, query: str) -> str:
        """
        Extract intent from user query.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Intent as string
        """
        extracted_intent = self.intent_extractor.extract_intent(query)
        return extracted_intent.intent.value
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better matching.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        # First extract intent to guide expansion
        extracted_intent = self.intent_extractor.extract_intent(query)
        
        # Use intent extractor's expansion method
        intent_variations = self.intent_extractor.expand_query(query, extracted_intent)
        
        # Add our own template-based expansions
        template_variations = self._generate_template_variations(query, extracted_intent)
        
        # Combine and deduplicate
        all_variations = intent_variations + template_variations
        
        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for variation in all_variations:
            if variation.lower() not in seen:
                seen.add(variation.lower())
                unique_variations.append(variation)
        
        return unique_variations
    
    def detect_language(self, query: str) -> str:
        """
        Detect the language of the query.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Language code as string
        """
        detection_result = self.language_detector.detect_language(query)
        return detection_result.primary_language.value
    
    def extract_composite_components(self, query: str, context: Optional[ConversationContext] = None) -> Dict[str, str]:
        """
        Extract composite key components from the query.
        Components: audience, category, intent, condition.
        
        This will eventually use Gemini to perform precise extraction.
        For now, we implement a baseline extraction with placeholders.
        """
        # Baseline extraction logic
        # In a real scenario, this would call self.intent_extractor.extract_with_gemini(...)
        
        # Default components
        components = {
            'audience': 'any',
            'category': 'general',
            'intent': 'information',
            'condition': 'default'
        }
        
        # Try to infer audience from context if available
        if context and context.user_preferences:
            components['audience'] = context.user_preferences.get('audience', 'any')
            
        # Extract intent using existing intent extractor
        extracted_intent = self.intent_extractor.extract_intent(query)
        # Normalize 'info' or any variation to 'information'
        intent_val = extracted_intent.intent.value
        if intent_val in ['info', 'information', 'unknown']:
            components['intent'] = 'information'
        else:
            components['intent'] = intent_val
        
        # Simple keyword-based category extraction
        query_lower = query.lower()
        if 'leave' in query_lower:
            components['category'] = 'leave'
        elif 'payroll' in query_lower or 'salary' in query_lower:
            components['category'] = 'payroll'
        elif 'policy' in query_lower:
            components['category'] = 'policy'
            
        return components
    
    def preprocess_query(self, query: str) -> ProcessedQuery:
        """
        Complete query preprocessing pipeline.
        
        Args:
            query: Raw user query
            
        Returns:
            ProcessedQuery object with all preprocessing results
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query,
                corrected_query="",
                intent="unknown",
                expanded_queries=[],
                language="en",
                confidence=0.0,
                embedding=None,
                ngram_keywords=[]
            )
        
        logger.info(f"Processing query: '{query}'")
        
        # Step 1: Detect language
        language_result = self.language_detector.detect_language(query)
        
        # Step 2: Normalize informal grammar and incomplete sentences
        normalized_query = self.normalize_informal_grammar(query)
        
        # Step 3: Handle mixed language queries if needed
        if language_result.is_mixed_language:
            normalized_query = self.language_detector.normalize_mixed_language_query(
                normalized_query, language_result
            )
        
        # Step 4: Correct typos
        corrected_query, typo_confidence = self.typo_corrector.correct_typos(normalized_query)
        
        # Step 5: Extract intent
        extracted_intent = self.intent_extractor.extract_intent(corrected_query)
        
        # Step 6: Generate query expansions
        expanded_queries = self.expand_query(corrected_query)
        
        # Step 7: Extract composite key components
        components = self.extract_composite_components(corrected_query)
        
        # Step 8: Calculate overall confidence
        confidence = self._calculate_processing_confidence(
            language_result, extracted_intent, corrected_query, query, typo_confidence
        )
        
        # Step 9: Generate N-Gram keywords for precision matching (Requirement: 90% overlap)
        ngram_keywords = list(generate_ngrams(corrected_query))
        
        result = ProcessedQuery(
            original_query=query,
            corrected_query=corrected_query,
            intent=extracted_intent.intent.value,
            expanded_queries=expanded_queries,
            language=language_result.primary_language.value,
            confidence=confidence,
            embedding=None,  # Will be set by vectorizer
            components=components,
            ngram_keywords=ngram_keywords
        )
        
        logger.info(f"Query processing complete: intent={result.intent}, "
                   f"language={result.language}, confidence={result.confidence:.2f}")
        
        return result
    
    def normalize_informal_grammar(self, query: str) -> str:
        """
        Normalize informal grammar and incomplete sentences.
        
        This method handles common informal patterns, missing words,
        and grammatical shortcuts to improve query understanding.
        
        Args:
            query: Query with potentially informal grammar
            
        Returns:
            Normalized query string
        """
        if not query or not query.strip():
            return query
        
        normalized = query.strip()
        
        # Apply informal pattern corrections
        for pattern, replacement in self.informal_patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Handle sentence fragments and add appropriate punctuation
        normalized = self._fix_sentence_structure(normalized)
        
        # Clean up extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        logger.debug(f"Grammar normalization: '{query}' -> '{normalized}'")
        
        return normalized
    
    def _fix_sentence_structure(self, query: str) -> str:
        """Fix incomplete sentence structures."""
        if not query:
            return query
        
        # Handle missing punctuation
        if not query.endswith(('.', '?', '!')):
            # Add question mark if it looks like a question
            question_indicators = [
                'what', 'how', 'where', 'when', 'who', 'why', 'which',
                'can', 'is', 'are', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might'
            ]
            
            first_word = query.split()[0].lower() if query.split() else ""
            if first_word in question_indicators:
                query += '?'
            else:
                query += '.'
        
        # Handle capitalization
        if query and query[0].islower():
            query = query[0].upper() + query[1:]
        
        return query
    
    def _generate_template_variations(self, query: str, intent: ExtractedIntent) -> List[str]:
        """Generate query variations using templates based on intent."""
        variations = []
        
        # Get templates for the detected intent
        intent_key = intent.intent.value.lower()
        templates = self.expansion_templates.get(intent_key, [])
        
        # Extract main keywords for template substitution
        main_keywords = intent.keywords[:3] if intent.keywords else [query]
        
        for template in templates:
            for keyword_set in [main_keywords, [query]]:
                for keywords in keyword_set:
                    if isinstance(keywords, list):
                        keyword_phrase = ' '.join(keywords)
                    else:
                        keyword_phrase = keywords
                    
                    variation = template.format(query=keyword_phrase)
                    variations.append(variation)
        
        return variations
    
    def _calculate_processing_confidence(self, language_result: LanguageDetectionResult,
                                       intent: ExtractedIntent, corrected_query: str,
                                       original_query: str, typo_confidence: float = 1.0) -> float:
        """Calculate overall confidence in query processing."""
        confidence_factors = []
        
        # Language detection confidence
        confidence_factors.append(language_result.confidence * 0.3)
        
        # Intent extraction confidence
        confidence_factors.append(intent.confidence * 0.4)
        
        # Typo correction confidence (use the provided confidence from typo corrector)
        confidence_factors.append(typo_confidence * 0.2)
        
        # Query completeness (has meaningful content)
        if len(corrected_query.split()) >= 2:
            completeness_confidence = 1.0
        else:
            completeness_confidence = 0.6
        
        confidence_factors.append(completeness_confidence * 0.1)
        
        # Calculate weighted average
        total_confidence = sum(confidence_factors)
        
        return min(total_confidence, 1.0)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity between two strings."""
        if not str1 or not str2:
            return 0.0
        
        if str1 == str2:
            return 1.0
        
        # Simple character-based similarity
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str2 if len(str1) > len(str2) else str1
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters
        matches = sum(1 for a, b in zip(longer, shorter) if a == b)
        return matches / len(longer)
    
    def process_informal_query(self, query: str) -> ProcessedQuery:
        """
        Specialized processing for informal queries with poor grammar.
        
        This method provides enhanced processing for queries that may have
        informal grammar, incomplete sentences, or colloquial expressions.
        
        Args:
            query: Informal query string
            
        Returns:
            ProcessedQuery with enhanced informal processing
        """
        # Start with standard preprocessing
        processed = self.preprocess_query(query)
        
        # Apply additional informal processing
        enhanced_query = self._enhance_informal_understanding(processed.corrected_query)
        
        # Re-extract intent with enhanced query
        enhanced_intent = self.intent_extractor.extract_intent(enhanced_query)
        
        # Generate additional expansions for informal queries
        informal_expansions = self._generate_informal_expansions(enhanced_query, enhanced_intent)
        
        # Combine with existing expansions
        all_expansions = processed.expanded_queries + informal_expansions
        
        # Remove duplicates
        unique_expansions = []
        seen = set()
        for expansion in all_expansions:
            if expansion.lower() not in seen:
                seen.add(expansion.lower())
                unique_expansions.append(expansion)
        
        # Update the processed query
        processed.corrected_query = enhanced_query
        processed.intent = enhanced_intent.intent.value
        processed.expanded_queries = unique_expansions
        
        return processed
    
    def _enhance_informal_understanding(self, query: str) -> str:
        """Apply additional enhancements for informal query understanding."""
        enhanced = query
        
        # Handle common informal question patterns
        informal_question_patterns = {
            r'^(whats|what\'s)\b': 'what is',
            r'^(hows|how\'s)\b': 'how is',
            r'^(wheres|where\'s)\b': 'where is',
            r'^(whens|when\'s)\b': 'when is',
            r'^(whos|who\'s)\b': 'who is',
            r'^(whys|why\'s)\b': 'why is',
            r'\b(dont|don\'t)\b': 'do not',
            r'\b(cant|can\'t)\b': 'cannot',
            r'\b(wont|won\'t)\b': 'will not',
            r'\b(isnt|isn\'t)\b': 'is not',
            r'\b(arent|aren\'t)\b': 'are not',
        }
        
        for pattern, replacement in informal_question_patterns.items():
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _generate_informal_expansions(self, query: str, intent: ExtractedIntent) -> List[str]:
        """Generate additional expansions specifically for informal queries."""
        expansions = []
        
        # Add more casual variations
        casual_templates = [
            'help me {query}',
            'i need to {query}',
            'how do i {query}',
            'can you {query}',
            'show me how to {query}',
            'i want to know about {query}',
            'tell me about {query}'
        ]
        
        for template in casual_templates:
            expansion = template.format(query=query)
            expansions.append(expansion)
        
        # Add keyword-only variations
        if intent.keywords:
            expansions.extend(intent.keywords[:3])
        
        return expansions
    
    def process_with_context(self, query: str, context: Optional[ConversationContext] = None) -> ProcessedQuery:
        """
        Process query with conversation context for improved understanding.
        
        This method enhances query processing by utilizing conversation history,
        current topic, and user preferences to better understand the query intent
        and generate more relevant expansions.
        
        Args:
            query: Raw user query
            context: Optional conversation context from previous interactions
            
        Returns:
            ProcessedQuery with context-enhanced processing
        """
        # Start with standard preprocessing
        processed = self.preprocess_query(query)
        
        if context is None:
            return processed
        
        # Enhance query understanding with context
        context_enhanced_query = self._apply_context_understanding(processed, context)
        
        # Generate context-aware expansions
        context_expansions = self._generate_context_aware_expansions(context_enhanced_query, context)
        
        # Combine with existing expansions
        all_expansions = processed.expanded_queries + context_expansions
        
        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for expansion in all_expansions:
            if expansion.lower() not in seen:
                seen.add(expansion.lower())
                unique_expansions.append(expansion)
        
        # Update processed query with context enhancements
        processed.corrected_query = context_enhanced_query
        processed.expanded_queries = unique_expansions
        
        # Increase confidence if context was helpful
        if context.history and self._is_context_relevant(processed, context):
            processed.confidence = min(processed.confidence * 1.1, 1.0)
        
        logger.info(f"Context-aware processing complete for query: '{query}' "
                   f"(session: {context.session_id if context else 'none'})")
        
        return processed
    
    def detect_ambiguity(self, query: str, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """
        Detect ambiguous queries that need clarification.
        
        Analyzes the query to identify potential ambiguities and suggests
        clarifying questions or multiple interpretations.
        
        Args:
            query: User query to analyze
            context: Optional conversation context
            
        Returns:
            Dictionary containing ambiguity analysis and suggestions
        """
        ambiguity_result = {
            'is_ambiguous': False,
            'ambiguity_type': None,
            'confidence': 1.0,
            'clarifying_questions': [],
            'possible_interpretations': [],
            'suggested_responses': []
        }
        
        # Process the query first
        processed = self.preprocess_query(query)
        
        # Check for various types of ambiguity
        ambiguity_checks = [
            self._check_pronoun_ambiguity(query, context),
            self._check_incomplete_query_ambiguity(query, processed),
            self._check_multiple_intent_ambiguity(query, processed),
            self._check_context_dependent_ambiguity(query, context),
            self._check_vague_terms_ambiguity(query)
        ]
        
        # Combine ambiguity results
        for check_result in ambiguity_checks:
            if check_result['is_ambiguous']:
                ambiguity_result['is_ambiguous'] = True
                ambiguity_result['ambiguity_type'] = check_result.get('type', 'unknown')
                ambiguity_result['confidence'] = min(ambiguity_result['confidence'], 
                                                   check_result.get('confidence', 0.5))
                ambiguity_result['clarifying_questions'].extend(
                    check_result.get('clarifying_questions', [])
                )
                ambiguity_result['possible_interpretations'].extend(
                    check_result.get('interpretations', [])
                )
        
        # Generate suggested responses for ambiguous queries
        if ambiguity_result['is_ambiguous']:
            ambiguity_result['suggested_responses'] = self._generate_clarification_responses(
                ambiguity_result
            )
        
        logger.debug(f"Ambiguity detection for '{query}': {ambiguity_result['is_ambiguous']} "
                    f"(type: {ambiguity_result['ambiguity_type']})")
        
        return ambiguity_result
    
    def handle_follow_up_question(self, query: str, context: ConversationContext) -> ProcessedQuery:
        """
        Handle follow-up questions by leveraging conversation context.
        
        This method specializes in processing queries that reference previous
        interactions, using pronouns, or building on earlier topics.
        
        Args:
            query: Follow-up query from user
            context: Conversation context with history
            
        Returns:
            ProcessedQuery enhanced for follow-up understanding
        """
        # Process with context first
        processed = self.process_with_context(query, context)
        
        # Detect follow-up patterns
        follow_up_info = self._analyze_follow_up_patterns(query, context)
        
        if follow_up_info['is_follow_up']:
            # Enhance query with follow-up context
            enhanced_query = self._resolve_follow_up_references(query, context, follow_up_info)
            
            # Re-process the enhanced query
            enhanced_processed = self.preprocess_query(enhanced_query)
            
            # Merge the results, keeping the best parts of both
            processed.corrected_query = enhanced_processed.corrected_query
            processed.intent = enhanced_processed.intent
            
            # Combine expansions
            combined_expansions = processed.expanded_queries + enhanced_processed.expanded_queries
            processed.expanded_queries = list(dict.fromkeys(combined_expansions))  # Remove duplicates
            
            # Boost confidence for successful follow-up resolution
            processed.confidence = min(processed.confidence * 1.15, 1.0)
            
            logger.info(f"Follow-up question processed: '{query}' -> '{enhanced_query}'")
        
        return processed
    
    def _apply_context_understanding(self, processed: ProcessedQuery, context: ConversationContext) -> str:
        """Apply conversation context to enhance query understanding."""
        enhanced_query = processed.corrected_query
        
        # Use current topic to add context
        if context.current_topic and len(enhanced_query.split()) <= 3:
            # Short queries might benefit from topic context
            topic_enhanced = f"{enhanced_query} about {context.current_topic}"
            enhanced_query = topic_enhanced
        
        # Resolve pronouns using recent history
        if context.history:
            enhanced_query = self._resolve_pronouns_with_history(enhanced_query, context.history)
        
        return enhanced_query
    
    def _generate_context_aware_expansions(self, query: str, context: ConversationContext) -> List[str]:
        """Generate query expansions that leverage conversation context."""
        expansions = []
        
        # Add topic-based expansions
        if context.current_topic:
            topic_expansions = [
                f"{query} {context.current_topic}",
                f"{context.current_topic} {query}",
                f"how does {query} relate to {context.current_topic}"
            ]
            expansions.extend(topic_expansions)
        
        # Add history-based expansions
        if context.history:
            recent_queries = [interaction.get('query', '') for interaction in context.history[-3:]]
            for recent_query in recent_queries:
                if recent_query and len(recent_query.split()) > 2:
                    # Extract key terms from recent queries
                    recent_keywords = self._extract_key_terms(recent_query)
                    for keyword in recent_keywords[:2]:  # Limit to avoid too many expansions
                        expansions.append(f"{query} {keyword}")
        
        return expansions
    
    def _is_context_relevant(self, processed: ProcessedQuery, context: ConversationContext) -> bool:
        """Check if conversation context is relevant to current query."""
        if not context.history:
            return False
        
        # Check if query contains pronouns or references
        pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        query_words = processed.corrected_query.lower().split()
        
        has_pronouns = any(pronoun in query_words for pronoun in pronouns)
        
        # Check if query is very short (might need context)
        is_short = len(query_words) <= 2
        
        # Check if current topic is mentioned
        topic_mentioned = (context.current_topic and 
                          any(word in processed.corrected_query.lower() 
                              for word in context.current_topic.lower().split('_')))
        
        return has_pronouns or is_short or topic_mentioned
    
    def _check_pronoun_ambiguity(self, query: str, context: Optional[ConversationContext]) -> Dict[str, Any]:
        """Check for ambiguous pronouns that need clarification."""
        pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        query_lower = query.lower()
        
        # Use regex to find pronouns with word boundaries
        import re
        found_pronouns = []
        for pronoun in pronouns:
            if re.search(r'\b' + pronoun + r'\b', query_lower):
                found_pronouns.append(pronoun)
        

        
        if not found_pronouns:
            return {'is_ambiguous': False}
        
        # Check if context can resolve pronouns
        if context and context.history:
            # If we have recent context, pronouns are still ambiguous but resolvable
            return {
                'is_ambiguous': False,  # Context can resolve it
                'type': 'pronoun_with_context'
            }
        
        return {
            'is_ambiguous': True,
            'type': 'pronoun_ambiguity',
            'confidence': 0.7,
            'clarifying_questions': [
                f"What specifically are you referring to when you say '{pronoun}'?"
                for pronoun in found_pronouns[:2]
            ],
            'interpretations': [
                f"Query refers to something mentioned earlier: '{query}'"
            ]
        }
    
    def _check_incomplete_query_ambiguity(self, query: str, processed: ProcessedQuery) -> Dict[str, Any]:
        """Check for incomplete queries that need more information."""
        # Very short queries might be incomplete
        if len(query.split()) <= 2:
            return {
                'is_ambiguous': True,
                'type': 'incomplete_query',
                'confidence': 0.6,
                'clarifying_questions': [
                    f"Could you provide more details about '{query}'?",
                    f"What specifically would you like to know about '{query}'?"
                ],
                'interpretations': [
                    f"User wants information about: {query}",
                    f"User needs help with: {query}"
                ]
            }
        
        return {'is_ambiguous': False}
    
    def _check_multiple_intent_ambiguity(self, query: str, processed: ProcessedQuery) -> Dict[str, Any]:
        """Check for queries with multiple possible intents."""
        # Simple heuristic: if query contains multiple question words or conjunctions
        question_words = ['what', 'how', 'where', 'when', 'who', 'why', 'which']
        conjunctions = ['and', 'or', 'but']
        
        query_lower = query.lower()
        question_count = sum(1 for word in question_words if word in query_lower)
        conjunction_count = sum(1 for word in conjunctions if word in query_lower)
        
        if question_count > 1 or conjunction_count > 0:
            return {
                'is_ambiguous': True,
                'type': 'multiple_intent',
                'confidence': 0.8,
                'clarifying_questions': [
                    "I notice you're asking about multiple things. Which would you like me to address first?",
                    "Would you like me to break this down into separate questions?"
                ],
                'interpretations': [
                    "User has multiple related questions",
                    "User wants comprehensive information on a complex topic"
                ]
            }
        
        return {'is_ambiguous': False}
    
    def _check_context_dependent_ambiguity(self, query: str, context: Optional[ConversationContext]) -> Dict[str, Any]:
        """Check for queries that depend heavily on context."""
        context_dependent_phrases = [
            'like before', 'as mentioned', 'the same', 'similar to',
            'like that', 'like this', 'as usual', 'again'
        ]
        
        query_lower = query.lower()
        has_context_dependency = any(phrase in query_lower for phrase in context_dependent_phrases)
        
        if has_context_dependency and (not context or not context.history):
            return {
                'is_ambiguous': True,
                'type': 'context_dependent',
                'confidence': 0.9,
                'clarifying_questions': [
                    "Could you provide more context about what you're referring to?",
                    "I don't have the previous context. Could you be more specific?"
                ],
                'interpretations': [
                    "User is referencing something from earlier conversation",
                    "User expects system to remember previous interaction"
                ]
            }
        
        return {'is_ambiguous': False}
    
    def _check_vague_terms_ambiguity(self, query: str) -> Dict[str, Any]:
        """Check for vague terms that could have multiple meanings."""
        vague_terms = [
            'thing', 'stuff', 'issue', 'problem', 'matter', 'situation',
            'something', 'anything', 'everything', 'nothing'
        ]
        
        query_lower = query.lower()
        found_vague_terms = [term for term in vague_terms if term in query_lower]
        
        if found_vague_terms:
            return {
                'is_ambiguous': True,
                'type': 'vague_terms',
                'confidence': 0.6,
                'clarifying_questions': [
                    f"Could you be more specific about the '{term}' you're referring to?"
                    for term in found_vague_terms[:2]
                ],
                'interpretations': [
                    f"User is asking about something general: {query}"
                ]
            }
        
        return {'is_ambiguous': False}
    
    def _generate_clarification_responses(self, ambiguity_result: Dict[str, Any]) -> List[str]:
        """Generate appropriate clarification responses for ambiguous queries."""
        responses = []
        
        ambiguity_type = ambiguity_result.get('ambiguity_type')
        clarifying_questions = ambiguity_result.get('clarifying_questions', [])
        
        if ambiguity_type == 'pronoun_ambiguity':
            responses.append(
                "I notice you're using pronouns like 'it' or 'this'. Could you clarify what you're referring to?"
            )
        elif ambiguity_type == 'incomplete_query':
            responses.append(
                "Your question seems brief. Could you provide more details so I can help you better?"
            )
        elif ambiguity_type == 'multiple_intent':
            responses.append(
                "I see you have multiple questions. Would you like me to address them one at a time?"
            )
        elif ambiguity_type == 'context_dependent':
            responses.append(
                "It seems like you're referring to something from earlier. Could you provide more context?"
            )
        elif ambiguity_type == 'vague_terms':
            responses.append(
                "I'd like to help, but could you be more specific about what you're looking for?"
            )
        
        # Add the specific clarifying questions
        responses.extend(clarifying_questions[:2])  # Limit to avoid overwhelming user
        
        return responses
    
    def _analyze_follow_up_patterns(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze if query is a follow-up and extract relevant patterns."""
        follow_up_indicators = [
            'also', 'too', 'additionally', 'furthermore', 'moreover',
            'what about', 'how about', 'and', 'plus', 'besides'
        ]
        
        continuation_words = [
            'continue', 'more', 'next', 'then', 'after', 'following'
        ]
        
        reference_words = [
            'it', 'this', 'that', 'they', 'them', 'these', 'those',
            'above', 'previous', 'earlier', 'before'
        ]
        
        query_lower = query.lower()
        
        has_follow_up_indicators = any(indicator in query_lower for indicator in follow_up_indicators)
        has_continuation_words = any(word in query_lower for word in continuation_words)
        has_reference_words = any(word in query_lower for word in reference_words)
        
        is_follow_up = has_follow_up_indicators or has_continuation_words or has_reference_words
        
        return {
            'is_follow_up': is_follow_up,
            'has_indicators': has_follow_up_indicators,
            'has_continuations': has_continuation_words,
            'has_references': has_reference_words,
            'confidence': 0.8 if is_follow_up else 0.0
        }
    
    def _resolve_follow_up_references(self, query: str, context: ConversationContext, 
                                    follow_up_info: Dict[str, Any]) -> str:
        """Resolve references in follow-up questions using conversation context."""
        enhanced_query = query
        
        if not context.history:
            return enhanced_query
        
        # Get the most recent interaction
        recent_interaction = context.history[-1] if context.history else None
        
        if recent_interaction:
            recent_query = recent_interaction.get('query', '')
            recent_response = recent_interaction.get('response', '')
            
            # Extract key terms from recent interaction
            recent_keywords = self._extract_key_terms(recent_query)
            
            # Replace pronouns with likely references
            pronoun_replacements = {
                'it': recent_keywords[0] if recent_keywords else 'the topic',
                'this': recent_keywords[0] if recent_keywords else 'this topic',
                'that': recent_keywords[0] if recent_keywords else 'that topic',
                'they': ' and '.join(recent_keywords[:2]) if len(recent_keywords) >= 2 else 'they',
                'them': ' and '.join(recent_keywords[:2]) if len(recent_keywords) >= 2 else 'them'
            }
            
            for pronoun, replacement in pronoun_replacements.items():
                pattern = r'\b' + pronoun + r'\b'
                enhanced_query = re.sub(pattern, replacement, enhanced_query, flags=re.IGNORECASE)
        
        # Add context from current topic
        if context.current_topic and len(enhanced_query.split()) <= 4:
            enhanced_query = f"{enhanced_query} regarding {context.current_topic}"
        
        return enhanced_query
    
    def _resolve_pronouns_with_history(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Resolve pronouns using conversation history."""
        if not history:
            return query
        
        # Get recent keywords from history
        recent_keywords = []
        for interaction in history[-3:]:  # Look at last 3 interactions
            query_text = interaction.get('query', '')
            if query_text:
                keywords = self._extract_key_terms(query_text)
                recent_keywords.extend(keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in recent_keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        # Replace pronouns with most likely references
        enhanced_query = query
        if unique_keywords:
            # Create pronoun replacement map
            primary_keyword = unique_keywords[0] if unique_keywords else "it"
            secondary_keyword = unique_keywords[1] if len(unique_keywords) > 1 else primary_keyword
            
            pronoun_map = {
                'it': primary_keyword,
                'this': primary_keyword,
                'that': secondary_keyword,
                'they': ' and '.join(unique_keywords[:2]) if len(unique_keywords) > 1 else primary_keyword,
                'them': ' and '.join(unique_keywords[:2]) if len(unique_keywords) > 1 else primary_keyword
            }
            
            for pronoun, replacement in pronoun_map.items():
                pattern = r'\b' + pronoun + r'\b'
                enhanced_query = re.sub(pattern, replacement, enhanced_query, flags=re.IGNORECASE)
        
        return enhanced_query
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for context resolution."""
        if not text:
            return []
        
        # Simple keyword extraction - remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'how', 'where', 'when',
            'who', 'why', 'which', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords