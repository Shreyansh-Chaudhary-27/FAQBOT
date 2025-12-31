"""
Intent Extraction and Natural Language Understanding Module

This module provides intent extraction and natural language understanding capabilities
for user queries. It handles informal grammar, incomplete sentences, and extracts
the underlying meaning and intent from user input.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    QUESTION = "question"           # User is asking a question
    INFORMATION = "information"     # User wants information about something
    HELP = "help"                  # User needs help or support
    PROBLEM = "problem"            # User is reporting a problem or issue
    INSTRUCTION = "instruction"     # User wants instructions or how-to
    COMPARISON = "comparison"       # User wants to compare options
    DEFINITION = "definition"       # User wants a definition or explanation
    LOCATION = "location"          # User is asking about location/where
    TIME = "time"                  # User is asking about time/when
    PERSON = "person"              # User is asking about who
    REASON = "reason"              # User is asking why/reason
    METHOD = "method"              # User is asking how
    CONFIRMATION = "confirmation"   # User wants confirmation
    UNKNOWN = "unknown"            # Intent cannot be determined


@dataclass
class ExtractedIntent:
    """Container for extracted intent information."""
    intent: QueryIntent
    confidence: float
    keywords: List[str]
    entities: Dict[str, List[str]]
    question_type: Optional[str]
    modifiers: List[str]


class IntentExtractor:
    """
    Extracts intent and meaning from user queries using pattern matching
    and natural language understanding techniques.
    """
    
    def __init__(self):
        self.question_patterns = self._load_question_patterns()
        self.intent_keywords = self._load_intent_keywords()
        self.entity_patterns = self._load_entity_patterns()
        self.stop_words = self._load_stop_words()
        
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for different types of questions."""
        return {
            'what': [
                r'\bwhat\s+is\b', r'\bwhat\s+are\b', r'\bwhat\s+does\b', r'\bwhat\s+do\b',
                r'\bwhat\s+can\b', r'\bwhat\s+should\b', r'\bwhat\s+would\b', r'\bwhat\s+will\b',
                r'\bwhat\s+about\b', r'\bwhat\s+if\b', r'\bwhat\s+kind\b', r'\bwhat\s+type\b'
            ],
            'how': [
                r'\bhow\s+do\b', r'\bhow\s+can\b', r'\bhow\s+to\b', r'\bhow\s+does\b',
                r'\bhow\s+should\b', r'\bhow\s+would\b', r'\bhow\s+will\b', r'\bhow\s+much\b',
                r'\bhow\s+many\b', r'\bhow\s+long\b', r'\bhow\s+often\b'
            ],
            'where': [
                r'\bwhere\s+is\b', r'\bwhere\s+are\b', r'\bwhere\s+can\b', r'\bwhere\s+do\b',
                r'\bwhere\s+should\b', r'\bwhere\s+would\b', r'\bwhere\s+will\b'
            ],
            'when': [
                r'\bwhen\s+is\b', r'\bwhen\s+are\b', r'\bwhen\s+can\b', r'\bwhen\s+do\b',
                r'\bwhen\s+should\b', r'\bwhen\s+would\b', r'\bwhen\s+will\b', r'\bwhen\s+did\b'
            ],
            'who': [
                r'\bwho\s+is\b', r'\bwho\s+are\b', r'\bwho\s+can\b', r'\bwho\s+should\b',
                r'\bwho\s+would\b', r'\bwho\s+will\b', r'\bwho\s+did\b'
            ],
            'why': [
                r'\bwhy\s+is\b', r'\bwhy\s+are\b', r'\bwhy\s+can\b', r'\bwhy\s+do\b',
                r'\bwhy\s+should\b', r'\bwhy\s+would\b', r'\bwhy\s+will\b', r'\bwhy\s+did\b'
            ],
            'which': [
                r'\bwhich\s+is\b', r'\bwhich\s+are\b', r'\bwhich\s+one\b', r'\bwhich\s+ones\b',
                r'\bwhich\s+should\b', r'\bwhich\s+would\b', r'\bwhich\s+will\b'
            ],
            'yes_no': [
                r'\bis\s+it\b', r'\bare\s+they\b', r'\bcan\s+i\b', r'\bcan\s+you\b',
                r'\bshould\s+i\b', r'\bwould\s+it\b', r'\bwill\s+it\b', r'\bdoes\s+it\b',
                r'\bdo\s+you\b', r'\bdid\s+you\b'
            ]
        }
    
    def _load_intent_keywords(self) -> Dict[QueryIntent, List[str]]:
        """Load keywords associated with different intents."""
        return {
            QueryIntent.HELP: [
                'help', 'assist', 'support', 'guide', 'aid', 'advice', 'guidance',
                'stuck', 'confused', 'lost', 'need help', 'can you help'
            ],
            QueryIntent.PROBLEM: [
                'problem', 'issue', 'error', 'bug', 'broken', 'not working', 'failed',
                'trouble', 'difficulty', 'wrong', 'incorrect', 'fix', 'repair',
                'solve', 'resolve', 'troubleshoot'
            ],
            QueryIntent.INSTRUCTION: [
                'how to', 'step by step', 'instructions', 'guide', 'tutorial',
                'walkthrough', 'procedure', 'process', 'method', 'way to',
                'show me', 'teach me', 'explain how'
            ],
            QueryIntent.INFORMATION: [
                'information', 'details', 'about', 'regarding', 'concerning',
                'tell me', 'know about', 'learn about', 'find out', 'discover'
            ],
            QueryIntent.DEFINITION: [
                'what is', 'what are', 'define', 'definition', 'meaning',
                'means', 'explain', 'description', 'clarify', 'understand'
            ],
            QueryIntent.COMPARISON: [
                'compare', 'comparison', 'difference', 'differences', 'versus',
                'vs', 'better', 'best', 'worse', 'worst', 'which is',
                'pros and cons', 'advantages', 'disadvantages'
            ],
            QueryIntent.CONFIRMATION: [
                'confirm', 'verify', 'check', 'make sure', 'is it true',
                'correct', 'right', 'accurate', 'valid', 'confirm that'
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for entity extraction."""
        return {
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'],
            'url': [r'https?://[^\s]+', r'www\.[^\s]+'],
            'money': [r'\$\d+(?:\.\d{2})?', r'\b\d+\s*dollars?\b', r'\b\d+\s*cents?\b'],
            'date': [r'\b\d{1,2}/\d{1,2}/\d{4}\b', r'\b\d{4}-\d{2}-\d{2}\b'],
            'time': [r'\b\d{1,2}:\d{2}(?:\s*[AP]M)?\b'],
            'number': [r'\b\d+\b']
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to filter out during processing."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'me', 'my', 'you', 'your',
            'we', 'our', 'they', 'their', 'this', 'these', 'those', 'but',
            'or', 'if', 'then', 'so', 'can', 'could', 'should', 'would'
        }
    
    def extract_intent(self, query: str) -> ExtractedIntent:
        """
        Extract intent and related information from a user query.
        
        Args:
            query: The user query string
            
        Returns:
            ExtractedIntent object containing intent and related information
        """
        if not query or not query.strip():
            return ExtractedIntent(
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                keywords=[],
                entities={},
                question_type=None,
                modifiers=[]
            )
        
        query_lower = query.lower().strip()
        
        # Extract question type
        question_type = self._identify_question_type(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Determine intent
        intent, confidence = self._determine_intent(query_lower, question_type, keywords)
        
        # Extract modifiers (urgency, politeness, etc.)
        modifiers = self._extract_modifiers(query_lower)
        
        result = ExtractedIntent(
            intent=intent,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            question_type=question_type,
            modifiers=modifiers
        )
        
        logger.info(f"Intent extraction: '{query}' -> {intent.value} (confidence: {confidence:.2f})")
        
        return result
    
    def _identify_question_type(self, query: str) -> Optional[str]:
        """Identify the type of question being asked."""
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return question_type
        return None
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from the query."""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, query, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query."""
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _determine_intent(self, query: str, question_type: Optional[str], keywords: List[str]) -> Tuple[QueryIntent, float]:
        """Determine the primary intent of the query."""
        intent_scores = {}
        
        # Score based on question type
        if question_type:
            if question_type in ['what', 'which']:
                intent_scores[QueryIntent.INFORMATION] = 0.7
                intent_scores[QueryIntent.DEFINITION] = 0.6
            elif question_type == 'how':
                intent_scores[QueryIntent.INSTRUCTION] = 0.8
                intent_scores[QueryIntent.METHOD] = 0.7
            elif question_type == 'where':
                intent_scores[QueryIntent.LOCATION] = 0.8
            elif question_type == 'when':
                intent_scores[QueryIntent.TIME] = 0.8
            elif question_type == 'who':
                intent_scores[QueryIntent.PERSON] = 0.8
            elif question_type == 'why':
                intent_scores[QueryIntent.REASON] = 0.8
            elif question_type == 'yes_no':
                intent_scores[QueryIntent.CONFIRMATION] = 0.7
        
        # Score based on intent keywords
        for intent, intent_keywords in self.intent_keywords.items():
            score = 0
            for keyword in intent_keywords:
                if keyword in query:
                    score += 0.3
                    # Bonus for exact keyword matches
                    if any(kw in keyword.split() for kw in keywords):
                        score += 0.2
            
            if score > 0:
                intent_scores[intent] = intent_scores.get(intent, 0) + min(score, 1.0)
        
        # Default scoring for general patterns
        if not intent_scores:
            if any(word in query for word in ['?', 'what', 'how', 'where', 'when', 'who', 'why']):
                intent_scores[QueryIntent.QUESTION] = 0.5
            else:
                intent_scores[QueryIntent.INFORMATION] = 0.3
        
        # Find the highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], min(best_intent[1], 1.0)
        
        return QueryIntent.UNKNOWN, 0.1
    
    def _extract_modifiers(self, query: str) -> List[str]:
        """Extract modifiers that affect how the query should be handled."""
        modifiers = []
        
        # Urgency indicators
        urgency_patterns = [
            r'\burgent\b', r'\basap\b', r'\bimmediately\b', r'\bquickly\b',
            r'\bright now\b', r'\bemergency\b', r'\bhelp\b.*\bnow\b'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                modifiers.append('urgent')
                break
        
        # Politeness indicators
        politeness_patterns = [
            r'\bplease\b', r'\bthank you\b', r'\bthanks\b', r'\bkindly\b',
            r'\bwould you\b', r'\bcould you\b', r'\bif you could\b'
        ]
        
        for pattern in politeness_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                modifiers.append('polite')
                break
        
        # Uncertainty indicators
        uncertainty_patterns = [
            r'\bmaybe\b', r'\bperhaps\b', r'\bi think\b', r'\bi guess\b',
            r'\bnot sure\b', r'\bunsure\b', r'\bmight be\b'
        ]
        
        for pattern in uncertainty_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                modifiers.append('uncertain')
                break
        
        return modifiers
    
    def expand_query(self, query: str, intent: ExtractedIntent) -> List[str]:
        """
        Generate query variations for better matching.
        
        Args:
            query: Original query
            intent: Extracted intent information
            
        Returns:
            List of query variations
        """
        variations = [query]
        query_lower = query.lower()
        
        # Add variations based on intent
        if intent.intent == QueryIntent.INSTRUCTION:
            # Add "how to" variations
            if not query_lower.startswith('how to'):
                variations.append(f"how to {query}")
            variations.append(f"instructions for {query}")
            variations.append(f"steps to {query}")
        
        elif intent.intent == QueryIntent.DEFINITION:
            # Add definition variations
            if not any(word in query_lower for word in ['what is', 'define']):
                variations.append(f"what is {query}")
                variations.append(f"define {query}")
        
        elif intent.intent == QueryIntent.PROBLEM:
            # Add problem-solving variations
            variations.append(f"fix {query}")
            variations.append(f"solve {query}")
            variations.append(f"troubleshoot {query}")
        
        # Add keyword-based variations
        if intent.keywords:
            # Create variations with different keyword combinations
            main_keywords = intent.keywords[:3]  # Use top 3 keywords
            if len(main_keywords) > 1:
                variations.append(' '.join(main_keywords))
        
        # Add question mark if missing and it's a question
        if intent.question_type and not query.endswith('?'):
            variations.append(f"{query}?")
        
        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for variation in variations:
            if variation.lower() not in seen:
                seen.add(variation.lower())
                unique_variations.append(variation)
        
        return unique_variations
    
    def normalize_informal_grammar(self, query: str) -> str:
        """
        Normalize informal grammar and incomplete sentences.
        
        Args:
            query: Query with potentially informal grammar
            
        Returns:
            Normalized query string
        """
        normalized = query.strip()
        
        # Handle common informal patterns
        informal_patterns = {
            # Missing articles
            r'\b(need|want|looking for)\s+(\w+)': r'\1 a \2',
            # Missing auxiliary verbs
            r'\b(you|i|we|they)\s+(going|coming|working)': r'\1 are \2',
            # Incomplete questions
            r'^(what|how|where|when|who|why)\s+about\s+': r'\1 is ',
            # Missing "to be" verbs
            r'\b(this|that|it)\s+(good|bad|working|broken)': r'\1 is \2',
        }
        
        for pattern, replacement in informal_patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Handle sentence fragments
        if not normalized.endswith(('.', '?', '!')):
            # Add question mark if it looks like a question
            if any(word in normalized.lower() for word in ['what', 'how', 'where', 'when', 'who', 'why', 'can', 'is', 'are', 'do', 'does']):
                normalized += '?'
            else:
                normalized += '.'
        
        return normalized