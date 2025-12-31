"""
Gemini AI-Powered Contextual Response Generator

This module provides advanced response generation using Google's Gemini AI for
contextual understanding, tone matching, and intelligent response synthesis.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from faq.rag.components.query_processor import ProcessedQuery

from faq.rag.interfaces.base import (
    ResponseGeneratorInterface, 
    FAQEntry, 
    Response, 
    ConversationContext
)
from faq.rag.components.vectorizer.gemini_service import GeminiGenerationService, GeminiServiceError
from faq.rag.components.response_generator.response_generator import BasicResponseGenerator
from faq.rag.utils.logging import get_rag_logger
from faq.rag.config.settings import rag_config


logger = get_rag_logger(__name__)


class GeminiResponseGeneratorError(Exception):
    """Custom exception for Gemini response generator errors."""
    pass


class GeminiResponseGenerator(ResponseGeneratorInterface):
    """
    Advanced response generator using Gemini AI for contextual responses.
    
    This implementation provides:
    - Context-aware response generation using Gemini AI
    - Tone and style matching capabilities
    - Intelligent multi-source synthesis
    - Fallback to basic template generation when AI is unavailable
    """
    
    def __init__(self):
        """Initialize the Gemini-powered response generator."""
        self.gemini_service = GeminiGenerationService()
        self.basic_generator = BasicResponseGenerator()  # Fallback generator
        self.context_integration_rules = {
            'max_history_items': 5  # Default value for max history items
        }
        self.prompt_templates = self._load_prompt_templates()

        logger.info("Gemini AI response generator initialized")

    def validate_candidate_relevance(self, query: str, faq: FAQEntry) -> bool:
        """
        Use Gemini to validate if a candidate FAQ is semantically relevant to the query.
        Requirement: Gemini decides if it is semantically correct for the question asked.
        """
        try:
            prompt = self.prompt_templates['semantic_validation'].format(
                query=query,
                faq_question=faq.question,
                faq_answer=faq.answer
            )
            
            response_text = self.gemini_service.generate_text(prompt, max_tokens=10).strip().upper()
            
            is_valid = "YES" in response_text
            logger.info(f"Semantic validation for query '{query}' against FAQ '{faq.question}': {is_valid} ({response_text})")
            return is_valid
            
        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            # In case of error, fall back to True to avoid losing potentially good matches 
            # (or False for strictness, but here we probably want to proceed if AI fails)
            return True 
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different response scenarios."""
        return {
            'contextual_response': """
You are an intelligent FAQ assistant. Generate a natural, helpful response based on the following information:

User Query: "{query}"
User Tone: {tone}
Context: {context}

Relevant FAQ Information:
{faq_content}

Instructions:
1. Match the user's tone and communication style
2. Provide a direct, helpful answer using the FAQ information
3. If multiple sources are provided, synthesize them coherently
4. Keep the response concise but complete
5. Maintain a {tone} tone throughout
6. If context suggests this is a follow-up question, reference previous discussion appropriately

            'semantic_validation': """
You are a semantic validator for a RAG system. Determine if the provided FAQ is a correct and appropriate answer to the user's question.

User Question: "{query}"
FAQ Question: "{faq_question}"
FAQ Answer: "{faq_answer}"

Does this FAQ accurately and directly address the user's question? 
Respond ONLY with "YES" or "NO".

Response:""",

            'multi_source_synthesis': """
You are synthesizing information from multiple FAQ sources to answer a user's question.

User Query: "{query}"
User Tone: {tone}

FAQ Sources:
{faq_sources}

Instructions:
1. Combine the information from all sources into a coherent, comprehensive answer
2. Avoid redundancy while ensuring completeness
3. Organize the information logically
4. Match the user's {tone} communication style
5. If sources conflict, acknowledge the differences
6. Prioritize the most relevant and reliable information

Synthesized Response:""",

            'context_aware_response': """
You are continuing a conversation with a user about FAQ topics.

Conversation Context:
{conversation_history}

Current Query: "{query}"
User Tone: {tone}

New FAQ Information:
{faq_content}

Instructions:
1. Reference the conversation context appropriately
2. Build upon previous exchanges naturally
3. Match the established tone and communication style
4. Provide new information while connecting to previous discussion
5. If this clarifies or expands on previous answers, make that connection clear

Contextual Response:""",

            'tone_matched_response': """
Rewrite the following FAQ response to match the user's communication style and tone.

Original FAQ Response: "{original_response}"
User Query: "{query}"
Detected User Tone: {tone}
User Style Characteristics: {style_characteristics}

Instructions:
1. Maintain all factual information from the original response
2. Adjust language formality to match the user's style
3. Use vocabulary and phrasing similar to the user's query
4. Keep the same level of detail but adapt presentation
5. Ensure the tone feels natural and appropriate

Tone-Matched Response:""",

            'uncertainty_response': """
Generate a helpful response when FAQ information doesn't fully answer the user's question.

User Query: "{query}"
User Tone: {tone}
Partial FAQ Information: {partial_info}
Confidence Level: {confidence}

Instructions:
1. Acknowledge what information is available
2. Be honest about limitations or uncertainty
3. Suggest alternative resources or next steps
4. Match the user's {tone} communication style
5. Remain helpful and supportive despite incomplete information

Response:"""
        }

    def _analyze_user_tone(self, query: str) -> str:
        """Placeholder for analyzing user tone."""
        # In a real implementation, this would use Gemini to analyze the tone
        # For now, we'll return a default tone.
        return "neutral"

    def _extract_style_characteristics(self, query: str) -> str:
        """Placeholder for extracting user style characteristics."""
        # In a real implementation, this would use Gemini to extract style characteristics
        # For now, we'll return a default.
        return "direct and informative"

    def generate_response(self, query: str, retrieved_faqs: List[FAQEntry], context: Optional[ConversationContext] = None, query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """
        Generate contextual response using Gemini AI.
        
        Args:
            query: Original user query
            retrieved_faqs: List of relevant FAQ entries
            context: Optional ConversationContext object to track conversation state.
            
        Returns:
            Response object with AI-generated contextual text
            
        Raises:
            GeminiResponseGeneratorError: If response generation fails
        """
        try:
            logger.debug(f"Generating Gemini AI response for query: '{query}' with {len(retrieved_faqs)} FAQs")
            
            # Analyze user tone and style
            user_tone = self._analyze_user_tone(query)
            style_characteristics = self._extract_style_characteristics(query)
            
            # Determine response strategy
            if not retrieved_faqs:
                # Increment no-match count if context is available
                if context:
                    context.consecutive_no_match_count += 1
                return self._generate_no_match_response_ai(query, user_tone, context, query_id, processed_query)
            else:
                # Reset no-match count on successful retrieval
                if context:
                    context.consecutive_no_match_count = 0
                if len(retrieved_faqs) == 1:
                    return self._generate_single_match_response_ai(query, retrieved_faqs[0], user_tone, style_characteristics, query_id, processed_query)
                else:
                    return self._generate_multi_source_response_ai(query, retrieved_faqs, user_tone, style_characteristics, query_id, processed_query)
                
        except GeminiServiceError as e:
            logger.warning(f"Gemini AI service error, falling back to basic generator: {e}")
            # If fallback, still consider it a "no match" for the AI, but don't increment if it's a service error
            if context and not retrieved_faqs: # Only increment if no FAQs were found AND it's an AI error
                 context.consecutive_no_match_count += 1
            return self.basic_generator.generate_response(query, retrieved_faqs, query_id=query_id, processed_query=processed_query)
        except GeminiResponseGeneratorError as e:
            # Check if this is a Gemini service error wrapped in our custom error
            if "API error" in str(e):
                logger.warning(f"Gemini AI service error, falling back to basic generator: {e}")
                if context and not retrieved_faqs:
                    context.consecutive_no_match_count += 1
                return self.basic_generator.generate_response(query, retrieved_faqs, query_id=query_id, processed_query=processed_query)
            else:
                logger.error(f"Failed to generate Gemini AI response: {e}")
                raise e
        except Exception as e:
            logger.error(f"Failed to generate Gemini AI response: {e}")
            raise GeminiResponseGeneratorError(f"AI response generation failed: {e}")
    
    def synthesize_multiple_sources(self, faqs: List[FAQEntry]) -> str:
        """
        Synthesize information from multiple FAQ sources using Gemini AI.
        
        Enhanced implementation for Requirements 5.3: Creates coherent, comprehensive
        responses that effectively combine information from multiple sources.
        
        Args:
            faqs: List of FAQ entries to synthesize
            
        Returns:
            AI-synthesized text combining information from all sources
        """
        try:
            if not faqs:
                return ""
            
            if len(faqs) == 1:
                return faqs[0].answer
            
            logger.debug(f"Performing enhanced AI synthesis of {len(faqs)} FAQ sources")
            
            # Enhanced synthesis with better prompting
            synthesis_result = self._perform_enhanced_ai_synthesis(faqs)
            
            return synthesis_result
            
        except GeminiServiceError as e:
            logger.warning(f"Gemini synthesis failed, using basic synthesis: {e}")
            return self.basic_generator.synthesize_multiple_sources(faqs)
        except Exception as e:
            logger.error(f"Failed to synthesize multiple sources: {e}")
            raise GeminiResponseGeneratorError(f"Multi-source synthesis failed: {e}")
    
    def _perform_enhanced_ai_synthesis(self, faqs: List[FAQEntry]) -> str:
        """
        Perform enhanced AI synthesis with improved prompting and analysis.
        
        This method creates more sophisticated prompts that guide the AI to:
        1. Identify complementary vs overlapping information
        2. Organize information hierarchically
        3. Create coherent narrative flow
        4. Avoid redundancy while ensuring completeness
        """
        # Analyze FAQ relationships for better synthesis
        content_analysis = self._analyze_faq_relationships(faqs)
        
        # Create enhanced synthesis prompt based on analysis
        enhanced_prompt = self._create_enhanced_synthesis_prompt(faqs, content_analysis)
        
        # Generate synthesized response with enhanced prompt
        synthesized_text = self.gemini_service.generate_response(
            enhanced_prompt, 
            max_tokens=rag_config.get_response_config()['max_response_length']
        )
        
        logger.debug(f"Enhanced AI synthesis completed for {len(faqs)} sources")
        return synthesized_text
    
    def _analyze_faq_relationships(self, faqs: List[FAQEntry]) -> Dict[str, Any]:
        """Analyze relationships between FAQs for better synthesis."""
        analysis = {
            'content_overlap': 'low',  # low, medium, high
            'information_type': 'mixed',  # complementary, overlapping, mixed
            'complexity_level': 'medium',  # low, medium, high
            'categories': list(set(faq.category for faq in faqs if faq.category)),
            'total_content_length': sum(len(faq.answer) for faq in faqs),
            'confidence_range': (min(faq.confidence_score for faq in faqs), 
                               max(faq.confidence_score for faq in faqs))
        }
        
        # Analyze content overlap using keyword similarity
        if len(faqs) >= 2:
            overlaps = []
            for i in range(len(faqs)):
                for j in range(i+1, len(faqs)):
                    overlap = self._calculate_content_overlap(faqs[i], faqs[j])
                    overlaps.append(overlap)
            
            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
            if avg_overlap > 0.7:
                analysis['content_overlap'] = 'high'
                analysis['information_type'] = 'overlapping'
            elif avg_overlap > 0.4:
                analysis['content_overlap'] = 'medium'
                analysis['information_type'] = 'mixed'
            else:
                analysis['content_overlap'] = 'low'
                analysis['information_type'] = 'complementary'
        
        # Determine complexity based on content length and structure
        if analysis['total_content_length'] > 1000:
            analysis['complexity_level'] = 'high'
        elif analysis['total_content_length'] > 400:
            analysis['complexity_level'] = 'medium'
        else:
            analysis['complexity_level'] = 'low'
        
        return analysis
    
    def _calculate_content_overlap(self, faq1: FAQEntry, faq2: FAQEntry) -> float:
        """Calculate content overlap between two FAQs."""
        # Combine keywords and answer words for analysis
        words1 = set()
        if faq1.keywords:
            words1.update(faq1.keywords)
        words1.update(faq1.answer.lower().split())
        
        words2 = set()
        if faq2.keywords:
            words2.update(faq2.keywords)
        words2.update(faq2.answer.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_enhanced_synthesis_prompt(self, faqs: List[FAQEntry], analysis: Dict[str, Any]) -> str:
        """Create enhanced synthesis prompt based on content analysis."""
        # Base synthesis prompt
        base_template = self._load_prompt_templates()['multi_source_synthesis']
        
        # Enhance prompt based on analysis
        if analysis['information_type'] == 'overlapping':
            synthesis_instructions = """
Instructions for OVERLAPPING content:
1. Identify common themes and consolidate redundant information
2. Prioritize the most accurate and comprehensive details
3. Create a unified response that eliminates repetition
4. Organize information from general to specific
5. Maintain all important details while removing redundancy
6. Use clear, coherent language that flows naturally"""
            
        elif analysis['information_type'] == 'complementary':
            synthesis_instructions = """
Instructions for COMPLEMENTARY content:
1. Organize information to build a comprehensive picture
2. Create logical flow between different pieces of information
3. Use transitional phrases to connect related concepts
4. Structure as a cohesive narrative rather than separate points
5. Ensure all unique information is preserved and integrated
6. Build from foundational concepts to more specific details"""
            
        else:  # mixed
            synthesis_instructions = """
Instructions for MIXED content:
1. First consolidate any overlapping information
2. Then integrate complementary information logically
3. Use clear section breaks or transitions between different aspects
4. Prioritize information by relevance and accuracy
5. Create a structured response that addresses all aspects
6. Maintain coherent flow while covering all important points"""
        
        # Add complexity-specific guidance
        if analysis['complexity_level'] == 'high':
            complexity_guidance = "\n7. Break complex information into digestible sections\n8. Use clear headings or bullet points for organization"
        elif analysis['complexity_level'] == 'low':
            complexity_guidance = "\n7. Keep the response concise and direct\n8. Focus on the most essential information"
        else:
            complexity_guidance = "\n7. Balance detail with clarity\n8. Organize information logically without overwhelming the reader"
        
        # Prepare FAQ sources with enhanced formatting
        faq_sources = self._format_faq_sources_for_enhanced_synthesis(faqs, analysis)
        
        # Create the enhanced prompt
        enhanced_prompt = f"""You are synthesizing information from multiple FAQ sources to create a comprehensive, coherent response.

Content Analysis:
- Information Type: {analysis['information_type']}
- Content Overlap: {analysis['content_overlap']}
- Complexity Level: {analysis['complexity_level']}
- Categories: {', '.join(analysis['categories']) if analysis['categories'] else 'General'}

{synthesis_instructions}{complexity_guidance}

FAQ Sources to Synthesize:
{faq_sources}

Create a synthesized response that combines all relevant information into a coherent, comprehensive answer:"""
        
        return enhanced_prompt
    
    def _format_faq_sources_for_enhanced_synthesis(self, faqs: List[FAQEntry], analysis: Dict[str, Any]) -> str:
        """Format FAQ sources with enhanced information for synthesis."""
        formatted_sources = []
        
        # Sort FAQs by confidence score for better synthesis
        sorted_faqs = sorted(faqs, key=lambda x: x.confidence_score, reverse=True)
        
        for i, faq in enumerate(sorted_faqs, 1):
            source = f"Source {i} (Confidence: {faq.confidence_score:.2f}):\n"
            source += f"Question: {faq.question}\n"
            source += f"Answer: {faq.answer}\n"
            source += f"Category: {faq.category or 'General'}\n"
            
            if faq.keywords:
                source += f"Key Topics: {', '.join(faq.keywords)}\n"
            
            # Add content analysis hints
            content_length = len(faq.answer.split())
            if content_length > 50:
                source += "Note: Comprehensive answer with detailed information\n"
            elif content_length < 15:
                source += "Note: Concise answer with essential information\n"
            
            formatted_sources.append(source)
        
        return "\n---\n".join(formatted_sources)
    
    def maintain_context(self, conversation_history: List[Dict[str, Any]]) -> ConversationContext:
        """
        Maintain conversation context with AI-enhanced understanding.
        
        Args:
            conversation_history: List of previous interactions
            
        Returns:
            Enhanced ConversationContext object
        """
        try:
            # Use basic context maintenance as foundation
            basic_context = self.basic_generator.maintain_context(conversation_history)
            
            # Enhance with AI-powered topic extraction and context analysis
            if conversation_history:
                enhanced_topic = self._extract_topic_with_ai(conversation_history)
                if enhanced_topic:
                    basic_context.current_topic = enhanced_topic
                
                # Analyze conversation patterns for better context
                conversation_patterns = self._analyze_conversation_patterns(conversation_history)
                basic_context.user_preferences.update(conversation_patterns)
            
            logger.debug(f"Enhanced context for session: {basic_context.session_id}")
            return basic_context
            
        except Exception as e:
            logger.warning(f"AI context enhancement failed, using basic context: {e}")
            return self.basic_generator.maintain_context(conversation_history)
    
    def calculate_confidence(self, response: Response) -> float:
        """
        Calculate confidence score with AI-enhanced analysis.
        
        Args:
            response: Response object to calculate confidence for
            
        Returns:
            Enhanced confidence score between 0.0 and 1.0
        """
        try:
            # Start with basic confidence calculation
            basic_confidence = self.basic_generator.calculate_confidence(response)
            
            # Enhance with AI-specific factors
            ai_confidence_boost = 0.0
            
            # Boost confidence for AI-generated responses
            if response.generation_method == 'rag':
                ai_confidence_boost += 0.1
            
            # Analyze response quality using AI metrics
            if response.text and len(response.text) > 50:  # Substantial response
                ai_confidence_boost += 0.05
            
            # Check for context usage
            if response.context_used:
                ai_confidence_boost += 0.1
            
            # Final confidence with AI enhancements
            final_confidence = min(1.0, basic_confidence + ai_confidence_boost)
            
            logger.debug(f"Enhanced confidence: {final_confidence:.3f} (basic: {basic_confidence:.3f})")
            return final_confidence
            
        except Exception as e:
            logger.warning(f"AI confidence enhancement failed, using basic calculation: {e}")
            return self.basic_generator.calculate_confidence(response)
    
    def generate_contextual_response(self, query: str, retrieved_faqs: List[FAQEntry],
                                   context: Optional[ConversationContext] = None,
                                   query_id: Optional[str] = None,
                                   processed_query: Optional[ProcessedQuery] = None) -> Response:
        """
        Generate response with full conversation context integration.
        
        Args:
            query: User query
            retrieved_faqs: Relevant FAQ entries
            context: Conversation context
            
        Returns:
            Context-aware response
        """
        try:
            user_tone = self._analyze_user_tone(query)
            
            # Prepare context information
            context_info = ""
            if context and context.history:
                context_info = self._format_conversation_context(context)
            
            # Prepare FAQ content
            faq_content = self._format_faq_content_for_prompt(retrieved_faqs)
            
            # Create contextual prompt
            prompt = self._load_prompt_templates()['context_aware_response'].format(
                conversation_history=context_info,
                query=query,
                tone=user_tone,
                faq_content=faq_content
            )
            
            # Generate contextual response
            response_text = self.gemini_service.generate_response(
                prompt,
                max_tokens=rag_config.get_response_config()['max_response_length']
            )
            
            # Create response object
            response = Response(
                text=response_text,
                confidence=0.8,  # High confidence for AI-generated contextual responses
                source_faqs=retrieved_faqs,
                context_used=bool(context and context.history),
                processing_time=0.0,
                generation_method='rag',
                query_id=query_id, # Use the passed query_id
                processed_query=processed_query, # Add processed_query
                metadata={
                    'ai_generated': True,
                    'context_used': bool(context),
                    'user_tone': user_tone,
                    'context_items': len(context.history) if context else 0
                }
            )
            
            # Calculate final confidence
            response.confidence = self.calculate_confidence(response)
            
            logger.debug(f"Generated contextual response with confidence: {response.confidence:.3f}")
            return response
            
        except GeminiServiceError as e:
            logger.warning(f"Contextual generation failed, using standard generation: {e}")
            return self.generate_response(query, retrieved_faqs)
        except Exception as e:
            logger.error(f"Failed to generate contextual response: {e}")
            raise GeminiResponseGeneratorError(f"Contextual response generation failed: {e}")
    

    
    def _generate_single_match_response_ai(self, query: str, faq: FAQEntry, 
                                         user_tone: str, style_characteristics: Dict[str, Any],
                                         query_id: Optional[str] = None, 
                                         processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate AI response for single FAQ match with uncertainty handling."""
        try:
            # Prepare FAQ content
            faq_content = f"Question: {faq.question}\nAnswer: {faq.answer}"
            if faq.keywords:
                faq_content += f"\nKeywords: {', '.join(faq.keywords)}"
            
            # Create contextual prompt
            prompt = self._load_prompt_templates()['contextual_response'].format(
                query=query,
                tone=user_tone,
                context="No previous context",
                faq_content=faq_content
            )
            
            # Generate response
            response_text = self.gemini_service.generate_response(
                prompt,
                max_tokens=rag_config.get_response_config()['max_response_length']
            )
            
            # Create response object
            response = Response(
                text=response_text,
                confidence=faq.confidence_score,
                source_faqs=[faq],
                context_used=False,
                processing_time=0.0,
                generation_method='rag',
                query_id=query_id or "unknown",
                processed_query=processed_query, 
                metadata={
                    'ai_generated': True,
                    'user_tone': user_tone,
                    'style_characteristics': style_characteristics,
                    'faq_category': faq.category
                }
            )
            
            response.confidence = self.calculate_confidence(response)
            
            # Apply uncertainty handling if confidence is low (Requirement 5.4)
            if response.confidence < 0.6:
                response = self.handle_low_confidence_ai_response(response, query, user_tone)
            
            return response
            
        except GeminiServiceError as e:
            logger.warning(f"Gemini service error in single match generation: {e}")
            raise e  # Re-raise to be caught by main generate_response method
        except Exception as e:
            logger.error(f"Failed to generate single match AI response: {e}")
            raise GeminiResponseGeneratorError(f"Single match AI generation failed: {e}")
    
    def _generate_multi_source_response_ai(self, query: str, faqs: List[FAQEntry], 
                                         user_tone: str, style_characteristics: Dict[str, Any],
                                         query_id: Optional[str] = None,
                                         processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate AI response for multiple FAQ matches with enhanced synthesis."""
        try:
            # Enhanced multi-source synthesis (Requirement 5.3)
            synthesized_text = self.synthesize_multiple_sources(faqs)
            
            # Create response object
            avg_confidence = sum(faq.confidence_score for faq in faqs) / len(faqs)
            
            response = Response(
                text=synthesized_text,
                confidence=avg_confidence,
                source_faqs=faqs,
                context_used=True,
                generation_method='multi_source_synthesis',
                query_id=query_id or "unknown",
                processed_query=processed_query,
                processing_time=0.0,
                metadata={
                    'ai_generated': True,
                    'multi_source': True,
                    'source_count': len(faqs),
                    'user_tone': user_tone
                }
            )
            
            response.confidence = self.calculate_confidence(response)
            
            # Apply uncertainty handling if confidence is low (Requirement 5.4)
            if response.confidence < 0.6:
                response = self.handle_low_confidence_ai_response(response, query, user_tone)
            
            return response
            
        except GeminiServiceError as e:
            logger.warning(f"Gemini service error in multi-source generation: {e}")
            raise e  # Re-raise to be caught by main generate_response method
        except Exception as e:
            logger.error(f"Failed to generate multi-source AI response: {e}")
            raise GeminiResponseGeneratorError(f"Multi-source AI generation failed: {e}")
    
    def _generate_no_match_response_ai(self, query: str, user_tone: str, context: Optional[ConversationContext] = None, query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """
        Generate an AI response when no relevant FAQs are found, with progressive messaging.
        """
        logger.debug(f"Generating no-match AI response for query: '{query}'")
        
        response_text = ""
        if context:
            if context.consecutive_no_match_count == 1:
                response_text = "Sorry, I can't understand your request. Please rephrase it."
            elif context.consecutive_no_match_count == 2:
                response_text = "There seems to be some confusion. Could you try asking in a different way?"
            else:
                response_text = "I'm having trouble understanding. For further assistance, please contact HR at example.email."
        else:
            # Default behavior if no context is provided or count is not tracked
            response_text = "Sorry, I couldn't find an answer to your question. Please try rephrasing it."

        # Attempt to use Gemini for a more nuanced response if it's the first or second time,
        # otherwise, use the hardcoded message for the third time.
        if context and context.consecutive_no_match_count < 3:
            no_match_prompt = f"""
            The user asked: "{query}"
            No relevant FAQ entries were found.
            Please generate a polite response indicating that you couldn't find an answer,
            and suggest rephrasing the question or trying a different query.
            Match the user's tone: {user_tone}.
            Current message: "{response_text}"
            """
            try:
                ai_response_text = self.gemini_service.generate_response(
                    no_match_prompt,
                    max_tokens=rag_config.get_response_config()['max_response_length']
                )
                response_text = ai_response_text
            except GeminiServiceError as e:
                logger.warning(f"Gemini service error for no-match response, falling back to basic: {e}")
                # Fallback to the hardcoded message if Gemini fails
            except Exception as e:
                logger.error(f"Failed to generate no-match AI response: {e}")
                # Fallback to the hardcoded message if Gemini fails

        return Response(
            text=response_text,
            confidence=0.1,  # Low confidence for no match
            source_faqs=[],
            context_used=False,
            generation_method='no_match_ai',
            query_id=query_id,
            processed_query=processed_query,
            processing_time=0.0,
            metadata={'ai_generated': True, 'no_match': True}
        )
    

    
    def handle_low_confidence_ai_response(self, response: Response, query: str, user_tone: str) -> Response:
        """
        Handle low confidence AI responses with enhanced uncertainty acknowledgment.
        
        Implements Requirement 5.4 for AI-generated responses.
        """
        if response.confidence >= 0.6:  # High confidence, no changes needed
            return response
        
        try:
            # Create enhanced prompt for low confidence scenarios
            enhancement_prompt = f"""You have generated a response to a user query, but the confidence level is low ({response.confidence:.2f}).

Original Query: "{query}"
User Tone: {user_tone}
Your Original Response: "{response.text}"
Confidence Level: {response.confidence:.2f}

Please enhance your response by:
1. Adding appropriate uncertainty acknowledgment
2. Maintaining the {user_tone} tone
3. Providing helpful suggestions for getting better information
4. Keeping the original helpful content while being transparent about limitations
5. Remaining supportive and solution-oriented

Enhanced Response:"""
            
            enhanced_text = self.gemini_service.generate_response(
                enhancement_prompt,
                max_tokens=rag_config.get_response_config()['max_response_length']
            )
            
            # Update response with enhanced text
            enhanced_response = Response(
                text=enhanced_text,
                confidence=response.confidence,
                source_faqs=response.source_faqs,
                context_used=response.context_used,
                generation_method=response.generation_method,
                query_id=response.query_id,
                processed_query=response.processed_query,
                processing_time=response.processing_time,
                metadata={
                    **response.metadata,
                    'uncertainty_handled': True,
                    'ai_enhanced': True,
                    'original_confidence': response.confidence,
                    'confidence_level': 'very_low' if response.confidence < 0.3 else 'low'
                }
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.warning(f"AI uncertainty enhancement failed, using basic handling: {e}")
            # Fall back to basic uncertainty handling
            return self.basic_generator.handle_low_confidence_response(response, query)
    
    def _format_faq_sources_for_synthesis(self, faqs: List[FAQEntry]) -> str:
        """Format FAQ entries for synthesis prompts."""
        formatted_sources = []
        
        for i, faq in enumerate(faqs, 1):
            source = f"Source {i}:\n"
            source += f"Question: {faq.question}\n"
            source += f"Answer: {faq.answer}\n"
            source += f"Category: {faq.category}\n"
            if faq.keywords:
                source += f"Keywords: {', '.join(faq.keywords)}\n"
            source += f"Confidence: {faq.confidence_score:.2f}\n"
            
            formatted_sources.append(source)
        
        return "\n---\n".join(formatted_sources)
    
    def _format_faq_content_for_prompt(self, faqs: List[FAQEntry]) -> str:
        """Format FAQ content for contextual prompts."""
        if not faqs:
            return "No FAQ information available"
        
        if len(faqs) == 1:
            faq = faqs[0]
            content = f"Question: {faq.question}\nAnswer: {faq.answer}"
            if faq.keywords:
                content += f"\nKeywords: {', '.join(faq.keywords)}"
            return content
        
        return self._format_faq_sources_for_synthesis(faqs)
    
    def _format_conversation_context(self, context: ConversationContext) -> str:
        """Format conversation context for prompts."""
        if not context.history:
            return "No previous conversation"
        
        # Get recent history items
        max_items = self.context_integration_rules['max_history_items']
        recent_history = context.history[-max_items:]
        
        formatted_history = []
        for i, interaction in enumerate(recent_history):
            item = f"Exchange {i+1}:\n"
            item += f"User: {interaction.get('query', 'N/A')}\n"
            item += f"Assistant: {interaction.get('response', 'N/A')[:200]}...\n"
            formatted_history.append(item)
        
        context_text = "\n".join(formatted_history)
        
        # Add current topic if available
        if context.current_topic:
            context_text = f"Current Topic: {context.current_topic}\n\n{context_text}"
        
        return context_text
    
    def _extract_topic_with_ai(self, conversation_history: List[Dict[str, Any]]) -> Optional[str]:
        """Extract conversation topic using AI analysis."""
        try:
            if not conversation_history:
                return None
            
            # Get recent queries for topic analysis
            recent_queries = [
                interaction.get('query', '') 
                for interaction in conversation_history[-3:]
                if interaction.get('query')
            ]
            
            if not recent_queries:
                return None
            
            # Simple topic extraction prompt
            queries_text = "\n".join(f"- {query}" for query in recent_queries)
            prompt = f"""
Analyze these recent user queries and identify the main topic or theme:

{queries_text}

Respond with just the main topic in 2-4 words (e.g., "account setup", "payment issues", "technical support"):"""
            
            topic = self.gemini_service.generate_response(prompt, max_tokens=20)
            
            # Clean and validate topic
            topic = topic.strip().lower()
            if len(topic) > 50 or not topic:
                return None
            
            logger.debug(f"AI extracted topic: {topic}")
            return topic
            
        except Exception as e:
            logger.debug(f"AI topic extraction failed: {e}")
            return None
    
    def _analyze_conversation_patterns(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns for user preferences."""
        patterns = {
            'preferred_tone': 'casual',
            'detail_preference': 'medium',
            'technical_level': 'medium'
        }
        
        if not conversation_history:
            return patterns
        
        # Analyze user queries for patterns
        queries = [interaction.get('query', '') for interaction in conversation_history]
        
        # Analyze tone consistency
        tones = [self._analyze_user_tone(query) for query in queries if query]
        if tones:
            # Most common tone
            tone_counts = {}
            for tone in tones:
                tone_counts[tone] = tone_counts.get(tone, 0) + 1
            patterns['preferred_tone'] = max(tone_counts, key=tone_counts.get)
        
        # Analyze query length for detail preference
        avg_length = sum(len(query.split()) for query in queries if query) / max(len(queries), 1)
        if avg_length > 15:
            patterns['detail_preference'] = 'high'
        elif avg_length < 5:
            patterns['detail_preference'] = 'low'
        
        return patterns
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Gemini response generator."""
        try:
            # Test AI service
            gemini_health = self.gemini_service.health_check()
            
            # Test basic functionality
            basic_health = self.basic_generator.get_generator_stats()
            
            return {
                'status': 'healthy' if gemini_health['status'] == 'healthy' else 'degraded',
                'ai_service': gemini_health,
                'fallback_available': True,
                'basic_generator': basic_health,
                'features': {
                    'contextual_responses': gemini_health['status'] == 'healthy',
                    'tone_matching': True,
                    'multi_source_synthesis': gemini_health['status'] == 'healthy',
                    'conversation_context': True
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'fallback_available': True
            }

    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics for health monitoring."""
        try:
            return self.basic_generator.get_generator_stats()
        except Exception:
            return {
                "total_generations": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "generator_type": "gemini_ai"
            }