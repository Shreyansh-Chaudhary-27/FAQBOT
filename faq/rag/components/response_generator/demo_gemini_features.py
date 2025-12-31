#!/usr/bin/env python
"""
Demonstration of Gemini AI-Powered Response Generator Features

This script demonstrates the enhanced capabilities of the Gemini response generator
including contextual responses, tone matching, and multi-source synthesis.
"""

import os
import sys
import django
from datetime import datetime

# Configure Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.interfaces.base import FAQEntry, ConversationContext
from faq.rag.components.response_generator.gemini_response_generator import GeminiResponseGenerator
from faq.rag.components.response_generator.response_generator import BasicResponseGenerator


def create_sample_faqs():
    """Create sample FAQ entries for demonstration."""
    return [
        FAQEntry(
            id="faq_1",
            question="How do I reset my password?",
            answer="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your email.",
            keywords=["password", "reset", "login", "email"],
            category="account",
            confidence_score=0.9,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        FAQEntry(
            id="faq_2",
            question="How do I change my password?",
            answer="You can change your password in the account settings. Navigate to Profile > Security > Change Password and enter your current password followed by your new password.",
            keywords=["password", "change", "security", "profile"],
            category="account",
            confidence_score=0.85,
            source_document="user_guide.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        FAQEntry(
            id="faq_3",
            question="What are the password requirements?",
            answer="Passwords must be at least 8 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character.",
            keywords=["password", "requirements", "security", "validation"],
            category="security",
            confidence_score=0.95,
            source_document="security_policy.docx",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]


def demonstrate_tone_matching():
    """Demonstrate tone analysis and matching capabilities."""
    print("🎭 TONE MATCHING DEMONSTRATION")
    print("=" * 50)
    
    generator = GeminiResponseGenerator()
    
    test_queries = [
        ("Could you please help me understand how to reset my password? Thank you.", "formal"),
        ("Hey, how do I reset my password?", "casual"),
        ("HELP! I can't access my account!", "urgent"),
        ("I'm confused about the password reset process", "confused"),
        ("This password thing is really frustrating me", "frustrated")
    ]
    
    for query, expected_tone in test_queries:
        detected_tone = generator._analyze_user_tone(query)
        style_chars = generator._extract_style_characteristics(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Expected Tone: {expected_tone}")
        print(f"Detected Tone: {detected_tone}")
        print(f"Style Characteristics: {style_chars}")
        print(f"Match: {'✓' if detected_tone == expected_tone else '✗'}")


def demonstrate_contextual_responses():
    """Demonstrate contextual response generation."""
    print("\n\n💬 CONTEXTUAL RESPONSE DEMONSTRATION")
    print("=" * 50)
    
    generator = GeminiResponseGenerator()
    faqs = create_sample_faqs()
    
    # Create conversation context
    conversation_history = [
        {
            "query": "I'm having trouble with my account",
            "response": "I can help you with account issues. What specific problem are you experiencing?"
        },
        {
            "query": "I forgot my password",
            "response": "I can help you reset your password. Let me provide the steps."
        }
    ]
    
    context = ConversationContext(
        session_id="demo_session",
        history=conversation_history,
        current_topic="password management",
        user_preferences={"preferred_tone": "casual"},
        last_activity=datetime.now(),
        context_embeddings=[]
    )
    
    # Test contextual response
    try:
        response = generator.generate_contextual_response(
            "How do I make sure it's secure?",
            [faqs[2]],  # Password requirements FAQ
            context
        )
        
        print(f"Query: 'How do I make sure it's secure?'")
        print(f"Context: Previous discussion about password reset")
        print(f"Response Method: {response.generation_method}")
        print(f"Context Used: {response.context_used}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Contextual response failed (expected with API limitations): {e}")


def demonstrate_multi_source_synthesis():
    """Demonstrate multi-source information synthesis."""
    print("\n\n🔄 MULTI-SOURCE SYNTHESIS DEMONSTRATION")
    print("=" * 50)
    
    generator = GeminiResponseGenerator()
    basic_generator = BasicResponseGenerator()
    faqs = create_sample_faqs()
    
    print("Query: 'Tell me about password management'")
    print("Sources: 3 FAQ entries about passwords")
    
    # Compare basic vs AI synthesis
    try:
        basic_synthesis = basic_generator.synthesize_multiple_sources(faqs)
        print(f"\nBasic Synthesis ({len(basic_synthesis)} chars):")
        print(basic_synthesis[:200] + "..." if len(basic_synthesis) > 200 else basic_synthesis)
        
        ai_synthesis = generator.synthesize_multiple_sources(faqs)
        print(f"\nAI Synthesis ({len(ai_synthesis)} chars):")
        print(ai_synthesis[:200] + "..." if len(ai_synthesis) > 200 else ai_synthesis)
        
    except Exception as e:
        print(f"AI synthesis failed (expected with API limitations): {e}")
        print("Falling back to basic synthesis...")
        basic_synthesis = basic_generator.synthesize_multiple_sources(faqs)
        print(f"Basic Synthesis: {basic_synthesis[:200]}...")


def demonstrate_confidence_calculation():
    """Demonstrate enhanced confidence calculation."""
    print("\n\n📊 CONFIDENCE CALCULATION DEMONSTRATION")
    print("=" * 50)
    
    generator = GeminiResponseGenerator()
    basic_generator = BasicResponseGenerator()
    faqs = create_sample_faqs()
    
    # Create sample responses
    responses = [
        {
            "text": "Short response",
            "method": "direct_match",
            "context_used": False,
            "source_faqs": [faqs[0]]
        },
        {
            "text": "This is a longer, more detailed response that provides comprehensive information about the topic.",
            "method": "rag",
            "context_used": True,
            "source_faqs": faqs[:2]
        },
        {
            "text": "Fallback response when no information is available",
            "method": "fallback",
            "context_used": False,
            "source_faqs": []
        }
    ]
    
    for i, resp_data in enumerate(responses, 1):
        from faq.rag.interfaces.base import Response
        
        response = Response(
            text=resp_data["text"],
            confidence=0.7,  # Base confidence
            source_faqs=resp_data["source_faqs"],
            context_used=resp_data["context_used"],
            generation_method=resp_data["method"],
            metadata={"ai_generated": True}
        )
        
        basic_confidence = basic_generator.calculate_confidence(response)
        ai_confidence = generator.calculate_confidence(response)
        
        print(f"\nResponse {i}:")
        print(f"  Method: {resp_data['method']}")
        print(f"  Context Used: {resp_data['context_used']}")
        print(f"  Sources: {len(resp_data['source_faqs'])}")
        print(f"  Basic Confidence: {basic_confidence:.3f}")
        print(f"  AI Enhanced Confidence: {ai_confidence:.3f}")
        print(f"  Improvement: {ai_confidence - basic_confidence:+.3f}")


def demonstrate_health_monitoring():
    """Demonstrate health check and monitoring capabilities."""
    print("\n\n🏥 HEALTH MONITORING DEMONSTRATION")
    print("=" * 50)
    
    generator = GeminiResponseGenerator()
    
    health = generator.health_check()
    
    print("System Health Status:")
    print(f"  Overall Status: {health['status']}")
    print(f"  AI Service Available: {health.get('ai_service', {}).get('status', 'unknown')}")
    print(f"  Fallback Available: {health['fallback_available']}")
    
    if 'features' in health:
        print("  Available Features:")
        for feature, available in health['features'].items():
            status = "✓" if available else "✗"
            print(f"    {status} {feature.replace('_', ' ').title()}")


def main():
    """Run all demonstrations."""
    print("🤖 GEMINI AI RESPONSE GENERATOR DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the enhanced capabilities of the Gemini AI-powered")
    print("response generator compared to the basic template-based generator.")
    print()
    
    try:
        demonstrate_tone_matching()
        demonstrate_contextual_responses()
        demonstrate_multi_source_synthesis()
        demonstrate_confidence_calculation()
        demonstrate_health_monitoring()
        
        print("\n\n✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The Gemini AI response generator provides:")
        print("• Intelligent tone analysis and matching")
        print("• Context-aware response generation")
        print("• Advanced multi-source synthesis")
        print("• Enhanced confidence scoring")
        print("• Robust fallback mechanisms")
        print("• Comprehensive health monitoring")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()