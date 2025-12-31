"""
Example Usage of Conversation Manager

This module demonstrates how to use the ConversationManager in a RAG system
for maintaining conversation context and session state.
"""

from datetime import datetime
from faq.rag.components.conversation_manager import ConversationManager


def example_conversation_flow():
    """
    Example of a typical conversation flow using the ConversationManager.
    """
    # Initialize conversation manager
    manager = ConversationManager(
        session_timeout_minutes=30,  # 30 minute session timeout
        max_history_length=20,       # Keep last 20 interactions
        cleanup_interval_minutes=10  # Clean up every 10 minutes
    )
    
    # Simulate a user starting a conversation
    session_id = "user-12345"
    context = manager.create_session(session_id)
    print(f"Created session: {context.session_id}")
    
    # First interaction
    interaction1 = {
        'query': 'How do I install Python?',
        'response': 'You can install Python by downloading it from python.org and following the installation instructions for your operating system.',
        'confidence': 0.95,
        'context_used': False,
        'metadata': {
            'topic': 'python_installation',
            'preferences': {'language': 'en', 'detail_level': 'beginner'}
        }
    }
    
    manager.update_context(session_id, interaction1)
    print("Added first interaction")
    
    # Follow-up question (context-aware)
    interaction2 = {
        'query': 'What about on Mac?',
        'response': 'On Mac, you can install Python using Homebrew with "brew install python" or download the installer from python.org. The Homebrew method is often preferred by developers.',
        'confidence': 0.88,
        'context_used': True,  # This response used previous context
        'metadata': {
            'topic': 'python_installation',
            'platform': 'mac'
        }
    }
    
    manager.update_context(session_id, interaction2)
    print("Added follow-up interaction")
    
    # Get current context
    current_context = manager.get_context(session_id)
    print(f"Current topic: {current_context.current_topic}")
    print(f"History length: {len(current_context.history)}")
    print(f"User preferences: {current_context.user_preferences}")
    
    # Get recent interactions for context-aware processing
    recent_interactions = manager.get_recent_interactions(session_id, count=3)
    print(f"Recent interactions: {len(recent_interactions)}")
    
    # Update user preferences
    new_preferences = {'notification_style': 'detailed', 'expertise_level': 'intermediate'}
    manager.update_user_preferences(session_id, new_preferences)
    print("Updated user preferences")
    
    # Get session statistics
    stats = manager.get_session_stats()
    print(f"Session stats: {stats}")
    
    # Simulate cleanup of expired sessions
    cleaned = manager.cleanup_expired_sessions()
    print(f"Cleaned up {cleaned} expired sessions")
    
    return manager


def example_multi_session_management():
    """
    Example of managing multiple concurrent sessions.
    """
    manager = ConversationManager()
    
    # Create multiple sessions
    sessions = []
    for i in range(3):
        session_id = f"user-{i+1}"
        context = manager.create_session(session_id)
        sessions.append(session_id)
        
        # Add some interactions to each session
        interaction = {
            'query': f'Question from user {i+1}',
            'response': f'Response for user {i+1}',
            'confidence': 0.8,
            'context_used': False,
            'metadata': {'user_id': i+1}
        }
        
        manager.update_context(session_id, interaction)
    
    print(f"Created {len(sessions)} sessions")
    
    # Get stats for all sessions
    stats = manager.get_session_stats()
    print(f"Total active sessions: {stats['active_sessions']}")
    print(f"Total interactions: {stats['total_interactions']}")
    
    # Reset one session
    manager.reset_session(sessions[0])
    print(f"Reset session: {sessions[0]}")
    
    # Verify reset
    context = manager.get_context(sessions[0])
    print(f"Session {sessions[0]} history length after reset: {len(context.history)}")
    
    return manager


if __name__ == '__main__':
    print("=== Example 1: Basic Conversation Flow ===")
    example_conversation_flow()
    
    print("\n=== Example 2: Multi-Session Management ===")
    example_multi_session_management()
    
    print("\nExamples completed successfully!")