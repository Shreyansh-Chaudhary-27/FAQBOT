"""
Conversation Manager Component

This module implements conversation context management for the RAG system,
providing session tracking, context storage, and conversation history management.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid
import logging

from faq.rag.interfaces.base import (
    ConversationManagerInterface, 
    ConversationContext
)


@dataclass
class Interaction:
    """Represents a single interaction in a conversation."""
    timestamp: datetime
    query: str
    response: str
    confidence: float
    context_used: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager(ConversationManagerInterface):
    """
    Manages conversation context and session state for the RAG system.
    
    Provides session tracking, conversation history management, and automatic
    cleanup of expired sessions based on configurable timeouts.
    """
    
    def __init__(self, 
                 session_timeout_minutes: int = 30,
                 max_history_length: int = 50,
                 cleanup_interval_minutes: int = 10):
        """
        Initialize the conversation manager.
        
        Args:
            session_timeout_minutes: Minutes before a session expires
            max_history_length: Maximum number of interactions to keep in history
            cleanup_interval_minutes: How often to run cleanup (in minutes)
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_history_length = max_history_length
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        # Thread-safe storage for conversation contexts
        self._contexts: Dict[str, ConversationContext] = {}
        self._lock = threading.RLock()
        
        # Cleanup tracking
        self._last_cleanup = datetime.now()
        
        # Logger for conversation management
        self.logger = logging.getLogger(__name__)
        
    def create_session(self, session_id: Optional[str] = None) -> ConversationContext:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID. If None, generates a new UUID.
            
        Returns:
            ConversationContext: New conversation context
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        with self._lock:
            # Check if session already exists
            if session_id in self._contexts:
                self.logger.warning(f"Session {session_id} already exists, returning existing context")
                return self._contexts[session_id]
            
            # Create new conversation context
            context = ConversationContext(
                session_id=session_id,
                history=[],
                current_topic=None,
                user_preferences={},
                last_activity=datetime.now(),
                context_embeddings=[]
            )
            
            self._contexts[session_id] = context
            self.logger.info(f"Created new conversation session: {session_id}")
            
            return context
    
    def update_context(self, session_id: str, interaction: Dict[str, Any]) -> None:
        """
        Update conversation context with new interaction.
        
        Args:
            session_id: Session identifier
            interaction: Dictionary containing interaction data
                        Expected keys: query, response, confidence, context_used, metadata
        """
        with self._lock:
            context = self._contexts.get(session_id)
            if context is None:
                self.logger.warning(f"Session {session_id} not found, creating new session")
                context = self.create_session(session_id)
            
            # Create interaction object
            new_interaction = Interaction(
                timestamp=datetime.now(),
                query=interaction.get('query', ''),
                response=interaction.get('response', ''),
                confidence=interaction.get('confidence', 0.0),
                context_used=interaction.get('context_used', False),
                metadata=interaction.get('metadata', {})
            )
            
            # Add to history
            context.history.append(new_interaction.__dict__)
            
            # Maintain history length limit
            if len(context.history) > self.max_history_length:
                context.history = context.history[-self.max_history_length:]
                self.logger.debug(f"Trimmed history for session {session_id} to {self.max_history_length} items")
            
            # Update last activity
            context.last_activity = datetime.now()
            
            # Update current topic if provided in metadata
            if 'topic' in interaction.get('metadata', {}):
                context.current_topic = interaction['metadata']['topic']
            
            # Update user preferences if provided
            if 'preferences' in interaction.get('metadata', {}):
                context.user_preferences.update(interaction['metadata']['preferences'])
            
            self.logger.debug(f"Updated context for session {session_id}")
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext or None if session doesn't exist or expired
        """
        with self._lock:
            context = self._contexts.get(session_id)
            
            if context is None:
                return None
            
            # Check if session has expired
            if self._is_session_expired(context):
                self.logger.info(f"Session {session_id} has expired, removing")
                del self._contexts[session_id]
                return None
            
            return context
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired conversation sessions.
        
        Returns:
            int: Number of sessions cleaned up
        """
        current_time = datetime.now()
        
        # Only run cleanup if enough time has passed
        if current_time - self._last_cleanup < self.cleanup_interval:
            return 0
        
        with self._lock:
            expired_sessions = []
            
            for session_id, context in self._contexts.items():
                if self._is_session_expired(context):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self._contexts[session_id]
                self.logger.info(f"Cleaned up expired session: {session_id}")
            
            self._last_cleanup = current_time
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
    
    def _is_session_expired(self, context: ConversationContext) -> bool:
        """
        Check if a session has expired based on last activity.
        
        Args:
            context: Conversation context to check
            
        Returns:
            bool: True if session has expired
        """
        return datetime.now() - context.last_activity > self.session_timeout
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions.
        
        Returns:
            Dict containing session statistics
        """
        with self._lock:
            active_sessions = len(self._contexts)
            total_interactions = sum(len(ctx.history) for ctx in self._contexts.values())
            
            # Calculate average session age
            if self._contexts:
                session_ages = [
                    (datetime.now() - ctx.last_activity).total_seconds() / 60
                    for ctx in self._contexts.values()
                ]
                avg_session_age_minutes = sum(session_ages) / len(session_ages)
            else:
                avg_session_age_minutes = 0
            
            return {
                'active_sessions': active_sessions,
                'total_interactions': total_interactions,
                'average_session_age_minutes': avg_session_age_minutes,
                'session_timeout_minutes': self.session_timeout.total_seconds() / 60,
                'max_history_length': self.max_history_length
            }
    
    def reset_session(self, session_id: str) -> bool:
        """
        Reset a specific session, clearing its history and context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if session was reset, False if session didn't exist
        """
        with self._lock:
            if session_id not in self._contexts:
                return False
            
            context = self._contexts[session_id]
            context.history = []
            context.current_topic = None
            context.user_preferences = {}
            context.context_embeddings = []
            context.last_activity = datetime.now()
            
            self.logger.info(f"Reset session: {session_id}")
            return True
    
    def get_recent_interactions(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent interactions for a session.
        
        Args:
            session_id: Session identifier
            count: Number of recent interactions to return
            
        Returns:
            List of recent interactions
        """
        context = self.get_context(session_id)
        if context is None:
            return []
        
        return context.history[-count:] if context.history else []
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences for a session.
        
        Args:
            session_id: Session identifier
            preferences: Dictionary of user preferences to update
            
        Returns:
            bool: True if preferences were updated, False if session doesn't exist
        """
        with self._lock:
            context = self._contexts.get(session_id)
            if context is None:
                return False
            
            context.user_preferences.update(preferences)
            context.last_activity = datetime.now()
            
            self.logger.debug(f"Updated preferences for session {session_id}")
            return True