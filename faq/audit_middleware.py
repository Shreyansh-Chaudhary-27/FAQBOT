"""
Audit logging middleware for the FAQ admin dashboard.

This middleware provides context-aware audit logging for encryption/decryption
operations and admin actions throughout the admin dashboard.
"""

import threading
from typing import Optional
from django.http import HttpRequest
from django.contrib.auth.models import User

# Thread-local storage for request context
_thread_local = threading.local()


class AdminAuditMiddleware:
    """
    Middleware to provide request context for audit logging throughout the admin dashboard.
    
    This middleware stores the current request and user in thread-local storage
    so that encryption/decryption operations can access audit context even when
    called from model properties or other contexts where request is not directly available.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Store request context for audit logging
        if request.path.startswith('/admin-dashboard/'):
            _thread_local.request = request
            _thread_local.admin_user = request.user if request.user.is_authenticated else None
        else:
            # Clear context for non-admin requests
            _thread_local.request = None
            _thread_local.admin_user = None
        
        try:
            response = self.get_response(request)
        finally:
            # Clean up thread-local storage after request
            _thread_local.request = None
            _thread_local.admin_user = None
        
        return response


class AuditContext:
    """
    Utility class to access audit context from anywhere in the admin dashboard.
    
    Provides static methods to get current request and admin user for audit logging
    purposes, particularly useful for encryption/decryption operations in models.
    """
    
    @staticmethod
    def get_current_request() -> Optional[HttpRequest]:
        """
        Get the current request from thread-local storage.
        
        Returns:
            HttpRequest: Current request object, or None if not in admin context
        """
        return getattr(_thread_local, 'request', None)
    
    @staticmethod
    def get_current_admin_user() -> Optional[User]:
        """
        Get the current admin user from thread-local storage.
        
        Returns:
            User: Current admin user object, or None if not authenticated or not in admin context
        """
        return getattr(_thread_local, 'admin_user', None)
    
    @staticmethod
    def is_admin_context() -> bool:
        """
        Check if we're currently in an admin dashboard context.
        
        Returns:
            bool: True if in admin dashboard context, False otherwise
        """
        request = AuditContext.get_current_request()
        return request is not None and request.path.startswith('/admin-dashboard/')
    
    @staticmethod
    def log_data_access(description: str, data_accessed: list, severity: str = 'LOW', 
                       additional_context: dict = None):
        """
        Convenience method to log data access events with current context.
        
        Args:
            description: Description of the data access event
            data_accessed: List of data types being accessed
            severity: Severity level of the event
            additional_context: Additional context data for the audit log
        """
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        if request and admin_user:
            from .models import AuditLog
            
            context_data = {
                'data_accessed': data_accessed,
                'access_method': 'model_property',
            }
            
            if additional_context:
                context_data.update(additional_context)
            
            AuditLog.log_admin_action(
                event_type='DATA_ACCESS',
                description=description,
                admin_user=admin_user,
                request=request,
                severity=severity,
                context_data=context_data
            )