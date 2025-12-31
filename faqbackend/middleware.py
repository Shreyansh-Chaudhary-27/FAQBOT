from django.http import HttpResponse
from django.shortcuts import redirect
from django.contrib.auth import logout
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
import logging

# Set up security logging
security_logger = logging.getLogger('admin_security')


class SimpleCORS:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse(status=204)
        else:
            response = self.get_response(request)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response


class AdminAuthMiddleware:
    """
    Middleware to protect admin dashboard routes with comprehensive authentication
    and security features.
    
    Implements Requirements 1.5, 5.2, 5.1, 5.3:
    - Automatic redirect to login for unauthenticated users
    - Session timeout handling and security logging
    - Protection of admin dashboard routes
    - Ensure separation from existing chatbot URLs
    - Implement read-only access controls for data viewing
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Session timeout in minutes (configurable)
        self.timeout_minutes = 30
        # Admin dashboard URL prefix
        self.admin_prefix = '/admin-dashboard/'
        # URLs that don't require authentication
        self.public_urls = [
            '/admin-dashboard/login/',
            '/admin-dashboard/logout/',
        ]
    
    def __call__(self, request):
        # Only apply to admin dashboard routes (Requirement 5.1: separation from chatbot URLs)
        if request.path.startswith(self.admin_prefix):
            # Ensure complete separation from existing chatbot URLs
            # Admin dashboard operates at /admin-dashboard/* while chatbot uses /api/* and /
            
            # Check if this is a public URL that doesn't require authentication
            if request.path not in self.public_urls:
                # Implement read-only access controls (Requirement 5.3)
                
                # EXCEPTION: Allow modifications for FAQ bot management (Requirement 6)
                # The admin must be able to create, update, and delete FAQs, Categories, and Tags
                is_management_path = (
                    '/admin-dashboard/faq/' in request.path or
                    '/admin-dashboard/category/' in request.path or
                    '/admin-dashboard/tag/' in request.path
                )
                
                # Only allow safe HTTP methods for data viewing (unless it's a management view)
                if request.method not in ['GET', 'HEAD'] and not is_management_path:
                    security_logger.warning(
                        f"Unauthorized {request.method} request to admin dashboard {request.path} by user {getattr(request.user, 'username', 'anonymous')} from IP {self.get_client_ip(request)}"
                    )
                    from django.http import HttpResponseForbidden
                    return HttpResponseForbidden("Admin dashboard only allows read-only access to data.")
                
                # Protect admin dashboard routes - require authentication
                if not request.user.is_authenticated:
                    # Log unauthorized access attempt
                    security_logger.warning(
                        f"Unauthorized access attempt to {request.path} from IP {self.get_client_ip(request)}"
                    )
                    
                    # Log to audit trail
                    try:
                        from faq.models import AuditLog
                        AuditLog.log_admin_action(
                            event_type='UNAUTHORIZED_ACCESS',
                            description=f'Unauthenticated access attempt to admin dashboard path: {request.path}',
                            admin_user=None,
                            request=request,
                            severity='MEDIUM',
                            context_data={
                                'access_type': 'unauthenticated',
                                'target_path': request.path,
                                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                            }
                        )
                    except Exception as e:
                        # If audit logging fails, log to security logger but don't break the flow
                        security_logger.error(f"Failed to log unauthorized access to audit trail: {e}")
                    
                    # Automatic redirect to login for unauthenticated users
                    try:
                        messages.info(request, 'Please log in to access the admin dashboard.')
                    except Exception:
                        # Messages framework not available (e.g., in tests)
                        pass
                    return redirect('admin_dashboard:login')
                
                # Check if user has staff permissions
                if not request.user.is_staff:
                    # Log insufficient permissions attempt
                    security_logger.warning(
                        f"Non-staff user {request.user.username} attempted to access {request.path} from IP {self.get_client_ip(request)}"
                    )
                    
                    # Log to audit trail
                    try:
                        from faq.models import AuditLog
                        AuditLog.log_admin_action(
                            event_type='UNAUTHORIZED_ACCESS',
                            description=f'Non-staff user attempted admin dashboard access: {request.user.username}',
                            admin_user=request.user,
                            request=request,
                            severity='MEDIUM',
                            context_data={
                                'access_type': 'insufficient_permissions',
                                'username': request.user.username,
                                'target_path': request.path,
                                'user_is_staff': False,
                            }
                        )
                    except Exception as e:
                        # If audit logging fails, log to security logger but don't break the flow
                        security_logger.error(f"Failed to log insufficient permissions to audit trail: {e}")
                    
                    # Redirect to login with error message
                    messages.error(request, 'Insufficient permissions to access admin dashboard.')
                    return redirect('admin_dashboard:login')
                
                # Enhanced session timeout handling and redirect logic (Requirements 1.5, 4.4)
                last_activity = request.session.get('last_activity')
                login_timestamp = request.session.get('login_timestamp')
                now = timezone.now()
                
                # Check for session timeout
                session_expired = False
                timeout_reason = None
                
                if last_activity:
                    try:
                        last_activity_time = timezone.datetime.fromisoformat(last_activity)
                        time_since_activity = now - last_activity_time
                        
                        if time_since_activity > timedelta(minutes=self.timeout_minutes):
                            # Session has timed out due to inactivity
                            session_expired = True
                            timeout_reason = 'inactivity'
                            security_logger.info(
                                f"Session timeout due to inactivity ({time_since_activity.total_seconds()//60:.0f} minutes) for user {request.user.username} from IP {self.get_client_ip(request)}"
                            )
                    except (ValueError, TypeError):
                        # Invalid last_activity format, treat as expired
                        session_expired = True
                        timeout_reason = 'invalid_session'
                        security_logger.warning(
                            f"Invalid session activity timestamp for user {request.user.username} - forcing logout"
                        )
                
                # Check for absolute session timeout (maximum session duration)
                if login_timestamp and not session_expired:
                    try:
                        login_time = timezone.datetime.fromisoformat(login_timestamp)
                        session_duration = now - login_time
                        max_session_duration = timedelta(hours=8)  # Maximum 8-hour sessions
                        
                        if session_duration > max_session_duration:
                            session_expired = True
                            timeout_reason = 'max_duration'
                            security_logger.info(
                                f"Session timeout due to maximum duration ({session_duration.total_seconds()//3600:.1f} hours) for user {request.user.username} from IP {self.get_client_ip(request)}"
                            )
                    except (ValueError, TypeError):
                        # Invalid login_timestamp, but don't force logout if last_activity is valid
                        pass
                
                # Handle session expiration with appropriate redirect and messaging
                if session_expired:
                    username = request.user.username  # Store before logout
                    user_for_audit = request.user  # Store user object for audit logging
                    
                    # Log session timeout to audit trail
                    try:
                        from faq.models import AuditLog
                        AuditLog.log_admin_action(
                            event_type='SESSION_TIMEOUT',
                            description=f'Admin session timeout for user {username} - reason: {timeout_reason}',
                            admin_user=user_for_audit,
                            request=request,
                            severity='LOW',
                            context_data={
                                'username': username,
                                'timeout_reason': timeout_reason,
                                'session_duration_minutes': self.timeout_minutes,
                                'logout_method': 'automatic_timeout',
                            }
                        )
                    except Exception as e:
                        # If audit logging fails, log to security logger but don't break the flow
                        security_logger.error(f"Failed to log session timeout to audit trail: {e}")
                    
                    logout(request)
                    
                    # Clear session data
                    request.session.flush()
                    
                    # Set appropriate timeout message based on reason
                    if timeout_reason == 'inactivity':
                        messages.warning(request, 'Your session has expired due to inactivity. Please log in again to continue.')
                    elif timeout_reason == 'max_duration':
                        messages.info(request, 'Your session has reached the maximum duration. Please log in again for security.')
                    else:
                        messages.warning(request, 'Your session has expired. Please log in again.')
                    
                    # Redirect to login with session timeout indicator
                    from django.urls import reverse
                    login_url = reverse('admin_dashboard:login')
                    return redirect(f"{login_url}?session_timeout=1")
                
                # Update last activity time for valid sessions
                request.session['last_activity'] = now.isoformat()
                
                # Update session metadata for monitoring
                request.session['request_count'] = request.session.get('request_count', 0) + 1
                
                # Log successful admin access for security audit
                if request.method == 'GET' and 'last_logged_access' not in request.session:
                    security_logger.info(
                        f"Admin access granted to user {request.user.username} for {request.path} from IP {self.get_client_ip(request)}"
                    )
                    # Prevent logging every request in the same session
                    request.session['last_logged_access'] = now.isoformat()
        
        response = self.get_response(request)
        return response
    
    def get_client_ip(self, request):
        """Get client IP address for security logging"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


# Keep the old middleware name for backward compatibility
# This can be removed once AdminAuthMiddleware is fully integrated
class AdminSessionTimeoutMiddleware:
    """
    Legacy middleware - replaced by AdminAuthMiddleware.
    Kept for backward compatibility during transition.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Session timeout in minutes (configurable)
        self.timeout_minutes = 30
    
    def __call__(self, request):
        # Only apply to admin dashboard routes
        if request.path.startswith('/admin-dashboard/') and request.path != '/admin-dashboard/login/':
            if request.user.is_authenticated and request.user.is_staff:
                # Check for session timeout
                last_activity = request.session.get('last_activity')
                now = timezone.now()
                
                if last_activity:
                    last_activity_time = timezone.datetime.fromisoformat(last_activity)
                    if now - last_activity_time > timedelta(minutes=self.timeout_minutes):
                        # Session has timed out
                        logout(request)
                        messages.warning(request, 'Your session has expired. Please log in again.')
                        return redirect('admin_dashboard:login')
                
                # Update last activity time
                request.session['last_activity'] = now.isoformat()
        
        response = self.get_response(request)
        return response
