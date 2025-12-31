from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, CreateView, UpdateView
from django.views import View
from django.contrib import messages
from django.urls import reverse, reverse_lazy
from django.http import HttpResponseForbidden, JsonResponse
from captcha.models import CaptchaStore
from captcha.helpers import captcha_image_url
from captcha.fields import CaptchaField
from django import forms
from .models import AuditLog, FAQEntry, RAGFAQEntry


class ReadOnlyAdminMixin:
    """
    Mixin to enforce read-only access controls for admin dashboard views.
    
    Implements Requirements 5.3:
    - Read-only access controls for viewing EndUser, UserRequest, and BotResponse data
    - Allows FAQ modification operations for knowledge base management
    - Prevents modification of user interaction data
    """
    
    def dispatch(self, request, *args, **kwargs):
        """
        Override dispatch to enforce selective read-only access controls.
        Allow modification operations only for FAQ management views.
        """
        # Check if this is an FAQ management view that allows modifications
        # Use resolver_match to check the URL name directly for reliability
        allowed_url_names = ['faq_create', 'faq_edit', 'faq_update', 'faq_delete']
        is_faq_management = (
            request.resolver_match and 
            request.resolver_match.url_name in allowed_url_names
        )
        
        # Allow all HTTP methods for FAQ management views
        if is_faq_management:
            return super().dispatch(request, *args, **kwargs)
        
        # Only allow safe HTTP methods for other views (user data is read-only)
        if request.method not in ['GET', 'HEAD']:
            # Log unauthorized modification attempt
            import logging
            security_logger = logging.getLogger('admin_security')
            security_logger.warning(
                f"Unauthorized {request.method} request to {request.path} by user {request.user.username} from IP {self.get_client_ip(request)}"
            )
            
            # Log to audit trail
            AuditLog.log_admin_action(
                event_type='SECURITY_VIOLATION',
                description=f'Unauthorized {request.method} request attempt to read-only admin dashboard section',
                admin_user=request.user if request.user.is_authenticated else None,
                request=request,
                severity='HIGH',
                context_data={
                    'violation_type': 'unauthorized_modification_attempt',
                    'attempted_method': request.method,
                    'target_path': request.path,
                    'user_authenticated': request.user.is_authenticated,
                }
            )
            
            return HttpResponseForbidden("This section of the admin dashboard only allows read-only access to data.")
        
        return super().dispatch(request, *args, **kwargs)
    
    def get_client_ip(self, request):
        """Get client IP address for security logging"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class AdminLoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    captcha = CaptchaField()


class FAQEntryForm(forms.ModelForm):
    """
    Form for creating and editing RAG FAQ entries with composite key components.
    """
    
    class Meta:
        model = RAGFAQEntry
        fields = ['question', 'answer', 'keywords', 'category', 'audience', 'intent', 'condition', 'composite_key']
        widgets = {
            'question': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter the FAQ question...',
                'required': True
            }),
            'answer': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 6,
                'placeholder': 'Enter the detailed answer...',
                'required': True
            }),
            'keywords': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter keywords separated by commas'
            }),
            'category': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., leave, payroll, policy'
            }),
            'audience': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., intern, full-time, any'
            }),
            'intent': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., info, request, policy'
            }),
            'condition': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., default, 3_months, *'
            }),
            'composite_key': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Auto-generated if left blank (audience::category::intent::condition)'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use existing fields but update labels
        self.fields['question'].label = 'Question'
        self.fields['answer'].label = 'Answer'
        self.fields['keywords'].label = 'Keywords (comma-separated)'
        self.fields['composite_key'].required = False
        
        # Add labels for new fields
        self.fields['category'].label = 'Category'
        self.fields['audience'].label = 'Audience'
        self.fields['intent'].label = 'Intent'
        self.fields['condition'].label = 'Condition'
    
    def clean_question(self):
        """Validate question field"""
        question = self.cleaned_data.get('question', '').strip()
        if not question:
            raise forms.ValidationError('Question cannot be empty.')
        if len(question) < 10:
            raise forms.ValidationError('Question must be at least 10 characters long.')
        return question
    
    def clean_answer(self):
        """Validate answer field"""
        answer = self.cleaned_data.get('answer', '').strip()
        if not answer:
            raise forms.ValidationError('Answer cannot be empty.')
        if len(answer) < 20:
            raise forms.ValidationError('Answer must be at least 20 characters long.')
        return answer
    
    def clean_keywords(self):
        """Validate and clean keywords field"""
        keywords = self.cleaned_data.get('keywords', '').strip()
        
        if not keywords:
            # Auto-generate keywords if empty
            # Import here to avoid circular dependencies
            try:
                from faq.rag.utils.text_processing import extract_keywords
                question = self.cleaned_data.get('question', '')
                answer = self.cleaned_data.get('answer', '')
                # Combine both for better context
                combined_text = f"{question} {answer}"
                extracted_list = extract_keywords(combined_text, max_keywords=8)
                keywords = ', '.join(extracted_list)
            except ImportError:
                # Fallback if utility not found
                pass
        
        if keywords:
            # Clean up keywords: remove extra spaces, normalize commas
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            if len(keyword_list) > 20:
                raise forms.ValidationError('Maximum 20 keywords allowed.')
            # Rejoin cleaned keywords
            keywords = ', '.join(keyword_list)
        
        return keywords


class AdminLoginView(View):
    """
    Admin login view with CAPTCHA protection and comprehensive error handling.
    
    Implements Requirements 1.3, 1.5, 4.3, 4.4:
    - Error message display for failed authentication
    - CAPTCHA regeneration on failed attempts
    - Session timeout handling and redirect logic
    """
    template_name = 'faq/admin/login.html'
    
    def get(self, request):
        """
        Handle GET requests to display login form.
        
        Implements session timeout redirect logic - if user was redirected here
        due to session timeout, appropriate message is already set by middleware.
        """
        if request.user.is_authenticated and request.user.is_staff:
            return redirect('admin_dashboard:dashboard')
        
        # Create fresh form with new CAPTCHA for each GET request
        form = AdminLoginForm()
        
        # Check if this is a session timeout redirect
        if 'session_timeout' in request.GET:
            messages.warning(request, 'Your session has expired due to inactivity. Please log in again.')
        
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        """
        Handle POST requests for authentication with comprehensive error handling.
        
        Implements Requirements 1.3, 4.3, 4.4:
        - Display specific error messages for different failure types
        - Regenerate CAPTCHA on failed attempts
        - Track failed login attempts for security
        """
        form = AdminLoginForm(request.POST)
        
        # Initialize error tracking
        authentication_failed = False
        captcha_failed = False
        form_validation_failed = False
        
        # Security logging
        import logging
        security_logger = logging.getLogger('admin_security')
        client_ip = self.get_client_ip(request)
        
        # Three-component validation: username, password, and CAPTCHA
        if form.is_valid():
            # All form fields are valid, now check authentication
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            # Log authentication attempt (sanitize username for logging)
            safe_username = username.encode('ascii', 'replace').decode('ascii') if username else 'unknown'
            security_logger.info(f"Authentication attempt for username '{safe_username}' from IP {client_ip}")
            
            # Authenticate user credentials
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                if user.is_staff:
                    # Successful authentication - log in the user
                    login(request, user)
                    
                    # Initialize session activity tracking for timeout handling (Requirement 1.5)
                    from django.utils import timezone
                    request.session['last_activity'] = timezone.now().isoformat()
                    request.session['login_timestamp'] = timezone.now().isoformat()
                    
                    # Clear any failed login attempt tracking
                    if 'failed_login_attempts' in request.session:
                        del request.session['failed_login_attempts']
                    
                    # Log successful authentication
                    security_logger.info(f"Successful authentication for user '{safe_username}' from IP {client_ip}")
                    
                    # Log successful login to audit trail
                    AuditLog.log_admin_action(
                        event_type='LOGIN',
                        description=f'Successful admin login for user {username}',
                        admin_user=user,
                        request=request,
                        severity='LOW',
                        context_data={
                            'username': username,
                            'login_method': 'username_password_captcha',
                            'session_id': request.session.session_key,
                        }
                    )
                    
                    # Success message and redirect
                    messages.success(request, f'Welcome back, {user.get_full_name() or username}!')
                    return redirect('admin_dashboard:dashboard')
                else:
                    # User exists but doesn't have staff permissions
                    authentication_failed = True
                    security_logger.warning(f"Authentication failed - insufficient permissions for user '{safe_username}' from IP {client_ip}")
                    
                    # Log unauthorized access attempt to audit trail
                    AuditLog.log_admin_action(
                        event_type='UNAUTHORIZED_ACCESS',
                        description=f'Login attempt by non-staff user: {username}',
                        admin_user=user,  # User exists but lacks permissions
                        request=request,
                        severity='MEDIUM',
                        context_data={
                            'username': username,
                            'reason': 'insufficient_permissions',
                            'user_is_staff': False,
                        }
                    )
                    
                    messages.error(request, 'Access denied. You do not have administrative privileges.')
            else:
                # Invalid username or password
                authentication_failed = True
                security_logger.warning(f"Authentication failed - invalid credentials for username '{safe_username}' from IP {client_ip}")
                
                # Log failed authentication attempt to audit trail
                AuditLog.log_admin_action(
                    event_type='UNAUTHORIZED_ACCESS',
                    description=f'Failed login attempt with invalid credentials for username: {username}',
                    admin_user=None,  # No valid user
                    request=request,
                    severity='MEDIUM',
                    context_data={
                        'username': username,
                        'reason': 'invalid_credentials',
                        'authentication_method': 'username_password',
                    }
                )
                
                messages.error(request, 'Invalid username or password. Please check your credentials and try again.')
        else:
            # Form validation failed
            form_validation_failed = True
            
            # Check specific validation failures for targeted error messages (Requirement 1.3)
            if 'captcha' in form.errors:
                captcha_failed = True
                security_logger.warning(f"CAPTCHA validation failed for IP {client_ip}")
                
                # Log CAPTCHA validation failure to audit trail
                AuditLog.log_admin_action(
                    event_type='SECURITY_VIOLATION',
                    description='CAPTCHA validation failed during login attempt',
                    admin_user=None,
                    request=request,
                    severity='MEDIUM',
                    context_data={
                        'username': form.data.get('username', ''),
                        'violation_type': 'captcha_failure',
                        'captcha_errors': form.errors.get('captcha', []),
                    }
                )
                
                messages.error(request, 'Security verification failed. Please enter the correct characters from the image.')
            
            if 'username' in form.errors:
                messages.error(request, 'Please enter a valid username.')
            
            if 'password' in form.errors:
                messages.error(request, 'Please enter your password.')
            
            # General form validation error if no specific errors were handled
            if not captcha_failed and not any(field in form.errors for field in ['username', 'password']):
                messages.error(request, 'Please correct the errors in the form and try again.')
        
        # Track failed login attempts for security monitoring
        if authentication_failed or captcha_failed or form_validation_failed:
            failed_attempts = request.session.get('failed_login_attempts', 0) + 1
            request.session['failed_login_attempts'] = failed_attempts
            
            # Log security event for multiple failed attempts
            if failed_attempts >= 3:
                security_logger.warning(f"Multiple failed login attempts ({failed_attempts}) from IP {client_ip}")
                messages.warning(request, 'Multiple failed login attempts detected. Please ensure you are entering the correct credentials.')
            
            # Implement progressive delay for repeated failures (basic rate limiting)
            if failed_attempts >= 5:
                security_logger.error(f"Excessive failed login attempts ({failed_attempts}) from IP {client_ip} - potential brute force attack")
                messages.error(request, 'Too many failed attempts. Please wait before trying again.')
        
        # CAPTCHA regeneration on failed attempts (Requirement 4.3, 4.4)
        # Create a new form instance to generate a fresh CAPTCHA
        # This ensures CAPTCHA is regenerated for each failed attempt
        new_form = AdminLoginForm()
        
        # Preserve form data for user convenience (except password for security)
        if hasattr(form, 'cleaned_data') and form.cleaned_data:
            new_form.initial = {
                'username': form.cleaned_data.get('username', ''),
                # Don't preserve password for security reasons
                # Don't preserve CAPTCHA as it should be regenerated
            }
        elif form.data:
            # If form validation failed, preserve username from raw data
            new_form.initial = {
                'username': form.data.get('username', ''),
            }
        
        # Add context for template to handle error states
        context = {
            'form': new_form,
            'authentication_failed': authentication_failed,
            'captcha_failed': captcha_failed,
            'failed_attempts': request.session.get('failed_login_attempts', 0),
        }
        
        return render(request, self.template_name, context)
    
    def get_client_ip(self, request):
        """Get client IP address for security logging"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


def admin_logout(request):
    """Admin logout view with audit logging"""
    # Log logout event before clearing session
    if request.user.is_authenticated:
        AuditLog.log_admin_action(
            event_type='LOGOUT',
            description=f'Admin logout for user {request.user.username}',
            admin_user=request.user,
            request=request,
            severity='LOW',
            context_data={
                'username': request.user.username,
                'session_id': request.session.session_key,
                'logout_method': 'manual',
            }
        )
    
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('admin_dashboard:login')


def refresh_captcha(request):
    """
    AJAX endpoint for refreshing CAPTCHA on failed attempts.
    
    Implements Requirements 4.3, 4.4:
    - CAPTCHA regeneration on failed attempts
    - Enhanced user experience for authentication errors
    """
    from django.http import JsonResponse
    from captcha.models import CaptchaStore
    from captcha.helpers import captcha_image_url
    import json
    
    if request.method == 'GET' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            # Generate new CAPTCHA
            new_key = CaptchaStore.generate_key()
            
            # Get the image URL for the new CAPTCHA
            image_url = captcha_image_url(new_key)
            
            # Return success response with new CAPTCHA data
            return JsonResponse({
                'success': True,
                'key': new_key,
                'image_url': image_url,
                'message': 'CAPTCHA refreshed successfully'
            })
            
        except Exception as e:
            # Log error and return failure response
            import logging
            logger = logging.getLogger('admin_security')
            logger.error(f"CAPTCHA refresh failed: {str(e)}")
            
            return JsonResponse({
                'success': False,
                'error': 'Failed to refresh CAPTCHA',
                'message': 'Please reload the page to get a new security verification'
            }, status=500)
    
    # Invalid request method or not AJAX
    return JsonResponse({
        'success': False,
        'error': 'Invalid request',
        'message': 'This endpoint only accepts AJAX GET requests'
    }, status=400)


class CategoryManagementView(LoginRequiredMixin, TemplateView):
    template_name = 'faq/admin/category_management.html'


class TagManagementView(LoginRequiredMixin, TemplateView):
    template_name = 'faq/admin/tag_management.html'


class FeedbackListView(LoginRequiredMixin, TemplateView):
    template_name = 'faq/admin/feedback_list.html'


class SettingsView(LoginRequiredMixin, TemplateView):
    template_name = 'faq/admin/settings.html'


class AdminDashboardView(LoginRequiredMixin, ReadOnlyAdminMixin, TemplateView):
    """Main admin dashboard view"""
    template_name = 'faq/admin/dashboard.html'
    login_url = '/admin-dashboard/login/'
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        """
        Add dashboard statistics and recent activity to template context.
        
        Implements dashboard home page with user statistics, navigation menu,
        and recent activity summary as required by Requirements 2.1.
        """
        context = super().get_context_data(**kwargs)
        
        # Log data access event for audit trail
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description='Admin dashboard accessed - viewing user statistics and recent activity',
            admin_user=self.request.user,
            request=self.request,
            severity='LOW',
            context_data={
                'view_name': 'AdminDashboardView',
                'data_accessed': ['user_statistics', 'recent_activity', 'system_status'],
                'access_type': 'dashboard_overview',
            }
        )
        
        # Import encrypted models for decrypted data access
        from .encrypted_models import EncryptedEndUser, EncryptedUserRequest, EncryptedBotResponse
        from .models import FAQEntry
        from django.utils import timezone
        from datetime import timedelta
        from django.db.models import Count, Q
        
        # User statistics - retrieve and display EndUser records with decrypted data
        total_users = EncryptedEndUser.objects.count()
        
        # Recent activity (last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_users = EncryptedEndUser.objects.filter(created_at__gte=week_ago).count()
        recent_requests = EncryptedUserRequest.objects.filter(created_at__gte=week_ago).count()
        recent_responses = EncryptedBotResponse.objects.filter(created_at__gte=week_ago).count()
        
        # Total interaction statistics
        total_requests = EncryptedUserRequest.objects.count()
        total_responses = EncryptedBotResponse.objects.count()
        total_faqs = FAQEntry.objects.count()
        
        # Recent activity summary - get latest interactions
        recent_interactions = EncryptedUserRequest.objects.select_related(
            'user', 'response'
        ).order_by('-created_at')[:5]
        
        # System status indicators
        # Check if there are any users without responses (pending requests)
        pending_requests = EncryptedUserRequest.objects.filter(response__isnull=True).count()
        
        # Calculate response rate
        response_rate = 0
        if total_requests > 0:
            response_rate = round((total_responses / total_requests) * 100, 1)
        
        # Most active users (users with most requests)
        active_users = EncryptedEndUser.objects.annotate(
            request_count=Count('requests')
        ).filter(request_count__gt=0).order_by('-request_count')[:5]
        
        context.update({
            # User statistics for dashboard display
            'total_users': total_users,
            'recent_users': recent_users,
            'total_requests': total_requests,
            'total_responses': total_responses,
            'total_faqs': total_faqs,
            
            # Recent activity summary
            'recent_requests': recent_requests,
            'recent_responses': recent_responses,
            'recent_interactions': recent_interactions,
            
            # System status information
            'pending_requests': pending_requests,
            'response_rate': response_rate,
            'active_users': active_users,
            
            # Navigation menu data (already handled by template)
            'current_page': 'dashboard',
        })
        
        return context


class UserHistoryView(LoginRequiredMixin, ReadOnlyAdminMixin, TemplateView):
    """
    User history view that lists all EndUser records with pagination,
    search and filtering capabilities, and displays decrypted user data.
    
    Implements Requirements 2.1, 2.3, 2.4:
    - Retrieve and display EndUser records with associated UserRequest and BotResponse data
    - Organize data showing user details, request text, bot responses, and timestamps
    - Implement pagination or filtering to manage display performance
    """
    template_name = 'faq/admin/user_history.html'
    login_url = '/admin-dashboard/login/'
    paginate_by = 20  # Number of users per page
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        """
        Add user history data with pagination, search, and filtering to template context.
        
        Implements:
        - List all EndUser records with pagination
        - Search and filtering by email and date range
        - Display user details with decrypted email and name fields
        - Show interaction counts and session information
        """
        context = super().get_context_data(**kwargs)
        
        # Get search and filter parameters for audit logging
        search_email = self.request.GET.get('search_email', '').strip()
        date_from = self.request.GET.get('date_from', '').strip()
        date_to = self.request.GET.get('date_to', '').strip()
        
        # Log data access event for audit trail
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description='User history accessed - viewing EndUser records with decrypted data',
            admin_user=self.request.user,
            request=self.request,
            severity='LOW',
            context_data={
                'view_name': 'UserHistoryView',
                'data_accessed': ['enduser_records', 'decrypted_emails', 'decrypted_names', 'interaction_counts'],
                'access_type': 'user_history_listing',
                'search_filters': {
                    'email_search': bool(search_email),
                    'date_from': bool(date_from),
                    'date_to': bool(date_to),
                },
                'page': self.request.GET.get('page', 1),
            }
        )
        
        # Import encrypted models for decrypted data access
        from .encrypted_models import EncryptedEndUser, EncryptedUserRequest, EncryptedBotResponse
        from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
        from django.db.models import Count, Q, Max
        from datetime import datetime
        
        # Get search and filter parameters
        search_email = self.request.GET.get('search_email', '').strip()
        date_from = self.request.GET.get('date_from', '').strip()
        date_to = self.request.GET.get('date_to', '').strip()
        page = self.request.GET.get('page', 1)
        
        # Start with all users, ordered by most recent activity
        users_queryset = EncryptedEndUser.objects.all().annotate(
            # Count total requests for each user
            request_count=Count('requests'),
            # Count total responses for each user
            response_count=Count('requests__response'),
            # Get the latest request timestamp
            last_activity=Max('requests__created_at')
        ).order_by('-last_activity', '-created_at')
        
        # Apply email search filter
        if search_email:
            # Search in both encrypted and potentially unencrypted email fields
            # This handles cases where data might not be encrypted yet
            users_queryset = users_queryset.filter(
                Q(email__icontains=search_email)
            )
        
        # Apply date range filters
        if date_from:
            try:
                date_from_parsed = datetime.strptime(date_from, '%Y-%m-%d').date()
                users_queryset = users_queryset.filter(created_at__date__gte=date_from_parsed)
            except ValueError:
                # Invalid date format, ignore filter
                pass
        
        if date_to:
            try:
                date_to_parsed = datetime.strptime(date_to, '%Y-%m-%d').date()
                users_queryset = users_queryset.filter(created_at__date__lte=date_to_parsed)
            except ValueError:
                # Invalid date format, ignore filter
                pass
        
        # Implement pagination for large datasets (Requirement 2.4)
        paginator = Paginator(users_queryset, self.paginate_by)
        
        try:
            users_page = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page
            users_page = paginator.page(1)
        except EmptyPage:
            # If page is out of range, deliver last page of results
            users_page = paginator.page(paginator.num_pages)
        
        # Prepare user data with decrypted fields and interaction counts
        users_data = []
        for user in users_page:
            # Get session information - latest session_id from user or requests
            latest_session = user.session_id
            if not latest_session and user.requests.exists():
                latest_request = user.requests.order_by('-created_at').first()
                latest_session = latest_request.session_id if latest_request else ''
            
            # Calculate interaction statistics
            total_requests = user.request_count or 0
            total_responses = user.response_count or 0
            
            # Get recent activity timestamp
            last_activity = user.last_activity or user.created_at
            
            users_data.append({
                'id': user.id,
                'decrypted_email': user.decrypted_email,  # Decrypted email field (Requirement 2.3)
                'decrypted_name': user.decrypted_name,    # Decrypted name field (Requirement 2.3)
                'session_id': latest_session,             # Session information
                'created_at': user.created_at,
                'last_activity': last_activity,
                'request_count': total_requests,          # Interaction counts
                'response_count': total_responses,        # Interaction counts
                'response_rate': round((total_responses / total_requests * 100) if total_requests > 0 else 0, 1)
            })
        
        # Get summary statistics for the filtered dataset
        total_users = users_queryset.count()
        total_requests_all = sum(user.request_count or 0 for user in users_queryset)
        total_responses_all = sum(user.response_count or 0 for user in users_queryset)
        
        context.update({
            'users_data': users_data,
            'users_page': users_page,
            'paginator': paginator,
            
            # Search and filter values for form persistence
            'search_email': search_email,
            'date_from': date_from,
            'date_to': date_to,
            
            # Summary statistics
            'total_users': total_users,
            'total_requests': total_requests_all,
            'total_responses': total_responses_all,
            
            # Navigation
            'current_page': 'user_history',
        })
        
        return context

class ConversationDetailView(LoginRequiredMixin, ReadOnlyAdminMixin, TemplateView):
    """
    Detailed conversation view for specific users.
    
    Implements Requirements 2.5:
    - Display complete conversation flow between users and the chatbot
    - Show UserRequest and BotResponse pairs chronologically
    - Display conversation thread visualization with timestamps
    """
    template_name = 'faq/admin/conversation_detail.html'
    login_url = '/admin-dashboard/login/'
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        """
        Add detailed conversation data for the specified user to template context.
        
        Implements:
        - Display UserRequest and BotResponse pairs chronologically
        - Show complete conversation flow with timestamps
        - Add conversation thread visualization
        """
        context = super().get_context_data(**kwargs)
        
        # Get user ID for audit logging
        user_id = self.kwargs.get('user_id')
        
        # Log data access event for audit trail
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'Conversation details accessed for user ID {user_id} - viewing decrypted conversation data',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',  # Higher severity for detailed personal data access
            context_data={
                'view_name': 'ConversationDetailView',
                'data_accessed': ['user_requests', 'bot_responses', 'decrypted_conversation_text'],
                'access_type': 'detailed_conversation_view',
                'target_user_id': user_id,
                'page': self.request.GET.get('page', 1),
            }
        )
        
        # Import encrypted models for decrypted data access
        from .encrypted_models import EncryptedEndUser, EncryptedUserRequest, EncryptedBotResponse
        from .models import BotResponse
        from django.shortcuts import get_object_or_404
        from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
        
        # Get user ID from URL parameters
        user_id = self.kwargs.get('user_id')
        
        # Get the specific user or return 404 if not found
        user = get_object_or_404(EncryptedEndUser, id=user_id)
        
        # Get pagination parameters
        page = self.request.GET.get('page', 1)
        per_page = 10  # Number of conversation pairs per page
        
        # Get all requests for this user, ordered chronologically
        # Use select_related to optimize database queries for related BotResponse
        user_requests = EncryptedUserRequest.objects.filter(
            user=user
        ).select_related('response').order_by('created_at')
        
        # Create conversation pairs (request + response) with timestamps
        conversation_pairs = []
        for request in user_requests:
            # Get the associated bot response (if it exists)
            try:
                response = request.response
                # Convert to encrypted model for decryption
                if response:
                    encrypted_response = EncryptedBotResponse.objects.get(id=response.id)
                    response_text = encrypted_response.decrypted_text
                    response_timestamp = response.created_at
                else:
                    response_text = None
                    response_timestamp = None
            except BotResponse.DoesNotExist:
                response = None
                response_text = None
                response_timestamp = None
            
            conversation_pairs.append({
                'request_id': request.id,
                'request_text': request.decrypted_text,  # Decrypted request text
                'request_timestamp': request.created_at,
                'session_id': request.session_id,
                'response_id': response.id if response else None,
                'response_text': response_text,  # Decrypted response text
                'response_timestamp': response_timestamp,
                'has_response': response is not None,
                # Calculate time between request and response for visualization
                'response_delay': (
                    (response_timestamp - request.created_at).total_seconds()
                    if response_timestamp and request.created_at else None
                )
            })
        
        # Implement pagination for large conversation histories
        paginator = Paginator(conversation_pairs, per_page)
        
        try:
            conversation_page = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page
            conversation_page = paginator.page(1)
        except EmptyPage:
            # If page is out of range, deliver last page of results
            conversation_page = paginator.page(paginator.num_pages)
        
        # Calculate conversation statistics for visualization
        total_requests = len(conversation_pairs)
        total_responses = sum(1 for pair in conversation_pairs if pair['has_response'])
        response_rate = round((total_responses / total_requests * 100) if total_requests > 0 else 0, 1)
        
        # Get unique session IDs for session-based filtering/visualization
        session_ids = list(set(pair['session_id'] for pair in conversation_pairs if pair['session_id']))
        session_ids.sort()
        
        # Calculate average response time (for conversation flow analysis)
        response_delays = [pair['response_delay'] for pair in conversation_pairs if pair['response_delay'] is not None]
        avg_response_time = (
            round(sum(response_delays) / len(response_delays), 2)
            if response_delays else None
        )
        
        # Get conversation timeline data (requests per day for visualization)
        from django.db.models import Count
        from django.db.models.functions import TruncDate
        
        timeline_data = EncryptedUserRequest.objects.filter(
            user=user
        ).extra(
            select={'date': 'date(created_at)'}
        ).values('date').annotate(
            request_count=Count('id')
        ).order_by('date')
        
        # Prepare user information with decrypted fields
        user_info = {
            'id': user.id,
            'decrypted_email': user.decrypted_email,
            'decrypted_name': user.decrypted_name,
            'session_id': user.session_id,
            'created_at': user.created_at,
            'total_requests': total_requests,
            'total_responses': total_responses,
            'response_rate': response_rate,
            'avg_response_time': avg_response_time,
            'session_count': len(session_ids),
        }
        
        context.update({
            # User information
            'user_info': user_info,
            'user': user,  # Keep original user object for template compatibility
            
            # Conversation data
            'conversation_pairs': conversation_page,
            'paginator': paginator,
            'total_conversation_pairs': total_requests,
            
            # Statistics for conversation thread visualization
            'total_requests': total_requests,
            'total_responses': total_responses,
            'response_rate': response_rate,
            'avg_response_time': avg_response_time,
            'session_ids': session_ids,
            'timeline_data': list(timeline_data),
            
            # Navigation
            'current_page': 'conversation_detail',
        })
        
        return context


class FAQManagementView(LoginRequiredMixin, ReadOnlyAdminMixin, TemplateView):
    """
    FAQ management view that displays all FAQEntry records with search and filtering.
    
    Implements Requirements 6.1, 6.2, 6.3, 6.4, 6.5:
    - Display all FAQEntry records with questions, answers, and keywords
    - Organize them in a searchable and filterable format
    - Provide read-only viewing of the current FAQ database
    - Display keywords in a readable format for content analysis
    - Display appropriate message when no FAQ entries exist
    """
    template_name = 'faq/admin/faq_management.html'
    login_url = '/admin-dashboard/login/'
    paginate_by = 25  # Number of FAQ entries per page
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        """
        Add FAQ data with search, filtering, pagination, and statistics to template context.
        
        Implements:
        - Display all FAQEntry records with questions, answers, and keywords (Requirement 6.1)
        - Searchable and filterable format (Requirement 6.2)
        - Read-only viewing of FAQ database (Requirement 6.3)
        - Keywords in readable format for content analysis (Requirement 6.4)
        - Handle empty FAQ database state (Requirement 6.5)
        """
        context = super().get_context_data(**kwargs)
        
        # Get search and filter parameters for audit logging
        search_query = self.request.GET.get('search_query', '').strip()
        keyword_filter = self.request.GET.get('keyword_filter', '').strip()
        sort_by = self.request.GET.get('sort_by', 'id').strip()
        min_answer_length = self.request.GET.get('min_answer_length', '').strip()
        has_keywords = self.request.GET.get('has_keywords', '').strip()
        export_format = self.request.GET.get('export', '').strip()
        
        # Log data access event for audit trail
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description='FAQ management accessed - viewing FAQ entries with search and filtering',
            admin_user=self.request.user,
            request=self.request,
            severity='LOW',
            context_data={
                'view_name': 'FAQManagementView',
                'data_accessed': ['faq_entries', 'faq_statistics', 'keyword_analysis'],
                'access_type': 'faq_management_listing',
                'search_filters': {
                    'search_query': bool(search_query),
                    'keyword_filter': bool(keyword_filter),
                    'sort_by': sort_by,
                    'min_answer_length': bool(min_answer_length),
                    'has_keywords': has_keywords,
                    'export_requested': bool(export_format),
                },
                'page': self.request.GET.get('page', 1),
            }
        )
        
        # Handle export requests
        if export_format in ['csv', 'json']:
            return self._handle_export(export_format, search_query, keyword_filter, 
                                     sort_by, min_answer_length, has_keywords)
        
        # Import FAQ model
        from .models import RAGFAQEntry
        from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
        from django.db.models import Q, Count
        from django.db.models.functions import Length
        import re
        
        # Start with all RAG FAQ entries
        faq_queryset = RAGFAQEntry.objects.all()
        
        # Apply search filter - search in questions and answers (Requirement 6.2)
        if search_query:
            faq_queryset = faq_queryset.filter(
                Q(question__icontains=search_query) | 
                Q(answer__icontains=search_query)
            )
        
        # Apply keyword filter (Requirement 6.2)
        if keyword_filter:
            faq_queryset = faq_queryset.filter(keywords__icontains=keyword_filter)
        
        # Apply minimum answer length filter
        if min_answer_length:
            try:
                min_length = int(min_answer_length)
                faq_queryset = faq_queryset.annotate(
                    answer_length=Length('answer')
                ).filter(answer_length__gte=min_length)
            except ValueError:
                # Invalid number, ignore filter
                pass
        
        # Apply keyword presence filter
        if has_keywords == 'yes':
            faq_queryset = faq_queryset.exclude(keywords='')
        elif has_keywords == 'no':
            faq_queryset = faq_queryset.filter(keywords='')
        
        # Apply sorting
        if sort_by == 'question':
            faq_queryset = faq_queryset.order_by('question')
        elif sort_by == 'keywords':
            faq_queryset = faq_queryset.order_by('keywords')
        elif sort_by == 'category':
            faq_queryset = faq_queryset.order_by('category')
        elif sort_by == 'composite_key':
            faq_queryset = faq_queryset.order_by('composite_key')
        elif sort_by == 'answer_length':
            faq_queryset = faq_queryset.annotate(
                answer_length=Length('answer')
            ).order_by('-answer_length')
        else:  # Default to 'id'
            faq_queryset = faq_queryset.order_by('id')
        
        # Get total count before pagination for statistics
        total_faqs = faq_queryset.count()
        
        # Implement pagination for large datasets (Requirement 6.2)
        page = self.request.GET.get('page', 1)
        paginator = Paginator(faq_queryset, self.paginate_by)
        
        try:
            faq_entries = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page
            faq_entries = paginator.page(1)
        except EmptyPage:
            # If page is out of range, deliver last page of results
            faq_entries = paginator.page(paginator.num_pages)
        
        # Process FAQ entries to add keywords list and analysis (Requirement 6.4)
        for faq in faq_entries:
            # Parse keywords into a list for readable format display
            faq.keywords_list = []
            if faq.keywords:
                # Split keywords by comma and clean up whitespace
                faq.keywords_list = [keyword.strip() for keyword in faq.keywords.split(',') if keyword.strip()]
        
        # Calculate statistics for content analysis (Requirement 6.4)
        if total_faqs > 0:
            # Get all FAQs for statistics (not just current page)
            all_faqs = RAGFAQEntry.objects.all()
            
            # Count FAQs with keywords
            faqs_with_keywords = all_faqs.exclude(keywords='').count()
            
            # Calculate average answer length
            total_answer_length = sum(len(faq.answer) for faq in all_faqs)
            avg_answer_length = round(total_answer_length / total_faqs) if total_faqs > 0 else 0
            
            # Count unique keywords
            all_keywords = set()
            for faq in all_faqs:
                if faq.keywords:
                    keywords = [keyword.strip() for keyword in faq.keywords.split(',') if keyword.strip()]
                    all_keywords.update(keywords)
            unique_keywords = len(all_keywords)
        else:
            faqs_with_keywords = 0
            avg_answer_length = 0
            unique_keywords = 0
        
        # Update context with FAQ data and statistics
        context.update({
            # FAQ entries data
            'faq_entries': faq_entries,  # Pass Page object properly for pagination
            'paginator': paginator,
            
            # Search and filter values for form persistence
            'search_query': search_query,
            'keyword_filter': keyword_filter,
            'sort_by': sort_by,
            'min_answer_length': min_answer_length,
            'has_keywords': has_keywords,
            
            # Statistics for dashboard display (Requirement 6.4)
            'total_faqs': total_faqs,
            'faqs_with_keywords': faqs_with_keywords,
            'avg_answer_length': avg_answer_length,
            'unique_keywords': unique_keywords,
            
            # Navigation
            'current_page': 'faq_management',
        })
        
        return context
    
    def _handle_export(self, export_format, search_query, keyword_filter, 
                      sort_by, min_answer_length, has_keywords):
        """
        Handle FAQ data export in CSV or JSON format.
        
        Implements export functionality for FAQ management.
        """
        from django.http import HttpResponse, JsonResponse
        from .models import RAGFAQEntry
        from django.db.models import Q
        from django.db.models.functions import Length
        import csv
        import json
        from datetime import datetime
        
        # Log export event for audit trail
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'FAQ data export requested in {export_format.upper()} format',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',  # Higher severity for data export
            context_data={
                'view_name': 'FAQManagementView',
                'data_accessed': ['faq_entries_export'],
                'access_type': 'faq_data_export',
                'export_format': export_format,
                'search_filters': {
                    'search_query': search_query,
                    'keyword_filter': keyword_filter,
                    'sort_by': sort_by,
                    'min_answer_length': min_answer_length,
                    'has_keywords': has_keywords,
                },
            }
        )
        
        # Apply same filtering logic as in get_context_data
        faq_queryset = RAGFAQEntry.objects.all()
        
        if search_query:
            faq_queryset = faq_queryset.filter(
                Q(question__icontains=search_query) | 
                Q(answer__icontains=search_query)
            )
        
        if keyword_filter:
            faq_queryset = faq_queryset.filter(keywords__icontains=keyword_filter)
        
        if min_answer_length:
            try:
                min_length = int(min_answer_length)
                faq_queryset = faq_queryset.annotate(
                    answer_length=Length('answer')
                ).filter(answer_length__gte=min_length)
            except ValueError:
                pass
        
        if has_keywords == 'yes':
            faq_queryset = faq_queryset.exclude(keywords='')
        elif has_keywords == 'no':
            faq_queryset = faq_queryset.filter(keywords='')
        
        # Apply sorting
        if sort_by == 'question':
            faq_queryset = faq_queryset.order_by('question')
        elif sort_by == 'keywords':
            faq_queryset = faq_queryset.order_by('keywords')
        elif sort_by == 'answer_length':
            faq_queryset = faq_queryset.annotate(
                answer_length=Length('answer')
            ).order_by('-answer_length')
        else:
            faq_queryset = faq_queryset.order_by('id')
        
        # Handle CSV export
        if export_format == 'csv':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="faq_entries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
            
            writer = csv.writer(response)
            writer.writerow(['ID', 'Question', 'Answer', 'Keywords', 'Category', 'Audience', 'Intent', 'Condition', 'Composite Key', 'Created At', 'Updated At'])
            
            for faq in faq_queryset:
                writer.writerow([
                    faq.id,
                    faq.question,
                    faq.answer,
                    faq.keywords,
                    faq.category,
                    faq.audience,
                    faq.intent,
                    faq.condition,
                    faq.composite_key,
                    faq.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    faq.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
                ])
            
            return response
        
        # Handle JSON export
        elif export_format == 'json':
            faq_data = []
            for faq in faq_queryset:
                keywords_list = []
                if faq.keywords:
                    keywords_list = [keyword.strip() for keyword in faq.keywords.split(',') if keyword.strip()]
                
                faq_data.append({
                    'id': faq.id,
                    'question': faq.question,
                    'answer': faq.answer,
                    'keywords': faq.keywords,
                    'category': faq.category,
                    'audience': faq.audience,
                    'intent': faq.intent,
                    'condition': faq.condition,
                    'composite_key': faq.composite_key,
                    'keywords_list': keywords_list,
                    'created_at': faq.created_at.isoformat(),
                    'updated_at': faq.updated_at.isoformat(),
                    'question_length': len(faq.question),
                    'answer_length': len(faq.answer),
                    'keyword_count': len(keywords_list),
                })
            
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_entries': len(faq_data),
                    'export_format': 'json',
                    'filters_applied': {
                        'search_query': search_query,
                        'keyword_filter': keyword_filter,
                        'sort_by': sort_by,
                        'min_answer_length': min_answer_length,
                        'has_keywords': has_keywords,
                    }
                },
                'faq_entries': faq_data
            }
            
            response = HttpResponse(
                json.dumps(export_data, indent=2),
                content_type='application/json'
            )
            response['Content-Disposition'] = f'attachment; filename="faq_entries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
            
            return response
        
        # Invalid export format
        return JsonResponse({'error': 'Invalid export format'}, status=400)


class FAQCreateView(LoginRequiredMixin, CreateView):
    """
    View for creating new FAQ entries.
    
    Implements Requirements 6.6, 6.8:
    - Validate question and answer fields and save entry to database
    - Log FAQ creation actions for audit purposes
    """
    model = RAGFAQEntry
    form_class = FAQEntryForm
    template_name = 'faq/admin/faq_form.html'
    success_url = reverse_lazy('admin_dashboard:faq_management')
    login_url = '/admin-dashboard/login/'
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def form_valid(self, form):
        """Handle successful form submission"""
        # Ensure a default 'Manual' document exists for manual entries
        from .models import RAGDocument
        manual_doc, created = RAGDocument.objects.get_or_create(
            file_name="Manual Entries",
            defaults={
                'file_path': 'manual_entry',
                'file_hash': 'manual_entry_hash',
                'file_size': 0,
                'status': 'completed'
            }
        )
        form.instance.document = manual_doc
        
        # Log FAQ creation attempt
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description='FAQ entry creation initiated',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',
            context_data={
                'view_name': 'FAQCreateView',
                'action': 'faq_creation',
                'question_length': len(form.cleaned_data.get('question', '')),
                'answer_length': len(form.cleaned_data.get('answer', '')),
                'keywords_provided': bool(form.cleaned_data.get('keywords', '')),
            }
        )
        
        response = super().form_valid(form)
        
        # Log successful FAQ creation
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'FAQ entry created successfully - ID: {self.object.id}',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',
            context_data={
                'view_name': 'FAQCreateView',
                'action': 'faq_created',
                'faq_id': self.object.id,
                'question': self.object.question[:100],  # First 100 chars for logging
                'keywords': self.object.keywords,
            }
        )
        
        messages.success(
            self.request, 
            f'FAQ entry created successfully! Question: "{self.object.question[:50]}..."'
        )
        return response
    
    def form_invalid(self, form):
        """Handle form validation errors"""
        # Log validation failure
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description='FAQ entry creation failed - validation errors',
            admin_user=self.request.user,
            request=self.request,
            severity='LOW',
            context_data={
                'view_name': 'FAQCreateView',
                'action': 'faq_creation_failed',
                'validation_errors': form.errors.as_json(),
            }
        )
        
        # Log to debug logger
        import logging
        logger = logging.getLogger('faq_admin')
        logger.error(f"FAQ creation failed: {form.errors}")
        
        messages.error(self.request, 'Please correct the errors below and try again.')
        return super().form_invalid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'page_title': 'Create New FAQ Entry',
            'form_action': 'Create',
            'current_page': 'faq_management',
        })
        return context


class FAQUpdateView(LoginRequiredMixin, UpdateView):
    """
    View for editing existing FAQ entries.
    
    Implements Requirements 6.7, 6.8:
    - Update FAQ entries while preserving creation timestamp
    - Log FAQ modification actions for audit purposes
    """
    model = RAGFAQEntry
    form_class = FAQEntryForm
    template_name = 'faq/admin/faq_form.html'
    success_url = reverse_lazy('admin_dashboard:faq_management')
    login_url = '/admin-dashboard/login/'
    context_object_name = 'faq'
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def form_valid(self, form):
        """Handle successful form submission"""
        # Store original values for audit logging
        original_faq = RAGFAQEntry.objects.get(pk=self.object.pk)
        
        # Log FAQ update attempt
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'FAQ entry update initiated - ID: {self.object.id}',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',
            context_data={
                'view_name': 'FAQUpdateView',
                'action': 'faq_update',
                'faq_id': self.object.id,
                'original_question': original_faq.question[:100],
                'new_question': form.cleaned_data.get('question', '')[:100],
                'question_changed': original_faq.question != form.cleaned_data.get('question', ''),
                'answer_changed': original_faq.answer != form.cleaned_data.get('answer', ''),
                'keywords_changed': original_faq.keywords != form.cleaned_data.get('keywords', ''),
            }
        )
        
        response = super().form_valid(form)
        
        # Log successful FAQ update
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'FAQ entry updated successfully - ID: {self.object.id}',
            admin_user=self.request.user,
            request=self.request,
            severity='MEDIUM',
            context_data={
                'view_name': 'FAQUpdateView',
                'action': 'faq_updated',
                'faq_id': self.object.id,
                'question': self.object.question[:100],
                'keywords': self.object.keywords,
            }
        )
        
        messages.success(
            self.request, 
            f'FAQ entry updated successfully! Question: "{self.object.question[:50]}..."'
        )
        return response
    
    def form_invalid(self, form):
        """Handle form validation errors"""
        # Log validation failure
        AuditLog.log_admin_action(
            event_type='DATA_ACCESS',
            description=f'FAQ entry update failed - validation errors for ID: {self.object.id}',
            admin_user=self.request.user,
            request=self.request,
            severity='LOW',
            context_data={
                'view_name': 'FAQUpdateView',
                'action': 'faq_update_failed',
                'faq_id': self.object.id,
                'validation_errors': form.errors.as_json(),
            }
        )
        
        messages.error(self.request, 'Please correct the errors below and try again.')
        return super().form_invalid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'page_title': f'Edit FAQ Entry #{self.object.id}',
            'form_action': 'Update',
            'current_page': 'faq_management',
        })
        return context


class FAQDeleteView(LoginRequiredMixin, View):
    """
    AJAX view for deleting FAQ entries.
    
    Implements FAQ deletion with audit logging.
    """
    login_url = '/admin-dashboard/login/'
    
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('admin_dashboard:login')
        return super().dispatch(request, *args, **kwargs)
    
    def post(self, request, pk):
        """Handle FAQ deletion via AJAX POST request"""
        try:
            faq = get_object_or_404(RAGFAQEntry, pk=pk)
            
            # Store FAQ details for audit logging before deletion
            faq_details = {
                'id': faq.id,
                'question': faq.question[:100],
                'answer_length': len(faq.answer),
                'keywords': faq.keywords,
                'created_at': faq.created_at.isoformat(),
            }
            
            # Log FAQ deletion attempt
            AuditLog.log_admin_action(
                event_type='DATA_ACCESS',
                description=f'FAQ entry deletion initiated - ID: {faq.id}',
                admin_user=request.user,
                request=request,
                severity='HIGH',  # Higher severity for deletion
                context_data={
                    'view_name': 'FAQDeleteView',
                    'action': 'faq_deletion',
                    'faq_details': faq_details,
                }
            )
            
            # Delete the FAQ entry
            faq.delete()
            
            # Log successful deletion
            AuditLog.log_admin_action(
                event_type='DATA_ACCESS',
                description=f'FAQ entry deleted successfully - ID: {faq_details["id"]}',
                admin_user=request.user,
                request=request,
                severity='HIGH',
                context_data={
                    'view_name': 'FAQDeleteView',
                    'action': 'faq_deleted',
                    'deleted_faq': faq_details,
                }
            )
            
            return JsonResponse({
                'success': True,
                'message': f'FAQ entry deleted successfully.',
                'faq_id': faq_details['id']
            })
            
        except RAGFAQEntry.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'FAQ entry not found.'
            }, status=404)
            
        except Exception as e:
            # Log deletion error
            AuditLog.log_admin_action(
                event_type='DATA_ACCESS',
                description=f'FAQ entry deletion failed - ID: {pk}, Error: {str(e)}',
                admin_user=request.user,
                request=request,
                severity='HIGH',
                context_data={
                    'view_name': 'FAQDeleteView',
                    'action': 'faq_deletion_failed',
                    'faq_id': pk,
                    'error': str(e),
                }
            )
            
            return JsonResponse({
                'success': False,
                'error': 'An error occurred while deleting the FAQ entry.'
            }, status=500)
