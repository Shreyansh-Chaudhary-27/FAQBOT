"""
Data encryption service for the FAQ admin dashboard.

This module provides secure encryption and decryption of sensitive user data
using Django's cryptography utilities with Fernet encryption.
"""

import base64
import logging
import binascii
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive data fields.
    
    Uses Fernet encryption with keys derived from Django's SECRET_KEY
    for secure field-level encryption of user data.
    """
    
    _fernet_instance = None
    
    @classmethod
    def _get_fernet(cls) -> Fernet:
        """
        Get or create a Fernet instance with key derived from Django SECRET_KEY.
        
        Returns:
            Fernet: Configured Fernet encryption instance
            
        Raises:
            ImproperlyConfigured: If SECRET_KEY is not properly configured
        """
        if cls._fernet_instance is None:
            try:
                # Use Django's SECRET_KEY as the base for key derivation
                secret_key = settings.SECRET_KEY.encode('utf-8')
                
                # Create a salt for key derivation (static for consistency)
                salt = b'faq_admin_dashboard_salt'
                
                # Derive a proper encryption key using PBKDF2
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(secret_key))
                
                cls._fernet_instance = Fernet(key)
                
            except Exception as e:
                logger.error(f"Failed to initialize encryption service: {e}")
                raise ImproperlyConfigured(
                    "Could not initialize encryption service. "
                    "Ensure SECRET_KEY is properly configured."
                ) from e
                
        return cls._fernet_instance
    
    @classmethod
    def encrypt(cls, plaintext: str, request=None, admin_user=None) -> Optional[str]:
        """
        Encrypt a plaintext string with audit logging.
        
        Args:
            plaintext: The string to encrypt
            request: Django request object for audit logging
            admin_user: Admin user performing the operation
            
        Returns:
            str: Base64-encoded encrypted string, or None if encryption fails
        """
        if not plaintext:
            return plaintext
            
        try:
            fernet = cls._get_fernet()
            encrypted_bytes = fernet.encrypt(plaintext.encode('utf-8'))
            encrypted_result = base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
            
            # Log successful encryption operation for audit trail
            cls._log_encryption_event(
                event_type='ENCRYPTION',
                description=f'Data encryption operation completed successfully',
                request=request,
                admin_user=admin_user,
                severity='LOW',
                success=True
            )
            
            return encrypted_result
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            
            # Log failed encryption operation for security audit
            cls._log_encryption_event(
                event_type='ENCRYPTION',
                description=f'Data encryption operation failed: {str(e)}',
                request=request,
                admin_user=admin_user,
                severity='HIGH',
                success=False,
                error_details=str(e)
            )
            
            # Return None to indicate encryption failure
            return None
    
    @classmethod
    def decrypt(cls, encrypted_text: str, request=None, admin_user=None) -> Optional[str]:
        """
        Decrypt an encrypted string with audit logging.
        
        Args:
            encrypted_text: Base64-encoded encrypted string
            request: Django request object for audit logging
            admin_user: Admin user performing the operation
            
        Returns:
            str: Decrypted plaintext string, or None if decryption fails
        """
        if not encrypted_text:
            return encrypted_text
            
        try:
            fernet = cls._get_fernet()
            
            # Log the encrypted text before decoding for debugging
            logger.debug(f"Attempting to decrypt: {encrypted_text[:50]}...") # Log first 50 chars
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            decrypted_result = decrypted_bytes.decode('utf-8')
            
            # Log successful decryption operation for audit trail
            cls._log_encryption_event(
                event_type='DECRYPTION',
                description=f'Data decryption operation completed successfully',
                request=request,
                admin_user=admin_user,
                severity='LOW',
                success=True
            )
            
            return decrypted_result
            
        except binascii.Error as e:
            logger.debug(f"Decryption failed (Base64 decoding error): {e}. Encrypted text: {encrypted_text[:100]}...")
            error_details = f"Base64 decoding error: {str(e)}"
            
            cls._log_encryption_event(
                event_type='DECRYPTION',
                description=f'Data decryption operation failed: {error_details}',
                request=request,
                admin_user=admin_user,
                severity='LOW',
                success=False,
                error_details=error_details
            )
            return None
            
        except InvalidToken as e:
            logger.debug(f"Decryption failed (Invalid Fernet token - incorrect padding or key): {e}. Encrypted text: {encrypted_text[:100]}...")
            error_details = f"Invalid Fernet token: {str(e)}"
            
            cls._log_encryption_event(
                event_type='DECRYPTION',
                description=f'Data decryption operation failed: {error_details}',
                request=request,
                admin_user=admin_user,
                severity='LOW',
                success=False,
                error_details=error_details
            )
            return None
            
        except Exception as e:
            logger.debug(f"Decryption failed (General error): {e}. Encrypted text: {encrypted_text[:100]}...")
            error_details = f"General decryption error: {str(e)}"
            
            # Log failed decryption operation for security audit
            cls._log_encryption_event(
                event_type='DECRYPTION',
                description=f'Data decryption operation failed: {error_details}',
                request=request,
                admin_user=admin_user,
                severity='HIGH',
                success=False,
                error_details=error_details
            )
            
            # Return None to indicate decryption failure
            return None
    
    @classmethod
    def _log_encryption_event(cls, event_type, description, request=None, admin_user=None, 
                             severity='LOW', success=True, error_details=None):
        """
        Log encryption/decryption events to audit trail.
        
        Args:
            event_type: Type of encryption event ('ENCRYPTION' or 'DECRYPTION')
            description: Description of the event
            request: Django request object for context
            admin_user: Admin user performing the operation
            severity: Severity level of the event
            success: Whether the operation was successful
            error_details: Error details if operation failed
        """
        try:
            # Import here to avoid circular imports
            from .models import AuditLog
            
            # Prepare context data
            context_data = {
                'operation_success': success,
                'encryption_service': 'Fernet',
            }
            
            if error_details:
                context_data['error_details'] = error_details
            
            # Log the encryption/decryption event
            AuditLog.log_admin_action(
                event_type=event_type,
                description=description,
                admin_user=admin_user,
                request=request,
                severity=severity,
                context_data=context_data
            )
            
        except Exception as e:
            # If audit logging fails, log to standard logger but don't fail the operation
            logger.error(f"Failed to log encryption event to audit trail: {e}")
            # Continue with the operation - audit logging failure shouldn't break encryption