"""
Encrypted proxy models for the FAQ admin dashboard.

This module provides proxy models that automatically decrypt sensitive fields
for display in the admin dashboard while maintaining the encrypted storage
in the database.
"""

from typing import Optional

from .encryption import EncryptionService
from .models import EndUser, UserRequest, BotResponse
from .audit_middleware import AuditContext


class EncryptedEndUser(EndUser):
    """
    Proxy model for EndUser with automatic decryption of sensitive fields.
    
    Provides decrypted access to email and name fields while maintaining
    the encrypted storage in the database.
    """
    
    class Meta:
        proxy = True
    
    @property
    def decrypted_email(self) -> Optional[str]:
        """
        Get the decrypted email address with audit logging.
        
        Returns:
            str: Decrypted email address, or original value if decryption fails
        """
        if not self.email:
            return self.email
        
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Decrypt with audit logging
        decrypted = EncryptionService.decrypt(
            self.email, 
            request=request, 
            admin_user=admin_user
        )
        
        # Log data access if in admin context
        if AuditContext.is_admin_context():
            AuditContext.log_data_access(
                description=f'Decrypted email field for EndUser ID {self.id}',
                data_accessed=['enduser_email'],
                severity='MEDIUM',
                additional_context={
                    'user_id': self.id,
                    'field_type': 'email',
                    'decryption_success': decrypted is not None,
                }
            )
        
        # Return original value if decryption fails (for backwards compatibility)
        return decrypted if decrypted is not None else self.email
    
    @property
    def decrypted_name(self) -> Optional[str]:
        """
        Get the decrypted name with audit logging.
        
        Returns:
            str: Decrypted name, or original value if decryption fails
        """
        if not self.name:
            return self.name
        
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Decrypt with audit logging
        decrypted = EncryptionService.decrypt(
            self.name, 
            request=request, 
            admin_user=admin_user
        )
        
        # Log data access if in admin context
        if AuditContext.is_admin_context():
            AuditContext.log_data_access(
                description=f'Decrypted name field for EndUser ID {self.id}',
                data_accessed=['enduser_name'],
                severity='MEDIUM',
                additional_context={
                    'user_id': self.id,
                    'field_type': 'name',
                    'decryption_success': decrypted is not None,
                }
            )
        
        # Return original value if decryption fails (for backwards compatibility)
        return decrypted if decrypted is not None else self.name
    
    def save(self, *args, **kwargs):
        """
        Override save to encrypt sensitive fields before storage.
        """
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Encrypt email if it's not already encrypted
        if self.email and not self._is_encrypted(self.email):
            encrypted_email = EncryptionService.encrypt(
                self.email, 
                request=request, 
                admin_user=admin_user
            )
            if encrypted_email is not None:
                self.email = encrypted_email
        
        # Encrypt name if it's not already encrypted
        if self.name and not self._is_encrypted(self.name):
            encrypted_name = EncryptionService.encrypt(
                self.name, 
                request=request, 
                admin_user=admin_user
            )
            if encrypted_name is not None:
                self.name = encrypted_name
        
        super().save(*args, **kwargs)
    
    def _is_encrypted(self, value: str) -> bool:
        """
        Check if a value appears to be encrypted.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value appears to be encrypted
        """
        if not value:
            return False
            
        # If it looks like a normal email or name, it's probably not encrypted
        if '@' in value and '.' in value and len(value) < 100:
            return False  # Looks like an email
        if value.replace(' ', '').replace('-', '').replace("'", '').isalpha() and len(value) < 100:
            return False  # Looks like a name
            
        try:
            import base64
            # Check if it's valid base64
            decoded = base64.urlsafe_b64decode(value.encode('utf-8'))
            # Our encryption format: base64(fernet_encrypted_data)
            # Fernet tokens are at least 60 bytes long when base64 decoded
            # and the base64 encoded version is much longer
            return len(decoded) >= 60 and len(value) > 80
        except Exception:
            return False


class EncryptedUserRequest(UserRequest):
    """
    Proxy model for UserRequest with automatic decryption of text field.
    
    Provides decrypted access to the request text while maintaining
    the encrypted storage in the database.
    """
    
    class Meta:
        proxy = True
    
    @property
    def decrypted_text(self) -> Optional[str]:
        """
        Get the decrypted request text with audit logging.
        
        Returns:
            str: Decrypted request text, or original value if decryption fails
        """
        if not self.text:
            return self.text
        
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Decrypt with audit logging
        decrypted = EncryptionService.decrypt(
            self.text, 
            request=request, 
            admin_user=admin_user
        )
        
        # Log data access if in admin context
        if AuditContext.is_admin_context():
            AuditContext.log_data_access(
                description=f'Decrypted request text for UserRequest ID {self.id}',
                data_accessed=['user_request_text'],
                severity='MEDIUM',
                additional_context={
                    'request_id': self.id,
                    'user_id': self.user_id,
                    'field_type': 'request_text',
                    'decryption_success': decrypted is not None,
                }
            )
        
        # Return original value if decryption fails (for backwards compatibility)
        return decrypted if decrypted is not None else self.text
    
    def save(self, *args, **kwargs):
        """
        Override save to encrypt text field before storage.
        """
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Encrypt text if it's not already encrypted
        if self.text and not self._is_encrypted(self.text):
            encrypted_text = EncryptionService.encrypt(
                self.text, 
                request=request, 
                admin_user=admin_user
            )
            if encrypted_text is not None:
                self.text = encrypted_text
        
        super().save(*args, **kwargs)
    
    def _is_encrypted(self, value: str) -> bool:
        """
        Check if a value appears to be encrypted.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value appears to be encrypted
        """
        if not value:
            return False
            
        try:
            import base64
            # Check if it's valid base64
            decoded = base64.urlsafe_b64decode(value.encode('utf-8'))
            # Our encryption format: base64(fernet_encrypted_data)
            # Fernet tokens are at least 60 bytes long when base64 decoded
            # and the base64 encoded version is much longer
            return len(decoded) >= 60 and len(value) > 80
        except Exception:
            return False


class EncryptedBotResponse(BotResponse):
    """
    Proxy model for BotResponse with automatic decryption of text field.
    
    Provides decrypted access to the response text while maintaining
    the encrypted storage in the database.
    """
    
    class Meta:
        proxy = True
    
    @property
    def decrypted_text(self) -> Optional[str]:
        """
        Get the decrypted response text with audit logging.
        
        Returns:
            str: Decrypted response text, or original value if decryption fails
        """
        if not self.text:
            return self.text
        
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Decrypt with audit logging
        decrypted = EncryptionService.decrypt(
            self.text, 
            request=request, 
            admin_user=admin_user
        )
        
        # Log data access if in admin context
        if AuditContext.is_admin_context():
            AuditContext.log_data_access(
                description=f'Decrypted response text for BotResponse ID {self.id}',
                data_accessed=['bot_response_text'],
                severity='MEDIUM',
                additional_context={
                    'response_id': self.id,
                    'request_id': self.request_id,
                    'field_type': 'response_text',
                    'decryption_success': decrypted is not None,
                }
            )
        
        # Return original value if decryption fails (for backwards compatibility)
        return decrypted if decrypted is not None else self.text
    
    def save(self, *args, **kwargs):
        """
        Override save to encrypt text field before storage.
        """
        # Get audit context for logging
        request = AuditContext.get_current_request()
        admin_user = AuditContext.get_current_admin_user()
        
        # Encrypt text if it's not already encrypted
        if self.text and not self._is_encrypted(self.text):
            encrypted_text = EncryptionService.encrypt(
                self.text, 
                request=request, 
                admin_user=admin_user
            )
            if encrypted_text is not None:
                self.text = encrypted_text
        
        super().save(*args, **kwargs)
    
    def _is_encrypted(self, value: str) -> bool:
        """
        Check if a value appears to be encrypted.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value appears to be encrypted
        """
        if not value:
            return False
            
        try:
            import base64
            # Check if it's valid base64
            decoded = base64.urlsafe_b64decode(value.encode('utf-8'))
            # Our encryption format: base64(fernet_encrypted_data)
            # Fernet tokens are at least 60 bytes long when base64 decoded
            # and the base64 encoded version is much longer
            return len(decoded) >= 60 and len(value) > 80
        except Exception:
            return False