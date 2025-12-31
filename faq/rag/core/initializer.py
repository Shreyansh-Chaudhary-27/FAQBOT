"""
RAG System Initializer

This module provides initialization utilities for the RAG system,
including component setup, configuration validation, and system startup.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..interfaces.base import RAGSystemInterface
from ..config.settings import rag_config
from ..utils.logging import get_rag_logger
from .factory import rag_factory


class RAGSystemInitializer:
    """
    Handles initialization and setup of the RAG system with comprehensive
    configuration validation and component verification.
    """
    
    def __init__(self):
        """Initialize the RAG system initializer."""
        self.logger = get_rag_logger('rag_initializer')
        self.config = rag_config.config
        
    def initialize_system(self, 
                         custom_components: Optional[Dict[str, Any]] = None,
                         validate_config: bool = True,
                         perform_health_check: bool = True) -> RAGSystemInterface:
        """
        Initialize a complete RAG system with validation and health checks.
        
        Args:
            custom_components: Optional dictionary of custom component implementations
            validate_config: Whether to validate configuration before initialization
            perform_health_check: Whether to perform health check after initialization
            
        Returns:
            RAGSystemInterface: Initialized RAG system
            
        Raises:
            RAGInitializationError: If initialization fails
        """
        self.logger.info("Starting RAG system initialization...")
        
        try:
            # Step 1: Validate configuration
            if validate_config:
                self._validate_configuration()
            
            # Step 2: Create RAG system with components
            if custom_components:
                rag_system = self._create_custom_system(custom_components)
            else:
                rag_system = rag_factory.create_default_system()
            
            # Step 3: Perform health check
            if perform_health_check:
                self._perform_initialization_health_check(rag_system)
            
            # Step 4: Log initialization success
            self._log_initialization_success(rag_system)
            
            return rag_system
            
        except Exception as e:
            self.logger.error(f"RAG system initialization failed: {e}")
            raise RAGInitializationError(f"Initialization failed: {e}")
    
    def _validate_configuration(self) -> None:
        """Validate RAG system configuration."""
        self.logger.info("Validating RAG system configuration...")
        
        validation_errors = []
        
        # Validate required configuration values
        required_configs = [
            ('similarity_threshold', float, 0.0, 1.0),
            ('max_results', int, 1, 100),
            ('vector_dimension', int, 1, 10000),
            ('session_timeout_minutes', int, 1, 1440),  # 1 minute to 24 hours
            ('max_conversation_history', int, 1, 1000)
        ]
        
        for config_name, expected_type, min_val, max_val in required_configs:
            if not hasattr(self.config, config_name):
                validation_errors.append(f"Missing required configuration: {config_name}")
                continue
            
            value = getattr(self.config, config_name)
            
            if not isinstance(value, expected_type):
                validation_errors.append(f"Configuration {config_name} must be {expected_type.__name__}, got {type(value).__name__}")
                continue
            
            if not (min_val <= value <= max_val):
                validation_errors.append(f"Configuration {config_name} must be between {min_val} and {max_val}, got {value}")
        
        # Validate Gemini API key
        if not self.config.gemini_api_key:
            validation_errors.append("Gemini API key is required but not configured")
        
        # Validate file paths and permissions
        try:
            # Check if we can create directories for vector store
            test_path = Path(self.config.vector_store_path)
            test_path.mkdir(exist_ok=True, parents=True)
            self.logger.debug("Vector store directory validation passed")
        except Exception as e:
            validation_errors.append(f"Cannot create vector store directory: {e}")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise RAGInitializationError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def _create_custom_system(self, custom_components: Dict[str, Any]) -> RAGSystemInterface:
        """Create RAG system with custom components."""
        self.logger.info("Creating RAG system with custom components")
        
        # Validate custom components
        valid_components = [
            'docx_scraper', 'query_processor', 'vectorizer',
            'vector_store', 'response_generator', 'conversation_manager'
        ]
        
        for component_name in custom_components:
            if component_name not in valid_components:
                raise RAGInitializationError(f"Invalid component name: {component_name}")
        
        # Create system with custom components
        return rag_factory.create_rag_system(**custom_components)
    
    def _perform_initialization_health_check(self, rag_system: RAGSystemInterface) -> None:
        """Perform health check on initialized system."""
        self.logger.info("Performing initialization health check...")
        
        try:
            health_results = rag_system.health_check()
            
            if health_results['overall_status'] == 'unhealthy':
                issues = health_results.get('issues', [])
                error_msg = f"System health check failed: {', '.join(issues)}"
                raise RAGInitializationError(error_msg)
            
            elif health_results['overall_status'] == 'degraded':
                issues = health_results.get('issues', [])
                self.logger.warning(f"System initialized with warnings: {', '.join(issues)}")
            
            else:
                self.logger.info("System health check passed")
                
        except Exception as e:
            raise RAGInitializationError(f"Health check failed: {e}")
    
    def _log_initialization_success(self, rag_system: RAGSystemInterface) -> None:
        """Log successful initialization with system information."""
        try:
            component_status = rag_system.get_component_status()
            available_components = [name for name, available in component_status.items() if available]
            
            self.logger.info(f"RAG system initialized successfully with {len(available_components)} components:")
            for component in available_components:
                self.logger.info(f"  ✓ {component}")
            
            # Log configuration summary
            self.logger.info(f"Configuration: similarity_threshold={self.config.similarity_threshold}, "
                           f"max_results={self.config.max_results}, "
                           f"vector_dimension={self.config.vector_dimension}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log initialization details: {e}")
    
    def quick_start(self) -> RAGSystemInterface:
        """
        Quick start method for development and testing.
        
        Returns:
            RAGSystemInterface: Basic RAG system for immediate use
        """
        self.logger.info("Quick starting RAG system...")
        
        try:
            # Create system with minimal validation
            rag_system = rag_factory.create_default_system()
            
            # Basic readiness check
            if not rag_system.is_ready():
                self.logger.warning("RAG system may not be fully functional")
            
            self.logger.info("RAG system quick start completed")
            return rag_system
            
        except Exception as e:
            self.logger.error(f"Quick start failed: {e}")
            raise RAGInitializationError(f"Quick start failed: {e}")
    
    def create_minimal_system(self) -> RAGSystemInterface:
        """
        Create a minimal RAG system with only essential components.
        
        Returns:
            RAGSystemInterface: Minimal RAG system
        """
        self.logger.info("Creating minimal RAG system...")
        
        try:
            # Create only essential components
            from ..components.response_generator.response_generator import BasicResponseGenerator
            
            minimal_components = {
                'response_generator': BasicResponseGenerator()
            }
            
            rag_system = rag_factory.create_rag_system(**minimal_components)
            
            self.logger.info("Minimal RAG system created")
            return rag_system
            
        except Exception as e:
            self.logger.error(f"Minimal system creation failed: {e}")
            raise RAGInitializationError(f"Minimal system creation failed: {e}")
    
    def validate_system_requirements(self) -> Dict[str, Any]:
        """
        Validate system requirements and dependencies.
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating system requirements...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'valid',
            'requirements': {},
            'warnings': [],
            'errors': []
        }
        
        # Check Python dependencies
        required_packages = [
            'numpy', 'scikit-learn', 'python-docx', 
            'google-generativeai', 'django'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                validation_results['requirements'][package] = 'available'
            except ImportError:
                validation_results['requirements'][package] = 'missing'
                validation_results['errors'].append(f"Required package missing: {package}")
        
        # Check configuration
        try:
            self._validate_configuration()
            validation_results['requirements']['configuration'] = 'valid'
        except RAGInitializationError as e:
            validation_results['requirements']['configuration'] = 'invalid'
            validation_results['errors'].append(f"Configuration invalid: {e}")
        
        # Check file system permissions
        try:
            test_path = Path("test_permissions")
            test_path.mkdir(exist_ok=True)
            test_path.rmdir()
            validation_results['requirements']['file_permissions'] = 'valid'
        except Exception as e:
            validation_results['requirements']['file_permissions'] = 'invalid'
            validation_results['errors'].append(f"File system permissions: {e}")
        
        # Determine overall status
        if validation_results['errors']:
            validation_results['overall_status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'warning'
        
        return validation_results


class RAGInitializationError(Exception):
    """Custom exception for RAG system initialization errors."""
    pass


# Global initializer instance
rag_initializer = RAGSystemInitializer()