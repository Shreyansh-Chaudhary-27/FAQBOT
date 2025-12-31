"""
Django management command to initialize the RAG system with Django integration.
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import logging

from faq.rag_django_service import RAGDjangoService, DjangoAnalyticsManager, DjangoFeedbackManager
from faq.rag.core.factory import RAGSystemFactory
from faq.rag.core.rag_system import RAGSystem


class Command(BaseCommand):
    help = 'Initialize the RAG system with Django integration'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--gemini-api-key',
            type=str,
            help='Gemini API key (overrides environment variable)',
        )
        parser.add_argument(
            '--vector-store-path',
            type=str,
            default='vector_store_data',
            help='Path to vector store data directory',
        )
        parser.add_argument(
            '--enable-monitoring',
            action='store_true',
            default=True,
            help='Enable performance monitoring',
        )
        parser.add_argument(
            '--test-connection',
            action='store_true',
            help='Test RAG system components after initialization',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Initializing RAG system with Django integration...'))
        
        try:
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Get configuration
            gemini_api_key = options.get('gemini_api_key') or getattr(settings, 'GEMINI_API_KEY', None)
            if not gemini_api_key:
                raise CommandError(
                    'Gemini API key not provided. Set GEMINI_API_KEY in settings or use --gemini-api-key'
                )
            
            vector_store_path = options['vector_store_path']
            enable_monitoring = options['enable_monitoring']
            
            # Create Django-integrated managers
            self.stdout.write('Creating Django-integrated analytics and feedback managers...')
            analytics_manager = DjangoAnalyticsManager()
            feedback_manager = DjangoFeedbackManager()
            
            # Initialize RAG system factory with Django managers
            self.stdout.write('Initializing RAG system factory...')
            factory = RAGSystemFactory(
                gemini_api_key=gemini_api_key,
                vector_store_path=vector_store_path,
                analytics_manager=analytics_manager,
                feedback_manager=feedback_manager,
                enable_performance_monitoring=enable_monitoring
            )
            
            # Create RAG system
            self.stdout.write('Creating RAG system...')
            rag_system = factory.create_rag_system()
            
            # Create Django service
            self.stdout.write('Creating Django integration service...')
            django_service = RAGDjangoService(rag_system)
            
            # Test system if requested
            if options['test_connection']:
                self.stdout.write('Testing RAG system components...')
                self._test_rag_system(rag_system, django_service)
            
            # Store system reference (in a real application, you'd want to use a proper singleton or service locator)
            self._store_system_reference(rag_system, django_service)
            
            self.stdout.write(
                self.style.SUCCESS('RAG system initialized successfully with Django integration!')
            )
            
            # Display system status
            self._display_system_status(rag_system)
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise CommandError(f'Failed to initialize RAG system: {e}')
    
    def _test_rag_system(self, rag_system: RAGSystem, django_service: RAGDjangoService):
        """Test RAG system components."""
        try:
            # Test system health
            self.stdout.write('  Testing system health...')
            health = rag_system.health_check()
            if health['overall_status'] != 'healthy':
                self.stdout.write(
                    self.style.WARNING(f"  System health: {health['overall_status']}")
                )
                for issue in health.get('issues', []):
                    self.stdout.write(f"    - {issue}")
            else:
                self.stdout.write(self.style.SUCCESS('  System health: OK'))
            
            # Test Django service
            self.stdout.write('  Testing Django service...')
            analytics = django_service.get_system_analytics(days=1)
            self.stdout.write(f"  Analytics retrieved: {len(analytics)} metrics")
            
            # Test basic query processing (if system is ready)
            if rag_system.is_ready():
                self.stdout.write('  Testing query processing...')
                test_response = rag_system.answer_query("What is this system?")
                self.stdout.write(f"  Test query confidence: {test_response.confidence:.2f}")
            else:
                self.stdout.write(self.style.WARNING('  System not ready for query processing'))
            
            self.stdout.write(self.style.SUCCESS('  All tests passed!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  Test failed: {e}'))
            raise
    
    def _store_system_reference(self, rag_system: RAGSystem, django_service: RAGDjangoService):
        """Store system reference for later use."""
        # In a real application, you'd want to use Django's cache framework
        # or a proper service locator pattern
        try:
            from django.core.cache import cache
            cache.set('rag_system', rag_system, timeout=None)
            cache.set('rag_django_service', django_service, timeout=None)
            self.stdout.write('  System references stored in cache')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'  Could not store system references: {e}'))
    
    def _display_system_status(self, rag_system: RAGSystem):
        """Display system status information."""
        try:
            self.stdout.write('\n' + '='*50)
            self.stdout.write('RAG SYSTEM STATUS')
            self.stdout.write('='*50)
            
            # Component status
            component_status = rag_system.get_component_status()
            self.stdout.write('\nComponent Status:')
            for component, available in component_status.items():
                status = 'Available' if available else 'Not Available'
                style = self.style.SUCCESS if available else self.style.WARNING
                self.stdout.write(f'  {component}: {style(status)}')
            
            # System statistics
            stats = rag_system.get_system_stats()
            self.stdout.write('\nSystem Statistics:')
            self.stdout.write(f"  Status: {stats.get('system_status', 'unknown')}")
            self.stdout.write(f"  Initialized: {stats.get('system_info', {}).get('initialized', False)}")
            
            perf_metrics = stats.get('performance_metrics', {})
            self.stdout.write(f"  Queries Processed: {perf_metrics.get('queries_processed', 0)}")
            self.stdout.write(f"  Documents Processed: {perf_metrics.get('documents_processed', 0)}")
            self.stdout.write(f"  Error Rate: {perf_metrics.get('error_rate', 0):.2f}%")
            
            # Vector store stats
            vector_stats = stats.get('vector_store_stats', {})
            if vector_stats and 'error' not in vector_stats:
                self.stdout.write(f"  Vector Store Vectors: {vector_stats.get('total_vectors', 0)}")
            
            self.stdout.write('\n' + '='*50)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Could not display system status: {e}'))