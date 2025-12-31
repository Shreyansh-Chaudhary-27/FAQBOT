from django.apps import AppConfig


class FaqConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'faq'
    
    def ready(self):
        """
        Import signals when the app is ready.
        This ensures automatic FAQ synchronization with the RAG system.
        """
        import faq.signals  # noqa
