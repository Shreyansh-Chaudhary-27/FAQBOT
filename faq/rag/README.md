# RAG System - Retrieval-Augmented Generation for FAQ Processing

## Overview

This RAG (Retrieval-Augmented Generation) system provides intelligent FAQ processing capabilities by combining document scraping, semantic search, and AI-powered response generation. The system can extract FAQ data from DOCX documents, create semantic embeddings, and provide contextual responses to user queries.

## Architecture

The system follows a modular architecture with the following components:

### Core Components

- **DOCX Scraper**: Extracts FAQ data from Microsoft Word documents
- **Query Processor**: Handles user queries with typo correction and intent understanding
- **FAQ Vectorizer**: Creates semantic embeddings using Gemini AI
- **Vector Store**: Manages vector storage and similarity search
- **Response Generator**: Generates contextual responses using retrieved FAQ content
- **Conversation Manager**: Maintains conversation context and session management

### System Structure

```
faq/rag/
├── __init__.py                 # Package exports and main API
├── README.md                   # This documentation
├── interfaces/
│   ├── __init__.py
│   └── base.py                 # Abstract interfaces and data models
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration management
├── core/
│   ├── __init__.py
│   ├── rag_system.py          # Main system orchestrator
│   └── factory.py             # Component factory and DI
├── components/
│   ├── __init__.py
│   ├── docx_scraper/          # Document processing components
│   ├── query_processor/       # Query processing components
│   ├── vectorizer/            # Embedding generation components
│   ├── vector_store/          # Vector storage components
│   ├── response_generator/    # Response generation components
│   └── conversation_manager/  # Context management components
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Centralized logging utilities
│   ├── text_processing.py     # Text processing functions
│   └── validation.py          # Validation utilities
└── tests/
    ├── __init__.py
    ├── test_structure.py       # Structure and import tests
    └── test_integration.py     # Integration tests
```

## Dependencies

The system requires the following dependencies:

- **python-docx**: For DOCX document processing
- **numpy**: For numerical operations and embeddings
- **scikit-learn**: For machine learning utilities
- **google-generativeai**: For Gemini AI integration
- **Django**: For web framework integration
- **hypothesis**: For property-based testing

## Configuration

The system uses environment variables and Django settings for configuration:

### Required Environment Variables

- `GEMINI_API_KEY`: API key for Gemini AI services

### Optional Configuration

- `GEMINI_MODEL`: Gemini model to use (default: "gemini-pro")
- `GEMINI_EMBEDDING_MODEL`: Embedding model (default: "models/embedding-001")
- `RAG_VECTOR_DIMENSION`: Vector dimension (default: 768)
- `RAG_SIMILARITY_THRESHOLD`: Similarity threshold (default: 0.7)
- `RAG_MAX_RESULTS`: Maximum search results (default: 10)

## Usage

### Basic Usage

```python
from faq.rag import RAGSystem, rag_factory

# Create RAG system (components will be implemented in subsequent tasks)
rag_system = rag_factory.create_rag_system(
    docx_scraper=your_docx_scraper,
    query_processor=your_query_processor,
    vectorizer=your_vectorizer,
    vector_store=your_vector_store,
    response_generator=your_response_generator,
    conversation_manager=your_conversation_manager
)

# Process a document
faqs = rag_system.process_document("path/to/document.docx")

# Answer a query
response = rag_system.answer_query("What is RAG?")
print(response.text)
```

### Data Models

The system uses well-defined data models:

```python
from faq.rag import FAQEntry, ProcessedQuery, Response

# FAQ Entry
faq = FAQEntry(
    id="faq-1",
    question="What is RAG?",
    answer="Retrieval-Augmented Generation",
    keywords=["rag", "ai", "generation"],
    category="AI",
    confidence_score=0.9,
    source_document="faqs.docx",
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# Processed Query
query = ProcessedQuery(
    original_query="what is rag?",
    corrected_query="What is RAG?",
    intent="information_request",
    expanded_queries=["What is RAG?", "Define RAG"],
    language="en",
    confidence=0.95
)
```

## Testing

The system includes comprehensive tests:

```bash
# Run structure tests
python manage.py test faq.rag.tests.test_structure

# Run integration tests
python manage.py test faq.rag.tests.test_integration

# Run all RAG tests
python manage.py test faq.rag.tests
```

## Development Status

### ✅ Completed (Task 1)

- [x] Project structure and directory organization
- [x] Dependency installation and verification
- [x] Abstract interfaces and data models
- [x] Configuration management system
- [x] Logging and utility modules
- [x] Core system orchestrator
- [x] Component factory and dependency injection
- [x] Comprehensive test suite
- [x] Documentation

### 🔄 Next Steps (Subsequent Tasks)

- [ ] DOCX Scraper implementation
- [ ] Query Processor implementation
- [ ] FAQ Vectorizer implementation
- [ ] Vector Store implementation
- [ ] Response Generator implementation
- [ ] Conversation Manager implementation
- [ ] API endpoints and Django integration
- [ ] Demo interface and documentation

## Logging

The system provides comprehensive logging:

```python
from faq.rag.utils.logging import get_rag_logger, log_performance

# Get component logger
logger = get_rag_logger('my_component')
logger.info("Processing started")

# Log performance metrics
log_performance('vectorizer', 'embedding_generation', 0.5, {'dimension': 768})
```

## Error Handling

The system implements robust error handling across all components with graceful degradation and detailed error reporting.

## Contributing

When implementing new components:

1. Follow the abstract interfaces defined in `interfaces/base.py`
2. Add comprehensive logging using the RAG logging utilities
3. Include proper error handling and validation
4. Write both unit tests and property-based tests
5. Update documentation as needed

## License

This RAG system is part of the FAQ backend application.