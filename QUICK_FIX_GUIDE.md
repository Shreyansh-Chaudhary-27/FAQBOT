# Quick Fix Guide for RAG FAQ Search

## Issues Fixed

### 1. ✅ ProcessedQuery Attribute Error
**Error**: `'ProcessedQuery' object has no attribute 'processed_text'`

**Fix**: Updated `faq/rag/components/response_generator/response_generator.py` line 101:
```python
# Changed from:
actual_query = processed_query.processed_text if processed_query else query

# To:
actual_query = processed_query.corrected_query if processed_query else query
```

### 2. ⚠️ Vector Search Finding 0 FAQs
**Issue**: Embeddings stored with old Gemini model (768 dim) don't match new local embeddings (384 dim)

**Solution**: Re-vectorize all FAQs with local embeddings

## Steps to Fix Vector Search

1. **Clear old embeddings**:
```powershell
Remove-Item "vector_store_data\*.pkl" -Force
```

2. **Clear Django RAG tables**:
```python
python manage.py shell -c "from faq.models import RAGFAQEntry; RAGFAQEntry.objects.all().delete()"
```

3. **Re-sync with local embeddings**:
```powershell
python master_rag_sync.py
```

4. **Restart Django server** to reload the RAG system with new embeddings

## Testing

After re-sync, test with:
```python
python test_rag_local.py
```

Query: "How do I apply for an internship?"
Expected: Should find relevant FAQs with similarity > 0.5

## Current Status

- ✅ Local embedding service configured (all-MiniLM-L6-v2, 384 dim)
- ✅ ProcessedQuery attribute fixed
- ⏳ Waiting for re-vectorization to complete
- ⏳ Need to restart Django server after re-sync

## Note

The local embedding model loads slowly on first use (downloads ~90MB model).
Subsequent uses will be fast as the model is cached.
