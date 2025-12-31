# Pinecone Vector Database Setup Guide

This project has been updated to support a **Persistent Vector Database** using **Pinecone**. This ensures that document embeddings persist across restart and redeployments (like on Render), preventing "No results found" errors.

## 1. Prerequisites

You must have a Pinecone account (Free Tier is sufficient).
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up / Log in.
3. Create a **Index**:
   - **Name**: `faq-index` (or your preferred name)
   - **Dimensions**: `384` (Matches the default MiniLM model used)
   - **Metric**: `cosine`
   - **Cloud/Region**: Choose the one closest to your users (e.g., AWS us-east-1).

## 2. Environment Variables

You **MUST** set the following environment variables in your Hosting Dashboard (Render) or local `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API Key | `pcsk_...` |
| `PINECONE_INDEX_NAME` | The name of the index you created | `faq-index` |
| `PINECONE_ENVIRONMENT` | (Optional) Environment setting | `us-east-1` |

> **Note**: If `PINECONE_API_KEY` is NOT set, the system will fallback to the old Local/SQLite storage (which is not persistent on Render).

## 3. Deployment Steps

1. **Add Environment Variables**: Go to Render Dashboard -> Environment -> Add the variables above.
2. **Re-deploy**: Trigger a manual deploy or push code to master.
3. **Verify**: Check the logs (Render Logs or `admin_security.log`) for:
   `Initializing Pinecone Vector Store (Persistent)`
   `Successfully connected to Pinecone index: faq-index`

## 4. Troubleshooting

*   **Error: "Index not found"**: Ensure you created the index manually in the Pinecone Console with dimensions **384**.
*   **Error: "Connection failed"**: Check your API Key.
*   **"No results found" initially**: You may need to trigger the ingestion process (`master_rag_sync.py`) once to populate the new Pinecone index.
    *   Run `python backend/master_rag_sync.py` locally (with env vars set) OR triggers it via the Admin Dashboard if available.

## 5. Metadata Handling

*   The system loads **metadata** (Question, Answer, IDs) into local memory for fast N-Gram keyword search, but **Embeddings** are kept in Pinecone (not RAM).
*   This ensures the application memory usage remains low while retrieval is fast and persistent.
