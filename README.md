# AI Document Q&A API

A FastAPI application that accepts PDF or text file uploads and answers 
questions about them using RAG (Retrieval-Augmented Generation).

## Tech Stack
- FastAPI — API framework
- ChromaDB — local vector store
- sentence-transformers — free local embeddings (all-MiniLM-L6-v2)
- Groq + Llama 3.3 70b — LLM for answer generation
- PyMuPDF — PDF text extraction

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
uvicorn main:app --reload
```

## API Endpoints

### POST /upload
Upload a PDF or TXT file.
Returns a document_id for use in /ask requests.

### POST /ask
Ask a question about an uploaded document.
```json
{
  "document_id": "a3f8b2c1",
  "question": "What is the main topic of this document?",
  "top_k": 3
}
```

### GET /documents
List all uploaded documents.

### DELETE /documents/{document_id}
Delete a document and all its chunks.

## Example Usage
1. Upload: POST /upload with your PDF file
2. Copy the document_id from the response
3. Ask: POST /ask with document_id and your question
4. Receive answer with source chunks and similarity scores

## v2 Updates — Harness + Context Engineering

- All LLM calls now route through LLMHarness class
- Context budget allocation: chunks get 45% of available token budget
- Automatic retry with exponential backoff on transient failures  
- Fallback to llama-3.1-8b-instant if primary model fails
- Structured logging on every LLM call
- New /stats endpoint returns session-level cost and token usage
- budget_report field in /ask response shows exact token allocation per call
