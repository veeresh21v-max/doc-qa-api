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
