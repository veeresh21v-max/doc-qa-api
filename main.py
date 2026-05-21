import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pdf_extractor import extract_text_from_pdf, extract_text_from_txt
from rag import (
    chunk_text,
    store_document,
    retrieve_chunks,
    build_rag_prompt,
    generate_answer,
    list_documents,
    delete_document,
    count_tokens,
)

load_dotenv()
app = FastAPI(title="AI Document Q&A API")

# ── Pydantic Models ────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    document_id: str
    question:    str
    top_k:       int = 3

class SourceChunk(BaseModel):
    chunk_id:   str
    text:       str
    similarity: float

class AskResponse(BaseModel):
    document_id:   str
    question:      str
    answer:        str
    source_chunks: list[SourceChunk]
    input_tokens:  int
    output_tokens: int

class UploadResponse(BaseModel):
    document_id:  str
    filename:     str
    chunk_count:  int
    total_tokens: int
    message:      str

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.
    Extracts text, chunks it, embeds chunks, stores in ChromaDB.
    Returns document_id for use in future /ask requests.
    """
    # Validate file type
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_txt(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate unique document ID
    document_id = str(uuid.uuid4())[:8]

    # Chunk the text
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # Count total tokens for reporting
    total_tokens = sum(count_tokens(chunk) for chunk in chunks)

    # Embed and store in ChromaDB
    chunk_count = store_document(document_id, file.filename, chunks)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=chunk_count,
        total_tokens=total_tokens,
        message=f"Document uploaded successfully. Use document_id '{document_id}' to ask questions."
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about an uploaded document.
    Retrieves relevant chunks and generates an answer using RAG.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant chunks from ChromaDB
    try:
        chunks = retrieve_chunks(
            document_id=request.document_id,
            question=request.question,
            top_k=request.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build RAG prompt with retrieved context
    prompt = build_rag_prompt(request.question, chunks)

    # Generate answer using Groq LLM
    answer, input_tokens, output_tokens = generate_answer(prompt)

    # Format source chunks for response
    source_chunks = [
        SourceChunk(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"][:200] + "...",
            # Return first 200 chars of each chunk
            # Full text would make response too large
            similarity=chunk["similarity"]
        )
        for chunk in chunks
    ]

    return AskResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        source_chunks=source_chunks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

@app.get("/documents")
async def get_documents():
    """List all uploaded documents."""
    documents = list_documents()
    return {
        "documents":  documents,
        "total_count": len(documents)
    }

@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    """Delete a document and all its chunks from ChromaDB."""
    success = delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found."
        )
    return {"message": f"Document {document_id} deleted successfully."}

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

# Run: uvicorn main:app --reload