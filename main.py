import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pdf_extractor import extract_text_from_pdf, extract_text_from_txt
from rag import (
    chunk_text,
    store_document,
    retrieve_chunks,
    generate_answer,
    list_documents,
    delete_document,
    get_harness_stats,
    count_tokens,
)

load_dotenv()
app = FastAPI(title="AI Document Q&A API v2 — with Harness + Context Engineering")

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
    document_id:    str
    question:       str
    answer:         str
    source_chunks:  list[SourceChunk]
    input_tokens:   int
    output_tokens:  int
    budget_report:  dict    # NEW — context budget breakdown

class UploadResponse(BaseModel):
    document_id:  str
    filename:     str
    chunk_count:  int
    total_tokens: int
    message:      str

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported.")

    file_bytes = await file.read()

    try:
        text = (extract_text_from_pdf(file_bytes)
                if file.filename.endswith(".pdf")
                else extract_text_from_txt(file_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import uuid
    document_id  = str(uuid.uuid4())[:8]
    chunks       = chunk_text(text, chunk_size=500, overlap=50)
    total_tokens = sum(count_tokens(c) for c in chunks)
    chunk_count  = store_document(document_id, file.filename, chunks)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=chunk_count,
        total_tokens=total_tokens,
        message=f"Uploaded. Use document_id '{document_id}' to ask questions."
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks = retrieve_chunks(request.document_id, request.question, request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    answer, input_tokens, output_tokens, budget_report = generate_answer(
        question=request.question,
        retrieved_chunks=chunks,
    )

    source_chunks = [
        SourceChunk(
            chunk_id=c["chunk_id"],
            text=c["text"][:200] + "...",
            similarity=c["similarity"]
        )
        for c in chunks
    ]

    return AskResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        source_chunks=source_chunks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        budget_report=budget_report,
    )

@app.get("/documents")
async def get_documents():
    documents = list_documents()
    return {"documents": documents, "total_count": len(documents)}

@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return {"message": f"Document {document_id} deleted."}

@app.get("/stats")
async def get_stats():
    """
    NEW endpoint — returns harness session stats.
    Total calls, tokens used, cost, avg latency.
    """
    return get_harness_stats()

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "v2", "harness": "enabled"}
