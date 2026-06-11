import os
import uuid
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llm_harness import LLMHarness

load_dotenv()

# ── Initialize components ──────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client   = chromadb.PersistentClient(path="./chroma_db")
enc             = tiktoken.get_encoding("cl100k_base")

# Single harness instance shared across all RAG calls
harness = LLMHarness(
    primary_model="llama-3.3-70b-versatile",
    fallback_model="llama-3.1-8b-instant",
    log_to_console=True,
)

# ── Context budget constants ───────────────────────────────────────────────
CONTEXT_LIMIT       = 8192
OUTPUT_RESERVE      = 512
SYSTEM_PROMPT_BUDGET = 200   # tokens reserved for system prompt
CHUNK_BUDGET_RATIO  = 0.45   # retrieved chunks get 45% of remaining budget
SIMILARITY_THRESHOLD = 0.4

SYSTEM_PROMPT = """You are a document assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have information about that in the uploaded document."
Do not use any knowledge outside the provided context.
Do not make up information."""

# ── Token counting ─────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# ── Chunking ───────────────────────────────────────────────────────────────
def chunk_text(
    text:       str,
    chunk_size: int = 500,
    overlap:    int = 50
) -> list[str]:
    """Split document text into overlapping token-based chunks."""
    tokens = enc.encode(text)
    chunks = []
    start  = 0

    while start < len(tokens):
        end            = min(start + chunk_size, len(tokens))
        chunk_text_str = enc.decode(tokens[start:end])
        chunks.append(chunk_text_str)
        start += chunk_size - overlap

    return chunks

# ── Embedding ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts).tolist()

# ── Storage ────────────────────────────────────────────────────────────────
def store_document(
    document_id: str,
    filename:    str,
    chunks:      list[str]
) -> int:
    collection = chroma_client.get_or_create_collection(
        name=f"doc_{document_id}",
        metadata={"hnsw:space": "cosine"}
    )
    embeddings = embed_texts(chunks)
    chunk_ids  = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"document_id": document_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
    )
    return len(chunks)

# ── Retrieval ──────────────────────────────────────────────────────────────
def retrieve_chunks(
    document_id: str,
    question:    str,
    top_k:       int = 3
) -> list[dict]:
    try:
        collection = chroma_client.get_collection(name=f"doc_{document_id}")
    except Exception:
        raise ValueError(f"Document {document_id} not found.")

    query_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        similarity = round(1 - results["distances"][0][i], 4)
        if similarity >= SIMILARITY_THRESHOLD:
            chunks.append({
                "text":       results["documents"][0][i],
                "similarity": similarity,
                "chunk_id":   results["ids"][0][i],
                "metadata":   results["metadatas"][0][i]
            })
    return chunks

# ── Context budget allocation ──────────────────────────────────────────────
def allocate_context(
    question:         str,
    retrieved_chunks: list[dict],
) -> tuple[list[dict], dict]:
    """
    Applies context budget allocation before building the prompt.
    Decides how many chunks fit within the allocated token budget.

    Returns:
    - filtered_chunks: chunks that fit within budget
    - budget_report: token breakdown for logging
    """
    available      = CONTEXT_LIMIT - OUTPUT_RESERVE
    system_tokens  = count_tokens(SYSTEM_PROMPT) + 4
    question_tokens = count_tokens(question) + 4
    used            = system_tokens + question_tokens

    # Chunk budget = 45% of available space
    chunk_budget   = int(available * CHUNK_BUDGET_RATIO)
    chunk_used     = 0
    filtered_chunks = []

    for chunk in retrieved_chunks:
        chunk_tokens = count_tokens(chunk["text"]) + 4
        if chunk_used + chunk_tokens <= chunk_budget:
            filtered_chunks.append(chunk)
            chunk_used += chunk_tokens
        else:
            print(f"[CONTEXT] Chunk dropped — budget full "
                  f"({chunk_used}/{chunk_budget} tokens used)")
            break

    used += chunk_used

    budget_report = {
        "context_limit":   CONTEXT_LIMIT,
        "available":       available,
        "system_tokens":   system_tokens,
        "question_tokens": question_tokens,
        "chunk_tokens":    chunk_used,
        "chunks_used":     len(filtered_chunks),
        "chunks_dropped":  len(retrieved_chunks) - len(filtered_chunks),
        "total_used":      used,
        "output_reserve":  OUTPUT_RESERVE,
        "tokens_remaining": available - used,
    }

    return filtered_chunks, budget_report

# ── RAG prompt builder ─────────────────────────────────────────────────────
def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (f"The user asked: '{question}'\n"
                f"No relevant context found above similarity threshold. "
                f"Respond: I don't have information about that in the uploaded document.")

    context_parts = [
        f"[Chunk {i+1} | Similarity: {c['similarity']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n".join(context_parts)

    return f"""Context from the uploaded document:
{context}

Question: {question}

Answer:"""

# ── Answer generation via harness ─────────────────────────────────────────
def generate_answer(
    question:         str,
    retrieved_chunks: list[dict]
) -> tuple[str, int, int, dict]:
    """
    Generates answer using harness with context budget allocation.
    Returns: (answer, input_tokens, output_tokens, budget_report)
    """
    # Apply context budget allocation
    filtered_chunks, budget_report = allocate_context(question, retrieved_chunks)

    print(f"[CONTEXT BUDGET] {budget_report}")

    # Build prompt
    prompt = build_rag_prompt(question, filtered_chunks)

    # Call through harness — gets retry, fallback, logging automatically
    result = harness.call_with_system(
        system_prompt=SYSTEM_PROMPT,
        user_message=prompt,
        task_type="rag_answer",
        max_tokens=OUTPUT_RESERVE,
    )

    return (
        result.text,
        result.input_tokens,
        result.output_tokens,
        budget_report,
    )

# ── List and delete documents ──────────────────────────────────────────────
def list_documents() -> list[str]:
    collections = chroma_client.list_collections()
    return [c.name.replace("doc_", "") for c in collections]

def delete_document(document_id: str) -> bool:
    try:
        chroma_client.delete_collection(name=f"doc_{document_id}")
        return True
    except Exception:
        return False

# ── Harness stats accessor ─────────────────────────────────────────────────
def get_harness_stats() -> dict:
    """Return session-level cost and token stats from the harness."""
    return harness.session_summary()
