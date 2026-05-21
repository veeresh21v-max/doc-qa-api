import os
import uuid
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Initialize all components ──────────────────────────────────────────────
groq_client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client  = chromadb.PersistentClient(path="./chroma_db")
enc            = tiktoken.get_encoding("cl100k_base")

MODEL              = "llama-3.3-70b-versatile"
SIMILARITY_THRESHOLD = 0.4
# Chunks below this similarity score are dropped
# They are not relevant enough to include in the prompt

# ── Token counting ─────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return len(enc.encode(text))

# ── Chunking ───────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping token-based chunks.

    text       : full document text
    chunk_size : maximum tokens per chunk
    overlap    : tokens repeated between adjacent chunks
                 prevents losing information at chunk boundaries

    Returns list of text chunk strings.
    """
    tokens = enc.encode(text)
    # Tokenize the entire document first

    chunks = []
    start  = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        # End of this chunk — capped at document length

        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        # Decode back to string for storage

        chunks.append(chunk_text_str)
        start += chunk_size - overlap
        # Move forward by chunk_size minus overlap
        # The overlap tokens are repeated in the next chunk

    return chunks

# ── Embedding ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    texts   : list of strings to embed
    Returns : list of embedding vectors (list of floats)
    """
    embeddings = embedding_model.encode(texts)
    # Returns numpy array of shape (len(texts), 384)
    return embeddings.tolist()
    # ChromaDB requires Python lists not numpy arrays

# ── Storage ────────────────────────────────────────────────────────────────
def store_document(
    document_id: str,
    filename: str,
    chunks: list[str]
) -> int:
    """
    Embed all chunks and store them in ChromaDB.
    Each document gets its own collection.

    document_id : unique identifier for this document
    filename    : original filename for metadata
    chunks      : list of text chunks to store

    Returns number of chunks stored.
    """
    # Create a collection named after this document
    collection_name = f"doc_{document_id}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
        # Use cosine similarity for all searches
    )

    # Generate embeddings for all chunks at once
    # More efficient than embedding one by one
    embeddings = embed_texts(chunks)

    # Create unique IDs for each chunk
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

    # Store everything in ChromaDB
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                "document_id": document_id,
                "filename":    filename,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    return len(chunks)

# ── Retrieval ──────────────────────────────────────────────────────────────
def retrieve_chunks(
    document_id: str,
    question: str,
    top_k: int = 3
) -> list[dict]:
    """
    Embed the question and find the most similar chunks
    from this document's collection.

    document_id : which document to search
    question    : user's question
    top_k       : maximum number of chunks to retrieve

    Returns list of dicts with text, similarity, chunk_id.
    """
    collection_name = f"doc_{document_id}"

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(f"Document {document_id} not found.")

    # Embed the question using the same model used for chunks
    # Critical: must be the same model — different models are incompatible
    query_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        # Cannot retrieve more chunks than exist
        include=["documents", "distances", "metadatas"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        similarity = round(1 - results["distances"][0][i], 4)
        # ChromaDB returns distance — convert to similarity

        if similarity >= SIMILARITY_THRESHOLD:
            # Only include chunks above the threshold
            chunks.append({
                "text":       results["documents"][0][i],
                "similarity": similarity,
                "chunk_id":   results["ids"][0][i],
                "metadata":   results["metadatas"][0][i]
            })

    return chunks

# ── RAG Prompt Builder ─────────────────────────────────────────────────────
def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the RAG prompt by injecting retrieved chunks as context.

    question : user's question
    chunks   : retrieved relevant chunks

    Returns complete prompt string ready for LLM.
    """
    if not chunks:
        return f"""The user asked: "{question}"
There is no relevant information in the uploaded document to answer this question.
Respond with: "I don't have information about that in the uploaded document." """

    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1} | Similarity: {chunk['similarity']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return f"""You are a document assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have information about that in the uploaded document."
Do not use any knowledge outside the provided context.
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

# ── Answer Generation ──────────────────────────────────────────────────────
def generate_answer(prompt: str) -> tuple[str, int, int]:
    """
    Call the Groq LLM with the RAG prompt.

    prompt  : complete RAG prompt with context injected
    Returns : tuple of (answer_text, input_tokens, output_tokens)
    """
    response = groq_client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    answer        = response.choices[0].message.content
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return answer, input_tokens, output_tokens

# ── List and Delete ────────────────────────────────────────────────────────
def list_documents() -> list[str]:
    """Return list of all document collection names."""
    collections = chroma_client.list_collections()
    return [c.name.replace("doc_", "") for c in collections]
    # Strip "doc_" prefix to return just the document_id

def delete_document(document_id: str) -> bool:
    """Delete a document's ChromaDB collection."""
    collection_name = f"doc_{document_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False