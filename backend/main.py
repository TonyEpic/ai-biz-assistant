from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
from pydantic import BaseModel
import io, os, uuid
import time
import requests
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import pandas as pd
import chromadb

app = FastAPI(title="AI Biz Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config / Env ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
os.makedirs(PERSIST_DIR, exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# Pick a model you have locally, e.g. "llama3:8b", "mistral", "phi3"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

USE_DISTANCE_THRESHOLD = 1  # simple guardrail; lower = stricter

# --- Chroma (persistent) ---
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name="documents")

# ---------- Utils ----------
def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    text = text.strip()
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def _extract_pdf(file_bytes: bytes) -> Tuple[str, int]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = [(page.get_text() or "") for page in doc]
    return "\n".join(texts), len(texts)

def _extract_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buf:
        doc = DocxDocument(buf)
        paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def _extract_csv(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buf:
        try:
            df = pd.read_csv(buf)
            return df.to_csv(index=False)
        except Exception:
            buf.seek(0)
            return buf.read().decode("utf-8", errors="ignore")

def _index_chunks(doc_id: str, filename: str, chunks: List[str]):
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)

def _collection_empty() -> bool:
    try:
        items = collection.get(include=[])
        return len(items.get("ids", [])) == 0
    except Exception:
        return True

def _call_ollama(prompt: str, temperature: float = 0.1) -> str:
    """Call local Ollama; returns plain text. Raises on failure."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"response": "...", "done": true, ...}
    return data.get("response", "").strip()

# ---------- Endpoints ----------
@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug_count")
def debug_count():
    try:
        items = collection.get(include=[])
        return {"ids": len(items.get("ids", []))}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        name_lower = (file.filename or "").lower()
        base_doc_id = f"{os.path.splitext(file.filename)[0]}-{uuid.uuid4().hex[:8]}"

        if name_lower.endswith(".pdf"):
            text, page_count = _extract_pdf(raw)
        elif name_lower.endswith(".docx"):
            text, page_count = _extract_docx(raw), None
        elif name_lower.endswith(".csv"):
            text, page_count = _extract_csv(raw), None
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or CSV.")

        if not text.strip():
            return {"filename": file.filename, "message": "No extractable text found", "chunks": 0}

        chunks = _chunk_text(text, max_chars=1000, overlap=150)
        if not chunks:
            return {"filename": file.filename, "message": "No chunks produced", "chunks": 0}

        _index_chunks(base_doc_id, file.filename, chunks)

        return {
            "filename": file.filename,
            "doc_id": base_doc_id,
            "pages": page_count,
            "chunks_indexed": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

# ----- Day 3: /ask -----
class AskRequest(BaseModel):
    query: str
    k: int = 4
    use_llm: bool = False

@app.post("/ask")
def ask(req: AskRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")
    if _collection_empty():
        return {
            "answer": "No documents are indexed yet. Upload a file first.",
            "sources": [],
            "latency_ms": 0,
        }

    t0 = time.time()
    # Retrieve top-k; include distances for a simple confidence check
    res = collection.query(
        query_texts=[req.query],
        n_results=max(1, min(req.k, 8)),
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Build sources (first 300 chars as snippet)
    sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not doc:
            continue
        sources.append({
            "source": meta.get("source", "document"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "snippet": doc[:300].strip().replace("\n", " "),
        })

    # Guardrail: if top hit is too far, return “insufficient context”
    if not sources or (dists and dists[0] > USE_DISTANCE_THRESHOLD):
        return {
            "answer": "I don't have enough relevant context in the indexed documents to answer that.",
            "sources": sources,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # Compose a prompt with context
    context_lines = []
    for i, s in enumerate(sources, start=1):
        context_lines.append(f"[S{i} • {s['source']}] {s['snippet']}")
    context_block = "\n\n".join(context_lines)

    system_preamble = (
        "You are a precise assistant. Only answer using the provided CONTEXT. "
        "If the answer is not in the context, say you don't know. "
        "Cite sources inline like [S1], [S2] matching the context items."
    )
    prompt = (
        f"{system_preamble}\n\n"
        f"QUESTION: {req.query}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "FINAL ANSWER (with inline citations):"
    )

    # If use_llm is off, return a deterministic extractive summary
    if not req.use_llm:
        bullet = " • "
        extractive = "Here are the most relevant excerpts:\n" + "\n".join(
            [f"{bullet}{s['snippet']} [{s['source']}]" for s in sources[:3]]
        )
        return {
            "answer": extractive,
            "sources": sources,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # Call local LLM via Ollama
    try:
        answer = _call_ollama(prompt, temperature=0.1)
    except Exception as e:
        # Fallback to extractive if LLM unavailable
        extractive = "LLM unavailable. Showing relevant excerpts instead:\n" + "\n".join(
            [f"• {s['snippet']} [{s['source']}]" for s in sources[:3]]
        )
        return {
            "answer": extractive + f"\n\n(Note: LLM error: {e})",
            "sources": sources,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # Always append a sources list (extra safety)
    unique_sources = sorted({s["source"] for s in sources})
    answer = answer.strip() + "\n\nSources: " + ", ".join(unique_sources)

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": int((time.time() - t0) * 1000),
    }
    
@app.get("/debug_chunks")
def debug_chunks():
    items = collection.get(include=["documents", "metadatas"])
    docs = items.get("documents", [])
    metas = items.get("metadatas", [])
    if not docs:
        return {"chunks": []}
    sample = []
    for doc_list, meta_list in zip(docs, metas):
        for text, m in zip(doc_list, meta_list):
            sample.append({"doc_id": m.get("doc_id"), "source": m.get("source"), "snippet": text[:120]})
            if len(sample) >= 5:
                break
    return {"chunks": sample}

