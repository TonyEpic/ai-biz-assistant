from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple
import io, os, uuid
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

# --- Chroma init (persist to disk automatically) ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
os.makedirs(PERSIST_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name="documents")

# --- Utils ---
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
    # No client.persist() needed with PersistentClient

@app.get("/ping")
def ping():
    return {"message": "pong"}

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
        if any(name_lower.endswith(x) for x in [".md", ".txt"]) and "plan" in name_lower:
            return {"filename": file.filename, "message": "Skipped (project doc)", "chunks": 0}

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
