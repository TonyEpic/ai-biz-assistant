from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
from pydantic import BaseModel
import io, os, uuid, time, re, json
from datetime import datetime, timezone
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
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.json")
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
USE_DISTANCE_THRESHOLD = 1.0  # a bit looser

# --- Chroma (persistent) ---
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name="documents")

# ---------- Metrics (JSON store) ----------
def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _load_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"totals": {"uploads": 0, "ask": 0, "email": 0, "errors": 0}, "events": []}
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"totals": {"uploads": 0, "ask": 0, "email": 0, "errors": 0}, "events": []}

def _save_metrics(m):
    # keep only last 200 events to avoid growing unbounded
    m["events"] = m.get("events", [])[-200:]
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def _inc_total(key: str, by: int = 1):
    m = _load_metrics()
    m["totals"][key] = int(m["totals"].get(key, 0)) + by
    _save_metrics(m)

def _log_event(ev: dict):
    m = _load_metrics()
    m.setdefault("events", []).append(ev)
    _save_metrics(m)

def _log_ok(kind: str, detail: dict, latency_ms: int):
    _inc_total(kind, 1)
    _log_event({"ts": _now_iso(), "type": kind, "ok": True, "latency_ms": latency_ms, **detail})

def _log_err(kind: str, err: str, detail: dict, latency_ms: int = 0):
    _inc_total("errors", 1)
    _log_event({"ts": _now_iso(), "type": kind, "ok": False, "error": err, "latency_ms": latency_ms, **detail})

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
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def _query_chroma(text_query: str, k: int):
    res = collection.query(
        query_texts=[text_query],
        n_results=max(1, min(k, 8)),
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not doc:
            continue
        sources.append({
            "source": meta.get("source", "document"),
            "doc_id": meta.get("doc_id"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "snippet": doc[:500].strip().replace("\n", " "),
        })
    return sources, dists

def _context_block_from_sources(sources: List[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[S{i} • {s['source']}] {s['snippet']}")
    return "\n\n".join(lines)

def _parse_subject_body(text: str):
    subj = ""
    body = text.strip()
    m_subj = re.search(r"(?i)^\s*subject\s*:\s*(.+)$", text, re.MULTILINE)
    m_body = re.search(r"(?is)^\s*body\s*:\s*(.+)$", text)
    if m_subj:
        subj = m_subj.group(1).strip()
    if m_body:
        body = m_body.group(1).strip()
    if not subj:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        subj = first_line[:120]
    return subj, body

# ---------- Models ----------
class AskRequest(BaseModel):
    query: str
    k: int = 4
    use_llm: bool = False

class DraftEmailRequest(BaseModel):
    goal: str
    tone: str = "neutral"
    recipient: Optional[str] = None
    k: int = 4
    use_llm: bool = True

# ---------- Endpoints ----------
@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/llm_health")
def llm_health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        tags = [m.get("name") for m in r.json().get("models", [])]
        return {"ok": True, "url": OLLAMA_URL, "model": OLLAMA_MODEL, "available_models": tags}
    except Exception as e:
        return {"ok": False, "url": OLLAMA_URL, "model": OLLAMA_MODEL, "error": str(e)}

@app.get("/debug_count")
def debug_count():
    try:
        items = collection.get(include=[])
        return {"ids": len(items.get("ids", []))}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug_chunks")
def debug_chunks():
    try:
        items = collection.get(include=["documents", "metadatas"])
        docs = items.get("documents", [])
        metas = items.get("metadatas", [])
        sample = []
        for doc_list, meta_list in zip(docs, metas):
            for text, m in zip(doc_list, meta_list):
                sample.append({
                    "doc_id": m.get("doc_id"),
                    "source": m.get("source"),
                    "snippet": (text or "")[:200]
                })
                if len(sample) >= 5:
                    break
        return {"chunks": sample}
    except Exception as e:
        return {"error": str(e), "chunks": []}


@app.get("/metrics")
def get_metrics(last_n: int = 50):
    m = _load_metrics()
    events = m.get("events", [])[-max(1, min(last_n, 200)):]
    return {"totals": m.get("totals", {}), "events": events}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    t0 = time.time()
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

        latency = int((time.time() - t0) * 1000)
        _log_ok("uploads", {"filename": file.filename, "doc_id": base_doc_id, "chunks": len(chunks)}, latency)

        return {
            "filename": file.filename,
            "doc_id": base_doc_id,
            "pages": page_count,
            "chunks_indexed": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        _log_err("uploads", str(e), {"filename": getattr(file, "filename", "unknown")}, latency)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query is empty.")
        if _collection_empty():
            return {"answer": "No documents are indexed yet. Upload a file first.", "sources": [], "latency_ms": 0}

        sources, dists = _query_chroma(req.query, req.k)
        if not sources or (dists and dists[0] > USE_DISTANCE_THRESHOLD):
            latency = int((time.time() - t0) * 1000)
            _log_ok("ask", {"query": req.query, "use_llm": req.use_llm, "matched": False, "top_dist": float(dists[0]) if dists else None}, latency)
            return {
                "answer": "I don't have enough relevant context in the indexed documents to answer that.",
                "sources": sources,
                "latency_ms": latency,
            }

        if not req.use_llm:
            extractive = "Here are the most relevant excerpts:\n" + "\n".join(
                [f"• {s['snippet']} [{s['source']}]" for s in sources[:3]]
            )
            latency = int((time.time() - t0) * 1000)
            _log_ok("ask", {"query": req.query, "use_llm": False, "matched": True, "top_dist": float(dists[0])}, latency)
            return {"answer": extractive, "sources": sources, "latency_ms": latency}

        context_block = _context_block_from_sources(sources)
        system_preamble = (
            "You are a precise assistant. Only answer using the provided CONTEXT. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources inline like [S1], [S2] matching the context items."
        )
        prompt = f"{system_preamble}\n\nQUESTION: {req.query}\n\nCONTEXT:\n{context_block}\n\nFINAL ANSWER (with inline citations):"
        answer = _call_ollama(prompt, temperature=0.1)

        unique_sources = sorted({s["source"] for s in sources})
        answer = answer.strip() + "\n\nSources: " + ", ".join(unique_sources)

        latency = int((time.time() - t0) * 1000)
        _log_ok("ask", {"query": req.query, "use_llm": True, "matched": True, "top_dist": float(dists[0])}, latency)
        return {"answer": answer, "sources": sources, "latency_ms": latency}
    except HTTPException:
        raise
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        _log_err("ask", str(e), {"query": req.query, "use_llm": req.use_llm}, latency)
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}")

@app.post("/draft-email")
def draft_email(req: DraftEmailRequest):
    t0 = time.time()
    try:
        if not req.goal or not req.goal.strip():
            raise HTTPException(status_code=400, detail="Goal is empty.")
        if _collection_empty():
            subject = f"{req.goal.strip()[:80]}"
            body = (
                f"Hi{(' ' + req.recipient) if req.recipient else ''},\n\n"
                f"{req.goal.strip()}.\n\n"
                f"Best regards,\n"
                f"{'Your team' if not req.recipient else req.recipient.split('@')[0]}"
            )
            latency = int((time.time() - t0) * 1000)
            _log_ok("email", {"goal": req.goal, "use_llm": req.use_llm, "matched": False}, latency)
            return {"subject": subject, "body": body, "sources": [], "latency_ms": latency}

        sources, dists = _query_chroma(req.goal, req.k)
        context_block = _context_block_from_sources(sources) if sources else ""

        if not req.use_llm:
            bullets = "\n".join([f"- {s['snippet']}" for s in sources[:3]]) if sources else "- (no context snippets)"
            subject = f"{req.goal.strip()[:80]}"
            body = (
                f"Hi{(' ' + req.recipient) if req.recipient else ''},\n\n"
                f"{req.goal.strip()}.\n\n"
                f"Key points:\n{bullets}\n\n"
                f"Regards,\nYour team"
            )
            latency = int((time.time() - t0) * 1000)
            _log_ok("email", {"goal": req.goal, "use_llm": False, "matched": bool(sources)}, latency)
            return {"subject": subject, "body": body, "sources": sources, "latency_ms": latency}

        style = {
            "neutral": "neutral and concise",
            "professional": "professional and concise",
            "friendly": "friendly and approachable",
            "formal": "formal and clear",
        }.get(req.tone.lower(), "neutral and concise")

        system_preamble = (
            "You write concise, actionable emails. "
            "Use ONLY the provided CONTEXT; do not invent facts. "
            "If key details are missing, add a short TODO line for the sender to fill in."
        )
        user_instruction = (
            f"Goal: {req.goal}\n"
            f"Recipient: {req.recipient or 'N/A'}\n"
            f"Tone: {style}\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            "Return exactly two sections with these headers:\n"
            "SUBJECT: <one line>\n"
            "BODY: <email body in 1-2 short paragraphs, include a clear next step>"
        )
        llm_text = _call_ollama(f"{system_preamble}\n\n{user_instruction}", temperature=0.2)
        subject, body = _parse_subject_body(llm_text)

        latency = int((time.time() - t0) * 1000)
        _log_ok("email", {"goal": req.goal, "use_llm": True, "matched": bool(sources)}, latency)
        return {"subject": subject, "body": body, "sources": sources, "latency_ms": latency}
    except HTTPException:
        raise
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        _log_err("email", str(e), {"goal": req.goal, "use_llm": req.use_llm}, latency)
        raise HTTPException(status_code=500, detail=f"Email draft failed: {e}")
