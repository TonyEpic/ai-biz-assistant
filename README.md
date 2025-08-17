# AI Business Assistant

A web app where you can upload documents (PDF, DOCX, CSV), ask questions, and get precise answers with source citations.  
It can also generate follow-up emails in different tones based on the document content.  

---

## Features
- Upload files (PDF, DOCX, CSV)
- Ask questions and get answers grounded in your documents
- Source citations (document + page snippets)
- Draft email generator (formal, friendly, neutral)
- Simple Streamlit UI + FastAPI backend
- Runs fully locally (optionally with OpenAI API)

---

## Tech Stack
- Backend: FastAPI (Python)
- Frontend: Streamlit
- AI: Ollama (local LLMs) or OpenAI (optional)
- Vector DB: Chroma
- Infra: Docker, GitHub Actions (CI/CD)

---

## Project Structure
```

ai-biz-assistant/
│   README.md
│   requirements.txt
│   .gitignore
│
├── backend/
│   main.py
├── frontend/
│   app.py
└── data/          # uploaded files + cache

````

---

## Quickstart (Windows)

```powershell
# 1) Clone the repo
git clone https://github.com/<your-username>/ai-biz-assistant.git
cd ai-biz-assistant

# 2) Create virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run backend
uvicorn backend.main:app --reload --port 8000

# 5) Run frontend (open new terminal)
streamlit run frontend/app.py
````

* Backend runs on: [http://127.0.0.1:8000/ping](http://127.0.0.1:8000/ping)
* Frontend runs on: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---


