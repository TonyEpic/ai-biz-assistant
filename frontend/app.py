import streamlit as st
import requests
import json

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Business Assistant", page_icon="🗂️")
st.title("AI Business Assistant")

# -------- Upload --------
st.subheader("1) Upload documents")
uploaded_files = st.file_uploader("Upload PDF, DOCX, or CSV", type=["pdf", "docx", "csv"], accept_multiple_files=True)
if uploaded_files and st.button("Process Uploads"):
    for f in uploaded_files:
        with st.spinner(f"Indexing {f.name}..."):
            try:
                files = {"file": (f.name, f.getvalue(), f"type")}
                res = requests.post(f"{BACKEND}/upload", files=files, timeout=120)
                st.write(res.json() if res.status_code == 200 else res.text)
            except Exception as e:
                st.error(f"Error {f.name}: {e}")

st.divider()

# -------- Ask --------
st.subheader("2) Ask a question")
query = st.text_input("Question about your documents:")
col1, col2, col3 = st.columns([1,1,2])
with col1: k = st.number_input("Top-K", min_value=1, max_value=8, value=4, step=1)
with col2: use_llm = st.checkbox("Use local LLM (Ollama)", value=False)
with col3: st.caption("Tip: uncheck if Ollama isn't running.")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving..."):
            try:
                payload = {"query": query, "k": int(k), "use_llm": bool(use_llm)}
                res = requests.post(f"{BACKEND}/ask", json=payload, timeout=180)
                data = res.json()
                st.markdown("**Answer**")
                st.write(data.get("answer", ""))
                st.markdown("**Sources**")
                for i, s in enumerate(data.get("sources", []), start=1):
                    with st.expander(f"[S{i}] {s.get('source')} (distance={s.get('distance'):.3f})"):
                        st.write(s.get("snippet", ""))
                st.caption(f"Latency: {data.get('latency_ms', 0)} ms")
            except Exception as e:
                st.error(f"Backend error: {e}")

st.divider()

# -------- Draft Email --------
st.subheader("3) Draft email")
goal = st.text_input("Goal (what should the email achieve?)", placeholder="Inform the team about the prototype deadline and ask for status.")
tone = st.selectbox("Tone", ["neutral", "professional", "friendly", "formal"], index=1)
recipient = st.text_input("Recipient (optional)", placeholder="team@example.com")
ek = st.number_input("Top-K context", min_value=1, max_value=8, value=4, step=1)
use_llm_email = st.checkbox("Use local LLM (Ollama) for email", value=True)

if st.button("Generate Email"):
    if not goal.strip():
        st.warning("Please enter a goal.")
    else:
        with st.spinner("Composing email..."):
            try:
                payload = {"goal": goal, "tone": tone, "recipient": recipient, "k": int(ek), "use_llm": bool(use_llm_email)}
                res = requests.post(f"{BACKEND}/draft-email", json=payload, timeout=180)
                data = res.json()
                subj = data.get("subject", "(no subject)")
                body = data.get("body", "")
                st.markdown("**Subject**")
                st.text_input("subject_out", value=subj, label_visibility="collapsed")
                st.markdown("**Body**")
                st.text_area("body_out", value=body, height=220, label_visibility="collapsed")

                # Download as .txt
                download_text = f"Subject: {subj}\n\n{body}"
                st.download_button("Download email (.txt)", data=download_text.encode("utf-8"), file_name="email_draft.txt")

                # Show sources (optional)
                if data.get("sources"):
                    st.markdown("**Context used**")
                    for i, s in enumerate(data["sources"][:3], start=1):
                        with st.expander(f"[S{i}] {s.get('source')} (distance={s.get('distance'):.3f})"):
                            st.write(s.get("snippet", ""))
                st.caption(f"Latency: {data.get('latency_ms', 0)} ms")
            except Exception as e:
                st.error(f"Backend error: {e}")

st.divider()

# -------- Debug --------
st.subheader("Debug")
cols = st.columns(3)
with cols[0]:
    if st.button("Count indexed chunks"):
        try:
            st.info(requests.get(f"{BACKEND}/debug_count", timeout=10).json())
        except Exception as e:
            st.error(e)
with cols[1]:
    if st.button("Show sample chunks"):
        try:
            st.info(requests.get(f"{BACKEND}/debug_chunks", timeout=10).json())
        except Exception as e:
            st.error(e)
with cols[2]:
    if st.button("LLM health"):
        try:
            st.info(requests.get(f"{BACKEND}/llm_health", timeout=10).json())
        except Exception as e:
            st.error(e)
