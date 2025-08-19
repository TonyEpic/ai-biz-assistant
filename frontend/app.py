import streamlit as st
import requests

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Business Assistant", page_icon="🗂️")
st.title("AI Business Assistant")

# -------- Upload --------
st.subheader("1) Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or CSV",
    type=["pdf", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Uploads"):
    for f in uploaded_files:
        with st.spinner(f"Indexing {f.name}..."):
            try:
                files = {"file": (f.name, f.getvalue(), f"type")}
                res = requests.post(f"{BACKEND}/upload", files=files, timeout=120)
                if res.status_code == 200:
                    st.success(res.json())
                else:
                    st.error(f"Failed {f.name}: {res.status_code} {res.text}")
            except Exception as e:
                st.error(f"Error {f.name}: {e}")

st.divider()

# -------- Ask --------
st.subheader("2) Ask a question")
query = st.text_input("Question about your documents:")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    k = st.number_input("Top-K", min_value=1, max_value=8, value=4, step=1)
with col2:
    use_llm = st.checkbox("Use local LLM (Ollama)", value=False)
with col3:
    st.caption("Tip: leave unchecked if Ollama isn't installed yet.")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving..."):
            try:
                payload = {"query": query, "k": int(k), "use_llm": bool(use_llm)}
                res = requests.post(f"{BACKEND}/ask", json=payload, timeout=180)
                if res.status_code != 200:
                    st.error(f"{res.status_code} {res.text}")
                else:
                    data = res.json()
                    st.markdown("**Answer**")
                    st.write(data.get("answer", ""))

                    st.markdown("**Sources**")
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No sources returned.")
                    else:
                        for i, s in enumerate(sources, start=1):
                            with st.expander(f"[S{i}] {s.get('source')} (distance={s.get('distance'):.3f})"):
                                st.write(s.get("snippet", ""))
                    st.caption(f"Latency: {data.get('latency_ms', 0)} ms")
            except Exception as e:
                st.error(f"Backend unreachable: {e}")

st.divider()

# -------- Debug --------
st.subheader("Debug")
if st.button("Count indexed chunks"):
    try:
        res = requests.get(f"{BACKEND}/debug_count", timeout=10)
        st.info(res.json())
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
