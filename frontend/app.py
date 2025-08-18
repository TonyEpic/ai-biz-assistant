import streamlit as st
import requests

st.set_page_config(page_title="AI Business Assistant", page_icon="🗂️")
st.title("AI Business Assistant")

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
                res = requests.post("http://127.0.0.1:8000/upload", files=files, timeout=120)
                if res.status_code == 200:
                    st.success(res.json())
                else:
                    st.error(f"Failed {f.name}: {res.status_code} {res.text}")
            except Exception as e:
                st.error(f"Error {f.name}: {e}")

st.divider()
st.subheader("Debug")
if st.button("Count indexed chunks"):
    try:
        res = requests.get("http://127.0.0.1:8000/debug_count", timeout=10)
        st.info(res.json())
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
