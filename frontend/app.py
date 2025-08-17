import streamlit as st
import requests

st.set_page_config(page_title="AI Business Assistant", page_icon="🗂️")
st.title("AI Business Assistant (Day 1)")

st.write("Click to test the backend connection (/ping).")

if st.button("Test Backend"):
    try:
        res = requests.get("http://localhost:8000/ping", timeout=10)
        st.success("Backend responded!")
        st.json(res.json())
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
