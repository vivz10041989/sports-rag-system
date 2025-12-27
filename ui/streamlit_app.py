# ui/streamlit_app.py

import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Sports RAG Q&A", layout="centered")

st.title("🏏 Sports RAG Assistant")

query = st.text_input("Ask a question about sports documents:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            response = requests.post(
                API_URL,
                json={"question": query},
                timeout=60
            )

        if response.status_code == 200:
            answer = response.json().get("answer", "")
            st.success("Answer")
            st.write(answer)
        else:
            st.error("Error from RAG API")
