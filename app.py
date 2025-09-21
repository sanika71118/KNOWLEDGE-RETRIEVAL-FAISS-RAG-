# app.py
import streamlit as st
from rag_backend import main_query

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("Document Retrieval-Augmented Generation (RAG) QA")

question = st.text_input("Ask a question:")

if st.button("Submit") and question:
    with st.spinner("Fetching answer..."):
        answer, sources = main_query(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, s in enumerate(sources, 1):
        st.write(f"{i}. {s['source']} -> {s['text'][:200]}{'...' if len(s['text'])>200 else ''}")
