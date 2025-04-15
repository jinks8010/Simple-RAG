import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF Q&A with RAG")

st.title("🔍 Ask Questions from a PDF")

rag = RAGEngine()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading PDF and indexing..."):
        rag.load_from_pdf(uploaded_file)
        st.success("PDF processed and ready!")

    user_query = st.text_input("Ask a question based on the uploaded PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            top_k=3
            
            context = rag.retrieve(user_query, top_k=top_k)
            st.subheader("📄 Retrieved Context")
            st.write(context)

            answer = rag.generate_answer(user_query, top_k=top_k)
            st.subheader("💡 Answer")
            st.write(answer)
