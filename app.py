import os
import streamlit as st
from rag.ingestion import load_documents, split_documents
from rag.retriever import build_index, load_index, get_retriever, INDEX_PATH
from rag.qa_chain import build_qa_chain

st.set_page_config(page_title="MediQuery", page_icon="🩺", layout="wide")
st.title("🩺 MediQuery – Clinical Literature Q&A")
st.caption("Ask evidence-based questions over your medical documents. All inference runs locally.")

with st.sidebar:
    st.header("Document Ingestion")
    if st.button("Index documents in data/"):
        with st.spinner("Loading and indexing documents..."):
            docs = load_documents("data")
            if not docs:
                st.warning("No PDF or TXT files found in data/")
            else:
                chunks = split_documents(docs)
                build_index(chunks)
                st.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")

    st.markdown("---")
    st.markdown("**Model**: Mistral-7B-Instruct (Ollama)")
    st.markdown("**Embeddings**: BioBERT (HuggingFace)")
    st.markdown("**Vector store**: FAISS")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a clinical question..."):
    if not os.path.exists(INDEX_PATH):
        st.error("No index found. Click 'Index documents' in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                vectorstore = load_index()
                retriever = get_retriever(vectorstore)
                chain = build_qa_chain(retriever)
                result = chain({"question": query})

                answer = result.get("answer", "No answer generated.")
                sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Source passages"):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"**[{i}]** `{doc.metadata.get('source', 'unknown')}` p.{doc.metadata.get('page', '?')}")
                            st.text(doc.page_content[:400] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
