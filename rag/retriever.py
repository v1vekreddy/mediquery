import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
INDEX_PATH = "faiss_index"


def build_index(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def load_index() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def get_retriever(vectorstore: FAISS, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})
