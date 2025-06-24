# rag_pipeline/vectorstore.py
from langchain_community.vectorstores import FAISS

from langchain_community.vectorstores import FAISS

def build_vectorstore(chunks, embedding_model):
    # Ensure you're only passing plain strings
    clean_chunks = [chunk["page_content"] if isinstance(chunk, dict) else chunk for chunk in chunks]
    return FAISS.from_texts(clean_chunks, embedding_model)
