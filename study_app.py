# streamlit_rag_multimodal_app.py

import streamlit as st
from loaders.pdf_loader import extract_text_from_pdf
from loaders.image_ocr_loader import extract_text_from_image
from loaders.youtube_loader import get_youtube_transcript
from rag_pipeline.embeddings import get_embedding_model
from rag_pipeline.vectorstore import build_vectorstore
from rag_pipeline.splitter import split_text
from rag_pipeline.rag_chain import get_rag_chain

from langchain_community.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="🧠 Multimodal RAG App", layout="centered")

st.title("📚🖼️🎥 Study _App")
st.markdown("Upload **PDF**, **Image**, paste **YouTube Link**, and ask a question.")

# --- Upload Inputs ---
pdf_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload Image", type=["png", "jpg", "jpeg"])
youtube_url = st.text_input("📺 Paste YouTube URL (e.g. https://www.youtube.com/watch?v=...)")
user_question = st.text_input("💬 Your Question", placeholder="Ask something based on uploaded data...")

if st.button("🧠 Get Answer") and user_question:
    with st.spinner("🔍 Processing..."):

        pdf_text = ""
        image_text = ""
        youtube_text = ""

        # Save and extract PDF
        if pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                tmp_pdf_path = tmp_pdf.name
            try:
                pdf_text = extract_text_from_pdf(tmp_pdf_path)
                st.success("✅ PDF text extracted.")
            except Exception as e:
                st.error(f"❌ PDF error: {e}")
            os.remove(tmp_pdf_path)

        # Save and extract image
        if image_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                tmp_img.write(image_file.read())
                tmp_img_path = tmp_img.name
            try:
                image_text = extract_text_from_image(tmp_img_path)
                st.success("✅ Image OCR complete.")
            except Exception as e:
                st.error(f"❌ Image error: {e}")
            os.remove(tmp_img_path)

        # Extract YouTube transcript
        if youtube_url.strip():
            try:
                youtube_text = get_youtube_transcript(youtube_url)
                st.success("✅ YouTube transcript retrieved.")
            except Exception as e:
                st.error(f"❌ YouTube error: {e}")

        # Combine and check
        full_text = "\n".join([pdf_text, image_text, youtube_text]).strip()
        if not full_text:
            st.error("❌ No valid input content found.")
        else:
            # Process RAG
            try:
                chunks = split_text(full_text)
                embedding_model = get_embedding_model()
                vectorstore = build_vectorstore(chunks, embedding_model)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
                )

                rag_chain = get_rag_chain(retriever, llm)

                st.info("💬 Answering your question...")
                answer = rag_chain.invoke({"question": user_question})

                st.subheader("📘 Question")
                st.write(user_question)

                st.subheader("🧠 Answer")
                st.success(answer)
            except Exception as e:
                st.error(f"❌ RAG processing error: {e}")
