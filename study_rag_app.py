# run_rag_multimodal.py

from loaders.pdf_loader import extract_text_from_pdf
from loaders.image_ocr_loader import extract_text_from_image
from loaders.youtube_loader import get_youtube_transcript
from rag_pipeline.embeddings import get_embedding_model
from rag_pipeline.vectorstore import build_vectorstore

from rag_pipeline.splitter import split_text
from rag_pipeline.embeddings import get_embedding_model
from rag_pipeline.vectorstore import build_vectorstore
from rag_pipeline.rag_chain import get_rag_chain

from langchain_community.llms import HuggingFaceEndpoint  # ✅ Correct!
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()


def main(pdf_path, image_path, youtube_video_id, user_question):
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        print("✅ PDF loaded.")
    except Exception as e:
        print(f"❌ PDF error: {e}")
        pdf_text = ""

    try:
        image_text = extract_text_from_image(image_path)
        print("✅ Image OCR complete.")
    except Exception as e:
        print(f"❌ Image error: {e}")
        image_text = ""

    try:
        youtube_text = get_youtube_transcript(youtube_video_id)
        print("✅ YouTube transcript retrieved.")
    except Exception as e:
        print(f"❌ YouTube error: {e}")
        youtube_text = ""

    full_text = "\n".join([pdf_text, image_text, youtube_text]).strip()

    if not full_text:
        print("❌ No valid input content found.")
        return

    chunks = split_text(full_text)  # returns List[str]
    embedding_model = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    rag_chain = get_rag_chain(retriever, llm)

    print("\n🧠 Asking your question...")
    answer = rag_chain.invoke({"question": user_question})

    print("\n📘 Question:", user_question)
    print("🧠 Answer:", answer)


if __name__ == "__main__":
    # Example usage
    main(
        pdf_path="D://AI_study_companion//data//ML cheetsheet.pdf",
        image_path="D://AI_study_companion//data//rcb.jpeg",
        youtube_video_id="https://www.youtube.com/shorts/w4gKzgELc0U",  # Replace with real ID
        user_question="Summarize all content provided."
    )
