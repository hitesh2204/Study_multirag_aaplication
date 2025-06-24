# 🧠 Multimodal RAG App (PDF + Image + YouTube + Question Answering)

This Streamlit app allows users to upload **PDFs**, **Images**, paste **YouTube video URLs**, and ask natural language questions. It uses a **Retrieval-Augmented Generation (RAG)** pipeline powered by HuggingFace models to return intelligent, context-aware answers.

---

## 🚀 Features

- 📄 Extract text from uploaded PDFs
- 🖼️ Perform OCR on uploaded images
- 📺 Retrieve transcripts from YouTube videos
- 💬 Ask any question based on the combined input content
- 🤖 Uses a HuggingFace LLM (`flan-t5-base`) for generating answers
- ⚡ Modular backend design with loaders, embeddings, vectorstore, and chain setup

---

## 🧱 Folder Structure

multimodal_rag_app/
├── streamlit_rag_multimodal_app.py # 🚀 Streamlit frontend
├── loaders/
│ ├── pdf_loader.py # Extracts text from PDFs
│ ├── image_ocr_loader.py # Extracts text from images using OCR
│ └── youtube_loader.py # Gets transcripts from YouTube videos
├── rag_pipeline/
│ ├── splitter.py # Splits text into chunks
│ ├── embeddings.py # Embedding model initialization
│ ├── vectorstore.py # Builds vectorstore using FAISS
│ └── rag_chain.py # Defines the RAG chain with LLM
└── README.md

## 🛠️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/multimodal-rag-app.git
cd multimodal-rag-app

Create a virtual environment

bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Set HuggingFace API Key

You can either:
Export the environment variable:

HUGGINGFACEHUB_API_TOKEN=your_token_here

▶️ Run the App
streamlit run streamlit_rag_multimodal_app.py

🧠 Example Use Case
Upload a PDF of a research paper

Upload an image of handwritten notes or a diagram

Provide a YouTube video link (lecture, short, etc.)

Ask: "Summarize the key points" or "Explain the diagram in the image"

📦 Dependencies
Key Python packages:

streamlit

pytesseract

Pillow

opencv-python

langchain

langchain-community

huggingface_hub

youtube-transcript-api

faiss-cpu or faiss-gpu

🧑‍💻 Author
Hitesh Yerekar
Machine Learning Engineer | AI + RAG Systems Builder