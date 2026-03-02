# File: backend/services/rag/vectorizer.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def create_vector_store(text_content: str):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=500,
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)

        if not chunks:
            return "Error: Tidak ada teks yang bisa diindeks dari PDF."

        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        return vector_store

    except Exception as e:
        print(f"Error saat membuat vector store: {e}")
        return f"Error: Gagal membuat index vector. Detail: {str(e)}"