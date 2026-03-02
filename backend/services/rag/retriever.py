from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Helper untuk menggabungkan konten dokumen relevan menjadi satu string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_answer(question: str, vector_store, llm_text):
    """
    Menjawab pertanyaan menggunakan alur kerja RAG.

    Args:
        question: Pertanyaan dari pengguna.
        vector_store: Objek FAISS vector store yang aktif.
        llm_text: Objek LLM (misal: ChatGoogleGenerativeAI) yang sudah diinisialisasi.

    Returns:
        Jawaban dari LLM atau pesan error.
    """
    if not hasattr(vector_store, 'as_retriever'):
        return "Error: Vector store tidak valid atau belum diinisialisasi."
            
    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})

        # Ambil dokumen yang relevan SEBELUM chain dijalankan
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"\n--- Dokumen Relevan yang Ditemukan untuk Pertanyaan: '{question}' ---")
        for i, doc in enumerate(relevant_docs):
            print(f"--- Dokumen {i+1} ---")
            print(doc.page_content)
            print("-" * 20)
        print("-------------------------------------------------------------\n")

        template = """Anda adalah asisten AI yang menjawab pertanyaan hanya berdasarkan konteks yang diberikan. 
        Jika Anda tidak tahu jawabannya dari konteks, katakan saja Anda tidak tahu. Jawab dengan ringkas.

        Konteks:
        {context}

        Pertanyaan: {question}

        Jawaban:"""
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm_text
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        
        return answer

    except Exception as e:
        print(f"Error saat menjalankan RAG chain: {e}")
        return f"Error: Gagal memproses pertanyaan RAG. Detail: {str(e)}"