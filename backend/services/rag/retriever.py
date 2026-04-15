from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    """Helper untuk menggabungkan konten dokumen relevan menjadi satu string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_answer(question: str, vector_store, llm_text):
    """
    Menjawab pertanyaan menggunakan alur kerja RAG.

    Alur:
        1. Retrieve top-K dokumen relevan dari vector store (1x call)
        2. Format dokumen ke string context
        3. Inject context + question ke prompt template
        4. LLM generate jawaban berdasarkan context
        5. Return jawaban sebagai string

    Args:
        question  : Pertanyaan / extraction prompt dari pengguna.
        vector_store: Objek FAISS vector store yang aktif.
        llm_text  : Objek LLM yang sudah diinisialisasi.

    Returns:
        Jawaban dari LLM (str), atau pesan error jika gagal.
    """
    if not hasattr(vector_store, 'as_retriever'):
        return "Error: Vector store tidak valid atau belum diinisialisasi."

    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})

        # ── Step 1: Retrieve dokumen (SEKALI saja) ──────────────────────────
        relevant_docs = retriever.invoke(question)

        # Logging ringkas untuk debug
        print(f"\n--- RAG Retrieval: '{question[:60]}...' ---")
        for i, doc in enumerate(relevant_docs):
            print(f"  [{i+1}] {doc.page_content[:120]}...")
        print("─" * 60)

        if not relevant_docs:
            return "Error: Tidak ada dokumen relevan yang ditemukan di knowledge base."

        # ── Step 2: Format context dari dokumen yang sudah diambil ──────────
        context_str = format_docs(relevant_docs)

        # ── Step 3: Build prompt + chain (pakai context yang sudah ada) ──────
        template = """Anda adalah asisten AI yang menjawab pertanyaan hanya berdasarkan konteks yang diberikan.
        Jika Anda tidak tahu jawabannya dari konteks, katakan saja Anda tidak tahu. Jawab dengan ringkas.

        Konteks:
        {context}

        Pertanyaan: {question}

        Jawaban:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Gunakan RunnableLambda agar context tidak perlu di-retrieve ulang
        rag_chain = (
            {
                "context": RunnableLambda(lambda _: context_str),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm_text
            | StrOutputParser()
        )

        # ── Step 4: Invoke chain dengan question saja ────────────────────────
        answer = rag_chain.invoke(question)
        return answer

    except Exception as e:
        print(f"Error saat menjalankan RAG chain: {e}")
        return f"Error: Gagal memproses pertanyaan RAG. Detail: {str(e)}"