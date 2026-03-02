import os
import base64
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import memory_manager
from backend.services.memory import persistent_memory

from backend.services.eda.main import get_eda_tools
from backend.services.visualization.main import get_visualization_tools
from backend.services.ml.main import get_ml_tools
from backend.services.rag.main import get_rag_tools
from backend.services.optimizer.main import get_optimizer_tools

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(model="openai/gpt-oss-120b")

def run_agent_flow(session_id: str, prompt: str, new_file_path: Optional[str], new_dataset_name: Optional[str]):
    column_list = []
    file_path_to_use = new_file_path
    file_type = None

    if new_file_path and new_dataset_name:
        if new_dataset_name.endswith('.csv'):
            file_type = 'csv'
            try:
                with open(new_file_path, 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                if df is not None:
                    column_list = df.columns.tolist()

                persistent_memory.save_dataset_path(session_id, "__latest_csv", new_file_path)
            except Exception as e:
                print(f"Gagal membaca file CSV baru untuk kolom: {e}")
        
        elif new_dataset_name.endswith('.pdf'):
            file_type = 'pdf'
            persistent_memory.save_dataset_path(session_id, "__latest_pdf", new_file_path)

    elif not new_file_path:
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                file_path_to_use = dataset_info['path']
                file_type = 'csv'
                with open(file_path_to_use, 'rb') as f:
                    csv_contents_bytes = f.read()
                df = _read_csv_with_fallback(csv_contents_bytes)
                if df is not None:
                    column_list = df.columns.tolist()
            except Exception as e:
                print(f"Gagal memuat kolom dari file CSV di LTM: {e}")

    context = {}
    
    eda_tools = get_eda_tools(session_id, context)
    vis_tools = get_visualization_tools(session_id, context)
    ml_tools = get_ml_tools(session_id, context)
    rag_tools = get_rag_tools(session_id, context, llm)
    optimizer_tools = get_optimizer_tools(session_id, context)
    
    tools = eda_tools + vis_tools + ml_tools + rag_tools + optimizer_tools
    
    memory_stm = memory_manager.get_or_create_memory(session_id)
    chat_history = memory_stm.load_memory_variables({})['chat_history']
    
    columns_str = ", ".join(column_list) if column_list else "Tidak ada file CSV yang dikonfirmasi."
    
    system_prompt = f"""Anda adalah AI data analyst agent yang canggih yang dioperasikan melalui arsitektur ReAct (Reasoning and Acting).
Anda memiliki akses ke berbagai tool untuk Exploratory Data Analysis (EDA), Machine Learning, RAG, Visualisasi, dan Optimizer Destinasi.

Kolom yang tersedia di dataset saat ini (jika ada):
{columns_str}

SOP PEMBUATAN ITINERARY PERJALANAN (TRAVEL OPTIMIZATION)
PENTING: Dataset CSV (kolom di atas) adalah SUMBER KEBENARAN UTAMA untuk ketersediaan destinasi, harga, dan rating. Dokumen PDF (jika ada) HANYA referensi tambahan dari pengguna untuk preferensi.
Jika pengguna secara eksplisit meminta pembuatan Itinerary Wisata, JADWALKAN Chain of Thought Anda secara KETAT:
Langkah 1 (Neuro RAG): Opsional. Jika ada referensi dokumen PDF tambahan dari pengguna, jalankan `rag_semantic_filter` untuk menangkap preferensi abstraknya. Jika tidak ada dokumen PDF, LEWATI langkah ini dan langsung ekstrak kata kunci kategori/lokasi dari prompt pengguna. KATA KUNCI harus ada di dalam kolom dataset CSV.
Langkah 2 (Data Reality): Lakukan observasi pada batas bawah dataset csv (budget minimal/maksimal, dst). Jika budget pengguna mustahil dijangkau, Anda WAJIB melakukan LOGICAL PUSHBACK (Tolak dengan menyajikan alasan angka statistiknya).
Langkah 3 (Neuro ML): Gunakan fungsi `predict_match_score` ke dalam daftar destinasi yang menjanjikan, untuk mengetahui seberapa yakin probabilitas tempat tersebut relevan secara machine-learning dengan input user. Ekstrak data 'Feature Importance'-nya.
Langkah 4 (Symbolic Optimizer): Masukkan keyword yg diekstrak dari langkah 1 (dalam bentuk array JSON) beserta budget pengguna ke alat kalkulator `budget_optimizer_tool`. Alat ini pasti tidak berhalusinasi. Jika gagal mencari list, kembalikan ke Langkah 1 dengan kata filter yg lebih lebar (Self-Correction).
Langkah 5 (XAI Vis): SETELAH list didapat dari Langkah 4, WAJIB mempresentasikan pilihan akhir melalui alat `plot_itinerary_scatter` (ini untuk mereduksi opaque/kotak hitam di mata user) lalu berikan jawaban narasinya merujuk pada gambar plot tersebut dan ML Feature Importance pada Langkah 3.

Aturan Penggunaan Tool UMUM & Alur Berpikir (CoT):
Anda harus selalu berpikir secara bertahap menggunakan pola ReAct (Reasoning and Acting):
1. THOUGHT (Pemikiran): Pahami masalah. "Apa yang pengguna minta? Apa yang harus saya lakukan pertama kali?"
2. ACTION (Aksi): Panggil tool yang sesuai berdasarkan pemikiran tersebut.
3. OBSERVATION (Observasi): Evaluasi hasil dari tool (sistem akan memberikannya kepada Anda).
4. REFLECTION (Refleksi): "Apakah hasil observasi ini sudah menjawab masalah? Atau adakah error? Jika kurang, langkah apa selanjutnya?"
5. Ulangi siklus ini sampai masalah terpecahkan, lalu berikan jawaban akhir.

- Jangan pernah menyerah hanya dengan satu kali percobaan. Jika tool gagal, ubah argumennya dan coba lagi, atau gunakan tool lain.
- JANGAN berhalusinasi argumen tool (misalnya nama kolom yang tidak ada di daftar atas).
- Jawab dengan bahasa Indonesia yang natural, informatif, dan ringkas.
- Jika pengguna bertanya sesuatu secara konversasional tentang riwayat obrolan yang tidak membutuhkan operasi data kompleks, cukup jawab langsung saja dari riwayat, tanpa memanggil tool.

---"""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10 # Extended to accommodate Neuro-Symbolic 5-steps loop
    )
    
    try:
        response = agent_executor.invoke({
            "input": prompt,
            "chat_history": chat_history,
        })
        agent_response = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])
        
        reasoning_log = []
        for step_idx, (action, observation) in enumerate(intermediate_steps):
            log_entry = {
                "step": step_idx + 1,
                "thought": action.log.split("Action:")[0].strip() if hasattr(action, 'log') else "Memutuskan aksi...",
                "tool_called": action.tool,
                "tool_input": action.tool_input,
                "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
            }
            reasoning_log.append(log_entry)
        
        inputs = {"input": prompt}
        outputs = {"output": agent_response}
        memory_stm.save_context(inputs, outputs)
        persistent_memory.save_chat_history(session_id, memory_stm)
        print(f"--- [STM] Konteks disimpan ke cache memori sesi {session_id} ---")

        result = {
            "summary": agent_response,
            "reasoning_log": reasoning_log
        }
        
        if "last_tool_name" in context:
            result["tool_name"] = context["last_tool_name"]
        
        if "last_tool_output" in context:
            result["data"] = context["last_tool_output"]
        if "last_image_bytes" in context:
            image_bytes = context["last_image_bytes"]
            result["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            result["image_format"] = "png"
            
            from backend.services.agent.interpretation import get_interpretation
            tool_name = context.get("last_tool_name", "custom plot")
            
            tool_params = context.get("last_tool_params", {})
            try:
                interpretation = get_interpretation(session_id, tool_name, {"tool_name": tool_name, **tool_params}, image_bytes=image_bytes)
                result["summary"] += "\n\n" + interpretation
            except Exception as e:
                print(f"Gagal generate interpretasi gambar: {e}")

        return result

    except Exception as e:
        return {"error": "Gagal menjalankan agen", "detail": str(e)}

class PlotPlan(BaseModel):
    plot_type: str = Field(description="Tipe plot, harus salah satu dari: bar, box, histogram, scatter.")
    x_col: str = Field(description="Nama kolom untuk sumbu X.")
    y_col: Optional[str] = Field(default=None, description="Nama kolom untuk sumbu Y.")
    hue_col: Optional[str] = Field(default=None, description="Nama kolom untuk pewarnaan (hue).")
    orientation: str = Field(default='v', description="Orientasi plot, 'v' untuk vertikal, 'h' untuk horizontal.")

def get_plot_plan(user_prompt: str) -> dict:
    parser = JsonOutputParser(pydantic_object=PlotPlan)
    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten yang tugasnya mengekstrak parameter untuk membuat plot dari permintaan pengguna.
        {format_instructions}
        Permintaan Pengguna: {user_input}"""
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "user_input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        return {"error": "Gagal mengekstrak parameter plot.", "detail": str(e)}