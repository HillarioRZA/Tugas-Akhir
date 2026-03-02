# File: backend/services/memory/persistent_memory.py
import os
import shutil
from tinydb import TinyDB, Query
from typing import Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

db = TinyDB('memory_db.json')
Q = Query()

model_registry = db.table('model_registry')
dataset_registry = db.table('dataset_registry')
chat_history = db.table('chat_history')

print("--- Persistent Memory Manager (TinyDB) Initialized ---")

def save_model_data(session_id: str, model_name: str, metrics: dict, model_path: str, preprocessor_path: str):
    """
    Menyimpan atau memperbarui data (path dan metrik) model ke TinyDB 
    berdasarkan session_id dan model_name.
    """
    model_data = {
        "session_id": session_id,
        "model_name": model_name,
        "metrics": metrics,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path
    }

    model_registry.upsert(model_data, (Q.session_id == session_id) & (Q.model_name == model_name))
    print(f"--- [LTM] Data Model '{model_name}' disimpan ke DB untuk sesi {session_id} ---")

def get_model_data(session_id: str, model_name: str) -> Optional[dict]:
    if not model_name:
        return None
    result = model_registry.get((Q.session_id == session_id) & (Q.model_name == model_name))
    return result

def save_dataset_path(session_id: str, dataset_name: str, dataset_path: str):
    dataset_data = {
        "session_id": session_id,
        "dataset_name": dataset_name, 
        "path": dataset_path
    }
    dataset_registry.upsert(dataset_data, (Q.session_id == session_id) & (Q.dataset_name == dataset_name))
    print(f"--- [LTM] Path Dataset '{dataset_name}' disimpan ke DB untuk sesi {session_id} ---")

def get_dataset_path(session_id: str, dataset_name: str) -> Optional[dict]:
    if not dataset_name:
        return None
    result = dataset_registry.get((Q.session_id == session_id) & (Q.dataset_name == dataset_name))
    return result

def save_chat_history(session_id: str, memory_object: ConversationBufferMemory):
    messages_dict = messages_to_dict(memory_object.chat_memory.messages)
    
    chat_history.upsert(
        {"session_id": session_id, "messages": messages_dict},
        Q.session_id == session_id
    )
    print(f"--- [LTM] Riwayat Chat disimpan ke DB untuk sesi {session_id} ---")

def load_chat_history(session_id: str) -> ConversationBufferMemory:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    data = chat_history.get(Q.session_id == session_id)
    
    if data and data.get('messages'):
        try:
            messages = messages_from_dict(data['messages'])
            memory.chat_memory.messages = messages
            print(f"--- [LTM] Riwayat Chat dimuat dari DB untuk sesi {session_id} ---")
        except Exception as e:
            print(f"Error memuat riwayat chat dari LTM: {e}. Membuat memori baru.")
    else:
        print(f"--- [LTM] Tidak ada riwayat chat LTM ditemukan. Membuat memori baru untuk sesi {session_id} ---")
            
    return memory

def clear_all_memory_for_session(session_id: str):
    print(f"--- [LTM] Memulai pembersihan total LTM untuk sesi {session_id} ---")

    model_registry.remove(Q.session_id == session_id)
    dataset_registry.remove(Q.session_id == session_id)
    chat_history.remove(Q.session_id == session_id)
    print(f"--- [LTM] Data TinyDB untuk sesi {session_id} dihapus ---")

    session_model_dir = os.path.join("saved_models", session_id)
    session_upload_dir = os.path.join("user_uploads", session_id)

    try:
        if os.path.exists(session_model_dir):
            shutil.rmtree(session_model_dir)
            print(f"--- [LTM] Folder model fisik '{session_model_dir}' dihapus ---")
    except Exception as e:
        print(f"Error menghapus folder model LTM: {e}")

    try:
        if os.path.exists(session_upload_dir):
            shutil.rmtree(session_upload_dir)
            print(f"--- [LTM] Folder upload fisik '{session_upload_dir}' dihapus ---")
    except Exception as e:
        print(f"Error menghapus folder upload LTM: {e}")