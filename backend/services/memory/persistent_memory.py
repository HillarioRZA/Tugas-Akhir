# File: backend/services/memory/persistent_memory.py
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from tinydb import TinyDB, Query
from typing import Dict, Any, Optional
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import messages_from_dict, messages_to_dict

# Fix C5: gunakan path absolut agar TinyDB tidak bergantung pada cwd saat server dijalankan
_MEMORY_DIR  = Path(__file__).parent          # .../backend/services/memory/
_PROJECT_ROOT = _MEMORY_DIR.parents[2]        # .../Data_Whisperer_v1.0/
_DB_PATH     = _MEMORY_DIR / "memory_db.json"  # absolut, selalu di folder memory/

db = TinyDB(str(_DB_PATH))
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

def save_chat_history(session_id: str, memory_object: ConversationBufferWindowMemory):
    """Simpan riwayat chat ke LTM (TinyDB). Menyertakan timestamp untuk auto-cleanup."""
    messages_dict = messages_to_dict(memory_object.chat_memory.messages)

    chat_history.upsert(
        {
            "session_id": session_id,
            "messages":   messages_dict,
            "updated_at": datetime.now(timezone.utc).isoformat(),  # Task 6: timestamp
        },
        Q.session_id == session_id
    )
    print(f"--- [LTM] Riwayat Chat disimpan ke DB untuk sesi {session_id} ---")

def load_chat_history(session_id: str) -> ConversationBufferWindowMemory:
    """
    Muat riwayat chat dari LTM (TinyDB) ke dalam WindowMemory.
    Task 6: Menggunakan ConversationBufferWindowMemory(k=10) — hanya 10 exchange terakhir
    yang dimuat ke konteks aktif, mencegah context window overflow pada sesi panjang.
    """
    # Task 6: window k=10 — hanya 10 pertukaran terakhir masuk ke context LLM
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True
    )

    data = chat_history.get(Q.session_id == session_id)

    if data and data.get('messages'):
        try:
            messages = messages_from_dict(data['messages'])
            # WindowMemory tetap muat semua pesan dari LTM, tapi saat inference
            # hanya k=10 terakhir yang dikirim ke LLM
            memory.chat_memory.messages = messages
            print(f"--- [LTM] Riwayat Chat dimuat dari DB untuk sesi {session_id} "
                  f"({len(messages)} pesan, window k=10) ---")
        except Exception as e:
            print(f"Error memuat riwayat chat dari LTM: {e}. Membuat memori baru.")
    else:
        print(f"--- [LTM] Tidak ada riwayat chat LTM. Membuat memori baru untuk sesi {session_id} ---")

    return memory


def cleanup_old_sessions(max_age_days: int = 7) -> int:
    """
    Task 6: Hapus riwayat chat dari LTM yang tidak diupdate lebih dari `max_age_days` hari.
    Dipanggil saat startup server untuk menjaga memory_db.json tetap bersih.

    Args:
        max_age_days: Ambang batas hari tidak aktif (default: 7 hari)

    Returns:
        Jumlah session yang dihapus.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    all_sessions = chat_history.all()
    removed_count = 0

    for record in all_sessions:
        updated_at_str = record.get("updated_at")
        if not updated_at_str:
            # Session lama (sebelum Task 6) tidak punya timestamp — skip aman
            continue

        try:
            updated_at = datetime.fromisoformat(updated_at_str)
            if updated_at < cutoff:
                session_id = record.get("session_id", "unknown")
                chat_history.remove(Q.session_id == session_id)
                print(f"--- [LTM] Auto-cleanup: sesi '{session_id}' dihapus "
                      f"(tidak aktif sejak {updated_at_str[:10]}) ---")
                removed_count += 1
        except (ValueError, TypeError):
            continue  # Skip record dengan timestamp rusak

    if removed_count:
        print(f"✅ [LTM] Cleanup selesai: {removed_count} sesi lama dihapus.")
    else:
        print(f"✅ [LTM] Cleanup selesai: tidak ada sesi yang perlu dihapus.")

    return removed_count

def clear_all_memory_for_session(session_id: str):
    print(f"--- [LTM] Memulai pembersihan total LTM untuk sesi {session_id} ---")

    model_registry.remove(Q.session_id == session_id)
    dataset_registry.remove(Q.session_id == session_id)
    chat_history.remove(Q.session_id == session_id)
    print(f"--- [LTM] Data TinyDB untuk sesi {session_id} dihapus ---")

    # Fix C5: pakai path absolut untuk folder model dan upload
    session_model_dir  = str(_PROJECT_ROOT / "saved_models" / session_id)
    session_upload_dir = str(_PROJECT_ROOT / "user_uploads"  / session_id)

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