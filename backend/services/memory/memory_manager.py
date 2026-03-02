from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
from backend.services.memory import persistent_memory

_session_memory: Dict[str, Dict[str, Any]] = {}

def _get_session_data(session_id: str) -> Dict[str, Any]:
    if session_id not in _session_memory:
        _session_memory[session_id] = {
            "active_vector_store": None,
            "chat_memory": None
        }
    return _session_memory[session_id]

def save_vector_store(session_id: str, vector_store: Any):
    session_data = _get_session_data(session_id)
    session_data["active_vector_store"] = vector_store
    print(f"--- Vector Store untuk sesi {session_id} disimpan ---")

def get_vector_store(session_id: str) -> Any | None:
    session_data = _get_session_data(session_id)
    return session_data.get("active_vector_store")

def clear_vector_store(session_id: str):
    if session_id in _session_memory:
        _session_memory[session_id]["active_vector_store"] = None
        print(f"--- Vector Store untuk sesi {session_id} dihapus ---")

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    session_data = _get_session_data(session_id)

    if session_data.get("chat_memory") is None:
        print(f"--- [STM] Memori chat tidak ada di cache. Mencoba memuat dari LTM... ---")

        ltm_memory = persistent_memory.load_chat_history(session_id)

        session_data["chat_memory"] = ltm_memory
        return ltm_memory
    else:
        print(f"--- [STM] Memori chat ditemukan di cache. ---")
        return session_data["chat_memory"]

def clear_chat_memory(session_id: str):
    if session_id in _session_memory:
        _session_memory[session_id]["chat_memory"] = None
        print(f"--- [STM] Memori Chat untuk sesi {session_id} dihapus dari cache ---")

def clear_all_memory_for_session(session_id: str):
    if session_id in _session_memory:
        del _session_memory[session_id]
        print(f"--- [STM] SEMUA memori cache untuk sesi {session_id} dihapus ---")
    else:
        print(f"--- [STM] Tidak ada memori cache ditemukan untuk sesi {session_id} ---")
        
    persistent_memory.clear_all_memory_for_session(session_id)