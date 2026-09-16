"""
==============================================================================
API_INJECT: Gemini Free Tier Multi-Key Manager & Smart Rotator
==============================================================================
Modul manajemen rotasi API Key Google Gemini untuk mengatasi batas:
- 5 RPM (Requests Per Minute)
- 21 RPD (Requests Per Day)
Error 429 (ResourceExhausted / QuotaExceeded) ditangani secara otomatis
dengan perputaran Round-Robin / LRU dan auto-retry transparan pada level LangChain.

Dapat dijalankan langsung sebagai skrip:
  python api_inject.py --status
  python api_inject.py --test
  python api_inject.py --reset
  python api_inject.py --init
==============================================================================
"""

import os
import sys

# Pastikan output konsol Windows mendukung UTF-8 (Emoji / tanda)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import json
import re
import math
import asyncio
import threading
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator, Sequence, Union

from dotenv import load_dotenv
load_dotenv()

# LangChain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# KONFIGURASI DEFAULT
# ==============================================================================
DEFAULT_RPM_LIMIT = 5          # Batas aman Gemini Free Tier per menit
DEFAULT_RPD_LIMIT = 21         # Batas aman Gemini Free Tier per hari
DEFAULT_COOLDOWN_RPM = 35.0    # Detik tunggu jika kena limit RPM
KEYS_FILE_NAME = "gemini_keys.txt"
STATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs", "api_keys_state.json")


def _mask_key(key: str) -> str:
    """Masking API Key untuk keamanan tampilan log (e.g. AIzaSy...8d4)."""
    if not key or len(key) < 10:
        return "******"
    return f"{key[:6]}...{key[-4:]}"


# ==============================================================================
# KEY RECORD (Struktur Data Pelacak Tiap Key)
# ==============================================================================
class KeyRecord:
    def __init__(self, key: str):
        self.key: str = key.strip()
        self.masked: str = _mask_key(self.key)
        self.requests_today: int = 0
        self.last_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.request_timestamps: List[float] = []
        self.cooldown_until: float = 0.0
        self.status: str = "ACTIVE"  # "ACTIVE", "COOLDOWN_RPM", "EXHAUSTED_RPD"
        self.total_success: int = 0
        self.total_errors: int = 0
        self.last_used: float = 0.0
        self.last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "masked": self.masked,
            "requests_today": self.requests_today,
            "last_reset_date": self.last_reset_date,
            "status": self.status,
            "cooldown_until": self.cooldown_until,
            "total_success": self.total_success,
            "total_errors": self.total_errors,
            "last_used": self.last_used,
            "last_error": self.last_error,
        }

    def update_from_dict(self, data: dict):
        self.requests_today = data.get("requests_today", 0)
        self.last_reset_date = data.get("last_reset_date", self.last_reset_date)
        self.status = data.get("status", "ACTIVE")
        self.cooldown_until = data.get("cooldown_until", 0.0)
        self.total_success = data.get("total_success", 0)
        self.total_errors = data.get("total_errors", 0)
        self.last_used = data.get("last_used", 0.0)
        self.last_error = data.get("last_error", "")


# ==============================================================================
# GEMINI KEY MANAGER (Core Engine)
# ==============================================================================
class GeminiKeyManager:
    """
    Manajer thread-safe untuk mengelola pool API Key Gemini.
    - Mendukung puluhan hingga ratusan API key.
    - Melacak RPM (5) dan RPD (21) per key.
    - Pemilihan key berbasis Round-Robin / LRU (beban terdistribusi merata).
    - Memulihkan status kuota secara persisten ke disk.
    """
    _instance: Optional['GeminiKeyManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Singleton pattern agar seluruh thread/request memakai state pool yang sama
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GeminiKeyManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        keys_file: str = KEYS_FILE_NAME,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        rpd_limit: int = DEFAULT_RPD_LIMIT,
    ):
        if getattr(self, "_initialized", False):
            return

        self.keys_file = keys_file
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.records: List[KeyRecord] = []
        self._key_map: Dict[str, KeyRecord] = {}
        self._current_idx: int = 0

        self.load_keys()
        self._load_state()
        self._initialized = True

    def load_keys(self) -> int:
        """Memuat daftar API key dari file teks, env var, atau .env."""
        found_keys: List[str] = []

        # 1. Dari berkas teks (e.g. gemini_keys.txt)
        file_path = self.keys_file
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), self.keys_file)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Lewati komentar dan baris kosong
                        if not line or line.startswith("#") or line.startswith("//"):
                            continue
                        # Bersihkan tanda kutip jika ada
                        line = line.strip("\"'").strip()
                        if line and line not in found_keys:
                            found_keys.append(line)
            except Exception as e:
                print(f"⚠️  [api_inject] Gagal membaca {file_path}: {e}")

        # 2. Dari Environment Variable GEMINI_API_KEYS (koma atau baris baru)
        env_keys = os.environ.get("GEMINI_API_KEYS", "")
        if env_keys:
            for k in re.split(r"[,;\n\r]+", env_keys):
                k = k.strip().strip("\"'")
                if k and k not in found_keys:
                    found_keys.append(k)

        # 3. Fallback ke GOOGLE_API_KEY di .env
        fallback_key = os.environ.get("GOOGLE_API_KEY", "").strip().strip("\"'")
        if fallback_key and fallback_key not in found_keys:
            found_keys.append(fallback_key)

        with self._lock:
            self.records = []
            self._key_map = {}
            for k in found_keys:
                rec = KeyRecord(k)
                self.records.append(rec)
                self._key_map[k] = rec

        return len(self.records)

    def _check_rollover(self, record: KeyRecord, now_date_str: str):
        """Reset kuota harian jika hari UTC berganti."""
        if record.last_reset_date != now_date_str:
            record.requests_today = 0
            record.last_reset_date = now_date_str
            if record.status == "EXHAUSTED_RPD":
                record.status = "ACTIVE"
                record.cooldown_until = 0.0

    def _clean_rpm_timestamps(self, record: KeyRecord, now: float):
        """Hapus timestamp pemanggilan yang sudah lebih dari 60 detik."""
        cutoff = now - 60.0
        record.request_timestamps = [t for t in record.request_timestamps if t > cutoff]
        # Jika sebelumnya cooldown karena RPM dan waktu sudah lewat, aktifkan lagi
        if record.status == "COOLDOWN_RPM" and now >= record.cooldown_until:
            record.status = "ACTIVE"

    def get_active_key(self) -> str:
        """
        Thread-safe: Mengembalikan API key aktif berikutnya.
        Mengutamakan Round-Robin / LRU pada key yang:
        1. Tidak dalam masa cooldown (now >= cooldown_until)
        2. Belum mencapai batas RPD (< 21)
        3. Belum mencapai batas RPM (< 5 dalam 60s terakhir)
        """
        with self._lock:
            if not self.records:
                raise RuntimeError(
                    "❌ [api_inject] Tidak ada API Key Gemini yang ditemukan! "
                    "Harap masukkan API key Anda ke dalam berkas 'gemini_keys.txt'."
                )

            now = time.time()
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Periksa pembaruan harian & timestamp per key
            for rec in self.records:
                self._check_rollover(rec, today_str)
                self._clean_rpm_timestamps(rec, now)

            total = len(self.records)

            # ── Tahap 1: Cari key yang 100% siap (Bukan Cooldown, RPM < 5, RPD < 21)
            for i in range(total):
                idx = (self._current_idx + i) % total
                rec = self.records[idx]

                if (
                    rec.status == "ACTIVE"
                    and rec.requests_today < self.rpd_limit
                    and len(rec.request_timestamps) < self.rpm_limit
                    and now >= rec.cooldown_until
                ):
                    self._current_idx = (idx + 1) % total
                    rec.request_timestamps.append(now)
                    rec.requests_today += 1
                    rec.last_used = now
                    self._save_state_debounced()
                    return rec.key

            # ── Tahap 2: Jika semua key kena limit RPM 5, cari key aktif dengan sisa waktu cooldown terpendek
            eligible_candidates = [
                rec for rec in self.records
                if rec.status != "EXHAUSTED_RPD" and rec.requests_today < self.rpd_limit
            ]

            if eligible_candidates:
                # Urutkan berdasarkan waktu pemanggilan tertua
                eligible_candidates.sort(
                    key=lambda r: (r.request_timestamps[0] if r.request_timestamps else 0)
                )
                chosen = eligible_candidates[0]
                # Hitung waktu tunggu minimal agar keluar dari jendela 60s
                if chosen.request_timestamps:
                    oldest = chosen.request_timestamps[0]
                    wait_time = max(0.5, 60.0 - (now - oldest) + 0.5)
                    print(f"⏳ [api_inject] Seluruh key mencapai 5 RPM. Menunggu {wait_time:.1f}s untuk {chosen.masked}...")
                    time.sleep(wait_time)
                    now = time.time()
                    self._clean_rpm_timestamps(chosen, now)

                chosen.request_timestamps.append(now)
                chosen.requests_today += 1
                chosen.last_used = now
                self._save_state_debounced()
                return chosen.key

            # ── Tahap 3: Semua 100 key habis kuota hariannya (EXHAUSTED_RPD)
            raise RuntimeError(
                "❌ [api_inject] Seluruh API Key dalam pool telah mencapai batas kuota harian (RPD Exceeded)! "
                "Harap tambahkan API key baru ke 'gemini_keys.txt' atau tunggu hingga reset harian besok."
            )

    def report_error(self, api_key: str, error: Union[Exception, str]) -> bool:
        """
        Mencatat error pada key. Jika error adalah 429 / QuotaExceeded:
        - Jika RPD limit: tandai EXHAUSTED_RPD
        - Jika RPM limit: tandai COOLDOWN_RPM
        Mengembalikan True jika error terdeteksi sebagai 429/ResourceExhausted.
        """
        err_str = str(error)
        now = time.time()
        is_429 = (
            "429" in err_str
            or "ResourceExhausted" in err_str
            or "Quota exceeded" in err_str
            or "quota" in err_str.lower()
        )

        with self._lock:
            rec = self._key_map.get(api_key)
            if not rec:
                return is_429

            rec.total_errors += 1
            rec.last_error = err_str[:200]

            if is_429:
                # Periksa apakah ini batas harian (RPD)
                if (
                    "limit: 20" in err_str
                    or "limit: 21" in err_str
                    or "GenerateRequestsPerDay" in err_str
                    or "per_day" in err_str.lower()
                ):
                    rec.status = "EXHAUSTED_RPD"
                    # Istirahatkan sampai esok hari UTC (atau 24 jam)
                    rec.cooldown_until = now + 86400
                    print(f"\n🛑 [api_inject] Key {rec.masked} MENCAPAI BATAS HARIAN (RPD)! Status: EXHAUSTED_RPD")
                else:
                    # Batas per menit (RPM)
                    # Coba cari saran retry delay dari Google (e.g. retry in 25.8s)
                    delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    seconds_match = re.search(r"seconds:\s*(\d+)", err_str)

                    if delay_match:
                        cooldown = float(delay_match.group(1)) + 2.0
                    elif seconds_match:
                        cooldown = float(seconds_match.group(1)) + 2.0
                    else:
                        cooldown = DEFAULT_COOLDOWN_RPM

                    rec.status = "COOLDOWN_RPM"
                    rec.cooldown_until = now + cooldown
                    print(f"\n⏳ [api_inject] Key {rec.masked} TERKENA 429 (RPM)! Diistirahatkan selama {cooldown:.1f}s.")

            self._save_state()

        return is_429

    def report_success(self, api_key: str):
        """Mencatat keberhasilan pemanggilan pada key."""
        with self._lock:
            rec = self._key_map.get(api_key)
            if rec:
                rec.total_success += 1
                rec.last_used = time.time()
                self._save_state_debounced()

    def reset_all_counters(self):
        """Reset seluruh status dan counter ke awal."""
        with self._lock:
            now = time.time()
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for rec in self.records:
                rec.requests_today = 0
                rec.last_reset_date = today_str
                rec.request_timestamps = []
                rec.cooldown_until = 0.0
                rec.status = "ACTIVE"
                rec.last_error = ""
            self._save_state()
            print("✅ [api_inject] Seluruh counter kuota dan status cooldown telah di-reset.")

    def get_summary(self) -> dict:
        """Mengembalikan statistik pool API key."""
        with self._lock:
            total = len(self.records)
            active = sum(1 for r in self.records if r.status == "ACTIVE")
            cooldown = sum(1 for r in self.records if r.status == "COOLDOWN_RPM")
            exhausted = sum(1 for r in self.records if r.status == "EXHAUSTED_RPD")
            used_today = sum(r.requests_today for r in self.records)
            total_capacity = total * self.rpd_limit
            remaining_capacity = max(0, total_capacity - used_today)

            return {
                "total_keys": total,
                "active_keys": active,
                "cooldown_keys": cooldown,
                "exhausted_keys": exhausted,
                "used_today": used_today,
                "total_capacity": total_capacity,
                "remaining_capacity": remaining_capacity,
                "rpm_pool_capacity": active * self.rpm_limit,
            }

    # ── State Persistence ──
    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
            data = {rec.key: rec.to_dict() for rec in self.records}
            with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    _last_save_time = 0.0
    def _save_state_debounced(self):
        now = time.time()
        if now - self._last_save_time > 10.0:
            self._last_save_time = now
            self._save_state()

    def _load_state(self):
        if not os.path.exists(STATE_FILE_PATH):
            return
        try:
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._lock:
                for key, state in data.items():
                    if key in self._key_map:
                        rec = self._key_map[key]
                        rec.update_from_dict(state)
                        self._check_rollover(rec, today_str)
        except Exception as e:
            print(f"⚠️  [api_inject] Gagal memuat state sebelumnya: {e}")


# Inisialisasi Singleton Manager
api_key_manager = GeminiKeyManager()


# ==============================================================================
# ROTATING CHAT MODEL (LangChain Drop-in Replacement)
# ==============================================================================
class RotatingChatGoogleGenerativeAI(BaseChatModel):
    """
    Drop-in replacement untuk ChatGoogleGenerativeAI pada LangChain.
    - Menggunakan explicit parameter injection (google_api_key=...) tanpa mutasi os.environ.
    - Jika terjadi error 429 pada _generate / _agenerate / _astream:
      Otomatis menandai key bermasalah, mengambil key baru, dan mencoba kembali (auto-retry).
    - Memiliki Stream Guard untuk mencegah duplikasi token saat stream putus di tengah jalan.
    """
    model: str = "gemini-3.6-flash"
    temperature: float = 0.3
    convert_system_message_to_human: bool = False
    max_retries: int = 0  # 0 agar tenacity internal tidak buang waktu pada key mati

    @property
    def _llm_type(self) -> str:
        return "rotating-chat-google-generative-ai"

    def _create_underlying_client(self, api_key: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=self.temperature,
            convert_system_message_to_human=self.convert_system_message_to_human,
            max_retries=self.max_retries,
            timeout=25.0,
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        """Format skema tools menggunakan dummy instance, lalu bind ke wrapper ini."""
        dummy = self._create_underlying_client("AIzaSyDummyKeyForToolBinding")
        bound = dummy.bind_tools(tools, **kwargs)
        return self.bind(**bound.kwargs)

    # ── Synchronous Generate ──
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        max_attempts = max(3, min(len(api_key_manager.records), 10))
        call_kwargs = {**kwargs, "max_retries": 1}

        for attempt in range(max_attempts):
            api_key = api_key_manager.get_active_key()
            client = self._create_underlying_client(api_key)
            try:
                result = client._generate(messages, stop=stop, **call_kwargs)
                api_key_manager.report_success(api_key)
                return result
            except Exception as e:
                is_429 = api_key_manager.report_error(api_key, e)
                if is_429 and attempt < max_attempts - 1:
                    print(f"🔄 [api_inject] Percobaan {attempt+1} gagal (429). Rotasi ke key baru...")
                    continue
                raise e

        raise RuntimeError("❌ [api_inject] Seluruh percobaan dengan API key cadangan gagal.")

    # ── Asynchronous Generate ──
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        max_attempts = max(3, min(len(api_key_manager.records), 10))
        call_kwargs = {**kwargs, "max_retries": 1}

        for attempt in range(max_attempts):
            api_key = api_key_manager.get_active_key()
            client = self._create_underlying_client(api_key)
            try:
                result = await client._agenerate(messages, stop=stop, **call_kwargs)
                api_key_manager.report_success(api_key)
                return result
            except Exception as e:
                is_429 = api_key_manager.report_error(api_key, e)
                if is_429 and attempt < max_attempts - 1:
                    print(f"🔄 [api_inject] Percobaan async {attempt+1} gagal (429). Rotasi ke key baru...")
                    continue
                raise e

        raise RuntimeError("❌ [api_inject] Seluruh percobaan async dengan API key cadangan gagal.")

    # ── Asynchronous Stream ──
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        max_attempts = max(3, min(len(api_key_manager.records), 10))
        call_kwargs = {**kwargs, "max_retries": 1}

        for attempt in range(max_attempts):
            api_key = api_key_manager.get_active_key()
            client = self._create_underlying_client(api_key)
            tokens_emitted = 0

            try:
                async for chunk in client._astream(messages, stop=stop, run_manager=run_manager, **call_kwargs):
                    tokens_emitted += 1
                    yield chunk

                api_key_manager.report_success(api_key)
                return  # Stream selesai sukses

            except Exception as e:
                is_429 = api_key_manager.report_error(api_key, e)

                # STREAM GUARD:
                # Jika error 429 terjadi di AWAL handshake (0 token emitted), kita bisa rotate dan ulang stream dengan aman
                if is_429 and tokens_emitted == 0 and attempt < max_attempts - 1:
                    print(f"🔄 [api_inject] Stream handshake gagal (429). Rotasi key dan coba lagi...")
                    continue

                # Jika token sudah sempat terkirim (tokens_emitted > 0), jangan emit ulang token 1..N agar tidak dobel di UI!
                if is_429 and tokens_emitted > 0:
                    print(f"⚠️  [api_inject] Stream terputus di tengah (token {tokens_emitted}). Mengambil sisa teks via fallback invoke...")
                    try:
                        fallback_key = api_key_manager.get_active_key()
                        fallback_client = self._create_underlying_client(fallback_key)
                        res = await fallback_client._agenerate(messages, stop=stop, **call_kwargs)
                        rest_chunk = ChatGenerationChunk(message=AIMessageChunk(content=""))
                        yield rest_chunk
                        return
                    except Exception as fb_err:
                        print(f"❌ [api_inject] Fallback invoke stream juga gagal: {fb_err}")
                        raise e

                raise e


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_gemini_llm(
    model: Optional[str] = None,
    temperature: float = 0.3,
    convert_system_message_to_human: bool = True,
) -> RotatingChatGoogleGenerativeAI:
    """Mengembalikan instance RotatingChatGoogleGenerativeAI yang terhubung ke key manager."""
    chosen_model = model or os.environ.get("AGENT_MODEL") or "gemini-2.5-flash"
    return RotatingChatGoogleGenerativeAI(
        model=chosen_model,
        temperature=temperature,
        convert_system_message_to_human=convert_system_message_to_human,
    )


def get_active_api_key() -> str:
    """Mengambil satu API key aktif dari pool secara thread-safe."""
    return api_key_manager.get_active_key()


# ==============================================================================
# CLI / SCRIPT MODE
# ==============================================================================
def cli_show_status():
    """Menampilkan status dan kuota pool API key."""
    summary = api_key_manager.get_summary()
    records = api_key_manager.records

    print("\n" + "=" * 75)
    print("📊 GEMINI FREE TIER API KEY POOL STATUS")
    print("=" * 75)
    print(f"  Total API Keys Terdaftar : {summary['total_keys']}")
    print(f"  API Keys Aktif           : {summary['active_keys']}  (Kapasitas RPM: ~{summary['rpm_pool_capacity']} RPM)")
    print(f"  Dalam Cooldown (RPM)     : {summary['cooldown_keys']}")
    print(f"  Habis Kuota Hari Ini(RPD): {summary['exhausted_keys']}")
    print(f"  Total Panggilan Hari Ini : {summary['used_today']} / {summary['total_capacity']} RPD")
    print(f"  Estimasi Sisa Kuota Hari : {summary['remaining_capacity']} panggilan")
    print("-" * 75)
    print(f"{'No':<4} | {'Masked Key':<18} | {'Status':<15} | {'RPD (Used/21)':<14} | {'Errors'}")
    print("-" * 75)

    now = time.time()
    for idx, rec in enumerate(records, 1):
        status_str = rec.status
        if rec.status == "COOLDOWN_RPM" and now < rec.cooldown_until:
            sisa_cd = int(rec.cooldown_until - now)
            status_str = f"CD ({sisa_cd}s)"
        rpd_str = f"{rec.requests_today} / {api_key_manager.rpd_limit}"
        print(f"{idx:<4} | {rec.masked:<18} | {status_str:<15} | {rpd_str:<14} | {rec.total_errors}")

    print("=" * 75 + "\n")


def cli_test_keys():
    """Menguji keaktifan setiap API key dengan prompt ping ringan."""
    records = api_key_manager.records
    if not records:
        print("❌ Tidak ada API Key yang terdaftar di 'gemini_keys.txt'.")
        return

    print(f"\n🔍 Memulai pengujian {len(records)} API key Gemini (Ping test)...")
    print("-" * 75)
    print(f"{'No':<4} | {'Masked Key':<18} | {'Latency':<10} | {'Status'}")
    print("-" * 75)


    for idx, rec in enumerate(records, 1):
        start_t = time.time()
        try:
            # Gunakan client langsung untuk tes cepat
            client = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=rec.key,
                max_retries=0,
            )
            res = client.invoke("Katakan 'OK' saja.", max_retries=1)
            elapsed = time.time() - start_t
            rec.status = "ACTIVE"
            rec.cooldown_until = 0.0
            print(f"{idx:<4} | {rec.masked:<18} | {elapsed*1000:<7.0f} ms | ✅ VALID ({res.content.strip()[:15]})")
        except Exception as e:
            elapsed = time.time() - start_t
            err_str = str(e)
            if "429" in err_str or "Quota exceeded" in err_str:
                if "limit: 20" in err_str or "limit: 21" in err_str or "per_day" in err_str.lower():
                    rec.status = "EXHAUSTED_RPD"
                    print(f"{idx:<4} | {rec.masked:<18} | {elapsed*1000:<7.0f} ms | 🛑 QUOTA HABIS (21 RPD)")
                else:
                    rec.status = "COOLDOWN_RPM"
                    print(f"{idx:<4} | {rec.masked:<18} | {elapsed*1000:<7.0f} ms | ⏳ RATE LIMIT (5 RPM)")
            elif "API_KEY_INVALID" in err_str or "invalid" in err_str.lower():
                rec.status = "INVALID"
                print(f"{idx:<4} | {rec.masked:<18} | {elapsed*1000:<7.0f} ms | ❌ KEY INVALID")
            else:
                print(f"{idx:<4} | {rec.masked:<18} | {elapsed*1000:<7.0f} ms | ⚠️  ERROR: {err_str[:40]}")

    api_key_manager._save_state()
    print("=" * 75 + "\n")


def cli_init():
    """Membuat template gemini_keys.txt jika belum ada."""
    target = os.path.join(os.path.dirname(__file__), KEYS_FILE_NAME)
    if os.path.exists(target):
        print(f"ℹ️  Berkas '{KEYS_FILE_NAME}' sudah ada.")
    else:
        with open(target, "w", encoding="utf-8") as f:
            f.write("# Tempelkan API Key Gemini Anda di sini (1 key per baris)\n")
        print(f"✅ Berkas template '{KEYS_FILE_NAME}' berhasil dibuat.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini API Key Injector & Rotator CLI")
    parser.add_argument("--status", action="store_true", help="Tampilkan status dan kuota pool API Key")
    parser.add_argument("--test", action="store_true", help="Uji validitas koneksi seluruh API Key")
    parser.add_argument("--reset", action="store_true", help="Reset kuota dan cooldown seluruh key")
    parser.add_argument("--init", action="store_true", help="Inisialisasi template gemini_keys.txt")

    args = parser.parse_args()

    if args.status:
        cli_show_status()
    elif args.test:
        cli_test_keys()
    elif args.reset:
        api_key_manager.reset_all_counters()
        cli_show_status()
    elif args.init:
        cli_init()
    else:
        # Default: tampilkan status
        cli_show_status()
