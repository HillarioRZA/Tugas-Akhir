"""
WebSocket Automated Test — Verifikasi event flow streaming.
Jalankan: python -m backend.testing.ws_test
Mendukung pengujian terhadap live server (ws://localhost:8000/ws/chat)
maupun in-process via TestClient jika server belum berjalan.
"""
import asyncio
import json
import sys
import uuid
from typing import List

WS_URL = "ws://127.0.0.1:8000/ws/chat"
TIMEOUT = 120


async def _test_ws_live():
    import websockets

    print("Target: Live WebSocket Server (" + WS_URL + ")")

    # Test 1: Basic Event Flow
    print("\nTest 1: Basic Event Flow...")
    flow_session = f"ws-live-{uuid.uuid4()}"
    async with websockets.connect(WS_URL, ping_interval=None, ping_timeout=None) as ws:
        await ws.send(json.dumps({
            "prompt": "Buatkan itinerary 1 hari ke Bangli budget 300000",
            "session_id": flow_session,
        }))

        events: List[str] = []
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT)
            event = json.loads(msg)
            events.append(event.get("type", ""))
            if event.get("type") in ("done", "error"):
                break

        print(f"  Received event sequence: {events}")
        if events[-1] == "error":
            print(f"  Error details: {event}")
        assert "tool_start" in events, f"tool_start tidak ditemukan dalam events: {events}"
        assert "tool_end" in events, f"tool_end tidak ditemukan dalam events: {events}"
        assert events[-1] == "done", f"Event terakhir bukan 'done', melainkan: {events[-1]} (detail: {event.get('detail')})"
        print("  ✓ Test 1 Passed: Complete event flow (tool_start -> tool_end -> done)")

    # Test 2: Error Handling
    print("\nTest 2: Error Handling...")
    async with websockets.connect(WS_URL, ping_interval=None, ping_timeout=None) as ws:
        await ws.send(json.dumps({
            "prompt": "",
            "session_id": f"ws-err-{uuid.uuid4()}",
        }))

        msg = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT)
        event = json.loads(msg)
        print(f"  Response event: {event}")
        assert event.get("type") in ("done", "error"), f"Unexpected event type: {event.get('type')}"
        assert event.get("type") == "error", f"Diharapkan 'error', diterima: {event.get('type')}"
        print("  ✓ Test 2 Passed: Error event properly emitted for empty prompt")


def _test_ws_inprocess():
    from fastapi.testclient import TestClient
    from backend.api.main import app

    print("Target: In-Process FastAPI TestClient (/ws/chat)")
    client = TestClient(app)

    # Test 1: Error Handling
    print("\nTest 1: Error Handling...")
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "prompt": "",
            "session_id": f"ws-inprocess-err-{uuid.uuid4()}",
        })
        data = ws.receive_json()
        print(f"  Response event: {data}")
        assert data.get("type") == "error", f"Diharapkan 'error', diterima: {data.get('type')}"
        print("  ✓ Test 1 Passed: Error event properly emitted for empty prompt")

    # Test 2: Basic Event Flow
    print("\nTest 2: Basic Event Flow...")
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "prompt": "Buatkan itinerary 1 hari ke Bangli budget 300000",
            "session_id": f"ws-inprocess-flow-{uuid.uuid4()}",
        })

        events: List[str] = []
        while True:
            data = ws.receive_json()
            events.append(data.get("type", ""))
            if data.get("type") in ("done", "error"):
                break

        print(f"  Received event sequence: {events}")
        assert "tool_start" in events, f"tool_start tidak ditemukan dalam events: {events}"
        assert "tool_end" in events, f"tool_end tidak ditemukan dalam events: {events}"
        assert events[-1] == "done", f"Event terakhir bukan 'done', melainkan: {events[-1]}"
        print("  ✓ Test 2 Passed: Complete event flow (tool_start -> tool_end -> done)")


async def main():
    print("=" * 60)
    print("WebSocket Automated Test Suite (WISTA)")
    print("=" * 60)

    try:
        import websockets
        # Coba koneksi ke live server terlebih dahulu
        try:
            async with websockets.connect(WS_URL, open_timeout=2):
                pass
            is_live = True
        except Exception:
            is_live = False
    except ImportError:
        is_live = False

    try:
        if is_live:
            await _test_ws_live()
        else:
            print("ℹ️  Live server di localhost:8000 tidak aktif. Menggunakan TestClient in-process mode.")
            _test_ws_inprocess()

        print("\n" + "=" * 60)
        print("ALL WEBSOCKET TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ WEBSOCKET TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
