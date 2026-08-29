import { useCallback, useEffect, useRef, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Konstanta
// ─────────────────────────────────────────────────────────────────────────────
const WS_URL            = 'ws://localhost:8000/ws/chat';
const RECONNECT_DELAY   = 3000;   // ms sebelum mencoba reconnect
const MAX_RECONNECT     = 5;      // batas maksimal percobaan reconnect

/**
 * useAgentWebSocket
 *
 * Custom hook untuk mengelola koneksi WebSocket ke WISTA backend.
 *
 * State yang dikembalikan:
 *   messages      — array pesan {id, role, text, time, reasoningSteps?,
 *                                itineraryData?, imageBase64?, isStreaming?}
 *   isConnected   — boolean, true saat WS terbuka
 *   isTyping      — boolean, true saat agent sedang mengetik / tool berjalan
 *   currentAction — string deskripsi tool yang sedang berjalan (mis. "budget_optimizer_tool")
 *   sessionId     — string UUID sesi saat ini
 *   error         — string pesan error terakhir atau null
 *
 * Fungsi yang dikembalikan:
 *   sendMessage(text, filePath?, datasetName?) — kirim pesan user ke backend
 *   clearMessages()                            — reset percakapan
 *   reconnect()                                — paksa reconnect manual
 */
export function useAgentWebSocket() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [messages,      setMessages]      = useState([]);
  const [isConnected,   setIsConnected]   = useState(false);
  const [isTyping,      setIsTyping]      = useState(false);
  const [currentAction, setCurrentAction] = useState('');
  const [sessionId,     setSessionId]     = useState(() => crypto.randomUUID());
  const [error,         setError]         = useState(null);

  // ── Refs (nilai tidak memicu re-render) ────────────────────────────────────
  const wsRef            = useRef(null);        // WebSocket instance
  const reconnectCount   = useRef(0);           // jumlah reconnect yang sudah dicoba
  const reconnectTimer   = useRef(null);        // timer untuk reconnect
  const shouldReconnect  = useRef(true);        // flag: apakah perlu reconnect?
  const msgIdCounter     = useRef(Date.now());  // ID counter untuk pesan

  // ─── Helper: ID unik ────────────────────────────────────────────────────────
  const nextId = () => ++msgIdCounter.current;

  // ─── Helper: format waktu (HH:MM) ──────────────────────────────────────────
  const nowTime = () =>
    new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

  // ─── Append token ke pesan agent terakhir (streaming) ──────────────────────
  const appendTokenToLastAgent = useCallback((token) => {
    setMessages((prev) => {
      // Cari pesan agent paling terakhir yang sedang streaming
      const lastIdx = [...prev].reverse().findIndex(
        (m) => m.role === 'agent' && m.isStreaming,
      );
      if (lastIdx === -1) return prev; // tidak ada → abaikan

      const realIdx = prev.length - 1 - lastIdx;
      const updated = [...prev];
      updated[realIdx] = {
        ...updated[realIdx],
        text: (updated[realIdx].text ?? '') + token,
      };
      return updated;
    });
  }, []);

  // ─── Finalisasi pesan agent (setelah event "done") ──────────────────────────
  const finalizeLastAgent = useCallback((payload) => {
    setMessages((prev) => {
      const lastIdx = [...prev].reverse().findIndex(
        (m) => m.role === 'agent' && m.isStreaming,
      );
      if (lastIdx === -1) return prev;

      const realIdx = prev.length - 1 - lastIdx;
      const updated = [...prev];
      updated[realIdx] = {
        ...updated[realIdx],
        // Jika summary berbeda dari teks streaming, pakai summary dari "done"
        text:           payload.summary || updated[realIdx].text,
        isStreaming:    false,
        reasoningSteps: payload.reasoning_log
          ? payload.reasoning_log.map((s, i) => ({
              step:   i + 1,
              tool:   s.tool_called ?? s.tool ?? '',
              status: s.observation ? 'success' : 'pending',
              detail: s.tool_input
                ? JSON.stringify(s.tool_input).slice(0, 120)
                : (s.observation ?? ''),
            }))
          : updated[realIdx].reasoningSteps,
        itineraryData:  payload.data         ?? updated[realIdx].itineraryData,
        imageBase64:    payload.image_base64  ?? updated[realIdx].imageBase64,
        hasItinerary:   !!payload.data,
        hasImage:       !!payload.image_base64,
      };
      return updated;
    });
  }, []);

  // ─── Handler event WebSocket ────────────────────────────────────────────────
  const handleEvent = useCallback((event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      console.warn('[WS] Pesan non-JSON diterima:', event.data);
      return;
    }

    const { type } = payload;

    switch (type) {

      // ── Streaming token LLM ──────────────────────────────────────────────
      case 'token':
        setIsTyping(true);
        appendTokenToLastAgent(payload.content ?? '');
        break;

      // ── Tool mulai dijalankan ────────────────────────────────────────────
      case 'tool_start': {
        const toolName = payload.tool ?? '';
        setCurrentAction(toolName);
        setIsTyping(true);
        // Tambahkan entri reasoning ke pesan terakhir secara live
        setMessages((prev) => {
          const lastIdx = [...prev].reverse().findIndex(
            (m) => m.role === 'agent' && m.isStreaming,
          );
          if (lastIdx === -1) return prev;
          const realIdx = prev.length - 1 - lastIdx;
          const updated = [...prev];
          const existing = updated[realIdx].reasoningSteps ?? [];
          updated[realIdx] = {
            ...updated[realIdx],
            reasoningSteps: [
              ...existing,
              {
                step:   existing.length + 1,
                tool:   toolName,
                status: 'running',
                detail: payload.input
                  ? JSON.stringify(payload.input).slice(0, 120)
                  : '',
              },
            ],
          };
          return updated;
        });
        break;
      }

      // ── Tool selesai ─────────────────────────────────────────────────────
      case 'tool_end': {
        const toolName = payload.tool ?? '';
        setMessages((prev) => {
          const lastIdx = [...prev].reverse().findIndex(
            (m) => m.role === 'agent' && m.isStreaming,
          );
          if (lastIdx === -1) return prev;
          const realIdx = prev.length - 1 - lastIdx;
          const updated = [...prev];
          const steps   = [...(updated[realIdx].reasoningSteps ?? [])];
          // Cari step running dengan nama tool ini dan tandai success
          const stepIdx = [...steps].reverse().findIndex(
            (s) => s.tool === toolName && s.status === 'running',
          );
          if (stepIdx !== -1) {
            const realStep = steps.length - 1 - stepIdx;
            steps[realStep] = {
              ...steps[realStep],
              status:      'success',
              observation: (payload.output ?? '').slice(0, 120),
            };
          }
          updated[realIdx] = { ...updated[realIdx], reasoningSteps: steps };
          return updated;
        });
        // Bersihkan currentAction hanya jika tool ini yang terakhir aktif
        setCurrentAction((prev) => (prev === toolName ? '' : prev));
        break;
      }

      // ── Gambar/plot diterima ─────────────────────────────────────────────
      case 'image':
        setMessages((prev) => {
          const lastIdx = [...prev].reverse().findIndex(
            (m) => m.role === 'agent' && m.isStreaming,
          );
          if (lastIdx === -1) return prev;
          const realIdx = prev.length - 1 - lastIdx;
          const updated = [...prev];
          updated[realIdx] = {
            ...updated[realIdx],
            imageBase64: payload.data,
            hasImage:    true,
          };
          return updated;
        });
        break;

      // ── Respons final — agent selesai ────────────────────────────────────
      case 'done':
        finalizeLastAgent(payload);
        setCurrentAction('');
        setIsTyping(false);
        break;

      // ── Error dari backend ───────────────────────────────────────────────
      case 'error':
        setError(payload.detail ?? 'Terjadi kesalahan pada backend.');
        setIsTyping(false);
        setCurrentAction('');
        // Tandai pesan terakhir sebagai error
        setMessages((prev) => {
          const lastIdx = [...prev].reverse().findIndex(
            (m) => m.role === 'agent' && m.isStreaming,
          );
          if (lastIdx === -1) return prev;
          const realIdx = prev.length - 1 - lastIdx;
          const updated = [...prev];
          updated[realIdx] = {
            ...updated[realIdx],
            isStreaming: false,
            isError:     true,
            text:        payload.detail ?? 'Terjadi kesalahan.',
          };
          return updated;
        });
        break;

      default:
        // Event tak dikenal — log saja untuk debugging
        console.debug('[WS] Event tidak dikenal:', type, payload);
    }
  }, [appendTokenToLastAgent, finalizeLastAgent]);

  // ─── Buka koneksi WebSocket ─────────────────────────────────────────────────
  const connect = useCallback(() => {
    // Tutup koneksi lama jika masih ada
    if (wsRef.current) {
      wsRef.current.onclose = null; // cegah reconnect loop
      wsRef.current.close();
    }

    console.log(`[WS] Menghubungkan ke ${WS_URL}…`);
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Terhubung ✓');
      setIsConnected(true);
      setError(null);
      reconnectCount.current = 0;
    };

    ws.onmessage = handleEvent;

    ws.onerror = (e) => {
      console.error('[WS] Error:', e);
      setError('Koneksi WebSocket error. Pastikan backend berjalan.');
    };

    ws.onclose = (e) => {
      setIsConnected(false);
      setIsTyping(false);
      setCurrentAction('');
      console.warn(`[WS] Terputus (code=${e.code}). Mencoba reconnect…`);

      if (!shouldReconnect.current) return;
      if (reconnectCount.current >= MAX_RECONNECT) {
        setError(`Gagal terhubung setelah ${MAX_RECONNECT} percobaan. Pastikan server berjalan.`);
        return;
      }

      reconnectCount.current += 1;
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };
  }, [handleEvent]);

  // ─── Mount / unmount ────────────────────────────────────────────────────────
  useEffect(() => {
    shouldReconnect.current = true;
    connect();

    return () => {
      shouldReconnect.current = false;
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  // ─── Kirim pesan ────────────────────────────────────────────────────────────
  const sendMessage = useCallback(
    (text, filePath = null, datasetName = null) => {
      if (!text?.trim()) return;

      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        setError('WebSocket belum terhubung. Tunggu sebentar lalu coba lagi.');
        return;
      }

      const time = nowTime();

      // 1. Tambahkan pesan user ke state
      const userMsg = {
        id:   nextId(),
        role: 'user',
        text: text.trim(),
        time,
      };

      // 2. Tambahkan placeholder pesan agent (streaming = true)
      const agentMsg = {
        id:            nextId(),
        role:          'agent',
        text:          '',           // akan diisi oleh token stream
        time,
        isStreaming:   true,
        reasoningSteps: [],
        hasItinerary:  false,
        hasImage:      false,
        imageBase64:   null,
        itineraryData: null,
        isError:       false,
      };

      setMessages((prev) => [...prev, userMsg, agentMsg]);
      setIsTyping(true);
      setCurrentAction('');
      setError(null);

      // 3. Kirim ke backend via WebSocket
      const payload = {
        prompt:       text.trim(),
        session_id:   sessionId,
        ...(filePath    && { file_path:    filePath }),
        ...(datasetName && { dataset_name: datasetName }),
      };

      ws.send(JSON.stringify(payload));
    },
    [sessionId],
  );

  // ─── Reset percakapan ───────────────────────────────────────────────────────
  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentAction('');
    setIsTyping(false);
    setError(null);
    // Buat session ID baru sehingga history backend juga terpisah
    setSessionId(crypto.randomUUID());
  }, []);

  // ─── Reconnect manual ──────────────────────────────────────────────────────
  const reconnect = useCallback(() => {
    reconnectCount.current = 0;
    setError(null);
    connect();
  }, [connect]);

  // ─── Return API hook ────────────────────────────────────────────────────────
  return {
    messages,
    isConnected,
    isTyping,
    currentAction,
    sessionId,
    error,
    sendMessage,
    clearMessages,
    reconnect,
  };
}
