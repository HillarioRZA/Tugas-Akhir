import { useCallback, useEffect, useRef, useState } from 'react';
import { INITIAL_SESSIONS } from '../data/dummy';

// ─────────────────────────────────────────────────────────────────────────────
// Konstanta
// ─────────────────────────────────────────────────────────────────────────────
const WS_URL             = 'ws://localhost:8000/ws/chat';
const RECONNECT_DELAY    = 3000;   // ms sebelum mencoba reconnect
const MAX_RECONNECT      = 5;      // batas maksimal percobaan reconnect
const STORAGE_KEY        = 'wista_sessions_v1';
const ACTIVE_SESSION_KEY = 'wista_active_session_id_v1';

/**
 * Helper untuk memuat sesi dari localStorage atau fallback ke INITIAL_SESSIONS.
 */
function loadInitialSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map((s) => ({
          ...s,
          messages: (s.messages || []).map((m) => {
            if (m.isStreaming) {
              return {
                ...m,
                isStreaming: false,
                text: m.text || '',
              };
            }
            return m;
          }),
        }));
      }
    }
  } catch (err) {
    console.warn('[WS] Gagal memuat sesi dari localStorage:', err);
  }
  return INITIAL_SESSIONS;
}

/**
 * Helper untuk menentukan active session ID awal.
 */
function loadInitialActiveId(sessionsList) {
  try {
    const saved = localStorage.getItem(ACTIVE_SESSION_KEY);
    if (saved && sessionsList.some((s) => String(s.id) === String(saved))) {
      return String(saved);
    }
  } catch {
    // fallback
  }
  return String(sessionsList[0]?.id || '1');
}

/**
 * Helper untuk menyimpan sesi ke localStorage dengan aman.
 */
function saveSessionsToStorage(updatedSessions, currentActiveId) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedSessions));
    if (currentActiveId != null) {
      localStorage.setItem(ACTIVE_SESSION_KEY, String(currentActiveId));
    }
  } catch (err) {
    console.warn('[WS] Gagal menyimpan sesi ke localStorage:', err);
    // Jika quota penuh (misal karena base64 gambar), buang base64 pada sesi lama
    try {
      const lightweight = updatedSessions.map((s, idx) => {
        if (idx > 1) {
          return {
            ...s,
            messages: s.messages?.map((m) => ({ ...m, imageBase64: null })) || [],
          };
        }
        return s;
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(lightweight));
    } catch {
      // ignore
    }
  }
}

/**
 * useAgentWebSocket
 *
 * Custom hook untuk mengelola koneksi WebSocket dan multi-sesi percakapan ke WISTA backend.
 */
export function useAgentWebSocket() {
  // ── Sesi & Percakapan State ────────────────────────────────────────────────
  const [sessions,        setSessions]        = useState(loadInitialSessions);
  const [activeSessionId, setActiveSessionId] = useState(() => loadInitialActiveId(sessions));

  const [messages, setMessages] = useState(() => {
    const active = sessions.find((s) => String(s.id) === String(activeSessionId));
    return active?.messages || [];
  });

  const [isConnected,   setIsConnected]   = useState(false);
  const [isTyping,      setIsTyping]      = useState(false);
  const [currentAction, setCurrentAction] = useState('');
  const [sessionId,     setSessionId]     = useState(() => String(activeSessionId));
  const [error,         setError]         = useState(null);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const wsRef              = useRef(null);
  const reconnectCount     = useRef(0);
  const reconnectTimer     = useRef(null);
  const shouldReconnect    = useRef(true);
  const msgIdCounter       = useRef(Date.now());
  const activeSessionIdRef = useRef(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  const messagesRef        = useRef(messages);
  messagesRef.current      = messages;

  const sessionsRef        = useRef(sessions);
  sessionsRef.current      = sessions;

  // ─── Helper: ID unik & Waktu ───────────────────────────────────────────────
  const nextId = () => ++msgIdCounter.current;
  const nowTime = () =>
    new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

  // ─── Append token ke pesan agent terakhir (streaming) ──────────────────────
  const appendTokenToLastAgent = useCallback((token) => {
    setMessages((prev) => {
      const lastIdx = [...prev].reverse().findIndex(
        (m) => m.role === 'agent' && m.isStreaming,
      );
      if (lastIdx === -1) return prev;

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
      let lastIdx = [...prev].reverse().findIndex(
        (m) => m.role === 'agent' && m.isStreaming,
      );
      if (lastIdx === -1) {
        lastIdx = [...prev].reverse().findIndex((m) => m.role === 'agent');
      }
      if (lastIdx === -1) return prev;

      const realIdx = prev.length - 1 - lastIdx;
      const updated = [...prev];
      const itinData = payload.data ?? updated[realIdx].itineraryData;
      const imgData  = payload.image_base64 ?? updated[realIdx].imageBase64;

      updated[realIdx] = {
        ...updated[realIdx],
        text:           payload.summary || updated[realIdx].text,
        isStreaming:    false,
        reasoningSteps: payload.reasoning_log
          ? payload.reasoning_log.map((s, i) => ({
              step:        s.step ?? (i + 1),
              tool:        s.tool ?? s.tool_called ?? '',
              status:      s.status ?? (s.observation ? 'success' : 'pending'),
              thought:     s.thought ?? '',
              action:      s.tool_input ?? null,
              observation: s.observation ?? '',
              detail:      s.tool_input
                ? JSON.stringify(s.tool_input)
                : (s.observation ?? ''),
            }))
          : updated[realIdx].reasoningSteps,
        itineraryData:  itinData,
        imageBase64:    imgData,
        hasItinerary:   !!itinData,
        hasImage:       !!imgData,
      };

      // Sinkronkan ke sessions dan localStorage
      const curId = activeSessionIdRef.current;
      setSessions((sPrev) => {
        const sUpdated = sPrev.map((s) =>
          String(s.id) === String(curId) ? { ...s, messages: updated } : s
        );
        saveSessionsToStorage(sUpdated, curId);
        return sUpdated;
      });

      return updated;
    });
  }, []);

  // ─── Handler event WebSocket ────────────────────────────────────────────────
  const handleEvent = useCallback((event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      try {
        const sanitized = event.data
          .replace(/:\s*NaN\b/g, ': null')
          .replace(/:\s*Infinity\b/g, ': null')
          .replace(/:\s*-Infinity\b/g, ': null');
        payload = JSON.parse(sanitized);
      } catch (err) {
        console.warn('[WS] Pesan non-JSON diterima:', event.data, err);
        return;
      }
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
        setMessages((prev) => {
          const lastIdx = [...prev].reverse().findIndex(
            (m) => m.role === 'agent' && m.isStreaming,
          );
          if (lastIdx === -1) return prev;
          const realIdx = prev.length - 1 - lastIdx;
          const updated = [...prev];
          const existing = updated[realIdx].reasoningSteps ?? [];
          // Tangkap teks pemikiran yang sempat distream sebelum pemanggilan tool
          const stepThought = payload.thought || updated[realIdx].text || '';

          updated[realIdx] = {
            ...updated[realIdx],
            // Bersihkan text utama agar chat bubble hanya untuk Final Answer yang rapi
            text: '',
            reasoningSteps: [
              ...existing,
              {
                step:        existing.length + 1,
                tool:        toolName,
                status:      'running',
                thought:     stepThought,
                action:      payload.input ?? null,
                observation: null,
                detail:      payload.input ? JSON.stringify(payload.input) : '',
              },
            ],
          };

          const curId = activeSessionIdRef.current;
          setSessions((sPrev) => {
            const sUpdated = sPrev.map((s) =>
              String(s.id) === String(curId) ? { ...s, messages: updated } : s
            );
            saveSessionsToStorage(sUpdated, curId);
            return sUpdated;
          });

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
          const stepIdx = [...steps].reverse().findIndex(
            (s) => s.tool === toolName && s.status === 'running',
          );
          if (stepIdx !== -1) {
            const realStep = steps.length - 1 - stepIdx;
            steps[realStep] = {
              ...steps[realStep],
              status:      'success',
              observation: payload.output ?? '',
            };
          }
          updated[realIdx] = { ...updated[realIdx], reasoningSteps: steps };

          const curId = activeSessionIdRef.current;
          setSessions((sPrev) => {
            const sUpdated = sPrev.map((s) =>
              String(s.id) === String(curId) ? { ...s, messages: updated } : s
            );
            saveSessionsToStorage(sUpdated, curId);
            return sUpdated;
          });

          return updated;
        });
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
        console.debug('[WS] Event tidak dikenal:', type, payload);
    }
  }, [appendTokenToLastAgent, finalizeLastAgent]);

  // ─── Buka koneksi WebSocket ─────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
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
      setMessages((prev) =>
        prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
      );
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

  // ─── Auto-Restore: Pulihkan jawaban dari server jika sesi aktif kosong/terputus ──
  useEffect(() => {
    if (!activeSessionId) return;

    // Jangan auto-restore jika agent sedang memproses/mengetik
    if (isTyping) return;

    const curMsgs = messagesRef.current;
    const lastMsg = curMsgs[curMsgs.length - 1];
    const needsRestore = lastMsg && lastMsg.role === 'agent' && !lastMsg.text?.trim();

    if (needsRestore) {
      fetch(`http://localhost:8000/api/agent/history/${activeSessionId}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data?.status === 'success' && Array.isArray(data.messages) && data.messages.length > 0) {
            const serverMsgs = data.messages;
            const lastServer = serverMsgs[serverMsgs.length - 1];
            if (lastServer?.role === 'agent' && lastServer?.text?.trim()) {
              setMessages((prev) => {
                const lastIdx = [...prev].reverse().findIndex((m) => m.role === 'agent');
                if (lastIdx === -1) return prev;
                const realIdx = prev.length - 1 - lastIdx;
                const updated = [...prev];
                updated[realIdx] = {
                  ...updated[realIdx],
                  text: lastServer.text,
                  isStreaming: false,
                  isError: false,
                };
                return updated;
              });

              setSessions((sPrev) => {
                const sUpdated = sPrev.map((s) => {
                  if (String(s.id) === String(activeSessionId)) {
                    const updatedMsgs = (s.messages || []).map((m, idx) =>
                      idx === s.messages.length - 1 && m.role === 'agent'
                        ? { ...m, text: lastServer.text, isStreaming: false, isError: false }
                        : m
                    );
                    return { ...s, messages: updatedMsgs };
                  }
                  return s;
                });
                saveSessionsToStorage(sUpdated, activeSessionId);
                return sUpdated;
              });
            }
          }
        })
        .catch((err) => console.warn('[WS] Gagal auto-restore riwayat dari server:', err));
    }
  }, [activeSessionId]);

  // ─── Pilih / Berpindah Sesi ─────────────────────────────────────────────────
  const selectSession = useCallback((targetId) => {
    const curId = activeSessionIdRef.current;
    if (String(targetId) === String(curId)) return;

    // 1. Simpan pesan sesi aktif saat ini ke sessions list
    const curMsgs = messagesRef.current;
    const currentSessions = sessionsRef.current;

    const updatedSessions = currentSessions.map((s) =>
      String(s.id) === String(curId) ? { ...s, messages: curMsgs } : s
    );

    // 2. Ambil pesan sesi target
    const targetSession = updatedSessions.find((s) => String(s.id) === String(targetId));
    const targetMsgs = targetSession?.messages || [];

    // 3. Simpan & perbarui state
    setSessions(updatedSessions);
    saveSessionsToStorage(updatedSessions, String(targetId));

    setActiveSessionId(String(targetId));
    setSessionId(String(targetId));
    setMessages(targetMsgs);

    setIsTyping(false);
    setCurrentAction('');
    setError(null);
  }, []);

  // ─── Buat Sesi Baru ─────────────────────────────────────────────────────────
  const createSession = useCallback(() => {
    const newId = crypto.randomUUID();
    const newSession = {
      id: newId,
      label: 'Sesi Baru',
      time: 'Baru saja',
      icon: '🗺️',
      messages: [],
    };

    // 1. Simpan pesan sesi aktif lama
    const curId = activeSessionIdRef.current;
    const curMsgs = messagesRef.current;
    const currentSessions = sessionsRef.current;

    const updatedOld = currentSessions.map((s) =>
      String(s.id) === String(curId) ? { ...s, messages: curMsgs } : s
    );

    const withNew = [newSession, ...updatedOld];

    // 2. Update state & storage
    setSessions(withNew);
    saveSessionsToStorage(withNew, newId);

    setActiveSessionId(newId);
    setSessionId(newId);
    setMessages([]);

    setIsTyping(false);
    setCurrentAction('');
    setError(null);
  }, []);

  // ─── Hapus Sesi ─────────────────────────────────────────────────────────────
  const deleteSession = useCallback((targetId) => {
    const currentSessions = sessionsRef.current;
    const filtered = currentSessions.filter((s) => String(s.id) !== String(targetId));

    if (filtered.length === 0) {
      createSession();
      return;
    }

    const curId = activeSessionIdRef.current;
    let nextActiveId = curId;

    if (String(curId) === String(targetId)) {
      const nextSession = filtered[0];
      nextActiveId = String(nextSession.id);
      const nextMessages = nextSession.messages || [];

      setActiveSessionId(nextActiveId);
      setSessionId(nextActiveId);
      setMessages(nextMessages);
      setIsTyping(false);
      setCurrentAction('');
      setError(null);
    }

    setSessions(filtered);
    saveSessionsToStorage(filtered, nextActiveId);
  }, [createSession]);

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

      // 1. Buat pesan user dan placeholder agent
      const userMsg = {
        id:   nextId(),
        role: 'user',
        text: text.trim(),
        time,
      };

      const agentMsg = {
        id:            nextId(),
        role:          'agent',
        text:          '',
        time,
        isStreaming:   true,
        reasoningSteps: [],
        hasItinerary:  false,
        hasImage:      false,
        imageBase64:   null,
        itineraryData: null,
        isError:       false,
      };

      const newMsgs = [...messagesRef.current, userMsg, agentMsg];
      setMessages(newMsgs);
      setIsTyping(true);
      setCurrentAction('');
      setError(null);

      // 2. Perbarui judul & waktu sesi jika sesi baru
      const curId = activeSessionIdRef.current;
      setSessions((prev) => {
        const updated = prev.map((s) => {
          if (String(s.id) === String(curId)) {
            const isDefault = !s.label || s.label === 'Sesi Baru' || !s.messages || s.messages.length === 0;
            const cleanText = text.trim();
            const newLabel = isDefault
              ? (cleanText.length > 22 ? cleanText.slice(0, 22) + '...' : cleanText)
              : s.label;
            return {
              ...s,
              label:    newLabel,
              time:     'Baru saja',
              messages: newMsgs,
            };
          }
          return s;
        });
        saveSessionsToStorage(updated, curId);
        return updated;
      });

      // 3. Kirim ke backend via WebSocket
      const payload = {
        prompt:       text.trim(),
        session_id:   curId,
        ...(filePath    && { file_path:    filePath }),
        ...(datasetName && { dataset_name: datasetName }),
      };

      ws.send(JSON.stringify(payload));
    },
    [],
  );

  // ─── Reset percakapan pada sesi aktif ───────────────────────────────────────
  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentAction('');
    setIsTyping(false);
    setError(null);

    const curId = activeSessionIdRef.current;
    setSessions((prev) => {
      const updated = prev.map((s) =>
        String(s.id) === String(curId) ? { ...s, messages: [] } : s
      );
      saveSessionsToStorage(updated, curId);
      return updated;
    });
  }, []);

  // ─── Reconnect manual ──────────────────────────────────────────────────────
  const reconnect = useCallback(() => {
    reconnectCount.current = 0;
    setError(null);
    connect();
  }, [connect]);

  // ─── Return API hook ────────────────────────────────────────────────────────
  return {
    sessions,
    activeSessionId,
    selectSession,
    createSession,
    deleteSession,
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
