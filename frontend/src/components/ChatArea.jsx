import { useEffect, useRef } from 'react';
import ItineraryCard from './ItineraryCard';
import ReasoningPanel from './ReasoningPanel';

// ─── Typing indicator — 3 dot bounce ────────────────────────────────────────
function TypingDots() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-7 h-7 rounded-full shrink-0 flex items-center justify-center text-xs font-bold"
        style={{ background: 'var(--accent)', boxShadow: '0 0 10px var(--accent-glow)', color: '#fff' }}>
        W
      </div>
      <div className="px-4 py-3 rounded-2xl rounded-tl-md"
        style={{ background: 'var(--agent-bubble)', border: '1px solid var(--border)' }}>
        <div className="flex gap-1 items-center h-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce"
              style={{
                background: 'var(--text-muted)',
                animationDelay: `${i * 0.15}s`,
                animationDuration: '0.9s',
              }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── User bubble ─────────────────────────────────────────────────────────────
function UserBubble({ msg }) {
  return (
    <div className="flex items-end justify-end gap-2.5">
      <div className="max-w-[72%]">
        <p className="text-xs text-right mb-1" style={{ color: 'var(--text-muted)' }}>
          Kamu · {msg.time}
        </p>
        <div className="rounded-2xl rounded-br-md px-4 py-2.5 text-sm leading-relaxed"
          style={{ background: 'var(--accent)', color: '#fff' }}>
          {msg.text}
        </div>
      </div>
      <div className="w-7 h-7 rounded-full shrink-0 flex items-center justify-center text-xs font-bold"
        style={{ background: 'var(--border)', color: 'var(--text-primary)' }}>
        U
      </div>
    </div>
  );
}

// ─── Agent bubble ─────────────────────────────────────────────────────────────
function AgentBubble({ msg }) {
  const isError = msg.isError;

  return (
    <div className="flex items-start gap-2.5">
      {/* Avatar */}
      <div className="w-7 h-7 rounded-full shrink-0 flex items-center justify-center text-xs font-bold mt-5"
        style={{
          background: isError ? '#f87171' : 'var(--accent)',
          boxShadow:  isError ? '0 0 10px rgba(248,113,113,0.3)' : '0 0 10px var(--accent-glow)',
          color: '#fff',
        }}>
        W
      </div>

      <div className="max-w-[80%] flex-1">
        <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
          WISTA · {msg.time}
          {msg.isStreaming && (
            <span className="ml-2 inline-flex items-center gap-1" style={{ color: 'var(--accent-light)' }}>
              <span className="w-1 h-1 rounded-full animate-pulse inline-block"
                style={{ background: 'var(--accent)' }} />
              mengetik...
            </span>
          )}
        </p>

        {/* Teks bubble — hanya tampil jika ada teks */}
        {(msg.text || msg.isStreaming) && (
          <div
            className="rounded-2xl rounded-tl-md px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap"
            style={{
              background: isError
                ? 'rgba(248,113,113,0.08)'
                : 'var(--agent-bubble)',
              border: `1px solid ${isError ? 'rgba(248,113,113,0.25)' : 'var(--border)'}`,
              color: isError ? '#f87171' : 'var(--text-secondary)',
            }}
          >
            {msg.text || <span style={{ color: 'var(--text-muted)' }}>▋</span>}
          </div>
        )}

        {/* Itinerary Card — tampil jika backend mengirim data itinerary */}
        {msg.hasItinerary && msg.itineraryData && (
          <div className="mt-3">
            <ItineraryCard data={msg.itineraryData} />
          </div>
        )}

        {/* Image plot */}
        {msg.hasImage && msg.imageBase64 && (
          <div className="mt-3 rounded-xl overflow-hidden"
            style={{ border: '1px solid var(--border)' }}>
            <img
              src={`data:image/png;base64,${msg.imageBase64}`}
              alt="Visualisasi agent"
              className="w-full object-contain"
            />
          </div>
        )}

        {/* Reasoning steps dropdown */}
        {msg.reasoningSteps && msg.reasoningSteps.length > 0 && (
          <ReasoningPanel steps={msg.reasoningSteps} />
        )}
      </div>
    </div>
  );
}

// ─── ChatArea ─────────────────────────────────────────────────────────────────
export default function ChatArea({ messages, isTyping }) {
  const bottomRef = useRef(null);

  // Auto-scroll ke bawah setiap kali ada pesan baru / token baru
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Apakah pesan agent terakhir sedang streaming?
  const lastMsg = messages[messages.length - 1];
  const lastIsStreaming = lastMsg?.role === 'agent' && lastMsg?.isStreaming;

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
      {/* Empty state */}
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-xl"
            style={{ background: 'var(--accent-glow)', border: '1px solid rgba(79,110,247,0.2)' }}>
            🗺️
          </div>
          <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Selamat datang di WISTA
          </p>
          <p className="text-xs max-w-xs" style={{ color: 'var(--text-muted)' }}>
            Tanyakan apapun tentang destinasi wisata Bali — itinerary, budget, atau rekomendasi tempat terbaik.
          </p>
        </div>
      )}

      {/* Messages */}
      {messages.map((msg) =>
        msg.role === 'user'
          ? <UserBubble key={msg.id} msg={msg} />
          : <AgentBubble key={msg.id} msg={msg} />
      )}

      {/* Typing dots — hanya tampil jika agent belum mulai streaming teks */}
      {isTyping && !lastIsStreaming && <TypingDots />}

      {/* Anchor untuk auto-scroll */}
      <div ref={bottomRef} />
    </div>
  );
}
