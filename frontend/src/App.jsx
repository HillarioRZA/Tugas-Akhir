import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import RightPanel from './components/RightPanel';
import { useAgentWebSocket } from './hooks/useAgentWebSocket';

/* ── Gear icon (inline SVG, zero dependency) ── */
function GearIcon({ active }) {
  return (
    <svg
      className="w-4 h-4 transition-transform duration-500"
      style={{
        color:     active ? 'var(--accent-light)' : 'var(--text-muted)',
        transform: active ? 'rotate(90deg)' : 'rotate(0deg)',
      }}
      fill="none" viewBox="0 0 24 24" stroke="currentColor"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
      />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
    </svg>
  );
}

export default function App() {
  const [activeSession,  setActiveSession]  = useState(1);
  const [showMLInsights, setShowMLInsights] = useState(false);

  // ── WebSocket hook — sumber data utama ──
  const {
    messages,
    isConnected,
    isTyping,
    currentAction,
    sessionId,
    error,
    sendMessage,
    clearMessages,
  } = useAgentWebSocket();

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--main-bg)' }}>

      {/* ── Left Sidebar ── */}
      <Sidebar activeSession={activeSession} onSessionClick={setActiveSession} />

      {/* ── Main Content ── */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Top Bar */}
        <header
          className="flex items-center justify-between px-6 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--main-bg)' }}
        >
          {/* Left: title + version */}
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Agent Workspace
            </h1>
            <span className="text-xs px-2 py-0.5 rounded-full font-medium"
              style={{ background: 'var(--card-bg)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
              v1.0.0
            </span>
          </div>

          {/* Right: status + session + toggle */}
          <div className="flex items-center gap-3">

            {/* Connection badge — dinamis dari hook */}
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-xs font-medium"
                style={{ color: isConnected ? '#34d399' : '#f87171' }}>
                {isConnected ? 'Online' : 'Offline'}
              </span>
            </div>

            {/* Session ID — dari hook */}
            <span className="text-xs font-mono px-2 py-1 rounded-lg"
              style={{ background: 'var(--card-bg)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
              {sessionId.slice(0, 8)}
            </span>

            {/* Clear chat */}
            {messages.length > 0 && (
              <button
                onClick={clearMessages}
                title="Bersihkan percakapan"
                className="flex items-center justify-center w-7 h-7 rounded-lg transition-all hover:bg-white/8 active:scale-90"
                style={{ border: '1px solid transparent' }}
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  style={{ color: 'var(--text-muted)' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}

            {/* Toggle ML Insights */}
            <div className="relative group">
              <button
                onClick={() => setShowMLInsights((v) => !v)}
                aria-label="Toggle ML Insights"
                className="flex items-center justify-center w-7 h-7 rounded-lg transition-all duration-150 hover:bg-white/8 active:scale-90"
                style={{
                  background: showMLInsights ? 'var(--accent-glow)' : 'transparent',
                  border: `1px solid ${showMLInsights ? 'rgba(79,110,247,0.3)' : 'transparent'}`,
                }}
              >
                <GearIcon active={showMLInsights} />
              </button>
              <div
                className="pointer-events-none absolute right-0 top-9 z-50 px-2.5 py-1.5 rounded-lg text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150"
                style={{
                  background: 'var(--card-bg)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                  boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                }}
              >
                {showMLInsights ? 'Sembunyikan' : 'Tampilkan'} ML Insights
              </div>
            </div>
          </div>
        </header>

        {/* ── Error banner ── */}
        {error && (
          <div className="mx-6 mt-3 px-4 py-2.5 rounded-xl text-xs flex items-center gap-2"
            style={{ background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.25)', color: '#f87171' }}>
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Chat + Input */}
        <div className="flex-1 flex flex-col min-h-0">
          <ChatArea messages={messages} isTyping={isTyping} />
          <ChatInput
            onSend={sendMessage}
            isConnected={isConnected}
            isTyping={isTyping}
            currentAction={currentAction}
          />
        </div>
      </div>

      {/* ── Right Panel — slide-in/out ── */}
      <div
        className="shrink-0 overflow-hidden flex flex-col"
        style={{
          width:         showMLInsights ? '208px' : '0px',
          opacity:       showMLInsights ? 1 : 0,
          borderLeft:    showMLInsights ? '1px solid var(--border)' : 'none',
          background:    'var(--main-bg)',
          transition:    'width 280ms cubic-bezier(0.4,0,0.2,1), opacity 200ms ease',
          pointerEvents: showMLInsights ? 'auto' : 'none',
        }}
      >
        <div style={{ width: '208px', minWidth: '208px' }}>
          <RightPanel />
        </div>
      </div>
    </div>
  );
}
