import { useState } from 'react';
import { DUMMY_SESSIONS } from '../data/dummy';

export default function Sidebar({
  activeSession,
  onSessionClick,
  onNewSession,
  onDeleteSession,
  sessions = DUMMY_SESSIONS,
  isTyping = false,
}) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <>
      <aside
        className="flex flex-col h-full w-56 shrink-0"
        style={{ background: 'var(--sidebar-bg)', borderRight: '1px solid var(--border)' }}
      >
        {/* Logo */}
        <div
          className="flex items-center gap-3 px-4 py-5 border-b"
          style={{ borderColor: 'var(--border)' }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold"
            style={{ background: 'var(--accent)', boxShadow: '0 0 16px var(--accent-glow)' }}
          >
            TA
          </div>
          <div>
            <p className="text-sm font-semibold leading-none tracking-tight" style={{ color: 'var(--text-primary)' }}>
              PROJECT TA
            </p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              AI Data Scientist
            </p>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-3 pt-4 pb-2">
          <button
            type="button"
            onClick={() => {
              if (isTyping) return;
              onNewSession?.();
            }}
            disabled={isTyping}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 hover:brightness-110 active:scale-95 cursor-pointer disabled:opacity-75 disabled:cursor-not-allowed"
            style={{ background: 'var(--accent)', color: '#fff' }}
          >
            <span className="text-base leading-none">+</span>
            <span>Sesi Baru</span>
          </button>
        </div>

        {/* Recent Sessions */}
        <div className="px-3 pt-3 flex-1 overflow-y-auto">
          <p
            className="text-xs font-semibold uppercase tracking-widest mb-2 px-1"
            style={{ color: 'var(--text-muted)' }}
          >
            Sesi Terbaru
          </p>
          <ul className="space-y-0.5">
            {sessions.map((s) => {
              const isActive = String(activeSession) === String(s.id);
              return (
                <li key={s.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => {
                      if (isTyping) return;
                      onSessionClick?.(s.id);
                    }}
                    disabled={isTyping}
                    className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all duration-150 cursor-pointer disabled:cursor-not-allowed hover:bg-white/5 pr-8"
                    style={{
                      background: isActive ? 'var(--accent-glow)' : 'transparent',
                      border: isActive
                        ? '1px solid rgba(79,110,247,0.25)'
                        : '1px solid transparent',
                    }}
                  >
                    <span className="text-sm mt-0.5">{s.icon || '🗺️'}</span>
                    <div className="flex-1 min-w-0">
                      <p
                        className="text-xs font-medium truncate leading-snug transition-colors"
                        style={{
                          color: isActive
                            ? 'var(--accent-light)'
                            : 'var(--text-secondary)',
                        }}
                      >
                        {s.label}
                      </p>
                      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        {s.time}
                      </p>
                    </div>
                  </button>

                  {/* Tombol Hapus Sesi Kecil (Muncul saat hover) */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (isTyping) return;
                      onDeleteSession?.(s.id);
                    }}
                    title="Hapus sesi"
                    disabled={isTyping}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 w-6 h-6 rounded flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-red-500/15 text-[var(--text-muted)] hover:text-red-400 transition-all duration-150 cursor-pointer disabled:cursor-not-allowed"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Settings Button */}
        <div className="px-3 py-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg w-full hover:bg-white/5 transition-all cursor-pointer"
          >
            <svg
              className="w-4 h-4 shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              style={{ color: 'var(--text-muted)' }}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Pengaturan
            </span>
          </button>
        </div>
      </aside>

      {/* Settings Modal (Overlay) */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm">
          <div
            className="w-full max-w-md rounded-2xl p-5 shadow-2xl transition-all"
            style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-3 border-b" style={{ borderColor: 'var(--border)' }}>
              <div className="flex items-center gap-2">
                <span className="text-base">⚙️</span>
                <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                  Pengaturan Sistem & Agen
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-white/10 text-[var(--text-muted)] hover:text-white transition-colors cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            {/* Modal Content */}
            <div className="py-4 space-y-4 text-xs">
              {/* Section 1: AI Engine Configuration */}
              <div>
                <p className="font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>
                  Konfigurasi Model & Agent
                </p>
                <div className="rounded-xl p-3 space-y-2" style={{ background: 'var(--card-bg)', border: '1px solid var(--border)' }}>
                  <div className="flex justify-between items-center">
                    <span style={{ color: 'var(--text-muted)' }}>Arsitektur</span>
                    <span className="font-mono text-emerald-400 font-medium">Neuro-Symbolic ReAct</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span style={{ color: 'var(--text-muted)' }}>Inference Mode</span>
                    <span className="font-mono text-indigo-300 font-medium">Deterministic Tool-Calling</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span style={{ color: 'var(--text-muted)' }}>Temperature</span>
                    <span className="font-mono" style={{ color: 'var(--text-primary)' }}>0.3 (Low Variance)</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span style={{ color: 'var(--text-muted)' }}>Verification Engine</span>
                    <span className="font-mono text-emerald-400 font-medium">Chain-of-Verification (Active)</span>
                  </div>
                </div>
              </div>

              {/* Section 2: Storage & Session Management */}
              <div>
                <p className="font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>
                  Penyimpanan Riwayat Sesi
                </p>
                <div className="rounded-xl p-3 space-y-2.5" style={{ background: 'var(--card-bg)', border: '1px solid var(--border)' }}>
                  <div className="flex justify-between items-center">
                    <span style={{ color: 'var(--text-muted)' }}>Total Sesi Tersimpan</span>
                    <span className="font-mono font-semibold text-emerald-300">{sessions.length} sesi</span>
                  </div>
                  <div className="pt-1">
                    <button
                      type="button"
                      onClick={() => {
                        if (window.confirm('Apakah Anda yakin ingin menghapus semua riwayat percakapan lokal dan mereset ke sesi baru?')) {
                          localStorage.removeItem('wista_sessions_v1');
                          localStorage.removeItem('wista_active_session_id_v1');
                          window.location.reload();
                        }
                      }}
                      className="w-full py-2 px-3 rounded-lg text-xs font-medium text-red-400 hover:bg-red-500/15 border border-red-500/30 transition-all cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                      Reset Semua Riwayat Percakapan
                    </button>
                  </div>
                </div>
              </div>

              {/* Section 3: About Application */}
              <div className="rounded-xl p-2.5 text-[11px] flex items-center justify-between" style={{ background: 'rgba(79,110,247,0.08)', border: '1px solid rgba(79,110,247,0.2)' }}>
                <span style={{ color: 'var(--text-secondary)' }}>PROJECT TA (AI Data Scientist)</span>
                <span className="text-indigo-400 font-mono font-medium">Tugas Akhir 2026</span>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 rounded-lg text-xs font-medium cursor-pointer transition-all hover:brightness-110 active:scale-95"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >
                Tutup
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

