import { DUMMY_SESSIONS } from '../data/dummy';

export default function Sidebar({ activeSession, onSessionClick }) {
  return (
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
          className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
          style={{ background: 'var(--accent)', boxShadow: '0 0 16px var(--accent-glow)' }}
        >
          W
        </div>
        <div>
          <p className="text-sm font-semibold leading-none" style={{ color: 'var(--text-primary)' }}>
            WISTA
          </p>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            AI Travel Agent
          </p>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="px-3 pt-4 pb-2">
        <button
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 hover:brightness-110 active:scale-95"
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
          {DUMMY_SESSIONS.map((s) => (
            <li key={s.id}>
              <button
                onClick={() => onSessionClick?.(s.id)}
                className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all duration-150"
                style={{
                  background: activeSession === s.id ? 'var(--accent-glow)' : 'transparent',
                  border: activeSession === s.id
                    ? '1px solid rgba(79,110,247,0.25)'
                    : '1px solid transparent',
                }}
              >
                <span className="text-sm mt-0.5">{s.icon}</span>
                <div className="flex-1 min-w-0">
                  <p
                    className="text-xs font-medium truncate leading-snug transition-colors"
                    style={{
                      color: activeSession === s.id
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
            </li>
          ))}
        </ul>
      </div>

      {/* Settings */}
      <div className="px-3 py-4 border-t" style={{ borderColor: 'var(--border)' }}>
        <button className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg w-full hover:bg-white/5 transition-all">
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
  );
}
