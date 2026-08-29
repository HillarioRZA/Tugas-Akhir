import { useState } from 'react';

const TOOL_ICONS = {
  budget_optimizer_tool: '⚙️',
  plot_itinerary_scatter: '📊',
  verify_output: '✅',
};

const TOOL_LABELS = {
  budget_optimizer_tool: 'Budget Optimizer',
  plot_itinerary_scatter: 'Scatter Plot XAI',
  verify_output: 'Verifikasi Output',
};

export default function ReasoningPanel({ steps }) {
  const [open, setOpen] = useState(false);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-3 rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', background: 'var(--card-bg)' }}>
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 hover:bg-white/5 transition-all"
      >
        <div className="flex items-center gap-2">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"
            style={{ color: 'var(--accent)' }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Reasoning Process
          </span>
          <span className="text-xs px-1.5 py-0.5 rounded-full"
            style={{ background: 'var(--accent-glow)', color: 'var(--accent-light)', fontSize: '10px' }}>
            {steps.length} langkah
          </span>
        </div>
        <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
          style={{ color: 'var(--text-muted)' }}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="px-3.5 pb-3.5 pt-1 space-y-2" style={{ borderTop: '1px solid var(--border)' }}>
          {steps.map((s) => (
            <div key={s.step} className="flex items-start gap-3">
              {/* Step number */}
              <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold"
                style={{ background: 'var(--accent-glow)', color: 'var(--accent-light)', border: '1px solid rgba(79,110,247,0.3)' }}>
                {s.step}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span>{TOOL_ICONS[s.tool] || '🔧'}</span>
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
                    {TOOL_LABELS[s.tool] || s.tool}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded-full"
                    style={{
                      background: s.status === 'success' ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
                      color: s.status === 'success' ? '#34d399' : '#f87171',
                      fontSize: '10px',
                    }}>
                    {s.status === 'success' ? '✓ sukses' : '✗ gagal'}
                  </span>
                </div>
                <p className="text-xs mt-0.5 font-mono" style={{ color: 'var(--text-muted)' }}>
                  {s.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
