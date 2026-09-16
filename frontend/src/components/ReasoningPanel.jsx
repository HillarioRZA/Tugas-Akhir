import { useState } from 'react';

const TOOL_INFO = {
  budget_optimizer_tool: { icon: '⚙️', label: 'Budget Optimizer' },
  plot_itinerary_scatter: { icon: '📊', label: 'Scatter Plot XAI' },
  plot_budget_breakdown:  { icon: '💰', label: 'Budget Breakdown Plot' },
  verify_output:          { icon: '🛡️', label: 'Verifikasi Kualitas & Safety' },
  eda_summary_tool:       { icon: '📈', label: 'EDA Summary Dataset' },
  rag_query_tool:         { icon: '📚', label: 'Knowledge Retrieval (RAG)' },
  ml_predict_tool:        { icon: '🤖', label: 'Inference Model ML' },
  custom_plot_tool:       { icon: '📉', label: 'Visualisasi Custom' },
};

function formatObservation(obs) {
  if (!obs) return 'Tidak ada data pengamatan.';
  if (typeof obs === 'object') {
    try {
      return JSON.stringify(obs, null, 2);
    } catch {
      return String(obs);
    }
  }
  if (typeof obs === 'string') {
    const trimmed = obs.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        const parsed = JSON.parse(trimmed);
        return JSON.stringify(parsed, null, 2);
      } catch {
        return obs;
      }
    }
  }
  return String(obs);
}

export default function ReasoningPanel({ steps }) {
  const [open, setOpen] = useState(false);

  if (!steps || steps.length === 0) return null;

  const hasRunningStep = steps.some((s) => s.status === 'running');

  return (
    <div
      className="mt-3 rounded-xl overflow-hidden transition-all"
      style={{ border: '1px solid var(--border)', background: 'var(--card-bg)' }}
    >
      {/* Header Toggle Button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 hover:bg-white/5 transition-all text-left"
      >
        <div className="flex items-center gap-2 flex-wrap">
          <div
            className="w-5 h-5 rounded-md flex items-center justify-center text-xs"
            style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}
          >
            🧠
          </div>
          <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Reasoning Process (ReAct)
          </span>
          <span
            className="text-[10px] font-medium px-2 py-0.5 rounded-full"
            style={{ background: 'var(--accent-glow)', color: 'var(--accent-light)' }}
          >
            {steps.length} langkah
          </span>
          {hasRunningStep ? (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30 flex items-center gap-1.5 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
              Sedang berjalan...
            </span>
          ) : (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              ✓ Selesai
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
            {open ? 'Tutup' : 'Lihat Detail'}
          </span>
          <svg
            className={`w-3.5 h-3.5 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            style={{ color: 'var(--text-muted)' }}
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Reasoning List */}
      {open && (
        <div className="px-3.5 pb-3.5 pt-2 space-y-3" style={{ borderTop: '1px solid var(--border)' }}>
          {steps.map((s, idx) => {
            const toolMeta = TOOL_INFO[s.tool] || { icon: '🔧', label: s.tool || 'Tool Execution' };
            const isRunning = s.status === 'running';
            const isSuccess = s.status === 'success' || (!isRunning && s.status !== 'error');

            return (
              <div
                key={s.step || idx}
                className="rounded-xl p-3 space-y-2.5 transition-all"
                style={{
                  background: 'rgba(15, 17, 23, 0.55)',
                  border: '1px solid var(--border)',
                }}
              >
                {/* Step Header */}
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold font-mono"
                      style={{
                        background: 'var(--accent-glow)',
                        color: 'var(--accent-light)',
                        border: '1px solid rgba(79,110,247,0.3)',
                      }}
                    >
                      {s.step || idx + 1}
                    </div>
                    <span className="text-sm">{toolMeta.icon}</span>
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {toolMeta.label}
                    </span>
                  </div>

                  {/* Status Badge */}
                  <div>
                    {isRunning ? (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30 flex items-center gap-1 font-mono">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping inline-block" />
                        Mengeksekusi...
                      </span>
                    ) : isSuccess ? (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-medium">
                        ✓ Berhasil
                      </span>
                    ) : (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 border border-red-500/30 font-medium">
                        ✗ Gagal
                      </span>
                    )}
                  </div>
                </div>

                {/* Section 1: 💭 Thought */}
                {s.thought && (
                  <div
                    className="rounded-lg p-2.5 text-xs leading-relaxed"
                    style={{
                      background: 'rgba(79, 110, 247, 0.08)',
                      borderLeft: '3px solid var(--accent)',
                      color: 'var(--text-primary)',
                    }}
                  >
                    <div
                      className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider mb-1"
                      style={{ color: 'var(--accent-light)' }}
                    >
                      <span>💭</span> Thought (Pemikiran)
                    </div>
                    <p className="whitespace-pre-wrap leading-relaxed text-[11px]" style={{ color: 'var(--text-primary)' }}>
                      {s.thought}
                    </p>
                  </div>
                )}

                {/* Section 2: ⚡ Action */}
                {(s.action || s.detail || s.tool) && (
                  <div
                    className="rounded-lg p-2.5 text-xs"
                    style={{ background: 'rgba(0, 0, 0, 0.3)', border: '1px solid var(--border)' }}
                  >
                    <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider mb-1 text-amber-400">
                      <span>⚡</span> Action: <span className="font-mono lowercase text-slate-300 font-normal">{s.tool}</span>
                    </div>
                    <pre
                      className="font-mono text-[11px] leading-relaxed text-slate-300 overflow-x-auto whitespace-pre-wrap max-h-32 p-1.5 rounded bg-black/30 border border-white/5"
                    >
                      {typeof s.action === 'object' && s.action !== null
                        ? JSON.stringify(s.action, null, 2)
                        : s.detail || JSON.stringify(s.action)}
                    </pre>
                  </div>
                )}

                {/* Section 3: 👁️ Observation */}
                <div
                  className="rounded-lg p-2.5 text-xs"
                  style={{ background: 'rgba(0, 0, 0, 0.3)', border: '1px solid var(--border)' }}
                >
                  <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider mb-1 text-emerald-400">
                    <span>👁️</span> Observation (Hasil / Pengamatan)
                  </div>
                  {isRunning ? (
                    <div className="flex items-center gap-2 text-slate-400 italic font-mono text-[11px] py-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping inline-block" />
                      Menunggu respon data dari tool...
                    </div>
                  ) : (
                    <pre
                      className="font-mono text-[11px] leading-relaxed text-slate-300 max-h-36 overflow-y-auto whitespace-pre-wrap p-1.5 rounded bg-black/30 border border-white/5"
                    >
                      {formatObservation(s.observation)}
                    </pre>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
