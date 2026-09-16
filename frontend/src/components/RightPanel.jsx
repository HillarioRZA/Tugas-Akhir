import { useState, useEffect } from 'react';
import { DUMMY_STATS } from '../data/dummy';

function StatBadge({ label, value, color }) {
  return (
    <div className="text-center">
      <p className="text-lg font-bold tracking-tight" style={{ color }}>{value}%</p>
      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)', fontSize: '10px' }}>{label}</p>
    </div>
  );
}

function FeatureBar({ name, value, color }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between mb-1">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{name}</span>
        <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-muted)' }}>{value}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.min(100, value * 3.5)}%`, background: color }}
        />
      </div>
    </div>
  );
}

export default function RightPanel() {
  const [stats, setStats] = useState(DUMMY_STATS);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function fetchMLInfo() {
      try {
        setIsLoading(true);
        const res = await fetch('/api/agent/ml-info');
        if (res.ok) {
          const data = await res.json();
          if (isMounted && data && data.accuracy) {
            setStats(data);
          }
        }
      } catch (err) {
        console.warn('[RightPanel] Gagal memuat live ML info, menggunakan cache lokal:', err);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    fetchMLInfo();
    return () => { isMounted = false; };
  }, []);

  const {
    model_name = "RandomForest (Calibrated)",
    accuracy = 94.5,
    f1_score = 94.4,
    roc_auc = 98.3,
    precision = 95.1,
    recall = 93.8,
    features = [],
    dataset = { file: "bali_tourist_clean_v3.csv", rows: 1452, columns: 15 },
    drift_status = "Optimal (No Drift)",
  } = stats;

  return (
    <aside className="flex flex-col gap-3 w-56 shrink-0 py-4 pr-4 overflow-y-auto">

      {/* Model Performance Card */}
      <div className="rounded-xl p-3.5" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Performa Model
          </p>
          <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium"
            style={{ background: 'rgba(79,110,247,0.12)', color: '#89b4fa', border: '1px solid rgba(79,110,247,0.2)' }}>
            Live ML
          </span>
        </div>

        <p className="text-[11px] font-mono text-emerald-400 mb-3 truncate" title={model_name}>
          {model_name}
        </p>

        {/* 3 Main Badges */}
        <div className="flex justify-around pb-2 mb-2" style={{ borderBottom: '1px solid var(--border)' }}>
          <StatBadge label="Accuracy" value={accuracy} color="#4f6ef7" />
          <StatBadge label="F1-Score" value={f1_score} color="#34d399" />
          <StatBadge label="ROC-AUC" value={roc_auc} color="#c084fc" />
        </div>

        {/* Secondary metrics */}
        <div className="flex justify-between text-[11px] pt-0.5 font-mono" style={{ color: 'var(--text-muted)' }}>
          <span>Precision: <strong className="text-slate-300">{precision}%</strong></span>
          <span>Recall: <strong className="text-slate-300">{recall}%</strong></span>
        </div>
      </div>

      {/* Feature Importance (XAI) */}
      <div className="rounded-xl p-3.5" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
            XAI Feature Weights
          </p>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            Gini MDI
          </span>
        </div>
        {features.slice(0, 5).map(f => (
          <FeatureBar key={f.name} {...f} />
        ))}
      </div>

      {/* Dataset & Drift Info */}
      <div className="rounded-xl p-3.5" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Info Dataset
          </p>
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
            Active
          </span>
        </div>

        {/* Model Drift Status Pill */}
        <div className="flex items-center justify-between text-[11px] px-2 py-1 rounded-lg mb-2.5"
          style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.25)' }}>
          <span style={{ color: 'var(--text-muted)' }}>Model Drift</span>
          <span className="font-semibold text-emerald-400 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            {drift_status}
          </span>
        </div>

        <table className="w-full text-xs">
          <tbody>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">File</td>
              <td className="text-right font-mono truncate pl-2 font-medium"
                style={{ color: 'var(--text-primary)', maxWidth: '100px' }} title={dataset.file}>
                {dataset.file}
              </td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">Total Baris</td>
              <td className="text-right font-semibold font-mono text-emerald-300">
                {dataset.rows.toLocaleString()}
              </td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">Total Kolom</td>
              <td className="text-right font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>
                {dataset.columns} fitur
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Agent Status Pill */}
      <div className="rounded-xl p-3 flex items-center gap-2"
        style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}>
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
        <div>
          <p className="text-xs font-semibold" style={{ color: '#34d399' }}>Agent Online</p>
          <p className="text-xs font-mono" style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
            ws://localhost:8000/ws/chat
          </p>
        </div>
      </div>

    </aside>
  );
}

