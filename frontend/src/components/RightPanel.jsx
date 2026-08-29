import { DUMMY_STATS } from '../data/dummy';

function StatBadge({ label, value, color }) {
  return (
    <div className="text-center">
      <p className="text-xl font-bold" style={{ color }}>{value}%</p>
      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{label}</p>
    </div>
  );
}

function FeatureBar({ name, value, color }) {
  return (
    <div className="mb-2.5">
      <div className="flex justify-between mb-1">
        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{name}</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{value}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value}%`, background: color }} />
      </div>
    </div>
  );
}

export default function RightPanel() {
  const { accuracy, precision, recall, features, dataset } = DUMMY_STATS;

  return (
    <aside className="flex flex-col gap-3 w-52 shrink-0 py-4 pr-4 overflow-y-auto">

      {/* Model Performance */}
      <div className="rounded-xl p-4" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <p className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
          Performa Model
        </p>
        <div className="flex justify-around">
          <StatBadge label="Accuracy" value={accuracy} color="#4f6ef7" />
          <StatBadge label="Precision" value={precision} color="#34d399" />
          <StatBadge label="Recall" value={recall} color="#fbbf24" />
        </div>
      </div>

      {/* Feature Importance */}
      <div className="rounded-xl p-4" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <p className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
          Feature Importance
        </p>
        {features.map(f => (
          <FeatureBar key={f.name} {...f} />
        ))}
      </div>

      {/* Dataset Info */}
      <div className="rounded-xl p-4" style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>
        <p className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
          Info Dataset
        </p>
        <table className="w-full text-xs">
          <tbody>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">File</td>
              <td className="text-right font-mono truncate pl-2" style={{ color: 'var(--text-primary)', maxWidth: '90px' }}>
                {dataset.file}
              </td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">Baris</td>
              <td className="text-right font-semibold" style={{ color: 'var(--text-primary)' }}>
                {dataset.rows.toLocaleString()}
              </td>
            </tr>
            <tr>
              <td style={{ color: 'var(--text-muted)' }} className="py-0.5">Kolom</td>
              <td className="text-right font-semibold" style={{ color: 'var(--text-primary)' }}>
                {dataset.columns}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Status pill */}
      <div className="rounded-xl p-3 flex items-center gap-2"
        style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}>
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
        <div>
          <p className="text-xs font-semibold" style={{ color: '#34d399' }}>Agent Online</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
            ws://localhost:8000/ws/chat
          </p>
        </div>
      </div>
    </aside>
  );
}
