import { useRef, useState } from 'react';

const HINTS = [
  'Itinerary 2 hari Bangli',
  'Wisata pantai Badung',
  'Analisis dataset wisata',
];

// Label yang lebih ramah untuk nama tool teknis
const ACTION_LABELS = {
  budget_optimizer_tool:  'Mengoptimasi budget & memilih destinasi',
  plot_itinerary_scatter: 'Membuat visualisasi peta perjalanan',
  plot_budget_breakdown:  'Membuat grafik alokasi budget',
  verify_output:          'Memverifikasi hasil itinerary',
  rag_tool:               'Mencari referensi wisata',
  default:                'Memproses permintaan',
};

function resolveActionLabel(action) {
  return ACTION_LABELS[action] ?? ACTION_LABELS.default;
}

export default function ChatInput({ onSend, isConnected, isTyping, currentAction }) {
  const [value, setValue] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const canSend = isConnected && !isTyping && !uploading && (value.trim().length > 0 || selectedFile !== null);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedTypes = ['.csv', '.pdf'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(ext)) {
      alert('Format file tidak didukung. Harap gunakan format CSV (.csv) atau PDF (.pdf).');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }
    setSelectedFile(file);
  };

  const handleAttachClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleSend = async () => {
    if (!canSend) return;
    const text = value.trim() || (selectedFile ? `Tolong analisis file ${selectedFile.name}` : '');

    let filePath = null;
    let fileName = null;

    if (selectedFile) {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const res = await fetch('/api/agent/upload', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`Upload gagal dengan status ${res.status}`);
        }

        const data = await res.json();
        filePath = data.file_path;
        fileName = selectedFile.name;
      } catch (err) {
        console.error('Upload failed:', err);
        alert(`Gagal mengunggah file: ${err.message}`);
        setUploading(false);
        return;
      }
      setUploading(false);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }

    onSend(text, filePath, fileName);
    setValue('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleHint = (hint) => {
    if (!isConnected || isTyping || uploading) return;
    onSend(hint);
  };

  return (
    <div className="px-6 py-4" style={{ borderTop: '1px solid var(--border)' }}>

      {/* ── Live action indicator ── */}
      {currentAction && (
        <div className="flex items-center gap-2 mb-2.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ background: 'var(--accent)' }} />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5"
              style={{ background: 'var(--accent)' }} />
          </span>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Agent sedang menjalankan:{' '}
            <span style={{ color: 'var(--accent-light)' }}>
              {resolveActionLabel(currentAction)}
            </span>
          </span>
        </div>
      )}

      {/* ── Selected File Chip ── */}
      {selectedFile && (
        <div
          className="flex items-center justify-between gap-2 mb-2 px-3 py-1.5 rounded-lg text-xs"
          style={{
            background: 'var(--card-bg)',
            border: '1px solid var(--accent)',
            boxShadow: '0 0 8px rgba(79, 110, 247, 0.2)',
          }}
        >
          <div className="flex items-center gap-2 truncate">
            <span style={{ color: 'var(--accent-light)' }}>📎</span>
            <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
              {selectedFile.name}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
          </div>
          <button
            type="button"
            onClick={() => {
              setSelectedFile(null);
              if (fileInputRef.current) fileInputRef.current.value = '';
            }}
            className="shrink-0 text-xs px-1 hover:opacity-70 transition-opacity"
            style={{ color: 'var(--text-muted)' }}
            title="Hapus file lampiran"
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Hint chips ── */}
      <div className="flex gap-2 mb-3 flex-wrap">
        {HINTS.map((hint) => (
          <button
            key={hint}
            onClick={() => handleHint(hint)}
            disabled={!isConnected || isTyping || uploading}
            className="text-xs px-3 py-1 rounded-full transition-all hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: 'var(--card-bg)',
              border: '1px solid var(--border)',
              color: 'var(--text-muted)',
            }}
          >
            {hint}
          </button>
        ))}
      </div>

      {/* ── Input row ── */}
      <div
        className="flex items-center gap-3 rounded-xl px-4 py-2.5 transition-all"
        style={{
          background: 'var(--card-bg)',
          border: `1px solid ${canSend ? 'var(--accent)' : 'var(--border)'}`,
          transition: 'border-color 200ms ease',
        }}
      >
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.pdf"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {/* Attachment button */}
        <button
          type="button"
          className="shrink-0 p-1 rounded-lg hover:bg-white/10 transition-all disabled:opacity-40"
          disabled={!isConnected || isTyping || uploading}
          onClick={handleAttachClick}
          title="Lampirkan file (.csv, .pdf)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"
            style={{ color: selectedFile ? 'var(--accent-light)' : 'var(--text-muted)' }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
        </button>

        {/* Text input */}
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!isConnected || isTyping || uploading}
          placeholder={
            !isConnected
              ? 'Menghubungkan ke server...'
              : uploading
              ? 'Mengunggah file lampiran...'
              : isTyping
              ? 'Agent sedang memproses...'
              : selectedFile
              ? `File siap: ${selectedFile.name} (ketik pesan atau tekan Enter)...`
              : 'Deskripsikan destinasi impian Anda atau tanyakan sesuatu...'
          }
          className="flex-1 bg-transparent text-sm outline-none disabled:cursor-not-allowed"
          style={{ color: 'var(--text-primary)' }}
        />

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={!canSend}
          className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:brightness-110 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:brightness-100"
          style={{ background: canSend ? 'var(--accent)' : 'var(--border)' }}
        >
          {isTyping || uploading ? (
            /* Spinner saat agent sedang bekerja atau sedang upload */
            <svg className="w-4 h-4 animate-spin text-white" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )}
        </button>
      </div>

      {/* Footer hint */}
      <p className="text-center text-xs mt-2.5" style={{ color: 'var(--text-muted)' }}>
        PROJECT TA · Powered by Neuro-Symbolic AI Data Scientist
      </p>
    </div>
  );
}
