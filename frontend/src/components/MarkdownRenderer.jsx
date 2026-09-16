import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Komponen untuk me-render output Markdown dari Agent secara rapi dan terstruktur:
 * - Parsing tabel Markdown (GFM) menjadi tabel HTML modern dan responsif.
 * - Format headings (#, ##, ###) dengan styling visual yang jelas & kontras.
 * - Callout box untuk blockquote (>).
 * - Bullet lists, ordered lists, code blocks, bold text, dan horizontal dividers.
 * - Indikator cursor berkedip saat respons sedang di-stream.
 */
function MarkdownRendererInner({ content, isStreaming }) {
  if (!content && isStreaming) {
    return (
      <span className="inline-block w-2 h-4 bg-[var(--accent)] animate-pulse rounded-xs align-middle" />
    );
  }

  if (!content) return null;

  return (
    <div className="markdown-body text-sm leading-relaxed text-[var(--text-primary)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-lg md:text-xl font-bold text-white mt-4 mb-2.5 pb-2 border-b border-[var(--border)] flex items-center gap-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base md:text-lg font-semibold text-white mt-5 mb-2.5 flex items-center gap-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm md:text-base font-semibold text-indigo-300 mt-4 mb-1.5">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="mb-2.5 leading-relaxed text-[var(--text-primary)]/95 last:mb-0">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-outside ml-4 space-y-1 my-2 text-[var(--text-secondary)]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside ml-4 space-y-1 my-2 text-[var(--text-secondary)]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed pl-1 text-[var(--text-secondary)]">
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-[var(--accent)] bg-[var(--accent-glow)]/40 rounded-r-xl px-4 py-2.5 my-3 text-[var(--text-primary)] text-sm not-italic shadow-xs">
              {children}
            </blockquote>
          ),
          hr: () => (
            <hr className="my-4 border-[var(--border)]" />
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-white">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-[var(--text-secondary)]">
              {children}
            </em>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--accent-light)] hover:text-white underline transition-colors"
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-3.5 rounded-xl border border-[var(--border)] bg-[#1e2130]/40 shadow-xs">
              <table className="min-w-full text-xs text-left border-collapse">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#16181f]/90 text-[var(--text-primary)] font-semibold border-b border-[var(--border)]">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="px-3.5 py-2.5 font-semibold text-white tracking-wide border-r border-[var(--border)]/40 last:border-r-0 whitespace-nowrap">
              {children}
            </th>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-[var(--border)]/60 text-[var(--text-secondary)]">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-white/[0.03] transition-colors odd:bg-transparent even:bg-white/[0.015]">
              {children}
            </tr>
          ),
          td: ({ children }) => (
            <td className="px-3.5 py-2.5 align-middle leading-normal border-r border-[var(--border)]/30 last:border-r-0">
              {children}
            </td>
          ),
          code: ({ inline, className, children, ...props }) => {
            if (inline) {
              return (
                <code
                  className="px-1.5 py-0.5 rounded bg-[var(--card-bg)] text-[var(--accent-light)] font-mono text-xs border border-[var(--border)]"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <div className="my-3 rounded-xl overflow-x-auto border border-[var(--border)] bg-[#12141c] p-3 text-xs font-mono text-slate-200">
                <pre className="m-0" {...props}>
                  <code>{children}</code>
                </pre>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>

      {/* Streaming cursor indicator */}
      {isStreaming && (
        <span className="inline-block w-1.5 h-3.5 ml-1 bg-[var(--accent)] animate-pulse align-middle rounded-xs" />
      )}
    </div>
  );
}

export const MarkdownRenderer = React.memo(MarkdownRendererInner);
export default MarkdownRenderer;
