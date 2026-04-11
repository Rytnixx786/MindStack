import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowPathIcon,
  Bars3BottomLeftIcon,
  BoltIcon,
  ChevronDownIcon,
  ClockIcon,
  DocumentTextIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AnimatePresence, motion } from 'framer-motion';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const HISTORY_LIMIT = 10;
const SUGGESTED_QUESTIONS = [
  { text: 'What is the refund policy?', scope: 'policies' },
  { text: 'Summarize onboarding requirements for new users.', scope: 'onboarding' },
  { text: 'What product limitations are documented?', scope: 'product' },
  { text: 'Which steps are required to get started quickly?', scope: 'onboarding' },
  { text: 'List key support contact channels and escalation path.', scope: 'policies' },
  { text: 'Compare enterprise and standard customer rules.', scope: 'policies' },
];

const QUERY_SCOPES = [
  { id: 'all', label: 'All docs' },
  { id: 'policies', label: 'Policies' },
  { id: 'onboarding', label: 'Onboarding' },
  { id: 'product', label: 'Product docs' },
];

const CONTEXT_SUMMARY = [
  'Best coverage: policies, onboarding, and product docs.',
  'Grounded answers include source citations and chunk references.',
  'If evidence is missing, MindStack will clearly say there is not enough context.',
];

const FADE_FAST = { duration: 0.15, ease: 'easeOut' };

function formatSeconds(ms) {
  return `${(ms / 1000).toFixed(3)}s`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeQuery(query) {
  return query.trim().replace(/\s+/g, ' ').toLowerCase();
}

function formatDayLabel(value) {
  if (!value) return '';
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return `${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')}`;
}

function statusTone(response) {
  if (!response) return 'neutral';
  if (response.answer === 'INSUFFICIENT_CONTEXT' || !response.answer_grounded) return 'danger';
  return 'good';
}

function statusText(response) {
  if (!response) return 'Awaiting answer';
  if (response.answer === 'INSUFFICIENT_CONTEXT' || !response.answer_grounded) {
    return 'Not enough context in sources';
  }
  return 'Grounded in sources';
}

function badgeClasses(tone, darkMode = true) {
  switch (tone) {
    case 'good':
      return darkMode
        ? 'bg-teal-500/14 text-teal-200 ring-1 ring-teal-500/30'
        : 'bg-teal-100 text-teal-800 ring-1 ring-teal-300';
    case 'danger':
      return darkMode
        ? 'bg-zinc-700/40 text-zinc-300 ring-1 ring-zinc-600/35'
        : 'bg-rose-100 text-rose-700 ring-1 ring-rose-200';
    default:
      return darkMode
        ? 'bg-zinc-800/80 text-zinc-400 ring-1 ring-zinc-700/30'
        : 'bg-slate-200 text-slate-700 ring-1 ring-slate-300';
  }
}

function createLatencySeries(history) {
  return history.slice(0, 8).reverse().map((item, index) => ({
    label: `${index + 1}`,
    value: item.latency_ms,
  }));
}

function buildFollowUpPrompts(latestResponse, latestQuery) {
  if (!latestResponse || !latestQuery) return [];
  const source = latestResponse?.citations?.[0]?.source;
  const base = latestQuery.replace(/\?+$/, '').trim();

  return [
    `Can you provide a concise checklist for: ${base}?`,
    `What are the limitations or exceptions related to: ${base}?`,
    source ? `Which additional details are available in ${source}?` : `Which source chunks best support: ${base}?`,
  ];
}

function Sparkline({ data, darkMode }) {
  if (!data.length) {
    return (
      <div className={`flex h-32 items-center justify-center rounded-2xl text-sm ${darkMode ? 'bg-zinc-900/62 text-zinc-500' : 'bg-white/85 text-slate-600 ring-1 ring-slate-200'}`}>
        No latency data yet
      </div>
    );
  }

  const width = 420;
  const height = 160;
  const padding = 18;
  const max = Math.max(...data.map((item) => item.value), 1);
  const axisTone = darkMode ? 'text-zinc-700' : 'text-slate-500';
  const points = data.map((item, index) => {
    const x = padding + (index * (width - padding * 2)) / Math.max(data.length - 1, 1);
    const y = height - padding - ((item.value / max) * (height - padding * 2));
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className={`rounded-2xl p-4 shadow-[0_10px_26px_rgba(0,0,0,0.2)] ${darkMode ? 'bg-zinc-900/62' : 'bg-white/88 ring-1 ring-slate-200'}`}>
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className={`text-sm font-medium ${darkMode ? 'text-zinc-200' : 'text-slate-800'}`}>Recent latencies</p>
          <p className={`text-xs ${darkMode ? 'text-zinc-500' : 'text-slate-600'}`}>Last {data.length} queries</p>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${darkMode ? 'bg-zinc-800/85 text-zinc-400' : 'bg-slate-200 text-slate-700'}`}>
          ms
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-40 w-full" role="img" aria-label="Recent query latency chart">
        <defs>
          <linearGradient id="lineGradient" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#2dd4bf" />
            <stop offset="100%" stopColor="#14b8a6" />
          </linearGradient>
        </defs>
        <line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} stroke="currentColor" className={axisTone} />
        <polyline fill="none" stroke="url(#lineGradient)" strokeWidth="4" strokeLinejoin="round" strokeLinecap="round" points={points} />
        {data.map((item, index) => {
          const x = padding + (index * (width - padding * 2)) / Math.max(data.length - 1, 1);
          const y = height - padding - ((item.value / max) * (height - padding * 2));
          return <circle key={`${item.label}-${index}`} cx={x} cy={y} r="4" fill="#14b8a6" />;
        })}
      </svg>
      <div className={`mt-2 flex items-center justify-between text-xs ${darkMode ? 'text-zinc-500' : 'text-slate-600'}`}>
        <span>Fastest</span>
        <span>Slowest</span>
      </div>
    </div>
  );
}

function MetricsTrendChart({ data, loading, darkMode }) {
  if (loading) {
    return (
      <div className="flex h-80 items-center justify-center rounded-2xl bg-zinc-900/62 text-sm text-zinc-500">
        Loading trend data...
      </div>
    );
  }

  if (!data.length) {
    return (
      <div className="flex h-80 items-center justify-center rounded-2xl bg-zinc-900/62 text-sm text-zinc-500">
        No trend data yet
      </div>
    );
  }

  const chartData = data.map((item) => ({
    ...item,
    day: formatDayLabel(item.date),
    grounded_rate_pct: Number((item.grounded_rate * 100).toFixed(2)),
    avg_latency_ms: Number((item.avg_latency_ms || 0).toFixed(2)),
  }));

  const axisColor = darkMode ? '#a1a1aa' : '#334155';
  const gridColor = darkMode ? 'rgba(113,113,122,0.36)' : 'rgba(148,163,184,0.4)';
  const tooltipBg = darkMode ? 'rgba(15, 23, 42, 0.96)' : 'rgba(248, 250, 252, 0.98)';
  const tooltipBorder = darkMode ? '1px solid rgba(71, 85, 105, 0.75)' : '1px solid rgba(148, 163, 184, 0.7)';
  const tooltipText = darkMode ? '#e4e4e7' : '#0f172a';
  const groundedLine = darkMode ? '#2dd4bf' : '#0f766e';
  const latencyLine = darkMode ? '#99f6e4' : '#14b8a6';

  return (
    <div className="h-80 w-full rounded-2xl bg-zinc-900/62 p-4 shadow-[0_10px_26px_rgba(0,0,0,0.2)]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis dataKey="day" tick={{ fontSize: 12, fill: axisColor }} />
          <YAxis yAxisId="left" tick={{ fontSize: 12, fill: axisColor }} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12, fill: axisColor }} tickFormatter={(value) => `${value}ms`} />
          <Tooltip
            contentStyle={{ background: tooltipBg, border: tooltipBorder, borderRadius: 14, color: tooltipText }}
            labelStyle={{ color: tooltipText, fontWeight: 600 }}
            itemStyle={{ color: tooltipText, fontWeight: 500 }}
            formatter={(value, name) => {
              if (name === 'Grounded rate') return [`${value}%`, name];
              return [`${value} ms`, name];
            }}
          />
          <Legend wrapperStyle={{ color: axisColor }} />
          <Line yAxisId="left" type="monotone" dataKey="grounded_rate_pct" name="Grounded rate" stroke={groundedLine} strokeWidth={2.4} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="avg_latency_ms" name="Avg latency" stroke={latencyLine} strokeWidth={2.1} dot={false} strokeOpacity={0.9} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function App() {
  const [query, setQuery] = useState('What is the refund policy?');
  const [streamMode, setStreamMode] = useState(true);
  const [scope, setScope] = useState('all');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [topKRetrieval, setTopKRetrieval] = useState(6);
  const [topKRerank, setTopKRerank] = useState(2);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streamingInProgress, setStreamingInProgress] = useState(false);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState({
    total_queries: 0,
    grounded_rate: 0,
    avg_latency_ms: 0,
    p95_latency_ms: 0,
    p50_latency_ms: 0,
    queries_last_24h: 0,
    grounded_rate_7d: 0,
  });
  const [trendData, setTrendData] = useState([]);
  const [trendLoading, setTrendLoading] = useState(false);
  const [metricsTab, setMetricsTab] = useState('overview');
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking');
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [filesToUpload, setFilesToUpload] = useState([]);
  const [darkMode, setDarkMode] = useState(() => {
    const stored = window.localStorage.getItem('mindstack-theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const answerRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    window.localStorage.setItem('mindstack-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    pingHealth();
    fetchMetrics();
    fetchTrend();
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      pingHealth();
      fetchMetrics();
      fetchTrend();
    }, 30000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (response && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [response]);

  async function parseErrorResponse(res, fallbackMessage) {
    try {
      const body = await res.json();
      if (body?.detail) {
        if (Array.isArray(body.detail)) {
          return `${fallbackMessage}: ${body.detail.map((item) => item.msg).join(', ')}`;
        }
        return `${fallbackMessage}: ${body.detail}`;
      }
    } catch {
      // Ignore JSON parse errors and fall back to status message.
    }
    return `${fallbackMessage} (HTTP ${res.status})`;
  }

  async function pingHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setApiStatus(res.ok ? 'online' : 'offline');
    } catch {
      setApiStatus('offline');
    }
  }

  async function fetchMetrics() {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMetrics(data);
      setApiStatus('online');
    } catch (err) {
      setApiStatus('offline');
      console.error('Failed to fetch metrics', err);
    }
  }

  async function fetchTrend() {
    setTrendLoading(true);
    try {
      const res = await fetch(`${API_BASE}/metrics/trend`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTrendData(Array.isArray(data) ? data : []);
      setApiStatus('online');
    } catch (err) {
      setApiStatus('offline');
      console.error('Failed to fetch metrics trend', err);
      setTrendData([]);
    } finally {
      setTrendLoading(false);
    }
  }

  function commitQueryResult(data, trimmed) {
    setApiStatus('online');
    setResponse(data);
    setHistory((current) => {
      const next = [{
        ...data,
        query: trimmed,
        normalizedQuery: normalizeQuery(trimmed),
        scope,
        top_k_retrieval: topKRetrieval,
        top_k_rerank: topKRerank,
      }, ...current.filter((item) => normalizeQuery(item.query) !== normalizeQuery(trimmed))];
      return next.slice(0, HISTORY_LIMIT);
    });
    fetchMetrics();
  }

  async function requestNonStreamingQuery(trimmed) {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: trimmed,
        top_k_retrieval: topKRetrieval,
        top_k_rerank: topKRerank,
      }),
    });

    if (!res.ok) {
      const message = await parseErrorResponse(res, 'Query failed');
      throw new Error(message);
    }

    return res.json();
  }

  async function handleStreamingQuery(trimmed) {
    const streamResult = {
      query: trimmed,
      answer: '',
      answer_grounded: false,
      citations: [],
      chunks_retrieved: 0,
      latency_ms: 0,
      model_used: 'mistral',
      prompt_version: '1.1',
      reranker_applied: true,
      llm_called: true,
      cached: false,
      timestamp: new Date().toISOString(),
    };

    setStreamingInProgress(true);
    setResponse(streamResult);

    let typingQueue = '';
    let typingTimer = null;

    const stopTypingPump = () => {
      if (typingTimer !== null) {
        window.clearInterval(typingTimer);
        typingTimer = null;
      }
    };

    const startTypingPump = () => {
      if (typingTimer !== null) return;
      typingTimer = window.setInterval(() => {
        if (!typingQueue) {
          if (doneReceived) {
            stopTypingPump();
          }
          return;
        }

        const batchSize = Math.max(2, Math.min(16, Math.ceil(typingQueue.length / 12)));
        streamResult.answer += typingQueue.slice(0, batchSize);
        typingQueue = typingQueue.slice(batchSize);
        setResponse({ ...streamResult });

        if (!typingQueue && doneReceived) {
          stopTypingPump();
        }
      }, 16);
    };

    const res = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: trimmed,
        top_k_retrieval: topKRetrieval,
        top_k_rerank: topKRerank,
      }),
    });

    if (!res.ok) {
      const message = await parseErrorResponse(res, 'Streaming query failed');
      throw new Error(message);
    }

    if (!res.body) {
      throw new Error('Streaming response has no body.');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let doneReceived = false;

    const processLine = (line) => {
      const trimmedLine = line.trim();
      if (!trimmedLine) return;

      const parsed = JSON.parse(trimmedLine);
      const event = parsed?.event;
      const data = parsed?.data || {};

      if (event === 'citations') {
        streamResult.citations = Array.isArray(data.citations) ? data.citations : [];
        streamResult.chunks_retrieved = Number(data.chunks_retrieved || 0);
        streamResult.answer_grounded = Boolean(data.answer_grounded);
        streamResult.reranker_applied = Boolean(data.reranker_applied ?? true);
        streamResult.prompt_version = data.prompt_version || streamResult.prompt_version;
        setResponse({ ...streamResult });
        return;
      }

      if (event === 'token') {
        typingQueue += String(data.token || '');
        startTypingPump();
        return;
      }

      if (event === 'done') {
        doneReceived = true;
        if (typeof data.answer === 'string' && !streamResult.answer && !typingQueue) {
          typingQueue += data.answer;
          startTypingPump();
        }
        if (typeof data.answer_grounded === 'boolean') {
          streamResult.answer_grounded = data.answer_grounded;
        }
        streamResult.latency_ms = Number(data.latency_ms || 0);
        streamResult.cached = Boolean(data.cached || false);
        streamResult.model_used = data.model_used || streamResult.model_used;
        streamResult.llm_called = Boolean(data.llm_called ?? streamResult.llm_called);
        streamResult.timestamp = new Date().toISOString();
        setResponse({ ...streamResult });
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        processLine(line);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      processLine(buffer);
    }

    while (typingQueue) {
      await new Promise((resolve) => window.setTimeout(resolve, 16));
    }
    stopTypingPump();

    if (!doneReceived) {
      stopTypingPump();
      throw new Error('Streaming ended before completion event.');
    }

    commitQueryResult({ ...streamResult }, trimmed);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError('');
    setUploadMessage('');

    try {
      if (streamMode) {
        try {
          await handleStreamingQuery(trimmed);
        } catch (streamErr) {
          console.warn('Streaming failed, falling back to non-streaming query.', streamErr);
          const fallbackData = await requestNonStreamingQuery(trimmed);
          commitQueryResult(fallbackData, trimmed);
        }
      } else {
        const data = await requestNonStreamingQuery(trimmed);
        commitQueryResult(data, trimmed);
      }
    } catch (err) {
      setApiStatus('offline');
      setError(err.message || 'Something went wrong while querying the API.');
    } finally {
      setStreamingInProgress(false);
      setLoading(false);
    }
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!filesToUpload.length) {
      setUploadMessage('Select at least one file to upload.');
      return;
    }

    setUploading(true);
    setUploadMessage('');
    setError('');

    try {
      const formData = new FormData();
      filesToUpload.forEach((file) => formData.append('files', file));

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const message = await parseErrorResponse(res, 'Upload failed');
        throw new Error(message);
      }

      const data = await res.json();
      const uploadedCount = Array.isArray(data?.files_uploaded) ? data.files_uploaded.length : filesToUpload.length;
      setApiStatus('online');
      setUploadMessage(`Uploaded ${uploadedCount} file(s) and refreshed index.`);
      setFilesToUpload([]);
      fetchMetrics();
      fetchTrend();
    } catch (err) {
      setApiStatus('offline');
      setUploadMessage(err.message || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  }

  const latencySeries = useMemo(() => createLatencySeries(history), [history]);
  const followUpPrompts = useMemo(() => buildFollowUpPrompts(response, query), [response, query]);
  const suggestedByScope = useMemo(() => {
    if (scope === 'all') return SUGGESTED_QUESTIONS;
    return SUGGESTED_QUESTIONS.filter((item) => item.scope === scope);
  }, [scope]);
  const groundedTone = statusTone(response);
  const groundedLabel = statusText(response);
  const responseSources = response?.citations ?? [];
  const apiOnline = apiStatus === 'online';

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={FADE_FAST}
      className={`min-h-screen bg-zinc-950 text-zinc-100 ${darkMode ? 'theme-dark' : 'theme-light'}`}
    >
      <div
        className={`pointer-events-none fixed inset-0 ${darkMode
          ? 'bg-[radial-gradient(circle_at_24%_18%,rgba(20,184,166,0.04),transparent_30%),radial-gradient(circle_at_78%_5%,rgba(56,189,248,0.025),transparent_22%)]'
          : 'bg-[radial-gradient(circle_at_18%_10%,rgba(20,184,166,0.13),transparent_30%),radial-gradient(circle_at_84%_2%,rgba(14,165,233,0.11),transparent_26%),linear-gradient(180deg,rgba(248,250,252,1),rgba(241,245,249,0.96))]'}
        `}
      />
      <div className="relative mx-auto max-w-[1400px] px-5 py-8 sm:px-8 sm:py-10 lg:px-12 lg:py-12">
        <header className="mb-8 flex items-start justify-between gap-6 rounded-3xl bg-zinc-900/80 px-7 py-6 shadow-[0_12px_34px_rgba(2,6,23,0.24)] backdrop-blur-xl">
          <div className="max-w-2xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-teal-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.2em] text-teal-300">
              <SparklesIcon className="h-4 w-4" />
              MindStack
            </div>
            <h1 className="font-display text-[2rem] font-medium leading-tight tracking-[-0.01em] text-zinc-50 sm:text-[2.4rem]">
              Enterprise-grade document answers with grounded evidence.
            </h1>
            <p className="mt-4 max-w-2xl text-[15px] leading-7 text-zinc-300 sm:text-base">
              Query your knowledge base, inspect citations, and monitor system quality from a clean, portfolio-ready dashboard.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ${apiOnline ? 'bg-emerald-500/10 text-emerald-300' : 'bg-rose-500/10 text-rose-300'}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${apiOnline ? 'bg-emerald-300 animate-pulse-soft' : 'bg-rose-300'}`} />
              {apiStatus === 'checking' ? 'Checking API' : apiOnline ? 'Connected' : 'API offline'}
            </span>
            <button
              onClick={() => setDarkMode((value) => !value)}
              className="rounded-full bg-zinc-800/90 px-3 py-1.5 text-xs font-medium text-zinc-200 transition duration-150 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-teal-600"
              aria-label="Toggle theme"
            >
              {darkMode ? 'Dark theme' : 'Light theme'}
            </button>
          </div>
        </header>

        <main className="grid gap-8 lg:grid-cols-[minmax(0,1.52fr)_minmax(330px,0.86fr)]">
          <section className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={FADE_FAST}
              className="rounded-3xl bg-zinc-900/82 p-7 shadow-[0_12px_32px_rgba(0,0,0,0.28)] backdrop-blur-xl"
            >
              <div className="mb-6 flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-medium tracking-[-0.01em] text-zinc-100">Ask a question</h2>
                  <p className="mt-1.5 text-sm leading-6 text-zinc-300">Grounded answers with citation context and latency feedback.</p>
                </div>
                <div className="flex items-center gap-2 rounded-full bg-zinc-800/85 px-3 py-2 text-xs font-medium text-zinc-300">
                  <BoltIcon className="h-4 w-4 text-teal-300" />
                  Optimized for repeated queries
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="rounded-3xl bg-zinc-900/60 p-5">
                  <div className="mb-2 text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">Context summary</div>
                  <ul className="space-y-2 text-sm leading-6 text-zinc-300">
                    {CONTEXT_SUMMARY.map((item) => (
                      <li key={item} className="flex items-start gap-2">
                        <span className="mt-2 h-1.5 w-1.5 rounded-full bg-teal-400/80" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-medium text-zinc-300">Question scope</span>
                    <span className="text-xs text-zinc-500">UI scope hint for targeted prompting</span>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                    {QUERY_SCOPES.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => setScope(item.id)}
                        className={`min-h-11 rounded-2xl px-3 py-2 text-sm font-medium transition duration-150 focus:outline-none focus:ring-2 focus:ring-teal-500/55 ${scope === item.id
                          ? 'bg-teal-500/12 text-teal-300 shadow-[0_8px_22px_rgba(20,184,166,0.16)]'
                          : 'bg-zinc-800/85 text-zinc-400 hover:-translate-y-0.5 hover:text-zinc-100'}`}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-zinc-300">Suggested questions</span>
                    <span className="text-xs text-zinc-500">Click to auto-fill</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {suggestedByScope.map((item) => (
                      <button
                        key={item.text}
                        type="button"
                        onClick={() => setQuery(item.text)}
                        className="rounded-full bg-zinc-800/90 px-3 py-1.5 text-sm text-zinc-300 transition duration-150 hover:-translate-y-0.5 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-teal-600"
                      >
                        {item.text}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3 rounded-3xl bg-zinc-900/60 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-medium text-zinc-300">Upload documents</span>
                    <span className="text-xs text-zinc-500">Supports PDF, TXT, MD, DOC, DOCX</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.txt,.md,.doc,.docx"
                      onChange={(event) => {
                        const nextFiles = Array.from(event.target.files || []);
                        setFilesToUpload(nextFiles);
                      }}
                      className="block w-full max-w-md cursor-pointer rounded-2xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-300 file:mr-3 file:rounded-full file:border-0 file:bg-zinc-800 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-zinc-200 hover:file:bg-zinc-700 focus:outline-none focus:ring-2 focus:ring-teal-600"
                    />
                    <button
                      type="button"
                      onClick={handleUpload}
                      disabled={uploading || !filesToUpload.length}
                      className="inline-flex min-h-10 items-center justify-center rounded-2xl bg-zinc-800/95 px-4 py-2 text-sm font-medium text-zinc-200 transition duration-150 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-teal-600 disabled:cursor-not-allowed disabled:opacity-55"
                    >
                      {uploading ? 'Uploading...' : 'Upload & Reindex'}
                    </button>
                  </div>
                  {filesToUpload.length > 0 && (
                    <p className="text-xs text-zinc-400">
                      {filesToUpload.length} file(s) selected
                    </p>
                  )}
                  {uploadMessage && (
                    <p className={`text-sm ${uploadMessage.toLowerCase().includes('failed') ? 'text-rose-300' : 'text-emerald-300'}`}>
                      {uploadMessage}
                    </p>
                  )}
                </div>

                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-zinc-300">Query</span>
                  <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) {
                        return;
                      }

                      event.preventDefault();
                      event.currentTarget.form?.requestSubmit();
                    }}
                    placeholder="Ask about policies, onboarding, pricing, or internal documentation..."
                    className="min-h-[180px] w-full resize-y rounded-2xl border border-zinc-700 bg-zinc-900 px-5 py-5 text-base leading-8 text-zinc-100 outline-none transition duration-150 placeholder:text-zinc-500 focus:ring-2 focus:ring-teal-600 focus:border-teal-600"
                  />
                </label>

                <div className="rounded-3xl bg-zinc-900/65 p-4">
                  <button
                    type="button"
                    onClick={() => setShowAdvanced((value) => !value)}
                    className="flex w-full items-center justify-between text-sm font-medium text-zinc-300 transition hover:text-teal-300"
                  >
                    <span>Advanced retrieval options</span>
                    <ChevronDownIcon className={`h-5 w-5 transition ${showAdvanced ? 'rotate-180' : ''}`} />
                  </button>
                  {showAdvanced && (
                    <div className="mt-4 grid gap-4 sm:grid-cols-2">
                      <label className="space-y-2">
                        <span className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">Top K retrieval</span>
                        <input
                          type="range"
                          min={1}
                          max={20}
                          value={topKRetrieval}
                          onChange={(event) => setTopKRetrieval(Number(event.target.value))}
                          className="w-full accent-sky-500"
                        />
                        <div className="text-sm text-zinc-400">{topKRetrieval} chunks</div>
                      </label>
                      <label className="space-y-2">
                        <span className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">Top K rerank</span>
                        <input
                          type="range"
                          min={1}
                          max={10}
                          value={topKRerank}
                          onChange={(event) => setTopKRerank(Number(event.target.value))}
                          className="w-full accent-teal-500"
                        />
                        <div className="text-sm text-zinc-400">{topKRerank} chunks</div>
                      </label>
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <motion.button
                    type="submit"
                    disabled={loading}
                    whileHover={{ y: -2 }}
                    whileTap={{ y: 0, scale: 0.99 }}
                    transition={FADE_FAST}
                    className="inline-flex min-h-12 items-center justify-center gap-2 rounded-2xl bg-teal-500 px-6 py-3 text-sm font-medium text-zinc-950 shadow-[0_12px_30px_rgba(20,184,166,0.28)] transition duration-150 hover:shadow-[0_16px_34px_rgba(20,184,166,0.32)] focus:outline-none focus:ring-2 focus:ring-teal-400 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loading ? <ArrowPathIcon className="h-5 w-5 animate-spin" /> : <SparklesIcon className="h-5 w-5" />}
                    {loading ? 'Generating...' : 'Ask MindStack'}
                  </motion.button>
                  <label className="inline-flex min-h-12 items-center gap-2 rounded-2xl bg-zinc-800/85 px-3 py-2 text-sm text-zinc-300">
                    <input
                      type="checkbox"
                      checked={streamMode}
                      onChange={(event) => setStreamMode(event.target.checked)}
                      className="h-4 w-4 accent-teal-500"
                    />
                    Stream
                  </label>
                  <div className="text-sm leading-6 text-zinc-500">Press Enter to submit. Use Shift+Enter for a new line.</div>
                </div>
              </form>

              {loading && (
                <div className="mt-6 rounded-3xl bg-zinc-900/70 p-4">
                  <div className="mb-3 flex items-center gap-3 text-sm font-medium text-teal-300">
                    <ArrowPathIcon className="h-4 w-4 animate-spin" />
                    Processing query
                  </div>
                  <div className="space-y-3">
                    <div className="h-4 w-11/12 animate-shimmer rounded-full bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%] dark:from-slate-800 dark:via-slate-700 dark:to-slate-800" />
                    <div className="h-4 w-9/12 animate-shimmer rounded-full bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%] dark:from-slate-800 dark:via-slate-700 dark:to-slate-800" />
                    <div className="h-4 w-10/12 animate-shimmer rounded-full bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200 bg-[length:200%_100%] dark:from-slate-800 dark:via-slate-700 dark:to-slate-800" />
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-6 rounded-3xl bg-rose-500/10 p-4 text-rose-200 shadow-sm">
                  <div className="mb-1 text-sm font-semibold">API error</div>
                  <div className="text-sm leading-6">{error}</div>
                </div>
              )}
            </motion.div>

            <motion.div
              ref={answerRef}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={FADE_FAST}
              className="rounded-3xl bg-zinc-900/82 p-7 shadow-[0_12px_32px_rgba(0,0,0,0.28)] backdrop-blur-xl"
            >
              <div className="mb-5 flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-xl font-medium tracking-[-0.01em] text-zinc-100">Answer</h2>
                  <p className="mt-1.5 text-sm leading-6 text-zinc-300">The latest grounded response with source trails.</p>
                </div>
                <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${badgeClasses(groundedTone, darkMode)} ${groundedTone === 'good' ? 'animate-pulse-soft' : ''}`}>
                  {groundedLabel}
                </span>
              </div>

              {!response ? (
                <div className="rounded-3xl bg-zinc-900/70 p-7 text-sm leading-7 text-zinc-300">
                  Start with a policy, onboarding, or product question. You will see grounded answers and lightweight citations here.
                </div>
              ) : (
                <AnimatePresence mode="wait">
                  <motion.div
                    key={response.timestamp || response.answer}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={FADE_FAST}
                    className="space-y-7"
                  >
                  <div className="rounded-3xl bg-zinc-900/75 p-6 shadow-[0_8px_24px_rgba(0,0,0,0.22)]">
                    <p className="text-[11px] font-medium uppercase tracking-[0.22em] text-teal-300">Answer text</p>
                    <p className="mt-4 max-w-4xl whitespace-pre-wrap text-[16px] leading-8 text-zinc-100">
                      {response.answer}
                      {streamingInProgress && <span className="streaming-cursor" aria-hidden>|</span>}
                    </p>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-3">
                    <StatTile icon={ClockIcon} label="Latency" value={formatSeconds(response.latency_ms)} />
                    <StatTile icon={Bars3BottomLeftIcon} label="Sources" value={`${response.chunks_retrieved} chunks`} />
                    <StatTile icon={DocumentTextIcon} label="Prompt" value={response.prompt_version || 'n/a'} />
                  </div>

                  <div className="space-y-3">
                    {followUpPrompts.length > 0 && (
                      <div className="rounded-3xl bg-zinc-900/68 p-4">
                        <div className="mb-2 text-[11px] font-medium uppercase tracking-[0.2em] text-zinc-500">Follow-up questions</div>
                        <div className="flex flex-wrap gap-2">
                          {followUpPrompts.map((item) => (
                            <button
                              key={item}
                              type="button"
                              onClick={() => setQuery(item)}
                              className="rounded-full bg-zinc-800/90 px-3 py-1.5 text-sm text-zinc-400 transition duration-150 hover:-translate-y-0.5 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-teal-500/55"
                            >
                              {item}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium uppercase tracking-[0.2em] text-zinc-500">Citations</h3>
                      <span className="text-xs text-zinc-500">Click to inspect supporting text</span>
                    </div>
                    {responseSources.length ? (
                      responseSources.map((citation, index) => (
                        <details key={`${citation.source}-${citation.chunk_index}-${index}`} className="group rounded-2xl bg-zinc-900/66 p-4 transition duration-150 hover:-translate-y-0.5 hover:scale-[1.01] hover:bg-zinc-800/78">
                          <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                            <div>
                              <div className="text-sm font-medium text-zinc-100">
                                {citation.source}
                              </div>
                              <div className="mt-1 text-xs text-zinc-500">Chunk {citation.chunk_index} • Score {typeof citation.reranker_score === 'number' ? citation.reranker_score.toFixed(2) : 'n/a'}</div>
                            </div>
                            <ChevronDownIcon className="h-5 w-5 flex-none text-zinc-500 transition group-open:rotate-180" />
                          </summary>
                          <div className="mt-4 rounded-2xl bg-zinc-900/85 p-4 text-sm leading-7 text-zinc-300">
                            {citation.text}
                          </div>
                        </details>
                      ))
                    ) : (
                      <div className="rounded-3xl bg-zinc-900/65 p-5 text-sm text-zinc-500">
                        No citations returned for this response.
                      </div>
                    )}
                  </div>
                  </motion.div>
                </AnimatePresence>
              )}
            </motion.div>
          </section>

          <motion.aside
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={FADE_FAST}
            className="space-y-7 lg:sticky lg:top-6 lg:self-start"
          >
            <div className="inline-flex w-full rounded-2xl bg-zinc-900/64 p-1">
              <button
                type="button"
                onClick={() => setMetricsTab('overview')}
                className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium transition duration-150 ${metricsTab === 'overview'
                  ? 'bg-teal-500/12 text-teal-300'
                  : 'text-zinc-500 hover:text-zinc-200'}`}
              >
                Overview
              </button>
              <button
                type="button"
                onClick={() => setMetricsTab('metrics')}
                className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium transition duration-150 ${metricsTab === 'metrics'
                  ? 'bg-teal-500/12 text-teal-300'
                  : 'text-zinc-500 hover:text-zinc-200'}`}
              >
                Metrics
              </button>
            </div>

            <div className="rounded-3xl bg-zinc-900/74 p-6 shadow-[0_12px_30px_rgba(0,0,0,0.24)] backdrop-blur-xl">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-medium tracking-[-0.01em] text-zinc-100">Metrics dashboard</h2>
                  <p className="mt-1 text-sm leading-6 text-zinc-400">Live API health and quality indicators.</p>
                </div>
                <button
                  type="button"
                  onClick={fetchMetrics}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-zinc-800/90 text-zinc-300 transition duration-150 hover:text-teal-300 focus:outline-none focus:ring-2 focus:ring-teal-500/55"
                  aria-label="Refresh metrics"
                >
                  <ArrowPathIcon className="h-5 w-5 transition-transform duration-150 hover:rotate-12" />
                </button>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                <MetricCard label="Total queries" value={metrics.total_queries ?? 0} accent="brand" />
                <MetricCard label="Grounded rate" value={formatPercent(metrics.grounded_rate || 0)} accent="teal" />
                <MetricCard label="Avg latency" value={`${(metrics.avg_latency_ms || 0).toFixed(0)}ms`} accent="brand" />
                <MetricCard label="P95 latency" value={`${(metrics.p95_latency_ms || 0).toFixed(0)}ms`} accent="teal" />
                <MetricCard label="P50 latency" value={`${(metrics.p50_latency_ms || 0).toFixed(0)}ms`} accent="brand" />
                <MetricCard label="Queries (24h)" value={metrics.queries_last_24h ?? 0} accent="teal" />
                <MetricCard label="Grounded rate (7d)" value={formatPercent(metrics.grounded_rate_7d || 0)} accent="brand" />
              </div>
            </div>

            {metricsTab === 'overview' ? (
              <>
                <Sparkline data={latencySeries} darkMode={darkMode} />

                <div className="rounded-3xl bg-zinc-900/74 p-6 shadow-[0_12px_30px_rgba(0,0,0,0.24)] backdrop-blur-xl">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h2 className="text-lg font-medium tracking-[-0.01em] text-zinc-100">History</h2>
                      <p className="mt-1 text-sm text-zinc-400">Last 10 queries, newest first.</p>
                    </div>
                    <span className="rounded-full bg-zinc-800/90 px-3 py-1 text-xs font-medium text-zinc-400">
                      {history.length}
                    </span>
                  </div>

                  <div className="space-y-3">
                    {history.length ? history.map((item) => (
                      <button
                        key={`${item.timestamp}-${item.query}`}
                        onClick={() => setQuery(item.query)}
                        className={`w-full rounded-3xl p-4 text-left transition duration-150 hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-teal-500/45 ${darkMode
                          ? 'bg-zinc-900/75 hover:bg-zinc-800/85'
                          : 'bg-slate-100/92 ring-1 ring-slate-200 hover:bg-slate-200/82'}`}
                      >
                        <div className="mb-2 flex flex-wrap items-center gap-2">
                          <span className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${item.cached
                            ? (darkMode ? 'bg-amber-500/18 text-amber-200' : 'bg-amber-100 text-amber-700 ring-1 ring-amber-200')
                            : (darkMode ? 'bg-zinc-700/50 text-zinc-300' : 'bg-slate-200 text-slate-700 ring-1 ring-slate-300')}`}>
                            {item.cached ? 'CACHED' : 'LIVE'}
                          </span>
                          <span className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${badgeClasses(statusTone(item), darkMode)}`}>
                            {statusTone(item) === 'good' ? 'Grounded' : 'Needs context'}
                          </span>
                          <span className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${darkMode ? 'bg-zinc-800/90 text-zinc-400' : 'bg-slate-200 text-slate-700 ring-1 ring-slate-300'}`}>
                            {formatSeconds(item.latency_ms)}
                          </span>
                        </div>
                        <div className={`line-clamp-2 text-sm leading-6 ${darkMode ? 'text-zinc-300' : 'text-slate-800'}`}>{item.query}</div>
                        <div className={`mt-2 text-xs ${darkMode ? 'text-zinc-500' : 'text-slate-600'}`}>
                          Scope: {QUERY_SCOPES.find((scopeItem) => scopeItem.id === item.scope)?.label || 'All docs'}
                        </div>
                      </button>
                    )) : (
                      <div className="rounded-3xl bg-zinc-900/70 p-5 text-sm text-zinc-500">
                        Query history will appear here after your first request.
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="rounded-3xl bg-zinc-900/74 p-6 shadow-[0_12px_30px_rgba(0,0,0,0.24)] backdrop-blur-xl">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-medium tracking-[-0.01em] text-zinc-100">30-day trend</h2>
                    <p className="mt-1 text-sm text-zinc-400">Grounded rate and average latency by day.</p>
                  </div>
                  <button
                    type="button"
                    onClick={fetchTrend}
                    className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-zinc-800/90 text-zinc-300 transition duration-150 hover:text-teal-300 focus:outline-none focus:ring-2 focus:ring-teal-500/55"
                    aria-label="Refresh trend"
                  >
                    <ArrowPathIcon className={`h-5 w-5 ${trendLoading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
                <MetricsTrendChart data={trendData} loading={trendLoading} darkMode={darkMode} />
              </div>
            )}
          </motion.aside>
        </main>
      </div>
    </motion.div>
  );
}

function MetricCard({ label, value, accent }) {
  const accentStyles = accent === 'teal'
    ? 'from-teal-500/16 via-teal-500/7 to-transparent text-teal-300'
    : 'from-zinc-700/20 via-zinc-700/8 to-transparent text-zinc-300';

  return (
    <motion.div
      layout
      transition={FADE_FAST}
      className="rounded-3xl bg-zinc-900/68 p-4 shadow-[0_8px_22px_rgba(0,0,0,0.2)]"
    >
      <div className={`rounded-2xl bg-gradient-to-r p-3 ${accentStyles}`}>
        <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">{label}</p>
        <p className="mt-2 text-xl font-medium text-zinc-100">{value}</p>
      </div>
    </motion.div>
  );
}

function StatTile({ icon: Icon, label, value }) {
  return (
    <div className="rounded-3xl bg-zinc-900/68 p-4">
      <div className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-zinc-800 text-zinc-200">
          <Icon className="h-5 w-5" />
        </span>
        <div>
          <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">{label}</div>
          <div className="mt-1 text-sm font-medium text-zinc-100">{value}</div>
        </div>
      </div>
    </div>
  );
}

export default App;