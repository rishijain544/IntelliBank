/**
 * Floating AI assistant chat widget.
 *
 * Conversation state lives in this component, not on the server: the backend
 * chat endpoint is stateless and the client replays recent turns for context.
 * That means no user can ever be served another session's history, and closing
 * the tab discards the conversation — appropriate for financial data.
 */
import { useMutation } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { Bot, ChevronDown, Loader2, Send, Sparkles, User as UserIcon } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { errorMessage, post } from '../lib/api';
import { useAuth } from '../store/auth';
import type { AssistantChatResponse, AssistantToolCall } from '../types/api';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: AssistantToolCall[];
  engine?: string;
  latencyMs?: number;
  degradedReason?: string | null;
  failed?: boolean;
}

const SUGGESTIONS = [
  'What is my total balance?',
  'How much did I spend on food this month?',
  'Do I have any loans?',
  'What payments are due soon?',
];

/** Maps a tool name to something a person would recognise. */
const TOOL_LABELS: Record<string, string> = {
  get_account_balances: 'Checked your balances',
  get_transactions: 'Read your transactions',
  get_spending_summary: 'Summarised your spending',
  get_loan_status: 'Looked up your loans',
  get_upcoming_dues: 'Checked upcoming payments',
};

let messageId = 0;
const nextId = () => (messageId += 1);

export default function AssistantWidget() {
  const user = useAuth((s) => s.user);
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the newest message in view as the conversation grows.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Escape closes the panel, matching normal dialog behaviour.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const send = useMutation({
    mutationFn: (text: string) =>
      post<AssistantChatResponse>('/assistant/chat', {
        message: text,
        // Only the recent tail is replayed: enough for follow-ups like
        // "and last month?", without growing the payload without bound.
        history: messages.slice(-8).map((m) => ({ role: m.role, content: m.content })),
      }),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          content: data.message,
          toolCalls: data.tool_calls,
          engine: data.engine,
          latencyMs: data.latency_ms,
          degradedReason: data.degraded_reason,
        },
      ]);
    },
    onError: (error) => {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          content: errorMessage(error),
          failed: true,
        },
      ]);
    },
  });

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || send.isPending) return;
    setMessages((prev) => [...prev, { id: nextId(), role: 'user', content: trimmed }]);
    setInput('');
    send.mutate(trimmed);
  }

  if (!user) return null;

  return (
    <>
      {/* ------------------------------ launcher ------------------------------ */}
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="fixed right-5 bottom-5 z-40 flex items-center gap-2 rounded-full bg-intelligence px-4 py-3 font-semibold text-ink shadow-lg shadow-intelligence/25 transition hover:bg-intelligence-deep"
          aria-label="Open the banking assistant"
        >
          <Sparkles className="h-5 w-5" aria-hidden />
          <span className="hidden sm:inline">Ask IntelliBank</span>
        </button>
      )}

      {/* ------------------------------- panel ------------------------------- */}
      {open && (
        <div
          className="card fixed right-4 bottom-4 z-40 flex h-[min(34rem,calc(100vh-2rem))] w-[min(24rem,calc(100vw-2rem))] flex-col overflow-hidden p-0 shadow-2xl"
          role="dialog"
          aria-label="Banking assistant"
        >
          <header className="flex items-center justify-between border-b border-line bg-surface/80 px-4 py-3">
            <div className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-intelligence/15 text-intelligence">
                <Bot className="h-4.5 w-4.5" aria-hidden />
              </span>
              <div className="leading-tight">
                <p className="text-sm font-semibold text-primary">Banking assistant</p>
                <p className="text-[11px] text-muted">Reads your data — never changes it</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="btn-ghost p-1.5"
              aria-label="Close the assistant"
            >
              <ChevronDown className="h-5 w-5" aria-hidden />
            </button>
          </header>

          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {messages.length === 0 && (
              <div className="py-3">
                <p className="text-sm text-muted">
                  Hello {user.full_name.split(' ')[0]}. Ask me about your balances, spending,
                  loans or upcoming payments.
                </p>
                <div className="mt-4 space-y-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => submit(suggestion)}
                      className="w-full rounded-lg border border-line px-3 py-2 text-left text-xs text-primary transition hover:border-gold/50 hover:bg-surface-raised"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={clsx('flex gap-2.5', message.role === 'user' && 'flex-row-reverse')}
              >
                <span
                  className={clsx(
                    'grid h-7 w-7 shrink-0 place-items-center rounded-full',
                    message.role === 'user'
                      ? 'bg-surface-raised text-primary'
                      : 'bg-intelligence/15 text-intelligence',
                  )}
                >
                  {message.role === 'user' ? (
                    <UserIcon className="h-3.5 w-3.5" aria-hidden />
                  ) : (
                    <Bot className="h-3.5 w-3.5" aria-hidden />
                  )}
                </span>

                <div className={clsx('min-w-0 flex-1', message.role === 'user' && 'text-right')}>
                  <div
                    className={clsx(
                      'inline-block max-w-full rounded-xl px-3 py-2 text-left text-sm',
                      message.role === 'user'
                        ? 'bg-gold text-ink'
                        : message.failed
                          ? 'bg-alert/10 text-alert'
                          : 'border border-intelligence/20 bg-intelligence/[0.07] text-primary',
                    )}
                  >
                    {/* Answers are short and may contain "- " lists, so newlines
                        are preserved rather than parsing markdown. */}
                    <p className="whitespace-pre-wrap break-words">{message.content}</p>
                  </div>

                  {message.role === 'assistant' && message.toolCalls?.length ? (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {message.toolCalls.map((call, index) => (
                        <span
                          key={`${call.name}-${index}`}
                          className={clsx(
                            'badge text-[10px]',
                            call.ok
                              ? 'bg-positive/10 text-positive'
                              : 'bg-alert/10 text-alert',
                          )}
                          title={`${call.name} (${call.duration_ms}ms)`}
                        >
                          {TOOL_LABELS[call.name] ?? call.name}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  {/* Be explicit when the answer came from the rule-based
                      fallback rather than the model. */}
                  {message.role === 'assistant' && message.engine === 'fallback' && (
                    <p className="mt-1 text-[10px] text-warning/80">
                      Offline mode — figures are live, wording is templated
                    </p>
                  )}
                </div>
              </div>
            ))}

            {send.isPending && (
              <div className="flex gap-2.5">
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-intelligence/15 text-intelligence">
                  <Bot className="h-3.5 w-3.5" aria-hidden />
                </span>
                <div className="flex items-center gap-2 rounded-xl bg-surface-raised px-3 py-2 text-sm text-muted">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                  Checking your accounts…
                </div>
              </div>
            )}
          </div>

          <form
            className="border-t border-line p-3"
            onSubmit={(event) => {
              event.preventDefault();
              submit(input);
            }}
          >
            <div className="flex gap-2">
              <input
                ref={inputRef}
                className="input py-2 text-sm"
                placeholder="Ask about your money…"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                maxLength={1000}
                disabled={send.isPending}
                aria-label="Message the banking assistant"
              />
              <button
                type="submit"
                className="btn-primary shrink-0 px-3 py-2"
                disabled={!input.trim() || send.isPending}
                aria-label="Send message"
              >
                <Send className="h-4 w-4" aria-hidden />
              </button>
            </div>
            <p className="mt-2 text-center text-[10px] text-faint">
              Answers come from your real account data. Simulated banking — not financial advice.
            </p>
          </form>
        </div>
      )}
    </>
  );
}
