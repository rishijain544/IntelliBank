/**
 * Shared presentational primitives.
 *
 * Deliberately plain components rather than a component library: the set of
 * patterns this app needs is small, and hand-rolling them keeps the bundle
 * light and the styling consistent with the Tailwind theme tokens.
 */
import { clsx } from 'clsx';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Info,
  Loader2,
  ShieldAlert,
  Sparkles,
  XCircle,
} from 'lucide-react';
import type { ReactNode } from 'react';

/* -------------------------------------------------------------------------- */
/* Layout                                                                      */
/* -------------------------------------------------------------------------- */

export function Card({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx('card p-5', className)} {...rest}>
      {children}
    </div>
  );
}

export function SectionHeading({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 className="text-lg font-semibold text-primary">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <header className="mb-7 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1.5 max-w-2xl text-sm text-muted">{subtitle}</p>}
      </div>
      {action}
    </header>
  );
}

/* -------------------------------------------------------------------------- */
/* Status                                                                      */
/* -------------------------------------------------------------------------- */

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={clsx('animate-spin', className ?? 'h-4 w-4')} aria-hidden />;
}

export function LoadingBlock({ label = 'Loading…', rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton h-12 w-full" />
      ))}
    </div>
  );
}

export function ErrorBlock({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="card border-alert/30 bg-alert/5 p-5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-alert" aria-hidden />
        <div className="flex-1">
          <p className="text-sm font-medium text-alert">Something went wrong</p>
          <p className="mt-1 text-sm text-alert/80">{message}</p>
          {onRetry && (
            <button type="button" onClick={onRetry} className="btn-secondary mt-3 py-1.5 text-xs">
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line px-6 py-14 text-center">
      {icon && <div className="mb-3 text-muted">{icon}</div>}
      <p className="font-medium text-primary">{title}</p>
      {description && <p className="mt-1.5 max-w-sm text-sm text-muted">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Badges                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Badge / meter tones.
 *
 * `intelligence` is reserved for AI-generated output: fraud scores, anomaly
 * callouts, credit-decision reasoning, assistant responses. Generic
 * informational states use `info` (gold), so the teal keeps its meaning as a
 * signal that a model produced the value.
 */
export type Tone =
  | 'neutral'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'
  | 'accent'
  | 'intelligence';

const TONE_CLASS: Record<Tone, string> = {
  neutral: 'bg-surface-raised/70 text-primary',
  success: 'bg-positive/15 text-positive',
  warning: 'bg-warning/15 text-warning',
  danger: 'bg-alert/15 text-alert',
  // Gold: general information and premium/trust framing, not model output.
  info: 'bg-gold/12 text-gold',
  accent: 'bg-gold/12 text-gold',
  // Teal: AI output only.
  intelligence: 'bg-intelligence/15 text-intelligence',
};

export function Badge({
  tone = 'neutral',
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return <span className={clsx('badge', TONE_CLASS[tone], className)}>{children}</span>;
}

/** Maps backend enum values onto badge tones in one place. */
const STATUS_TONES: Record<string, Tone> = {
  // transactions
  completed: 'success',
  pending: 'warning',
  held: 'warning',
  failed: 'danger',
  blocked: 'danger',
  reversed: 'neutral',
  // accounts / users
  active: 'success',
  frozen: 'danger',
  suspended: 'danger',
  closed: 'neutral',
  dormant: 'neutral',
  // kyc
  verified: 'success',
  submitted: 'info',
  not_started: 'neutral',
  rejected: 'danger',
  // loans
  approved: 'success',
  under_review: 'warning',
  disbursed: 'info',
  defaulted: 'danger',
  draft: 'neutral',
  // alerts
  open: 'warning',
  confirmed_fraud: 'danger',
  disputed: 'warning',
  resolved_legit: 'success',
  resolved_fraud: 'danger',
  dismissed: 'neutral',
  // severity
  low: 'neutral',
  medium: 'warning',
  high: 'danger',
  critical: 'danger',
  // drift
  stable: 'success',
  watch: 'warning',
  drifting: 'danger',
};

export function StatusBadge({ status }: { status: string }) {
  const tone = STATUS_TONES[status] ?? 'neutral';
  const label = status
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  return <Badge tone={tone}>{label}</Badge>;
}

/** Risk band A–E, coloured from safest to riskiest. */
export function RiskBandBadge({ band }: { band: string }) {
  const tone: Tone =
    band === 'A' ? 'success' : band === 'B' ? 'info' : band === 'C' ? 'warning' : 'danger';
  return <Badge tone={tone}>Band {band}</Badge>;
}

/* -------------------------------------------------------------------------- */
/* Notices                                                                     */
/* -------------------------------------------------------------------------- */

const NOTICE_STYLE: Record<
  'info' | 'success' | 'warning' | 'danger' | 'intelligence',
  { wrap: string; icon: ReactNode }
> = {
  // Gold for ordinary guidance; teal only when explaining model output.
  info: {
    wrap: 'border-gold/30 bg-gold/5 text-primary',
    icon: <Info className="h-4 w-4 text-gold" aria-hidden />,
  },
  intelligence: {
    wrap: 'border-intelligence/30 bg-intelligence/5 text-primary',
    icon: <Sparkles className="h-4 w-4 text-intelligence" aria-hidden />,
  },
  success: {
    wrap: 'border-positive/30 bg-positive/5 text-positive',
    icon: <CheckCircle2 className="h-4 w-4 text-positive" aria-hidden />,
  },
  warning: {
    wrap: 'border-warning/30 bg-warning/5 text-warning',
    icon: <ShieldAlert className="h-4 w-4 text-warning" aria-hidden />,
  },
  danger: {
    wrap: 'border-alert/30 bg-alert/5 text-alert',
    icon: <XCircle className="h-4 w-4 text-alert" aria-hidden />,
  },
};

export function Notice({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'success' | 'warning' | 'danger' | 'intelligence';
  title?: string;
  children: ReactNode;
}) {
  const style = NOTICE_STYLE[tone];
  return (
    <div className={clsx('flex gap-3 rounded-lg border p-3.5 text-sm', style.wrap)}>
      <span className="mt-0.5 shrink-0">{style.icon}</span>
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        <div className={clsx(title && 'mt-1', 'opacity-90')}>{children}</div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Form fields                                                                 */
/* -------------------------------------------------------------------------- */

export function Field({
  label,
  htmlFor,
  error,
  hint,
  required,
  children,
}: {
  label: string;
  htmlFor?: string;
  error?: string;
  hint?: string;
  required?: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      <label htmlFor={htmlFor} className="label">
        {label}
        {required && <span className="ml-1 text-alert">*</span>}
      </label>
      {children}
      {/* role=alert so screen readers announce validation failures. */}
      {error ? (
        <p role="alert" className="mt-1.5 text-xs text-alert">
          {error}
        </p>
      ) : hint ? (
        <p className="mt-1.5 text-xs text-muted">{hint}</p>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Stats                                                                       */
/* -------------------------------------------------------------------------- */

export function StatTile({
  label,
  value,
  hint,
  icon,
  tone = 'neutral',
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="card card-hover p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium tracking-wide text-muted uppercase">{label}</p>
        {icon && <span className={clsx('rounded-lg p-1.5', TONE_CLASS[tone])}>{icon}</span>}
      </div>
      {/* Both the value and the hint are numeric, so both take mono +
          tabular-nums. Tiles sit in a grid and would visibly misalign otherwise. */}
      <p className="tnum mt-2.5 text-2xl font-bold text-primary">{value}</p>
      {hint && <p className="tnum mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Pagination                                                                  */
/* -------------------------------------------------------------------------- */

export function Pagination({
  page,
  totalPages,
  total,
  pageSize,
  onPageChange,
}: {
  page: number;
  totalPages: number;
  total: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}) {
  if (total === 0) return null;

  const from = (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);

  return (
    <nav
      aria-label="Pagination"
      className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3"
    >
      <p className="tnum text-xs text-muted">
        Showing <span className="font-medium text-primary">{from}</span>–
        <span className="font-medium text-primary">{to}</span> of{' '}
        <span className="font-medium text-primary">{total.toLocaleString('en-IN')}</span>
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn-secondary px-2.5 py-1.5"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
          aria-label="Previous page"
        >
          <ChevronLeft className="h-4 w-4" aria-hidden />
        </button>
        <span className="tnum text-xs text-muted">
          {page} / {Math.max(totalPages, 1)}
        </span>
        <button
          type="button"
          className="btn-secondary px-2.5 py-1.5"
          disabled={page >= totalPages}
          onClick={() => onPageChange(page + 1)}
          aria-label="Next page"
        >
          <ChevronRight className="h-4 w-4" aria-hidden />
        </button>
      </div>
    </nav>
  );
}

/* -------------------------------------------------------------------------- */
/* Modal                                                                       */
/* -------------------------------------------------------------------------- */

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={onClose}
    >
      <div
        className={clsx(
          'card max-h-[90vh] w-full overflow-y-auto p-0',
          wide ? 'max-w-2xl' : 'max-w-md',
        )}
        // Stop clicks inside the panel from reaching the backdrop handler.
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
          <h3 className="font-semibold text-primary">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="btn-ghost -mr-2 p-1.5"
            aria-label="Close dialog"
          >
            <XCircle className="h-5 w-5" aria-hidden />
          </button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-line px-5 py-3.5">{footer}</div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Misc                                                                        */
/* -------------------------------------------------------------------------- */

/** Horizontal meter for scores and utilisation. */
export function Meter({
  value,
  max = 1,
  tone = 'info',
  label,
}: {
  value: number;
  max?: number;
  tone?: Tone;
  label?: string;
}) {
  const pct = Math.min(Math.max((value / max) * 100, 0), 100);
  const bar: Record<Tone, string> = {
    neutral: 'bg-faint',
    success: 'bg-positive',
    warning: 'bg-warning',
    danger: 'bg-alert',
    info: 'bg-gold',
    accent: 'bg-gold',
    intelligence: 'bg-intelligence',
  };
  return (
    <div>
      {label && <p className="mb-1 text-xs text-muted">{label}</p>}
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-surface-raised"
        role="meter"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label ?? 'Progress'}
      >
        <div className={clsx('h-full rounded-full transition-all', bar[tone])} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/** Small label used to mark simulated behaviour, so nothing looks like a real bank. */
export function SimulatedTag({ children = 'Simulated' }: { children?: ReactNode }) {
  return (
    <span className="badge border border-line-strong bg-surface-raised/80 text-[10px] tracking-wide text-muted uppercase">
      {children}
    </span>
  );
}
