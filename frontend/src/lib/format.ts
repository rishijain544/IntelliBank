/**
 * Formatting helpers.
 *
 * Money arrives from the API as a decimal *string*. It is parsed only here, at
 * the display boundary, so no arithmetic is ever performed on a float that came
 * from a currency value.
 */
import { format, formatDistanceToNowStrict, parseISO } from 'date-fns';

const inr = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const inrCompact = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  notation: 'compact',
  maximumFractionDigits: 1,
});

export function money(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  return Number.isFinite(n) ? inr.format(n) : '—';
}

/** Compact form for dashboard tiles, e.g. ₹6.8L. */
export function moneyCompact(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—';
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  return Number.isFinite(n) ? inrCompact.format(n) : '—';
}

/** Signed amount with an explicit +/- prefix, for ledger rows. */
export function moneySigned(value: string | number): string {
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  if (!Number.isFinite(n)) return '—';
  const sign = n > 0 ? '+' : n < 0 ? '−' : '';
  return `${sign}${inr.format(Math.abs(n))}`;
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-IN', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** For values already expressed as 0-100. */
export function percentRaw(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${value.toFixed(digits)}%`;
}

export function dateShort(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return format(parseISO(iso), 'dd MMM yyyy');
  } catch {
    return '—';
  }
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return format(parseISO(iso), 'dd MMM yyyy, HH:mm');
  } catch {
    return '—';
  }
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return `${formatDistanceToNowStrict(parseISO(iso))} ago`;
  } catch {
    return '—';
  }
}

/** snake_case -> Title Case, used for enum values from the API. */
export function titleCase(value: string | null | undefined): string {
  if (!value) return '—';
  return value
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function maskAccount(accountNumber: string | null | undefined): string {
  if (!accountNumber || accountNumber.length < 4) return '—';
  return `•••• ${accountNumber.slice(-4)}`;
}

export function initials(fullName: string | null | undefined): string {
  if (!fullName) return '?';
  const parts = fullName.trim().split(/\s+/);
  return ((parts[0]?.[0] ?? '') + (parts.at(-1)?.[0] ?? '')).toUpperCase() || '?';
}

/** Latency shown in ms below 1s, seconds above. */
export function latency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—';
  return ms < 1000 ? `${ms.toFixed(ms < 10 ? 2 : 0)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** Convert a 300-900 credit score to a coarse rating. */
export function scoreLabel(score: number | null | undefined): string {
  if (!score) return 'Not scored';
  if (score >= 800) return 'Excellent';
  if (score >= 740) return 'Very good';
  if (score >= 670) return 'Good';
  if (score >= 580) return 'Fair';
  return 'Poor';
}

export const CATEGORY_LABELS: Record<string, string> = {
  groceries: 'Groceries',
  dining: 'Dining',
  transport: 'Transport',
  shopping: 'Shopping',
  utilities: 'Utilities',
  entertainment: 'Entertainment',
  healthcare: 'Healthcare',
  education: 'Education',
  travel: 'Travel',
  rent: 'Rent',
  investment: 'Investment',
  cash: 'Cash',
  transfer: 'Transfer',
  other: 'Other',
};

/**
 * Category colours live in `lib/colors`, which reads the CSS design tokens, so
 * there is exactly one definition of the palette. Re-exported here because call
 * sites already import formatting helpers from this module.
 */
export { categoryColor } from './colors';

export function categoryLabel(category: string): string {
  return CATEGORY_LABELS[category] ?? titleCase(category);
}
