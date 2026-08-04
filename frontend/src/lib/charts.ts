/**
 * Recharts helpers.
 *
 * Recharts 3 types formatter arguments as `ValueType | undefined` (a value can be
 * absent for a hidden series), so passing `(v: number) => string` fails to
 * typecheck. Rather than casting at every call site — which would hide the real
 * possibility of an undefined value — these helpers coerce defensively once and
 * are reused everywhere.
 *
 * Colours come from `lib/colors`, which reads the CSS design tokens at runtime.
 * Charts therefore stay in sync with the rest of the app instead of carrying
 * their own hardcoded hex values.
 */
import type { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';

import { palette } from './colors';
import { money, moneyCompact, num } from './format';

/**
 * Tooltip styling.
 *
 * A getter rather than a constant: the tokens are read from the document, which
 * is not available at module-evaluation time.
 */
export const tooltipStyle = {
  get backgroundColor() {
    return palette.surfaceRaised;
  },
  get border() {
    return `1px solid ${palette.line}`;
  },
  borderRadius: '0.5rem',
  fontSize: '0.75rem',
  get color() {
    return palette.textPrimary;
  },
  fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

export const axisTick = {
  fontSize: 10,
  get fill() {
    return palette.textMuted;
  },
  fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

/**
 * Grid stroke.
 *
 * A function, not a constant: module evaluation happens before the stylesheet is
 * applied, so reading the token eagerly would capture the fallback forever.
 */
export function gridStroke(): string {
  return palette.line;
}

export const barCursor = {
  get fill() {
    return `color-mix(in oklab, ${palette.textMuted} 10%, transparent)`;
  },
} as const;

function toNumber(value: ValueType | undefined): number {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

/** Currency tooltip formatter. */
export function moneyFormatter(value: ValueType | undefined): string {
  return money(toNumber(value));
}

/** Currency tooltip formatter that also labels the series. */
export function labelledMoneyFormatter(label: string) {
  return (value: ValueType | undefined): [string, string] => [money(toNumber(value)), label];
}

/**
 * Formats currency, but leaves non-currency series (counts) as plain numbers.
 * Used on dual-axis charts where one line is money and the other is a count.
 */
export function mixedFormatter(moneySeries: string) {
  return (value: ValueType | undefined, name: NameType | undefined): string =>
    String(name) === moneySeries ? money(toNumber(value)) : num(toNumber(value));
}

/** Compact axis labels without the currency symbol, to keep axes narrow. */
export function compactAxisFormatter(value: number | string): string {
  return moneyCompact(toNumber(value as ValueType)).replace('₹', '');
}

/** Tooltip label formatter that safely stringifies a ReactNode label. */
export function labelFormatter(transform: (label: string) => string) {
  return (label: unknown): string => transform(label == null ? '' : String(label));
}
