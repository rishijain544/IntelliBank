/**
 * Single source of truth for colours used by JavaScript.
 *
 * Recharts needs concrete colour strings and cannot consume Tailwind classes, so
 * without this module chart colours drift out of sync with the CSS tokens the
 * rest of the app uses. Values are read from the live custom properties at
 * runtime, which means the design system stays authoritative: change a token in
 * index.css and the charts follow.
 *
 * Each lookup is memoised because `getComputedStyle` forces a style recalculation
 * and charts re-render frequently.
 */

const cache = new Map<string, string>();

/**
 * Resolve a CSS custom property to its computed value.
 *
 * The fallback matters for two cases: server-side rendering (no `document`) and
 * the first paint before the stylesheet has applied.
 */
function token(name: string, fallback: string): string {
  const cached = cache.get(name);
  if (cached) return cached;

  if (typeof document === 'undefined') return fallback;

  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  const resolved = value || fallback;
  cache.set(name, resolved);
  return resolved;
}

/** Clear memoised values. Only needed if tokens change at runtime. */
export function resetColorCache(): void {
  cache.clear();
}

/* -------------------------------------------------------------------------- */
/* Semantic palette                                                           */
/* -------------------------------------------------------------------------- */

export const palette = {
  get ink() {
    return token('--color-ink', '#0B0E14');
  },
  get surface() {
    return token('--color-surface', '#141926');
  },
  get surfaceRaised() {
    return token('--color-surface-raised', '#1B2233');
  },
  get line() {
    return token('--color-line', '#232B3E');
  },
  get textPrimary() {
    return token('--color-text-primary', '#F2F0E8');
  },
  get textMuted() {
    return token('--color-text-muted', '#8B92A5');
  },
  get textFaint() {
    return token('--color-text-faint', '#5D6478');
  },
  get gold() {
    return token('--color-gold', '#C9A15C');
  },
  /** AI output only: fraud scores, anomaly detection, credit reasoning. */
  get intelligence() {
    return token('--color-intelligence', '#4CD9C0');
  },
  get alert() {
    return token('--color-alert', '#E8604C');
  },
  get positive() {
    return token('--color-positive', '#7FB09A');
  },
  get warning() {
    return token('--color-warning', '#D99A4C');
  },
} as const;

/* -------------------------------------------------------------------------- */
/* Spending categories                                                        */
/* -------------------------------------------------------------------------- */

const CATEGORY_FALLBACK: Record<string, string> = {
  groceries: '#7FB09A',
  dining: '#D99A4C',
  transport: '#6B8FC9',
  shopping: '#A98BC9',
  utilities: '#4CD9C0',
  entertainment: '#C97FA8',
  healthcare: '#E8604C',
  education: '#8B7FC9',
  travel: '#5FB8C9',
  rent: '#C9A15C',
  investment: '#6FA88A',
  cash: '#8B92A5',
  transfer: '#7AA5D2',
  other: '#5D6478',
};

/** Stable colour for a spending category, so it matches across every chart. */
export function categoryColor(category: string): string {
  const key = category.toLowerCase();
  return token(`--color-cat-${key}`, CATEGORY_FALLBACK[key] ?? CATEGORY_FALLBACK.other);
}

/* -------------------------------------------------------------------------- */
/* Credit risk bands                                                          */
/* -------------------------------------------------------------------------- */

const BAND_FALLBACK: Record<string, string> = {
  A: '#7FB09A',
  B: '#A8BD7F',
  C: '#C9A15C',
  D: '#D9834C',
  E: '#E8604C',
};

export function bandColor(band: string): string {
  const key = band.toUpperCase();
  return token(`--color-band-${key.toLowerCase()}`, BAND_FALLBACK[key] ?? palette.textMuted);
}

/** Alert severity, mapped onto the warning/alert tokens. */
export function severityColor(severity: string): string {
  switch (severity.toLowerCase()) {
    case 'critical':
      return palette.alert;
    case 'high':
      return palette.alert;
    case 'medium':
      return palette.warning;
    default:
      return palette.textMuted;
  }
}
