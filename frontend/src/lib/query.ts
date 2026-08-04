/**
 * TanStack Query client.
 *
 * Defaults chosen for a banking UI: balances and alert queues must not be shown
 * stale for long, but a 401 or 403 should never be retried — the axios layer has
 * already attempted a token refresh by then, so retrying only delays the redirect.
 */
import { QueryClient } from '@tanstack/react-query';
import axios from 'axios';

function isAuthError(error: unknown): boolean {
  if (!axios.isAxiosError(error)) return false;
  const status = error.response?.status;
  return status === 401 || status === 403;
}

/** 4xx responses are client mistakes; retrying them is pointless. */
function shouldRetry(failureCount: number, error: unknown): boolean {
  if (isAuthError(error)) return false;
  if (axios.isAxiosError(error)) {
    const status = error.response?.status ?? 0;
    if (status >= 400 && status < 500) return false;
  }
  return failureCount < 2;
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      retry: shouldRetry,
      refetchOnWindowFocus: true,
      // Balances change server-side (transfers from other users), so a remount
      // should re-validate rather than trust the cache.
      refetchOnMount: true,
    },
    mutations: {
      retry: false,
    },
  },
});

/** Query keys in one place, so invalidation after a mutation cannot go stale. */
export const qk = {
  me: ['me'] as const,
  dashboard: ['dashboard'] as const,
  accounts: ['accounts'] as const,
  account: (id: number) => ['accounts', id] as const,
  transactions: (params: Record<string, unknown>) => ['transactions', params] as const,
  transaction: (id: number) => ['transactions', id] as const,
  beneficiaries: ['beneficiaries'] as const,
  cards: ['cards'] as const,
  loans: (page: number) => ['loans', page] as const,
  loan: (id: number) => ['loans', id] as const,
  insights: (days: number) => ['insights', days] as const,
  fraudAlerts: (params: Record<string, unknown>) => ['fraud', 'alerts', params] as const,
  fraudSummary: ['fraud', 'summary'] as const,
  notifications: (params: Record<string, unknown>) => ['notifications', params] as const,
  devices: ['profile', 'devices'] as const,
  mlStatus: ['ml', 'status'] as const,

  admin: {
    stats: ['admin', 'stats'] as const,
    users: (params: Record<string, unknown>) => ['admin', 'users', params] as const,
    user: (id: number) => ['admin', 'users', id] as const,
    fraudQueue: (params: Record<string, unknown>) => ['admin', 'fraud', params] as const,
    loanQueue: (params: Record<string, unknown>) => ['admin', 'loans', params] as const,
    models: (days: number) => ['admin', 'models', days] as const,
    analytics: (days: number) => ['admin', 'analytics', days] as const,
    audit: (params: Record<string, unknown>) => ['admin', 'audit', params] as const,
  },
} as const;

/**
 * Anything a money movement can touch.
 *
 * A transfer changes balances, history, the dashboard, alert queues and
 * notifications at once. Listing them here keeps call sites from forgetting one
 * and leaving a stale balance on screen.
 */
export async function invalidateAfterMoneyMovement(): Promise<void> {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: qk.dashboard }),
    queryClient.invalidateQueries({ queryKey: qk.accounts }),
    queryClient.invalidateQueries({ queryKey: ['transactions'] }),
    queryClient.invalidateQueries({ queryKey: ['fraud'] }),
    queryClient.invalidateQueries({ queryKey: ['notifications'] }),
    queryClient.invalidateQueries({ queryKey: ['insights'] }),
  ]);
}
