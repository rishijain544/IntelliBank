/**
 * Axios client with transparent access-token refresh.
 *
 * The backend issues a short-lived access token plus a rotating refresh token,
 * and it revokes the whole session family if a refresh token is ever replayed.
 * That makes concurrency a correctness concern, not a nicety: if three requests
 * 401 at once and each fires its own refresh, two of them replay a
 * now-consumed token and the user gets logged out of every device.
 *
 * So refreshes are funnelled through a single in-flight promise, and any request
 * that 401s while a refresh is running waits for that one result.
 */
import axios, {
  AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
  type InternalAxiosRequestConfig,
} from 'axios';
import type { ApiErrorBody, TokenResponse } from '../types/api';

/**
 * Base URL.
 *
 * Defaults to a relative path so requests go through the Vite dev proxy (see
 * vite.config.ts) and the app stays same-origin — which keeps dev behaviour
 * identical to a production deployment behind a reverse proxy, and means CORS
 * is not load-bearing. Set VITE_API_URL to point at a remote API instead.
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? '/api/v1';

const ACCESS_KEY = 'intellibank.access_token';
const REFRESH_KEY = 'intellibank.refresh_token';

export const tokenStore = {
  access: (): string | null => localStorage.getItem(ACCESS_KEY),
  refresh: (): string | null => localStorage.getItem(REFRESH_KEY),
  set(access: string, refresh: string) {
    localStorage.setItem(ACCESS_KEY, access);
    localStorage.setItem(REFRESH_KEY, refresh);
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
  },
};

export const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
});

/** Raised after a failed refresh so callers can distinguish it from a 403. */
export class SessionExpiredError extends Error {
  constructor() {
    super('Your session has expired. Please sign in again.');
    this.name = 'SessionExpiredError';
  }
}

type Listener = () => void;
const logoutListeners = new Set<Listener>();

/** The auth store subscribes here so a hard logout clears React state too. */
export function onForcedLogout(fn: Listener): () => void {
  logoutListeners.add(fn);
  return () => logoutListeners.delete(fn);
}

function forceLogout() {
  tokenStore.clear();
  logoutListeners.forEach((fn) => fn());
}

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = tokenStore.access();
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** Shared across all callers so a token is only ever consumed once. */
let refreshInFlight: Promise<string> | null = null;

async function refreshAccessToken(): Promise<string> {
  const refreshToken = tokenStore.refresh();
  if (!refreshToken) throw new SessionExpiredError();

  // A bare axios call, not `api`: going through the instance would re-enter
  // this interceptor and could recurse if the refresh itself 401s.
  const { data } = await axios.post<TokenResponse>(
    `${BASE_URL}/auth/refresh`,
    { refresh_token: refreshToken },
    { headers: { 'Content-Type': 'application/json' }, timeout: 30_000 },
  );
  tokenStore.set(data.access_token, data.refresh_token);
  return data.access_token;
}

interface RetriableConfig extends InternalAxiosRequestConfig {
  _retried?: boolean;
}

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ApiErrorBody>) => {
    const config = error.config as RetriableConfig | undefined;
    const status = error.response?.status;

    // Only a 401 is refreshable. A 403 means authenticated-but-not-allowed,
    // which refreshing cannot fix.
    if (status !== 401 || !config || config._retried) {
      return Promise.reject(error);
    }

    // Never try to refresh the auth endpoints themselves; a 401 from /login
    // means bad credentials, and a 401 from /refresh means the session is gone.
    const url = config.url ?? '';
    if (url.includes('/auth/login') || url.includes('/auth/refresh')) {
      if (url.includes('/auth/refresh')) forceLogout();
      return Promise.reject(error);
    }

    config._retried = true;

    try {
      refreshInFlight ??= refreshAccessToken().finally(() => {
        refreshInFlight = null;
      });
      const newToken = await refreshInFlight;
      config.headers.Authorization = `Bearer ${newToken}`;
      return api.request(config);
    } catch {
      forceLogout();
      return Promise.reject(new SessionExpiredError());
    }
  },
);

/**
 * Turn any thrown value into a message safe to render.
 *
 * FastAPI returns `detail` as a string for HTTPException but the custom
 * validation handler adds a `fields` map, so both shapes are handled.
 */
export function errorMessage(error: unknown): string {
  if (error instanceof SessionExpiredError) return error.message;

  if (axios.isAxiosError<ApiErrorBody>(error)) {
    const body = error.response?.data;
    if (body?.fields) {
      const first = Object.entries(body.fields)[0];
      if (first) return `${first[0]}: ${first[1]}`;
    }
    if (typeof body?.detail === 'string') return body.detail;
    if (error.code === 'ECONNABORTED') return 'The request timed out. Please try again.';
    if (!error.response) return 'Cannot reach the server. Is the backend running?';
    return error.message;
  }

  return error instanceof Error ? error.message : 'Something went wrong.';
}

/** Field-level validation errors, for inline form display. */
export function fieldErrors(error: unknown): Record<string, string> {
  if (axios.isAxiosError<ApiErrorBody>(error)) {
    return error.response?.data?.fields ?? {};
  }
  return {};
}

export async function get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const { data } = await api.get<T>(url, config);
  return data;
}

export async function post<T>(
  url: string,
  body?: unknown,
  config?: AxiosRequestConfig,
): Promise<T> {
  const { data } = await api.post<T>(url, body, config);
  return data;
}

export async function patch<T>(
  url: string,
  body?: unknown,
  config?: AxiosRequestConfig,
): Promise<T> {
  const { data } = await api.patch<T>(url, body, config);
  return data;
}

export async function del<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const { data } = await api.delete<T>(url, config);
  return data;
}

/**
 * Download an authenticated file.
 *
 * A plain anchor href cannot carry the bearer token, so the bytes are fetched
 * as a blob and handed to a temporary object URL.
 */
export async function downloadFile(
  url: string,
  fallbackName: string,
  config?: AxiosRequestConfig,
): Promise<void> {
  const response = await api.get(url, { ...config, responseType: 'blob' });

  const disposition = response.headers['content-disposition'] as string | undefined;
  const matched = disposition?.match(/filename="?([^";]+)"?/);
  const filename = matched?.[1] ?? fallbackName;

  const blobUrl = URL.createObjectURL(response.data as Blob);
  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(blobUrl);
}
