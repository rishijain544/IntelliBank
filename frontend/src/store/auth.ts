/**
 * Authentication state.
 *
 * Zustand rather than Context because the token is read by non-React code (the
 * axios interceptor) and a Context value cannot be consumed there without
 * plumbing a ref through the tree.
 *
 * The user object is persisted alongside the tokens so a page refresh renders
 * immediately instead of flashing a spinner while `/auth/me` resolves. That
 * cached copy is treated as a hint only: `bootstrap()` re-fetches it, and the
 * server re-checks role and status on every request, so a stale cached role
 * cannot grant access to anything.
 */
import { create } from 'zustand';

import { get, onForcedLogout, post, tokenStore } from '../lib/api';
import type { MessageResponse, TokenResponse, User } from '../types/api';

const USER_KEY = 'intellibank.user';

function cachedUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  } catch {
    return null;
  }
}

/** Thrown by `login` when the account has 2FA enabled and no code was supplied. */
export class TwoFactorRequiredError extends Error {
  constructor() {
    super('Enter the 6-digit code from your authenticator app.');
    this.name = 'TwoFactorRequiredError';
  }
}

interface LoginArgs {
  email: string;
  password: string;
  totpCode?: string;
}

interface AuthState {
  user: User | null;
  /** True until the initial session check finishes, so guards can wait. */
  initialising: boolean;
  login: (args: LoginArgs) => Promise<User>;
  logout: () => Promise<void>;
  bootstrap: () => Promise<void>;
  setUser: (user: User) => void;
  clear: () => void;
}

export const useAuth = create<AuthState>((set, storeGet) => ({
  user: cachedUser(),
  // Only block on bootstrap when there is a token worth validating.
  initialising: Boolean(tokenStore.access()),

  async login({ email, password, totpCode }) {
    try {
      const data = await post<TokenResponse>('/auth/login', {
        email,
        password,
        totp_code: totpCode ?? null,
      });
      tokenStore.set(data.access_token, data.refresh_token);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      set({ user: data.user, initialising: false });
      return data.user;
    } catch (error) {
      // 428 means the credentials were correct but a second factor is needed.
      // Surfaced as a distinct error so the form can reveal the code field
      // rather than showing a generic failure.
      const status = (error as { response?: { status?: number } })?.response?.status;
      if (status === 428) throw new TwoFactorRequiredError();
      throw error;
    }
  },

  async logout() {
    const refresh = tokenStore.refresh();
    try {
      // Best effort: the server-side revoke is what actually kills the session,
      // but a network failure must not trap the user in a signed-in shell.
      if (refresh) await post<MessageResponse>('/auth/logout', { refresh_token: refresh });
    } catch {
      /* ignored on purpose */
    } finally {
      storeGet().clear();
    }
  },

  async bootstrap() {
    if (!tokenStore.access()) {
      set({ user: null, initialising: false });
      return;
    }
    try {
      const user = await get<User>('/auth/me');
      localStorage.setItem(USER_KEY, JSON.stringify(user));
      set({ user, initialising: false });
    } catch {
      // The axios interceptor already attempted a refresh; reaching here means
      // the session is genuinely unusable.
      storeGet().clear();
    }
  },

  setUser(user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    set({ user });
  },

  clear() {
    tokenStore.clear();
    localStorage.removeItem(USER_KEY);
    set({ user: null, initialising: false });
  },
}));

// A failed token refresh happens inside the axios layer, which has no way to
// reach React state. This bridges that gap.
onForcedLogout(() => useAuth.getState().clear());

export const isAdmin = (user: User | null): boolean => user?.role === 'admin';

/** KYC-verified customers are the only ones allowed to move money. */
/**
 * Can the user perform basic, non-compliance-sensitive banking?
 *
 * True for any account in good standing, including a brand-new signup that has
 * not completed KYC (status `pending`). Covers opening an account, adding
 * simulated funds, internal transfers and dry-run loan quotes.
 *
 * These were previously KYC-gated, which indirectly disabled all three ML
 * features: with no account and no funds, there are no transactions for the
 * fraud model to score and no history for insights to analyse.
 */
export const canBank = (user: User | null): boolean =>
  user != null && user.status !== 'frozen' && user.status !== 'suspended';

/**
 * Can the user perform actions a real bank gates on verified identity?
 *
 * External (interbank) transfers, card issuance and binding loan applications.
 * The backend enforces this independently; this only controls the UI.
 */
export const isKycVerified = (user: User | null): boolean =>
  user?.status === 'active' && user?.kyc_status === 'verified';
