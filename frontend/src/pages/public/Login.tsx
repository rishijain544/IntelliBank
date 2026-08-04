import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, KeyRound, Loader2, ShieldCheck } from 'lucide-react';

import { Card, Field, Notice } from '../../components/ui';
import { errorMessage } from '../../lib/api';
import { TwoFactorRequiredError, useAuth } from '../../store/auth';

/** Demo accounts, surfaced so a reviewer can get in without reading the README. */
const DEMO = [
  { label: 'Customer (has fraud alerts)', email: 'priya@intellibank.dev', password: 'Demo@Pass123' },
  { label: 'Administrator', email: 'admin@intellibank.dev', password: 'Admin@Pass123' },
];

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const login = useAuth((s) => s.login);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [needsTotp, setNeedsTotp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const returnTo = (location.state as { from?: string } | null)?.from;

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const user = await login({ email, password, totpCode: totpCode || undefined });
      // Admins land in the console; everyone else goes where they were headed.
      const destination = returnTo ?? (user.role === 'admin' ? '/admin' : '/app');
      navigate(destination, { replace: true });
    } catch (err) {
      if (err instanceof TwoFactorRequiredError) {
        // Credentials were valid — reveal the code field rather than showing a failure.
        setNeedsTotp(true);
        setError(null);
      } else {
        setError(errorMessage(err));
      }
    } finally {
      setBusy(false);
    }
  }

  function useDemo(account: (typeof DEMO)[number]) {
    setEmail(account.email);
    setPassword(account.password);
    setError(null);
  }

  return (
    <div className="mx-auto flex max-w-md flex-col justify-center px-4 py-14">
      <div className="mb-7 text-center">
        <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-xl bg-gold/15 text-gold">
          <ShieldCheck className="h-6 w-6" aria-hidden />
        </span>
        <h1 className="text-2xl font-bold text-primary">Sign in to IntelliBank</h1>
        <p className="mt-1.5 text-sm text-muted">
          New here?{' '}
          <Link to="/register" className="font-medium text-gold hover:text-gold-bright">
            Open an account
          </Link>
        </p>
      </div>

      <Card>
        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          {error && <Notice tone="danger">{error}</Notice>}

          <Field label="Email address" htmlFor="email" required>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              className="input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </Field>

          <Field label="Password" htmlFor="password" required>
            <div className="relative">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="current-password"
                required
                className="input pr-11"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute top-1/2 right-2 -translate-y-1/2 rounded p-1.5 text-muted hover:text-primary"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff className="h-4 w-4" aria-hidden /> : <Eye className="h-4 w-4" aria-hidden />}
              </button>
            </div>
          </Field>

          {needsTotp && (
            <>
              <Notice tone="info" title="Two-factor authentication">
                Enter the 6-digit code from your authenticator app.
              </Notice>
              <Field label="Authentication code" htmlFor="totp" required>
                <input
                  id="totp"
                  type="text"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  maxLength={6}
                  required
                  autoFocus
                  className="input tnum text-center text-lg tracking-[0.4em]"
                  value={totpCode}
                  onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
                  placeholder="000000"
                />
              </Field>
            </>
          )}

          <button type="submit" className="btn-primary w-full py-2.5" disabled={busy}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <KeyRound className="h-4 w-4" aria-hidden />}
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </Card>

      <div className="mt-6">
        <p className="mb-2.5 text-center text-xs font-medium tracking-wide text-muted uppercase">
          Demo accounts
        </p>
        <div className="space-y-2">
          {DEMO.map((account) => (
            <button
              key={account.email}
              type="button"
              onClick={() => useDemo(account)}
              className="w-full rounded-lg border border-line bg-surface/60 px-3.5 py-2.5 text-left transition hover:border-line-strong hover:bg-surface-raised"
            >
              <p className="text-sm font-medium text-primary">{account.label}</p>
              <p className="font-mono text-xs text-muted">{account.email}</p>
            </button>
          ))}
        </div>
        <p className="mt-3 text-center text-xs text-faint">
          Simulated accounts in a demo environment. Never enter real credentials.
        </p>
      </div>
    </div>
  );
}
