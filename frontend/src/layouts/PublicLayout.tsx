import { Link, NavLink, Outlet } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';
import { clsx } from 'clsx';

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/features', label: 'Features' },
  { to: '/contact', label: 'Support' },
];

/** Shown on every public page so the project is never mistaken for a real bank. */
export function DisclaimerBar() {
  return (
    <div className="border-b border-warning/20 bg-warning/10 px-4 py-2 text-center text-xs text-warning/90">
      <strong className="font-semibold">Educational project.</strong> IntelliBank is a simulated
      banking platform. It holds no real money and is not a licensed financial institution.
    </div>
  );
}

export default function PublicLayout() {
  return (
    <div className="flex min-h-screen flex-col">
      <DisclaimerBar />

      <header className="sticky top-0 z-40 border-b border-line bg-ink/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5">
          <Link to="/" className="flex items-center gap-2.5 font-bold text-primary">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-gold/15 text-gold">
              <ShieldCheck className="h-4.5 w-4.5" aria-hidden />
            </span>
            IntelliBank
          </Link>

          <nav className="hidden items-center gap-1 sm:flex" aria-label="Primary">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    'rounded-lg px-3 py-2 text-sm font-medium transition',
                    isActive ? 'bg-surface-raised text-primary' : 'text-muted hover:text-primary',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Link to="/login" className="btn-ghost px-3 py-2 text-sm">
              Sign in
            </Link>
            <Link to="/register" className="btn-primary px-3.5 py-2 text-sm">
              Open account
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-line bg-ink">
        <div className="mx-auto max-w-6xl px-4 py-9">
          <div className="flex flex-wrap items-start justify-between gap-8">
            <div className="max-w-sm">
              <div className="flex items-center gap-2.5 font-bold text-primary">
                <span className="grid h-7 w-7 place-items-center rounded-lg bg-gold/15 text-gold">
                  <ShieldCheck className="h-4 w-4" aria-hidden />
                </span>
                IntelliBank
              </div>
              <p className="mt-3 text-sm text-muted">
                A portfolio project demonstrating production ML wired into real banking logic:
                fraud detection, credit scoring and spending anomaly detection.
              </p>
            </div>

            <div className="flex gap-12 text-sm">
              <div>
                <p className="mb-2.5 font-semibold text-primary">Product</p>
                <ul className="space-y-2 text-muted">
                  <li>
                    <Link to="/features" className="hover:text-primary">
                      Features
                    </Link>
                  </li>
                  <li>
                    <Link to="/register" className="hover:text-primary">
                      Open an account
                    </Link>
                  </li>
                  <li>
                    <Link to="/contact" className="hover:text-primary">
                      Support
                    </Link>
                  </li>
                </ul>
              </div>
              <div>
                <p className="mb-2.5 font-semibold text-primary">Technical</p>
                <ul className="space-y-2 text-muted">
                  <li>
                    <a href="/docs" className="hover:text-primary">
                      API documentation
                    </a>
                  </li>
                  <li>XGBoost · Isolation Forest</li>
                  <li>FastAPI · React · TypeScript</li>
                </ul>
              </div>
            </div>
          </div>

          <p className="mt-8 border-t border-line pt-6 text-xs text-faint">
            Simulated banking platform for educational and portfolio use only. No real funds, no
            payment network connectivity, and not a licensed financial institution. Do not enter
            real personal or financial information.
          </p>
        </div>
      </footer>
    </div>
  );
}
