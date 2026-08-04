/**
 * Admin console shell.
 *
 * Visually distinct from the customer app (amber accent, "Admin console" label)
 * so an operator can never mistake which context they are acting in — a
 * privileged action taken by accident is the failure mode worth designing against.
 */
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  BarChart3,
  Landmark,
  LayoutDashboard,
  LogOut,
  Menu,
  ShieldAlert,
  ShieldCheck,
  Users,
  X,
} from 'lucide-react';
import { useState } from 'react';
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom';

import { Badge } from '../components/ui';
import { get } from '../lib/api';
import { initials } from '../lib/format';
import { qk } from '../lib/query';
import { useAuth } from '../store/auth';
import type { AdminStats } from '../types/api';

const NAV = [
  { to: '/admin', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/admin/users', label: 'Users', icon: Users },
  { to: '/admin/fraud', label: 'Fraud queue', icon: ShieldAlert, badge: 'fraud' as const },
  { to: '/admin/loans', label: 'Loan queue', icon: Landmark, badge: 'loans' as const },
  { to: '/admin/analytics', label: 'Model analytics', icon: BarChart3 },
];

export default function AdminLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const { data: stats } = useQuery({
    queryKey: qk.admin.stats,
    queryFn: () => get<AdminStats>('/admin/stats'),
    refetchInterval: 60_000,
  });

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  function badgeCount(kind?: 'fraud' | 'loans') {
    if (kind === 'fraud') return stats?.fraud_alerts_open ?? 0;
    if (kind === 'loans') return stats?.loans_pending ?? 0;
    return 0;
  }

  const sidebar = (
    <nav className="flex h-full flex-col" aria-label="Admin navigation">
      <div className="flex items-center justify-between px-5 py-4">
        <Link to="/admin" className="flex items-center gap-2.5 font-bold text-primary">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-warning/15 text-warning">
            <ShieldCheck className="h-4.5 w-4.5" aria-hidden />
          </span>
          <span className="leading-tight">
            IntelliBank
            <span className="block text-[10px] font-normal text-muted">
              AI-Powered Banking for Everyone.
            </span>
            <span className="block text-[10px] font-semibold tracking-wide text-warning/80 uppercase">
              Admin console
            </span>
          </span>
        </Link>
        <button
          type="button"
          className="btn-ghost p-1.5 lg:hidden"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        >
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>

      <ul className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon, end, badge }) => {
          const count = badgeCount(badge);
          return (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition',
                    isActive
                      ? 'bg-warning/12 text-warning'
                      : 'text-muted hover:bg-surface-raised hover:text-primary',
                  )
                }
              >
                <Icon className="h-4.5 w-4.5 shrink-0" aria-hidden />
                <span className="flex-1">{label}</span>
                {count > 0 && <Badge tone="warning">{count}</Badge>}
              </NavLink>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-line p-3">
        <Link to="/app" className="btn-secondary w-full py-2 text-xs">
          Back to customer view
        </Link>
      </div>
    </nav>
  );

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 shrink-0 border-r border-line bg-surface/50 lg:block">
        <div className="sticky top-0 h-screen">{sidebar}</div>
      </aside>

      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/70" onClick={() => setOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-line bg-surface">
            {sidebar}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-warning/20 bg-ink/85 px-4 py-3 backdrop-blur">
          <button
            type="button"
            className="btn-ghost p-2 lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" aria-hidden />
          </button>

          <p className="hidden text-xs text-warning/80 sm:block">
            Privileged session — actions here are recorded in the audit trail
          </p>

          <div className="ml-auto flex items-center gap-2">
            <div className="flex items-center gap-2.5 rounded-lg border border-line bg-surface px-2.5 py-1.5">
              <span className="grid h-7 w-7 place-items-center rounded-full bg-warning/15 text-xs font-bold text-warning">
                {initials(user?.full_name)}
              </span>
              <div className="hidden leading-tight sm:block">
                <p className="text-xs font-semibold text-primary">{user?.full_name}</p>
                <p className="text-[11px] text-warning/80">Administrator</p>
              </div>
            </div>
            <button type="button" onClick={handleLogout} className="btn-ghost p-2" aria-label="Sign out">
              <LogOut className="h-4.5 w-4.5" aria-hidden />
            </button>
          </div>
        </header>

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
