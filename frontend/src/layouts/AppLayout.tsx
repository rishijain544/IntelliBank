/**
 * Signed-in customer shell: sidebar, notification bell and account menu.
 */
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  ArrowLeftRight,
  Bell,
  CreditCard,
  Landmark,
  LayoutDashboard,
  LineChart,
  ListOrdered,
  LogOut,
  Menu,
  Settings as SettingsIcon,
  ShieldAlert,
  ShieldCheck,
  Wallet,
  X,
} from 'lucide-react';
import { useState } from 'react';
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom';

import { Badge, SimulatedTag } from '../components/ui';
import { get } from '../lib/api';
import { initials } from '../lib/format';
import { qk } from '../lib/query';
import { useAuth } from '../store/auth';
import type { DashboardData, Notification, Page } from '../types/api';

const NAV = [
  { to: '/app', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/app/accounts', label: 'Accounts', icon: Wallet },
  { to: '/app/transactions', label: 'Transactions', icon: ListOrdered },
  { to: '/app/transfer', label: 'Transfer', icon: ArrowLeftRight },
  { to: '/app/cards', label: 'Cards', icon: CreditCard },
  { to: '/app/loans', label: 'Loans', icon: Landmark },
  { to: '/app/insights', label: 'Insights', icon: LineChart },
  { to: '/app/fraud-center', label: 'Security', icon: ShieldAlert },
  { to: '/app/settings', label: 'Settings', icon: SettingsIcon },
];

function NotificationBell() {
  const [open, setOpen] = useState(false);

  const { data } = useQuery({
    queryKey: qk.notifications({ page: 1, unread: true }),
    queryFn: () =>
      get<Page<Notification>>('/notifications', { params: { page: 1, page_size: 8, unread_only: true } }),
    // Alerts should surface without a manual refresh.
    refetchInterval: 60_000,
  });

  const unread = data?.total ?? 0;

  return (
    <div className="relative">
      <button
        type="button"
        className="btn-ghost relative p-2"
        onClick={() => setOpen((v) => !v)}
        aria-label={`Notifications${unread ? `, ${unread} unread` : ''}`}
        aria-expanded={open}
      >
        <Bell className="h-5 w-5" aria-hidden />
        {unread > 0 && (
          <span className="absolute top-1 right-1 grid h-4 min-w-4 place-items-center rounded-full bg-alert px-1 text-[10px] font-bold text-primary">
            {unread > 9 ? '9+' : unread}
          </span>
        )}
      </button>

      {open && (
        <>
          {/* Click-away target; keeps the panel open while interacting inside it. */}
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} aria-hidden />
          <div className="card absolute right-0 z-40 mt-2 w-80 p-0 shadow-2xl">
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <p className="text-sm font-semibold text-primary">Notifications</p>
              {unread > 0 && <Badge tone="danger">{unread} new</Badge>}
            </div>
            <div className="max-h-80 divide-y divide-line overflow-y-auto">
              {data?.items.length ? (
                data.items.map((n) => (
                  <Link
                    key={n.id}
                    to={n.action_url ?? '/app'}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-3 transition hover:bg-surface-raised/60"
                  >
                    <p className="text-sm font-medium text-primary">{n.title}</p>
                    <p className="mt-0.5 line-clamp-2 text-xs text-muted">{n.body}</p>
                  </Link>
                ))
              ) : (
                <p className="px-4 py-8 text-center text-sm text-muted">You are all caught up.</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function AppLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Drives the badge on the Security nav item.
  const { data: dashboard } = useQuery({
    queryKey: qk.dashboard,
    queryFn: () => get<DashboardData>('/dashboard'),
    staleTime: 60_000,
  });

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  const needsKyc = user?.kyc_status !== 'verified';

  const sidebar = (
    <nav className="flex h-full flex-col" aria-label="Account navigation">
      <div className="flex items-center justify-between px-5 py-4">
        <Link to="/app" className="flex items-center gap-2.5 font-bold text-primary">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gold/15 text-gold">
            <ShieldCheck className="h-4.5 w-4.5" aria-hidden />
          </span>
          <span className="leading-tight">
            IntelliBank
            <span className="block text-[10px] font-normal text-muted">
              AI-Powered Banking for Everyone.
            </span>
          </span>
        </Link>
        <button
          type="button"
          className="btn-ghost p-1.5 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
        >
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>

      <ul className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon, end }) => {
          const badge = to === '/app/fraud-center' ? dashboard?.open_fraud_alerts ?? 0 : 0;
          return (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition',
                    isActive
                      ? 'bg-gold/12 text-gold-bright'
                      : 'text-muted hover:bg-surface-raised hover:text-primary',
                  )
                }
              >
                <Icon className="h-4.5 w-4.5 shrink-0" aria-hidden />
                <span className="flex-1">{label}</span>
                {badge > 0 && <Badge tone="danger">{badge}</Badge>}
              </NavLink>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-line p-3">
        <SimulatedTag>Simulated environment</SimulatedTag>
      </div>
    </nav>
  );

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-line bg-surface/50 lg:block">
        <div className="sticky top-0 h-screen">{sidebar}</div>
      </aside>

      {/* Mobile drawer */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/70" onClick={() => setSidebarOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-line bg-surface">
            {sidebar}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-line bg-ink/85 px-4 py-3 backdrop-blur">
          <button
            type="button"
            className="btn-ghost p-2 lg:hidden"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" aria-hidden />
          </button>

          <div className="ml-auto flex items-center gap-2">
            <NotificationBell />

            <div className="flex items-center gap-2.5 rounded-lg border border-line bg-surface px-2.5 py-1.5">
              <span className="grid h-7 w-7 place-items-center rounded-full bg-gold/15 text-xs font-bold text-gold-bright">
                {initials(user?.full_name)}
              </span>
              <div className="hidden leading-tight sm:block">
                <p className="max-w-[10rem] truncate text-xs font-semibold text-primary">
                  {user?.full_name}
                </p>
                <p className="text-[11px] text-muted">{user?.email}</p>
              </div>
            </div>

            {user?.role === 'admin' && (
              <Link to="/admin" className="btn-secondary hidden px-3 py-2 text-xs sm:inline-flex">
                Admin
              </Link>
            )}

            <button type="button" onClick={handleLogout} className="btn-ghost p-2" aria-label="Sign out">
              <LogOut className="h-4.5 w-4.5" aria-hidden />
            </button>
          </div>
        </header>

        {needsKyc && (
          <div className="rounded-lg border border-warning/30 bg-warning/5 px-4 py-3 text-sm text-warning">
            Complete KYC verification to unlock external transfers, cards and formal loan applications.{' '}
            <Link to="/app/settings" className="font-semibold underline">
              Finish verification
            </Link>
          </div>
        )}

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
