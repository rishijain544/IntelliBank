import { useQuery } from '@tanstack/react-query';
import {
  ArrowDownLeft,
  ArrowUpRight,
  Gauge,
  Landmark,
  PiggyBank,
  Plus,
  ShieldAlert,
  Wallet,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import AssistantWidget from '../../components/AssistantWidget';
import {
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  SectionHeading,
  StatTile,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get } from '../../lib/api';
import {
  axisTick,
  compactAxisFormatter,
  labelFormatter,
  labelledMoneyFormatter,
  moneyFormatter,
  tooltipStyle,
} from '../../lib/charts';
import { palette } from '../../lib/colors';
import {
  categoryColor,
  categoryLabel,
  dateShort,
  maskAccount,
  money,
  moneyCompact,
  moneySigned,
  scoreLabel,
  titleCase,
} from '../../lib/format';
import { qk } from '../../lib/query';
import { useAuth } from '../../store/auth';
import type { DashboardData } from '../../types/api';

export default function Dashboard() {
  const user = useAuth((s) => s.user);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.dashboard,
    queryFn: () => get<DashboardData>('/dashboard'),
  });

  if (isLoading) return <LoadingBlock rows={6} label="Loading your dashboard" />;
  if (error) return <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />;
  if (!data) return null;

  const firstName = user?.full_name?.split(' ')[0] ?? 'there';

  // Recharts needs numbers; the API sends decimal strings.
  const spendSeries = data.daily_spend.map((d) => ({
    date: d.date.slice(5), // MM-DD keeps the axis readable
    amount: Number.parseFloat(d.amount),
  }));

  const categorySeries = data.category_breakdown
    .filter((c) => Number.parseFloat(c.total) > 0)
    .slice(0, 6)
    .map((c) => ({
      name: categoryLabel(c.category),
      value: Number.parseFloat(c.total),
      category: c.category,
    }));

  return (
    <div>
      <PageHeader
        title={`Welcome back, ${firstName}`}
        subtitle="Your accounts, recent activity and anything that needs attention."
        action={
          <Link to="/app/transfer" className="btn-primary px-4 py-2.5">
            <ArrowUpRight className="h-4 w-4" aria-hidden />
            Send money
          </Link>
        }
      />

      {/* ------------------------------- tiles ------------------------------- */}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Total balance"
          value={money(data.total_balance)}
          hint={`Across ${data.accounts.length} account${data.accounts.length === 1 ? '' : 's'}`}
          icon={<Wallet className="h-4 w-4" aria-hidden />}
          tone="info"
        />
        <StatTile
          label="Spent (30 days)"
          value={moneyCompact(data.spend_last_30d)}
          icon={<ArrowUpRight className="h-4 w-4" aria-hidden />}
          tone="warning"
        />
        <StatTile
          label="Received (30 days)"
          value={moneyCompact(data.received_last_30d)}
          icon={<ArrowDownLeft className="h-4 w-4" aria-hidden />}
          tone="success"
        />
        <StatTile
          label="Credit score"
          value={data.latest_credit_score ?? '—'}
          hint={data.latest_credit_score ? scoreLabel(data.latest_credit_score) : 'Apply for a loan to get scored'}
          icon={<Gauge className="h-4 w-4" aria-hidden />}
          tone="accent"
        />
      </div>

      {/* Security banner only appears when there is something to act on. */}
      {data.open_fraud_alerts > 0 && (
        <Link
          to="/app/fraud-center"
          className="mb-6 flex items-center gap-3 rounded-xl border border-alert/30 bg-alert/5 p-4 transition hover:bg-alert/10"
        >
          <ShieldAlert className="h-5 w-5 shrink-0 text-alert" aria-hidden />
          <div className="flex-1">
            <p className="text-sm font-semibold text-alert">
              {data.open_fraud_alerts} transaction{data.open_fraud_alerts === 1 ? '' : 's'} need your
              review
            </p>
            <p className="text-xs text-alert/70">
              Our fraud model flagged unusual activity. Confirm whether it was you.
            </p>
          </div>
          <span className="text-sm font-medium text-alert">Review →</span>
        </Link>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* ------------------------------ charts ------------------------------ */}
        <div className="space-y-6 lg:col-span-2">
          <Card>
            <SectionHeading title="Spending" subtitle="Daily outflow over the last 30 days" />
            {spendSeries.some((d) => d.amount > 0) ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={spendSeries} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                    <defs>
                      <linearGradient id="spendFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={palette.gold} stopOpacity={0.32} />
                        <stop offset="100%" stopColor={palette.gold} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="date"
                      tick={axisTick}
                      axisLine={false}
                      tickLine={false}
                      interval="preserveStartEnd"
                      minTickGap={24}
                    />
                    <YAxis
                      tick={axisTick}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={compactAxisFormatter}
                      width={52}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={labelledMoneyFormatter('Spent')}
                      labelFormatter={labelFormatter((label) => `Day ${label}`)}
                    />
                    <Area
                      type="monotone"
                      dataKey="amount"
                      stroke={palette.gold}
                      strokeWidth={2}
                      fill="url(#spendFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No spending yet" description="Transactions will appear here once you start using your account." />
            )}
          </Card>

          <Card>
            <SectionHeading
              title="Recent transactions"
              action={
                <Link to="/app/transactions" className="text-sm font-medium text-gold hover:text-gold-bright">
                  View all
                </Link>
              }
            />
            {data.recent_transactions.length ? (
              <ul className="divide-y divide-line">
                {data.recent_transactions.map((txn) => {
                  const amount = Number.parseFloat(txn.signed_amount);
                  return (
                    <li key={txn.id} className="flex items-center gap-3 py-3">
                      <span
                        className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg ${
                          amount >= 0 ? 'bg-positive/10 text-positive' : 'bg-surface-raised text-muted'
                        }`}
                      >
                        {amount >= 0 ? (
                          <ArrowDownLeft className="h-4 w-4" aria-hidden />
                        ) : (
                          <ArrowUpRight className="h-4 w-4" aria-hidden />
                        )}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-primary">
                          {txn.description ?? txn.merchant_name ?? titleCase(txn.merchant_category)}
                        </p>
                        <p className="tnum text-xs text-muted">
                          {dateShort(txn.occurred_at)} · {categoryLabel(txn.merchant_category)}
                        </p>
                      </div>
                      <div className="text-right">
                        <p
                          className={`tnum text-sm font-semibold ${
                            amount >= 0 ? 'text-positive' : 'text-primary'
                          }`}
                        >
                          {moneySigned(txn.signed_amount)}
                        </p>
                        {txn.status !== 'completed' && <StatusBadge status={txn.status} />}
                      </div>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <EmptyState
                title="No transactions yet"
                description="Fund your account to get started."
                action={
                  <Link to="/app/accounts" className="btn-secondary px-4 py-2">
                    Go to accounts
                  </Link>
                }
              />
            )}
          </Card>
        </div>

        {/* ----------------------------- sidebar ----------------------------- */}
        <div className="space-y-6">
          <Card>
            <SectionHeading
              title="Accounts"
              action={
                <Link to="/app/accounts" className="text-sm font-medium text-gold hover:text-gold-bright">
                  Manage
                </Link>
              }
            />
            <ul className="space-y-3">
              {data.accounts.map((account) => (
                <li key={account.id} className="rounded-lg border border-line bg-ink/40 p-3.5">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium text-primary">
                        {account.nickname ?? titleCase(account.account_type)}
                      </p>
                      <p className="font-mono text-xs text-muted">
                        {maskAccount(account.account_number)}
                      </p>
                    </div>
                    {account.is_primary && (
                      <span className="badge bg-gold/15 text-gold-bright">Primary</span>
                    )}
                  </div>
                  <p className="tnum mt-2 text-lg font-bold text-primary">{money(account.balance)}</p>
                  {Number.parseFloat(account.hold_amount) > 0 && (
                    <p className="mt-0.5 text-xs text-warning">
                      {money(account.hold_amount)} on hold
                    </p>
                  )}
                </li>
              ))}
              <li>
                <Link
                  to="/app/accounts"
                  className="flex items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong py-3 text-sm text-muted transition hover:border-gold/50 hover:text-primary"
                >
                  <Plus className="h-4 w-4" aria-hidden />
                  Open another account
                </Link>
              </li>
            </ul>
          </Card>

          <Card>
            <SectionHeading title="Where your money goes" subtitle="Last 30 days" />
            {categorySeries.length ? (
              <>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={categorySeries}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={44}
                        outerRadius={68}
                        paddingAngle={2}
                        stroke="none"
                      >
                        {categorySeries.map((entry) => (
                          <Cell key={entry.category} fill={categoryColor(entry.category)} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} formatter={moneyFormatter} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="mt-3 space-y-1.5">
                  {categorySeries.map((entry) => (
                    <li key={entry.category} className="flex items-center gap-2 text-xs">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: categoryColor(entry.category) }}
                        aria-hidden
                      />
                      <span className="flex-1 text-muted">{entry.name}</span>
                      <span className="tnum font-medium text-primary">{moneyCompact(entry.value)}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="py-6 text-center text-sm text-muted">No spending data yet.</p>
            )}
          </Card>

          <div className="grid grid-cols-2 gap-3">
            <Link to="/app/loans" className="card card-hover flex flex-col items-center gap-2 p-4 text-center">
              <Landmark className="h-5 w-5 text-positive" aria-hidden />
              <span className="text-xs font-medium text-primary">
                {data.active_loans > 0 ? `${data.active_loans} active loan${data.active_loans === 1 ? '' : 's'}` : 'Apply for a loan'}
              </span>
            </Link>
            <Link to="/app/insights" className="card card-hover flex flex-col items-center gap-2 p-4 text-center">
              <PiggyBank className="h-5 w-5 text-gold" aria-hidden />
              <span className="text-xs font-medium text-primary">Spending insights</span>
            </Link>
          </div>
        </div>
      </div>

      <AssistantWidget />
    </div>
  );
}
