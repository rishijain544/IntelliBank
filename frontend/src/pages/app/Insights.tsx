import { useMutation, useQuery } from '@tanstack/react-query';
import { ArrowDownLeft, ArrowUpRight, Lightbulb, Receipt, Sparkles, TrendingUp, X } from 'lucide-react';
import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import AssistantWidget from '../../components/AssistantWidget';
import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  Notice,
  PageHeader,
  SectionHeading,
  StatTile,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import {
  axisTick,
  barCursor,
  compactAxisFormatter,
  gridStroke,
  labelledMoneyFormatter,
  moneyFormatter,
  tooltipStyle,
} from '../../lib/charts';
import { palette } from '../../lib/colors';
import {
  categoryColor,
  categoryLabel,
  dateShort,
  money,
  moneyCompact,
  num,
  percentRaw,
  titleCase,
} from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import type { InsightsData, MessageResponse } from '../../types/api';

const PERIODS = [
  { days: 30, label: '30 days' },
  { days: 90, label: '90 days' },
  { days: 180, label: '6 months' },
  { days: 365, label: '1 year' },
] as const;

/**
 * Spending insights, driven by the anomaly model.
 *
 * The anomaly alerts here are informational by design — the model never blocks
 * money. Each one is phrased as an observation with the baseline it deviated
 * from, so the customer can judge it rather than just being told "unusual".
 */
export default function Insights() {
  const [days, setDays] = useState<number>(30);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.insights(days),
    queryFn: () => get<InsightsData>('/insights', { params: { days } }),
  });

  const dismiss = useMutation({
    mutationFn: (alertId: number) =>
      post<MessageResponse>(`/insights/anomalies/${alertId}/acknowledge`),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['insights'] });
    },
  });

  if (isLoading) return <LoadingBlock rows={6} label="Loading insights" />;
  if (error) return <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />;
  if (!data) return null;

  const categorySeries = data.category_breakdown
    .filter((c) => Number.parseFloat(c.total) > 0)
    .map((c) => ({
      name: categoryLabel(c.category),
      category: c.category,
      value: Number.parseFloat(c.total),
      count: c.count,
      percentage: c.percentage,
    }));

  const trendSeries = data.monthly_trends.map((t) => ({
    month: t.month,
    inflow: Number.parseFloat(t.inflow),
    outflow: Number.parseFloat(t.outflow),
    net: Number.parseFloat(t.net),
  }));

  const dailySeries = data.daily_spend.map((d) => ({
    date: d.date.slice(5),
    amount: Number.parseFloat(d.amount),
  }));

  const netChange = Number.parseFloat(data.net_change);

  return (
    <div>
      <PageHeader
        title="Insights"
        subtitle="Where your money goes, and anything that looks out of character."
        action={
          <div className="flex gap-1 rounded-lg border border-line p-1">
            {PERIODS.map((period) => (
              <button
                key={period.days}
                type="button"
                onClick={() => setDays(period.days)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                  days === period.days ? 'bg-gold text-ink' : 'text-muted hover:text-primary'
                }`}
              >
                {period.label}
              </button>
            ))}
          </div>
        }
      />

      {/* ---------------------------- anomaly alerts ---------------------------- */}
      {data.anomaly_alerts.length > 0 && (
        <div className="mb-6 space-y-3">
          {data.anomaly_alerts.map((alert) => (
            <div
              key={alert.id}
              className="flex items-start gap-3 rounded-xl border border-intelligence/25 bg-intelligence/[0.04] p-4"
            >
              <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-intelligence/15 text-intelligence">
                <Sparkles className="h-4.5 w-4.5" aria-hidden />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-medium text-primary">{alert.title}</p>
                  <Badge tone="accent">{titleCase(alert.anomaly_type)}</Badge>
                </div>
                <p className="mt-1 text-sm text-muted">{alert.message}</p>
                {alert.baseline_value !== null && alert.observed_value !== null && (
                  <p className="tnum mt-1.5 text-xs text-muted">
                    Usually {money(alert.baseline_value)} · this period {money(alert.observed_value)}
                    {alert.deviation_ratio !== null && ` · ${alert.deviation_ratio.toFixed(1)}x`}
                  </p>
                )}
              </div>
              <button
                type="button"
                className="rounded p-1.5 text-muted transition hover:text-primary"
                aria-label="Dismiss insight"
                onClick={() => dismiss.mutate(alert.id)}
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* -------------------------------- tiles -------------------------------- */}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Total spent"
          value={moneyCompact(data.total_spent)}
          hint={`${num(data.txn_count)} transactions`}
          icon={<ArrowUpRight className="h-4 w-4" aria-hidden />}
          tone="warning"
        />
        <StatTile
          label="Total received"
          value={moneyCompact(data.total_received)}
          icon={<ArrowDownLeft className="h-4 w-4" aria-hidden />}
          tone="success"
        />
        <StatTile
          label="Net change"
          value={moneyCompact(data.net_change)}
          hint={netChange >= 0 ? 'You saved money' : 'You spent more than you received'}
          icon={<TrendingUp className="h-4 w-4" aria-hidden />}
          tone={netChange >= 0 ? 'success' : 'danger'}
        />
        <StatTile
          label="Average transaction"
          value={money(data.avg_transaction)}
          hint={`Largest ${moneyCompact(data.largest_transaction)}`}
          icon={<Receipt className="h-4 w-4" aria-hidden />}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ------------------------------ category ------------------------------ */}
        <Card>
          <SectionHeading title="Spending by category" subtitle={`Last ${days} days`} />
          {categorySeries.length ? (
            <div className="grid gap-4 sm:grid-cols-[1fr_1.1fr]">
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={categorySeries}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={48}
                      outerRadius={76}
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
              <ul className="space-y-2 self-center">
                {categorySeries.slice(0, 7).map((entry) => (
                  <li key={entry.category}>
                    <div className="flex items-center gap-2 text-xs">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: categoryColor(entry.category) }}
                        aria-hidden
                      />
                      <span className="flex-1 text-muted">{entry.name}</span>
                      <span className="tnum font-medium text-primary">
                        {moneyCompact(entry.value)}
                      </span>
                      <span className="tnum w-11 text-right text-muted">
                        {percentRaw(entry.percentage, 0)}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <EmptyState title="No spending in this period" />
          )}
        </Card>

        {/* ------------------------------- daily ------------------------------- */}
        <Card>
          <SectionHeading title="Daily spending" subtitle="Outflow per day" />
          {dailySeries.some((d) => d.amount > 0) ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dailySeries} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke()} vertical={false} />
                  <XAxis
                    dataKey="date"
                    tick={axisTick}
                    axisLine={false}
                    tickLine={false}
                    interval="preserveStartEnd"
                    minTickGap={26}
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
                    cursor={barCursor}
                  />
                  <Bar dataKey="amount" fill={palette.gold} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState title="No daily spending data" />
          )}
        </Card>

        {/* ------------------------------ trends ------------------------------ */}
        <Card className="lg:col-span-2">
          <SectionHeading title="Money in versus money out" subtitle="Monthly, last 6 months" />
          {trendSeries.length ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendSeries} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke()} vertical={false} />
                  <XAxis
                    dataKey="month"
                    tick={axisTick}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={axisTick}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={compactAxisFormatter}
                    width={58}
                  />
                  <Tooltip contentStyle={tooltipStyle} formatter={moneyFormatter} />
                  <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                  <Line
                    type="monotone"
                    dataKey="inflow"
                    name="Money in"
                    stroke={palette.positive}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="outflow"
                    name="Money out"
                    stroke={palette.warning}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="net"
                    name="Net"
                    stroke={palette.gold}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState title="Not enough history yet" description="Monthly trends need at least one full month of activity." />
          )}
        </Card>

        {/* ---------------------------- top merchants ---------------------------- */}
        <Card>
          <SectionHeading title="Top merchants" subtitle={`Where you spent most in ${days} days`} />
          {data.top_merchants.length ? (
            <ul className="space-y-2.5">
              {data.top_merchants.map((merchant, index) => (
                <li key={merchant.merchant} className="flex items-center gap-3">
                  <span className="tnum grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-surface-raised text-xs font-bold text-muted">
                    {index + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-primary">{merchant.merchant}</p>
                    <p className="text-xs text-muted">
                      {merchant.count} transaction{merchant.count === 1 ? '' : 's'}
                    </p>
                  </div>
                  <span className="tnum text-sm font-semibold text-primary">
                    {money(merchant.total)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No merchant data yet" />
          )}
        </Card>

        {/* ------------------------------- summary ------------------------------- */}
        <Card>
          <SectionHeading title="Category detail" />
          {categorySeries.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Spending by category</caption>
                <thead className="table-header">
                  <tr>
                    <th scope="col" className="py-2.5 pr-3">Category</th>
                    <th scope="col" className="px-3 py-2.5 text-right">Total</th>
                    <th scope="col" className="px-3 py-2.5 text-right">Count</th>
                    <th scope="col" className="py-2.5 pl-3 text-right">Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.category_breakdown.map((row) => (
                    <tr key={row.category}>
                      <td className="py-2.5 pr-3">
                        <span className="flex items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: categoryColor(row.category) }}
                            aria-hidden
                          />
                          <span className="text-primary">{categoryLabel(row.category)}</span>
                        </span>
                      </td>
                      <td className="tnum px-3 py-2.5 text-right text-primary">{money(row.total)}</td>
                      <td className="tnum px-3 py-2.5 text-right text-muted">{row.count}</td>
                      <td className="tnum py-2.5 pl-3 text-right text-muted">
                        {percentRaw(row.percentage, 1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="No categories to show" />
          )}
        </Card>
      </div>

      <div className="mt-6">
        <Notice tone="info" title="How these insights are produced">
          An Isolation Forest model learns your personal spending baseline and flags genuine
          departures from it. It informs only — no transaction is ever blocked by this model, and
          the alert rate is anchored to a fixed percentile so a quiet week does not flood you with
          notifications.
        </Notice>
      </div>

      {data.anomaly_alerts.length === 0 && (
        <p className="mt-4 flex items-center justify-center gap-1.5 text-xs text-faint">
          <Lightbulb className="h-3.5 w-3.5" aria-hidden />
          No unusual spending detected in this period · last checked {dateShort(new Date().toISOString())}
        </p>
      )}

      <AssistantWidget />
    </div>
  );
}
