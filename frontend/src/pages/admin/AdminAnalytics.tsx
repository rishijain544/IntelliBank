import { useMutation, useQuery } from '@tanstack/react-query';
import { Activity, Brain, RefreshCw } from 'lucide-react';
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

import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  Notice,
  PageHeader,
  SectionHeading,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import {
  axisTick,
  barCursor,
  compactAxisFormatter,
  gridStroke,
  labelFormatter,
  mixedFormatter,
  moneyFormatter,
  tooltipStyle,
} from '../../lib/charts';
import { bandColor, palette, severityColor } from '../../lib/colors';
import { categoryColor, categoryLabel, latency, num, percent } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import type { AnalyticsData, MessageResponse, ModelPerformance } from '../../types/api';

const MODEL_LABELS: Record<string, string> = {
  fraud_xgb: 'Fraud detection',
  credit_xgb: 'Credit scoring',
  anomaly_iforest: 'Anomaly detection',
};

/**
 * Model analytics and drift monitoring.
 *
 * The point of this page is the gap between *training* metrics (what the model
 * scored on held-out data) and *live* behaviour (what it is doing in production).
 * A model that looked good at training time and has since drifted is the failure
 * mode this exists to make visible.
 */
function ModelPanel({ model }: { model: ModelPerformance }) {
  const label = MODEL_LABELS[model.model_name] ?? model.model_name;
  const training = model.training_metrics;

  const histogram = model.score_histogram.filter((b) => b.count > 0);

  return (
    <Card>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-primary">{label}</h3>
          <p className="font-mono text-[11px] text-muted">
            {model.model_name}
            {model.model_version && ` v${model.model_version}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={model.loaded ? 'success' : 'danger'}>
            {model.loaded ? 'Loaded' : 'Not loaded'}
          </Badge>
          {model.drift_status && <StatusBadge status={model.drift_status} />}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Training baseline */}
        <div>
          <p className="mb-2 text-[11px] font-semibold tracking-wide text-muted uppercase">
            Training baseline
          </p>
          <dl className="space-y-1.5 text-xs">
            {(
              [
                ['ROC-AUC', training.roc_auc, (v: number) => v.toFixed(4)],
                ['PR-AUC', training.pr_auc, (v: number) => v.toFixed(4)],
                ['Recall', training.recall, (v: number) => percent(v, 1)],
                ['Precision', training.precision, (v: number) => percent(v, 1)],
                ['Gini', training.gini, (v: number) => v.toFixed(4)],
                ['Calibration error', training.ece, (v: number) => v.toFixed(4)],
              ] as const
            )
              .filter(([, value]) => value !== undefined && value !== null)
              .map(([name, value, fmt]) => (
                <div key={name} className="flex justify-between">
                  <dt className="text-muted">{name}</dt>
                  <dd className="tnum text-primary">{fmt(value as number)}</dd>
                </div>
              ))}
            {model.training_latency?.p95_ms !== undefined && (
              <div className="flex justify-between">
                <dt className="text-muted">Latency p95</dt>
                <dd className="tnum text-primary">{latency(model.training_latency.p95_ms)}</dd>
              </div>
            )}
          </dl>
        </div>

        {/* Live behaviour */}
        <div>
          <p className="mb-2 text-[11px] font-semibold tracking-wide text-muted uppercase">
            Live behaviour
          </p>
          <dl className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <dt className="text-muted">Inferences</dt>
              <dd className="tnum text-primary">{num(model.live_inference_count)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Flagged</dt>
              <dd className="tnum text-primary">{num(model.live_flagged_count)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Labelled</dt>
              <dd className="tnum text-primary">{num(model.live_labelled_count)}</dd>
            </div>
            {model.live_mean_score !== null && (
              <div className="flex justify-between">
                <dt className="text-muted">Mean score</dt>
                <dd className="tnum text-primary">{model.live_mean_score.toFixed(4)}</dd>
              </div>
            )}
            {model.live_p95_latency_ms !== null && (
              <div className="flex justify-between">
                <dt className="text-muted">Latency p95</dt>
                <dd className="tnum text-primary">{latency(model.live_p95_latency_ms)}</dd>
              </div>
            )}
            {model.psi !== null && (
              <div className="flex justify-between">
                <dt className="text-muted">PSI</dt>
                <dd className="tnum text-primary">{model.psi.toFixed(4)}</dd>
              </div>
            )}
          </dl>
        </div>
      </div>

      {/* Realised performance only exists once analysts have labelled cases. */}
      {(model.realised_precision !== null || model.realised_recall !== null) && (
        <div className="mt-4 rounded-lg border border-line bg-ink/40 p-3">
          <p className="mb-1.5 text-[11px] font-semibold tracking-wide text-muted uppercase">
            Realised performance (from analyst labels)
          </p>
          <div className="flex gap-6 text-xs">
            {model.realised_precision !== null && (
              <span className="text-muted">
                Precision <span className="tnum font-medium text-primary">{percent(model.realised_precision, 1)}</span>
              </span>
            )}
            {model.realised_recall !== null && (
              <span className="text-muted">
                Recall <span className="tnum font-medium text-primary">{percent(model.realised_recall, 1)}</span>
              </span>
            )}
          </div>
        </div>
      )}

      {histogram.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-[11px] font-semibold tracking-wide text-muted uppercase">
            Live score distribution
          </p>
          <div className="h-28">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={model.score_histogram} margin={{ top: 2, right: 2, bottom: 0, left: -28 }}>
                <XAxis dataKey="bin" tick={axisTick} axisLine={false} tickLine={false} />
                <YAxis tick={axisTick} axisLine={false} tickLine={false} width={40} />
                <Tooltip contentStyle={tooltipStyle} cursor={barCursor} />
                <Bar dataKey="count" fill={palette.intelligence} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  );
}

export default function AdminAnalytics() {
  const [days, setDays] = useState(30);
  const [banner, setBanner] = useState<string | null>(null);

  const { data: models, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useQuery({
    queryKey: qk.admin.models(days),
    queryFn: () => get<ModelPerformance[]>('/admin/models', { params: { days } }),
  });

  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: qk.admin.analytics(days),
    queryFn: () => get<AnalyticsData>('/admin/analytics', { params: { days } }),
  });

  const reload = useMutation({
    mutationFn: () => post<MessageResponse>('/admin/models/reload'),
    onSuccess: async (data) => {
      setBanner(data.message);
      window.setTimeout(() => setBanner(null), 5000);
      await queryClient.invalidateQueries({ queryKey: ['admin'] });
    },
    onError: (err) => setBanner(errorMessage(err)),
  });

  const anyDrifting = models?.some((m) => m.drift_status === 'drifting');

  const volumeSeries =
    analytics?.daily_volume.map((d) => ({
      date: d.date.slice(5),
      volume: d.volume,
      count: d.count,
      flagged: d.flagged,
    })) ?? [];

  return (
    <div>
      <PageHeader
        title="Model analytics"
        subtitle="Training baselines versus live production behaviour, with drift monitoring."
        action={
          <div className="flex gap-2">
            <select
              className="input w-auto"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              aria-label="Analysis window"
            >
              <option value={7}>7 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
            <button
              type="button"
              className="btn-secondary px-3.5 py-2.5"
              onClick={() => reload.mutate()}
              disabled={reload.isPending}
            >
              <RefreshCw className={`h-4 w-4 ${reload.isPending ? 'animate-spin' : ''}`} aria-hidden />
              Reload artifacts
            </button>
          </div>
        }
      />

      {banner && (
        <div className="mb-5">
          <Notice tone="success">{banner}</Notice>
        </div>
      )}

      {anyDrifting && (
        <div className="mb-6">
          <Notice tone="warning" title="Score distribution has shifted">
            A model's live score distribution differs meaningfully from its training baseline
            (PSI &gt; 0.25). That can mean genuine population change, an upstream data issue, or a
            model that needs retraining. On seeded demo data this is expected, because the seeder
            deliberately injects fraud.
          </Notice>
        </div>
      )}

      {/* ------------------------------- models ------------------------------- */}
      <SectionHeading title="Model performance" subtitle={`Live window: last ${days} days`} />
      {modelsLoading ? (
        <LoadingBlock rows={5} label="Loading model metrics" />
      ) : modelsError ? (
        <ErrorBlock message={errorMessage(modelsError)} onRetry={() => void refetchModels()} />
      ) : models?.length ? (
        <div className="mb-8 grid gap-5 lg:grid-cols-3">
          {models.map((model) => (
            <ModelPanel key={model.model_name} model={model} />
          ))}
        </div>
      ) : (
        <EmptyState icon={<Brain className="h-9 w-9" aria-hidden />} title="No model data" />
      )}

      {/* ------------------------------ system ------------------------------ */}
      <SectionHeading title="System activity" />
      {analyticsLoading ? (
        <LoadingBlock rows={4} />
      ) : analytics ? (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card className="lg:col-span-2">
            <SectionHeading title="Transaction volume" subtitle="Daily value and flagged count" />
            {volumeSeries.length ? (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={volumeSeries} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
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
                      yAxisId="left"
                      tick={axisTick}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={compactAxisFormatter}
                      width={54}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={axisTick}
                      axisLine={false}
                      tickLine={false}
                      width={34}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={mixedFormatter('Volume')}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="volume"
                      name="Volume"
                      stroke={palette.gold}
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="flagged"
                      name="Flagged"
                      stroke={palette.alert}
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No transaction activity in this window" />
            )}
          </Card>

          <Card>
            <SectionHeading title="Credit risk bands" subtitle="Distribution of scored applications" />
            {analytics.credit_band_distribution.length ? (
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={analytics.credit_band_distribution}
                    margin={{ top: 4, right: 4, bottom: 0, left: -22 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={gridStroke()} vertical={false} />
                    <XAxis dataKey="band" tick={axisTick} axisLine={false} tickLine={false} />
                    <YAxis tick={axisTick} axisLine={false} tickLine={false} width={40} />
                    <Tooltip contentStyle={tooltipStyle} cursor={barCursor} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {analytics.credit_band_distribution.map((entry) => (
                        <Cell key={entry.band} fill={bandColor(entry.band)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No credit scores yet" description="Scores appear once customers apply for loans." />
            )}
          </Card>

          <Card>
            <SectionHeading title="Alert severity" subtitle="Fraud alerts by severity" />
            {analytics.fraud_severity_distribution.length ? (
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={analytics.fraud_severity_distribution}
                      dataKey="count"
                      nameKey="severity"
                      innerRadius={44}
                      outerRadius={72}
                      paddingAngle={2}
                      stroke="none"
                    >
                      {analytics.fraud_severity_distribution.map((entry) => (
                        <Cell
                          key={entry.severity}
                          fill={
                            severityColor(entry.severity)
                          }
                        />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No fraud alerts in this window" />
            )}
          </Card>

          <Card className="lg:col-span-2">
            <SectionHeading title="Spending categories" subtitle="Platform-wide transaction volume" />
            {analytics.category_distribution.length ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={analytics.category_distribution.slice(0, 10)}
                    margin={{ top: 4, right: 4, bottom: 0, left: -8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={gridStroke()} vertical={false} />
                    <XAxis
                      dataKey="category"
                      tick={{ ...axisTick, fontSize: 9 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v: string) => categoryLabel(v)}
                      angle={-25}
                      textAnchor="end"
                      height={52}
                    />
                    <YAxis
                      tick={axisTick}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={compactAxisFormatter}
                      width={54}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={moneyFormatter}
                      labelFormatter={labelFormatter(categoryLabel)}
                      cursor={barCursor}
                    />
                    <Bar dataKey="total" radius={[3, 3, 0, 0]}>
                      {analytics.category_distribution.slice(0, 10).map((entry) => (
                        <Cell key={entry.category} fill={categoryColor(entry.category)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No category data" />
            )}
          </Card>
        </div>
      ) : null}

      <div className="mt-6">
        <Notice tone="info" title="How drift is measured">
          Population Stability Index compares the live score distribution against the one captured at
          training time. Below 0.10 is stable, 0.10–0.25 warrants watching, and above 0.25 indicates
          meaningful drift. PSI is suppressed below 50 live scores, where the statistic is too noisy
          to act on.
        </Notice>
      </div>

      <p className="mt-4 flex items-center justify-center gap-1.5 text-xs text-faint">
        <Activity className="h-3.5 w-3.5" aria-hidden />
        Retrain with <code className="rounded bg-surface px-1.5 py-0.5 font-mono">python manage.py train</code>,
        then use “Reload artifacts” to pick up the new models without restarting the API.
      </p>
    </div>
  );
}
