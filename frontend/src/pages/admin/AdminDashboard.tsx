import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  Ban,
  Brain,
  Landmark,
  ShieldAlert,
  UserCheck,
  Users,
  Wallet,
} from 'lucide-react';
import { Link } from 'react-router-dom';

import {
  Badge,
  Card,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  SectionHeading,
  StatTile,
} from '../../components/ui';
import { errorMessage, get } from '../../lib/api';
import { dateTime, latency, moneyCompact, num, percent } from '../../lib/format';
import { qk } from '../../lib/query';
import type { AdminStats, ModelStatusEntry } from '../../types/api';

/** Compact model card for the overview; full detail lives in Analytics. */
function ModelCard({ label, entry }: { label: string; entry: ModelStatusEntry }) {
  const metrics = entry.metrics as Record<string, number | undefined>;
  const p95 = entry.latency_benchmark?.p95_ms;

  return (
    <div className="rounded-lg border border-line bg-ink/40 p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-primary">{label}</p>
        <Badge tone={entry.loaded ? 'success' : 'danger'}>{entry.loaded ? 'Loaded' : 'Missing'}</Badge>
      </div>
      <p className="mt-0.5 font-mono text-[11px] text-muted">
        {entry.name}
        {entry.version && ` v${entry.version}`}
      </p>

      {entry.loaded ? (
        <dl className="mt-3 space-y-1 text-xs">
          {metrics.roc_auc !== undefined && (
            <div className="flex justify-between">
              <dt className="text-muted">ROC-AUC</dt>
              <dd className="tnum text-primary">{metrics.roc_auc.toFixed(4)}</dd>
            </div>
          )}
          {metrics.recall !== undefined && (
            <div className="flex justify-between">
              <dt className="text-muted">Recall</dt>
              <dd className="tnum text-primary">{percent(metrics.recall, 1)}</dd>
            </div>
          )}
          {metrics.precision !== undefined && (
            <div className="flex justify-between">
              <dt className="text-muted">Precision</dt>
              <dd className="tnum text-primary">{percent(metrics.precision, 1)}</dd>
            </div>
          )}
          {p95 !== undefined && (
            <div className="flex justify-between">
              <dt className="text-muted">Latency p95</dt>
              <dd className="tnum text-primary">{latency(p95)}</dd>
            </div>
          )}
          {entry.trained_at && (
            <div className="flex justify-between">
              <dt className="text-muted">Trained</dt>
              <dd className="tnum text-muted">{dateTime(entry.trained_at).split(',')[0]}</dd>
            </div>
          )}
        </dl>
      ) : (
        <p className="mt-3 text-xs text-alert/80">
          Artifact not found. Scoring falls back to rules until this model is trained.
        </p>
      )}
    </div>
  );
}

export default function AdminDashboard() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.admin.stats,
    queryFn: () => get<AdminStats>('/admin/stats'),
  });

  if (isLoading) return <LoadingBlock rows={6} label="Loading platform statistics" />;
  if (error) return <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />;
  if (!data) return null;

  const anyModelMissing = Object.values(data.model_status).some((m) => !m.loaded);

  return (
    <div>
      <PageHeader
        title="Platform overview"
        subtitle="System-wide activity, review queues and model health."
      />

      {anyModelMissing && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-alert/30 bg-alert/5 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-alert" aria-hidden />
          <div>
            <p className="text-sm font-semibold text-alert">One or more models are not loaded</p>
            <p className="mt-0.5 text-xs text-alert/80">
              Risk scoring is degraded to rule-based fallbacks. Run{' '}
              <code className="rounded bg-surface px-1.5 py-0.5 font-mono">
                python manage.py train
              </code>{' '}
              then reload artifacts from the Analytics page.
            </p>
          </div>
        </div>
      )}

      {/* ------------------------------ customers ------------------------------ */}
      <SectionHeading title="Customers" />
      <div className="mb-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Total customers"
          value={num(data.total_users)}
          icon={<Users className="h-4 w-4" aria-hidden />}
          tone="info"
        />
        <StatTile
          label="Active"
          value={num(data.active_users)}
          icon={<UserCheck className="h-4 w-4" aria-hidden />}
          tone="success"
        />
        <StatTile
          label="Awaiting KYC"
          value={num(data.pending_kyc)}
          hint={data.pending_kyc > 0 ? 'Needs review' : 'All clear'}
          icon={<AlertTriangle className="h-4 w-4" aria-hidden />}
          tone={data.pending_kyc > 0 ? 'warning' : 'neutral'}
        />
        <StatTile
          label="Frozen"
          value={num(data.frozen_users)}
          icon={<Ban className="h-4 w-4" aria-hidden />}
          tone={data.frozen_users > 0 ? 'danger' : 'neutral'}
        />
      </div>

      {/* ------------------------------- money ------------------------------- */}
      <SectionHeading title="Money movement" />
      <div className="mb-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Total deposits held"
          value={moneyCompact(data.total_balance)}
          hint={`${num(data.total_accounts)} open accounts`}
          icon={<Wallet className="h-4 w-4" aria-hidden />}
          tone="info"
        />
        <StatTile
          label="Volume today"
          value={moneyCompact(data.txn_volume_today)}
          hint={`${num(data.txn_count_today)} transactions`}
          icon={<Activity className="h-4 w-4" aria-hidden />}
        />
        <StatTile
          label="Volume (30 days)"
          value={moneyCompact(data.txn_volume_30d)}
          hint={`${num(data.txn_count_30d)} transactions`}
          icon={<Activity className="h-4 w-4" aria-hidden />}
        />
        <StatTile
          label="Loans disbursed"
          value={moneyCompact(data.loans_disbursed_value)}
          hint={`${num(data.loans_approved)} approved`}
          icon={<Landmark className="h-4 w-4" aria-hidden />}
          tone="accent"
        />
      </div>

      {/* ------------------------------- queues ------------------------------- */}
      <SectionHeading title="Needs attention" />
      <div className="mb-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Link to="/admin/fraud" className="block">
          <StatTile
            label="Open fraud alerts"
            value={num(data.fraud_alerts_open)}
            hint="Awaiting analyst review"
            icon={<ShieldAlert className="h-4 w-4" aria-hidden />}
            tone={data.fraud_alerts_open > 0 ? 'danger' : 'success'}
          />
        </Link>
        <StatTile
          label="Confirmed fraud"
          value={num(data.fraud_confirmed)}
          hint={`of ${num(data.fraud_alerts_total)} total alerts`}
          icon={<ShieldAlert className="h-4 w-4" aria-hidden />}
          tone="warning"
        />
        <StatTile
          label="Blocked transactions"
          value={num(data.blocked_transactions)}
          hint="Stopped pre-settlement"
          icon={<Ban className="h-4 w-4" aria-hidden />}
          tone="warning"
        />
        <Link to="/admin/loans" className="block">
          <StatTile
            label="Loans pending"
            value={num(data.loans_pending)}
            hint="Awaiting credit decision"
            icon={<Landmark className="h-4 w-4" aria-hidden />}
            tone={data.loans_pending > 0 ? 'warning' : 'success'}
          />
        </Link>
      </div>

      {/* ------------------------------- models ------------------------------- */}
      <Card>
        <SectionHeading
          title="Model health"
          subtitle="Training metrics from the loaded artifacts"
          action={
            <Link to="/admin/analytics" className="text-sm font-medium text-warning hover:text-warning">
              Full analytics
            </Link>
          }
        />
        <div className="grid gap-4 md:grid-cols-3">
          <ModelCard label="Fraud detection" entry={data.model_status.fraud} />
          <ModelCard label="Credit scoring" entry={data.model_status.credit} />
          <ModelCard label="Anomaly detection" entry={data.model_status.anomaly} />
        </div>
        <p className="mt-4 flex items-center gap-1.5 text-xs text-faint">
          <Brain className="h-3.5 w-3.5" aria-hidden />
          Metrics are measured on held-out test data at training time. Live behaviour and drift are
          tracked separately in Analytics.
        </p>
      </Card>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <Link to="/admin/users" className="card card-hover flex items-center gap-3 p-4">
          <Users className="h-5 w-5 text-warning" aria-hidden />
          <span className="text-sm font-medium text-primary">Manage users</span>
        </Link>
        <Link to="/admin/fraud" className="card card-hover flex items-center gap-3 p-4">
          <ShieldAlert className="h-5 w-5 text-alert" aria-hidden />
          <span className="text-sm font-medium text-primary">Review fraud queue</span>
        </Link>
        <Link to="/admin/loans" className="card card-hover flex items-center gap-3 p-4">
          <Landmark className="h-5 w-5 text-positive" aria-hidden />
          <span className="text-sm font-medium text-primary">Approve loans</span>
        </Link>
      </div>
    </div>
  );
}
