import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { Ban, Brain, CheckCircle2, MinusCircle, ShieldAlert, ShieldCheck } from 'lucide-react';
import { useState } from 'react';

import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  Field,
  LoadingBlock,
  Meter,
  Modal,
  Notice,
  PageHeader,
  Pagination,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { dateTime, latency, money, percent, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import type { FraudAlert, Page } from '../../types/api';

const SEVERITIES = ['critical', 'high', 'medium', 'low'];

/**
 * Fraud review queue.
 *
 * This is the human-in-the-loop step that closes the ML feedback cycle: an
 * analyst verdict writes `Transaction.is_fraud_label`, which becomes the ground
 * truth for the next retraining run.
 *
 * "Dismiss" deliberately leaves the label NULL rather than guessing. Feeding an
 * uncertain case into training data as if it were known is worse than having no
 * label at all.
 */
export default function AdminFraudQueue() {
  const [page, setPage] = useState(1);
  const [severity, setSeverity] = useState('');
  const [minScore, setMinScore] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [selected, setSelected] = useState<FraudAlert | null>(null);
  const [note, setNote] = useState('');
  const [reverse, setReverse] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const params: Record<string, unknown> = { page, page_size: 15 };
  if (severity) params.severity = severity;
  if (minScore) params.min_score = Number(minScore);
  if (statusFilter) params.status = statusFilter;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.admin.fraudQueue(params),
    queryFn: () => get<Page<FraudAlert>>('/admin/fraud/queue', { params }),
  });

  const review = useMutation({
    mutationFn: ({
      alert,
      decision,
    }: {
      alert: FraudAlert;
      decision: 'fraud' | 'legitimate' | 'dismiss';
    }) =>
      post<FraudAlert>(`/admin/fraud/queue/${alert.id}/review`, {
        decision,
        note: note.trim() || null,
        reverse_transaction: decision === 'fraud' ? reverse : false,
      }),
    onSuccess: async () => {
      setSelected(null);
      setNote('');
      setReverse(false);
      setActionError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['admin', 'fraud'] }),
        queryClient.invalidateQueries({ queryKey: qk.admin.stats }),
        queryClient.invalidateQueries({ queryKey: ['admin', 'models'] }),
      ]);
    },
    onError: (err) => setActionError(errorMessage(err)),
  });

  const severityTone = (value: string) =>
    value === 'critical' || value === 'high' ? 'danger' : value === 'medium' ? 'warning' : 'neutral';

  return (
    <div>
      <PageHeader
        title="Fraud review queue"
        subtitle="Model-flagged transactions. Your verdict becomes the training label for the next retrain."
      />

      {actionError && (
        <div className="mb-5">
          <Notice tone="danger">{actionError}</Notice>
        </div>
      )}

      {/* ------------------------------ filters ------------------------------ */}
      <div className="mb-5 flex flex-wrap gap-2">
        <select
          className="input w-auto"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(1);
          }}
          aria-label="Filter by status"
        >
          <option value="">Actionable queue</option>
          <option value="open">Open only</option>
          <option value="confirmed_fraud">Customer-disputed</option>
          <option value="resolved_fraud">Resolved as fraud</option>
          <option value="resolved_legit">Resolved as legitimate</option>
          <option value="dismissed">Dismissed</option>
        </select>

        <select
          className="input w-auto"
          value={severity}
          onChange={(e) => {
            setSeverity(e.target.value);
            setPage(1);
          }}
          aria-label="Filter by severity"
        >
          <option value="">Any severity</option>
          {SEVERITIES.map((value) => (
            <option key={value} value={value}>
              {titleCase(value)}
            </option>
          ))}
        </select>

        <select
          className="input w-auto"
          value={minScore}
          onChange={(e) => {
            setMinScore(e.target.value);
            setPage(1);
          }}
          aria-label="Minimum risk score"
        >
          <option value="">Any score</option>
          <option value="0.5">≥ 50%</option>
          <option value="0.75">≥ 75%</option>
          <option value="0.9">≥ 90%</option>
        </select>
      </div>

      {/* ------------------------------- queue ------------------------------- */}
      {isLoading ? (
        <LoadingBlock rows={5} label="Loading review queue" />
      ) : error ? (
        <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
      ) : data?.items.length ? (
        <div className="space-y-3">
          {data.items.map((alert) => (
            <Card
              key={alert.id}
              className={clsx(
                'transition',
                alert.status === 'confirmed_fraud' && 'border-alert/40 bg-alert/[0.04]',
              )}
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs text-muted">{alert.alert_ref}</span>
                    <Badge tone={severityTone(alert.severity)}>{titleCase(alert.severity)}</Badge>
                    <StatusBadge status={alert.status} />
                    {alert.auto_blocked && <Badge tone="danger">Auto-blocked</Badge>}
                    {alert.customer_response && (
                      <Badge tone={alert.customer_response === 'disputed' ? 'danger' : 'info'}>
                        Customer: {alert.customer_response}
                      </Badge>
                    )}
                  </div>

                  <p className="tnum mt-2 font-semibold text-primary">
                    {alert.transaction
                      ? `${money(alert.transaction.amount)} · ${
                          alert.transaction.description ??
                          alert.transaction.merchant_name ??
                          titleCase(alert.transaction.merchant_category)
                        }`
                      : 'Flagged transaction'}
                  </p>
                  <p className="tnum mt-0.5 text-xs text-muted">
                    {alert.transaction && (
                      <>
                        {dateTime(alert.transaction.occurred_at)} ·{' '}
                        {alert.transaction.channel.toUpperCase()}
                        {alert.transaction.location_city && ` · ${alert.transaction.location_city}`}
                        {alert.transaction.location_country &&
                          alert.transaction.location_country !== 'IN' &&
                          ` (${alert.transaction.location_country})`}
                      </>
                    )}
                  </p>

                  {alert.triggered_rules && alert.triggered_rules.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap gap-1.5">
                      {alert.triggered_rules.slice(0, 3).map((rule) => (
                        <span key={rule} className="badge bg-warning/10 text-[10px] text-warning">
                          {rule}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="w-full shrink-0 sm:w-40">
                  <p className="text-[11px] tracking-wide text-intelligence uppercase">Model score</p>
                  <p className="tnum text-2xl font-bold text-intelligence">{percent(alert.risk_score, 1)}</p>
                  <div className="mt-2">
                    <Meter value={alert.risk_score} tone={severityTone(alert.severity)} />
                  </div>
                  {alert.inference_latency_ms !== null && (
                    <p className="tnum mt-1.5 text-[11px] text-faint">
                      scored in {latency(alert.inference_latency_ms)}
                    </p>
                  )}
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-line pt-3">
                {alert.final_label === null ? (
                  <button
                    type="button"
                    className="btn-secondary px-3.5 py-2 text-xs"
                    onClick={() => {
                      setSelected(alert);
                      setNote('');
                      setReverse(false);
                      setActionError(null);
                    }}
                  >
                    <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                    Review
                  </button>
                ) : (
                  <p className="text-xs text-muted">
                    Labelled{' '}
                    <span className={alert.final_label ? 'text-alert' : 'text-positive'}>
                      {alert.final_label ? 'fraud' : 'legitimate'}
                    </span>
                    {alert.review_note && ` · ${alert.review_note}`}
                  </p>
                )}
                <button
                  type="button"
                  className="btn-ghost ml-auto px-3 py-2 text-xs"
                  onClick={() => {
                    setSelected(alert);
                    setNote('');
                  }}
                >
                  Details
                </button>
              </div>
            </Card>
          ))}

          <Card className="p-0">
            <Pagination
              page={data.page}
              totalPages={data.total_pages}
              total={data.total}
              pageSize={data.page_size}
              onPageChange={setPage}
            />
          </Card>
        </div>
      ) : (
        <EmptyState
          icon={<ShieldCheck className="h-10 w-10" aria-hidden />}
          title="Queue is empty"
          description="No transactions are currently awaiting review."
        />
      )}

      {/* ------------------------------ review ------------------------------ */}
      <Modal
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={`Review ${selected?.alert_ref ?? ''}`}
        wide
        footer={
          selected?.final_label === null ? (
            <>
              <button
                type="button"
                className="btn-ghost px-3.5 py-2"
                onClick={() => selected && review.mutate({ alert: selected, decision: 'dismiss' })}
                disabled={review.isPending}
              >
                <MinusCircle className="h-4 w-4" aria-hidden />
                Dismiss
              </button>
              <button
                type="button"
                className="btn-secondary px-3.5 py-2"
                onClick={() => selected && review.mutate({ alert: selected, decision: 'legitimate' })}
                disabled={review.isPending}
              >
                <CheckCircle2 className="h-4 w-4" aria-hidden />
                Legitimate
              </button>
              <button
                type="button"
                className="btn-danger px-3.5 py-2"
                onClick={() => selected && review.mutate({ alert: selected, decision: 'fraud' })}
                disabled={review.isPending}
              >
                <Ban className="h-4 w-4" aria-hidden />
                Confirm fraud
              </button>
            </>
          ) : (
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setSelected(null)}>
              Close
            </button>
          )
        }
      >
        {selected && (
          <div className="space-y-4">
            {actionError && <Notice tone="danger">{actionError}</Notice>}

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-line bg-ink/50 p-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">Amount</p>
                <p className="tnum mt-0.5 font-bold text-primary">
                  {selected.transaction ? money(selected.transaction.amount) : '—'}
                </p>
              </div>
              <div className="rounded-lg border border-line bg-ink/50 p-3">
                <p className="text-[11px] tracking-wide text-intelligence uppercase">Risk score</p>
                <p className="tnum mt-0.5 font-bold text-intelligence">{percent(selected.risk_score, 2)}</p>
              </div>
              <div className="rounded-lg border border-line bg-ink/50 p-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">Source</p>
                <p className="mt-0.5 font-bold text-primary capitalize">{selected.decision_source}</p>
              </div>
            </div>

            {selected.transaction && (
              <dl className="divide-y divide-line text-sm">
                {[
                  ['Reference', <span className="font-mono text-xs">{selected.transaction.reference}</span>],
                  ['When', dateTime(selected.transaction.occurred_at)],
                  ['Channel', selected.transaction.channel.toUpperCase()],
                  ['Category', titleCase(selected.transaction.merchant_category)],
                  ...(selected.transaction.counterparty_name
                    ? [['Counterparty', selected.transaction.counterparty_name]]
                    : []),
                  ...(selected.transaction.location_city
                    ? [
                        [
                          'Location',
                          `${selected.transaction.location_city}${
                            selected.transaction.location_country
                              ? ` (${selected.transaction.location_country})`
                              : ''
                          }`,
                        ],
                      ]
                    : []),
                  ['Transaction status', <StatusBadge status={selected.transaction.status} />],
                ].map(([label, value], i) => (
                  <div key={i} className="flex items-center justify-between gap-3 py-2.5">
                    <dt className="text-muted">{label}</dt>
                    {/* Values in this list are references, dates and amounts,
                        so the whole column takes mono + tabular-nums. */}
                    <dd className="tnum text-right text-primary">{value}</dd>
                  </div>
                ))}
              </dl>
            )}

            {selected.reasons && selected.reasons.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-semibold tracking-wide text-muted uppercase">
                  Reasons
                </p>
                <ul className="space-y-1">
                  {selected.reasons.map((reason) => (
                    <li key={reason} className="flex gap-2 text-sm text-muted">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-warning" aria-hidden />
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {selected.top_factors && selected.top_factors.length > 0 && (
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-muted uppercase">
                  <Brain className="h-3.5 w-3.5" aria-hidden />
                  Model factors
                </p>
                <ul className="space-y-1.5">
                  {selected.top_factors.map((factor) => (
                    <li key={factor.feature} className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-muted">
                        {factor.label}
                        <span className="tnum ml-2 text-xs text-faint">
                          {factor.value.toFixed(2)}
                        </span>
                      </span>
                      <Badge tone={factor.direction === 'increases risk' ? 'danger' : 'success'}>
                        {factor.contribution > 0 ? '+' : ''}
                        {factor.contribution.toFixed(3)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {selected.customer_note && (
              <Notice tone="info" title={`Customer says (${selected.customer_response})`}>
                {selected.customer_note}
              </Notice>
            )}

            {selected.final_label === null && (
              <div className="space-y-3 border-t border-line pt-4">
                <Field
                  label="Review note"
                  htmlFor="reviewNote"
                  hint="Stored with the label and visible in the audit trail"
                >
                  <textarea
                    id="reviewNote"
                    rows={3}
                    className="input resize-none"
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    maxLength={1000}
                    placeholder="Confirmed with the customer by phone."
                  />
                </Field>

                {selected.transaction?.status === 'completed' && (
                  <label className="flex cursor-pointer items-start gap-2.5 text-sm text-primary">
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 accent-alert"
                      checked={reverse}
                      onChange={(e) => setReverse(e.target.checked)}
                    />
                    <span>
                      Reverse the transaction
                      <span className="block text-xs text-muted">
                        Posts a compensating credit rather than editing the original entry, keeping
                        the ledger append-only.
                      </span>
                    </span>
                  </label>
                )}

                <Notice tone="warning" title="This writes a training label">
                  Confirming fraud or legitimacy sets the ground-truth label used by the next
                  retraining run. Dismiss instead if you are genuinely unsure — an uncertain label is
                  worse than no label.
                </Notice>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
