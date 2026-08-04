import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  CheckCircle2,
  Fingerprint,
  Flag,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  ThumbsUp,
  XCircle,
} from 'lucide-react';
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
  StatTile,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { dateTime, latency, money, percent, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import { useAuth } from '../../store/auth';
import type { FraudAlert, Page } from '../../types/api';

interface FraudSummary {
  total_alerts: number;
  open_alerts: number;
  blocked_transactions: number;
  confirmed_fraud: number;
  two_factor_enabled: boolean;
  kyc_status: string;
}

/**
 * Fraud & Security Center.
 *
 * Where the fraud model's output becomes actionable for the customer: each alert
 * shows the score, the plain-language reasons and the model's top contributing
 * factors, then asks a single clear question — was this you?
 *
 * The customer's answer is treated as a signal, not as final truth. An analyst
 * still reviews it, because someone who has taken over an account would
 * otherwise be able to clear their own alerts.
 */
export default function FraudCenter() {
  const user = useAuth((s) => s.user);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<FraudAlert | null>(null);
  const [note, setNote] = useState('');
  const [formError, setFormError] = useState<string | null>(null);

  const params = { page, page_size: 10 };

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.fraudAlerts(params),
    queryFn: () => get<Page<FraudAlert>>('/fraud/alerts', { params }),
  });

  const { data: summary } = useQuery({
    queryKey: qk.fraudSummary,
    queryFn: () => get<FraudSummary>('/fraud/summary'),
  });

  const respond = useMutation({
    mutationFn: ({ alert, response }: { alert: FraudAlert; response: 'confirmed' | 'disputed' }) =>
      post<FraudAlert>(`/fraud/alerts/${alert.id}/respond`, {
        response,
        note: note.trim() || null,
      }),
    onSuccess: async () => {
      setSelected(null);
      setNote('');
      setFormError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['fraud'] }),
        queryClient.invalidateQueries({ queryKey: qk.dashboard }),
        queryClient.invalidateQueries({ queryKey: ['transactions'] }),
      ]);
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const severityTone = (severity: string) =>
    severity === 'critical' || severity === 'high' ? 'danger' : severity === 'medium' ? 'warning' : 'neutral';

  return (
    <div>
      <PageHeader
        title="Security Center"
        subtitle="Transactions our fraud model flagged, and the reasoning behind each one."
      />

      {/* ------------------------------ posture ------------------------------ */}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Needs your review"
          value={summary?.open_alerts ?? 0}
          icon={<Flag className="h-4 w-4" aria-hidden />}
          tone={summary?.open_alerts ? 'danger' : 'success'}
        />
        <StatTile
          label="Total alerts"
          value={summary?.total_alerts ?? 0}
          icon={<ShieldAlert className="h-4 w-4" aria-hidden />}
        />
        <StatTile
          label="Blocked transactions"
          value={summary?.blocked_transactions ?? 0}
          hint="Stopped before money moved"
          icon={<XCircle className="h-4 w-4" aria-hidden />}
          tone="warning"
        />
        <StatTile
          label="Two-factor auth"
          value={summary?.two_factor_enabled ? 'On' : 'Off'}
          hint={summary?.two_factor_enabled ? 'Your account is protected' : 'Enable it in Settings'}
          icon={<Fingerprint className="h-4 w-4" aria-hidden />}
          tone={summary?.two_factor_enabled ? 'success' : 'warning'}
        />
      </div>

      {!user?.two_factor_enabled && (
        <div className="mb-6">
          <Notice tone="warning" title="Two-factor authentication is off">
            Adding a second factor is the single most effective way to protect your account against
            credential theft.
          </Notice>
        </div>
      )}

      {/* ------------------------------- alerts ------------------------------- */}
      {isLoading ? (
        <LoadingBlock rows={4} label="Loading alerts" />
      ) : error ? (
        <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
      ) : data?.items.length ? (
        <div className="space-y-4">
          {data.items.map((alert) => {
            const needsAction = alert.status === 'open';
            return (
              <Card
                key={alert.id}
                className={clsx(needsAction && 'border-warning/30 bg-warning/[0.03]')}
              >
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs text-muted">{alert.alert_ref}</span>
                      <Badge tone={severityTone(alert.severity)}>{titleCase(alert.severity)}</Badge>
                      <StatusBadge status={alert.status} />
                      {alert.auto_blocked && <Badge tone="danger">Auto-blocked</Badge>}
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
                      {alert.transaction ? dateTime(alert.transaction.occurred_at) : dateTime(alert.created_at)}
                      {alert.transaction?.location_city && ` · ${alert.transaction.location_city}`}
                    </p>

                    {alert.reasons && alert.reasons.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {alert.reasons.slice(0, 3).map((reason) => (
                          <li key={reason} className="flex gap-2 text-xs text-muted">
                            <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-warning" aria-hidden />
                            {reason}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  <div className="w-full shrink-0 sm:w-44">
                    <p className="text-[11px] tracking-wide text-intelligence uppercase">Risk score</p>
                    <p className="tnum text-2xl font-bold text-intelligence">{percent(alert.risk_score, 1)}</p>
                    <div className="mt-2">
                      <Meter value={alert.risk_score} tone={severityTone(alert.severity)} />
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-line pt-4">
                  {needsAction ? (
                    <>
                      <button
                        type="button"
                        className="btn-secondary px-3.5 py-2 text-xs"
                        onClick={() => {
                          setSelected(alert);
                          setNote('');
                          setFormError(null);
                        }}
                      >
                        <ThumbsUp className="h-3.5 w-3.5" aria-hidden />
                        This was me
                      </button>
                      <button
                        type="button"
                        className="btn-danger px-3.5 py-2 text-xs"
                        onClick={() => {
                          setSelected(alert);
                          setNote('');
                          setFormError(null);
                        }}
                      >
                        <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                        I do not recognise this
                      </button>
                    </>
                  ) : (
                    <p className="text-xs text-muted">
                      {alert.customer_response
                        ? `You marked this as "${alert.customer_response}".`
                        : 'Reviewed by our security team.'}
                      {alert.final_label !== null &&
                        ` Final outcome: ${alert.final_label ? 'confirmed fraud' : 'legitimate'}.`}
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
                    View details
                  </button>
                </div>
              </Card>
            );
          })}

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
          title="No security alerts"
          description="Nothing on your account has been flagged. We will notify you the moment something looks unusual."
        />
      )}

      {/* ------------------------------ respond ------------------------------ */}
      <Modal
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={`Alert ${selected?.alert_ref ?? ''}`}
        wide
        footer={
          selected?.status === 'open' ? (
            <>
              <button
                type="button"
                className="btn-secondary px-4 py-2"
                onClick={() => selected && respond.mutate({ alert: selected, response: 'confirmed' })}
                disabled={respond.isPending}
              >
                <CheckCircle2 className="h-4 w-4" aria-hidden />
                This was me
              </button>
              <button
                type="button"
                className="btn-danger px-4 py-2"
                onClick={() => selected && respond.mutate({ alert: selected, response: 'disputed' })}
                disabled={respond.isPending}
              >
                <ShieldAlert className="h-4 w-4" aria-hidden />
                Report as fraud
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
            {formError && <Notice tone="danger">{formError}</Notice>}

            <div className="flex items-center justify-between gap-3 rounded-lg border border-line bg-ink/50 p-4">
              <div>
                <p className="text-xs text-muted">Transaction</p>
                <p className="text-lg font-bold text-primary">
                  {selected.transaction ? money(selected.transaction.amount) : '—'}
                </p>
                <p className="mt-0.5 text-xs text-muted">
                  {selected.transaction?.description ??
                    selected.transaction?.merchant_name ??
                    'Flagged activity'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-intelligence">Risk score</p>
                <p className="tnum text-lg font-bold text-intelligence">{percent(selected.risk_score, 2)}</p>
                <Badge tone={severityTone(selected.severity)}>{titleCase(selected.severity)}</Badge>
              </div>
            </div>

            {selected.reasons && selected.reasons.length > 0 && (
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-intelligence uppercase">
                  <Sparkles className="h-3 w-3" aria-hidden />
                  Why this was flagged
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
                <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-intelligence uppercase">
                  <Sparkles className="h-3 w-3" aria-hidden />
                  Model factors
                </p>
                <ul className="space-y-1.5">
                  {selected.top_factors.slice(0, 6).map((factor) => (
                    <li key={factor.feature} className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-muted">{factor.label}</span>
                      <Badge tone={factor.direction === 'increases risk' ? 'danger' : 'success'}>
                        {factor.direction === 'increases risk' ? 'Raised risk' : 'Lowered risk'}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <p className="tnum text-[11px] text-faint">
              {selected.model_name} v{selected.model_version}
              {selected.inference_latency_ms !== null &&
                ` · scored in ${latency(selected.inference_latency_ms)}`}
            </p>

            {selected.status === 'open' && (
              <>
                <Field label="Add a note" htmlFor="alertNote" hint="Optional — helps our team review faster">
                  <textarea
                    id="alertNote"
                    rows={3}
                    className="input resize-none"
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    maxLength={1000}
                    placeholder="I was travelling and made this purchase myself."
                  />
                </Field>

                <Notice tone="info">
                  Reporting this as fraud will hold the funds and escalate it to our security team
                  for review.
                </Notice>
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
