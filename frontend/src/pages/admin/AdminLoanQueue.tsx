import { useMutation, useQuery } from '@tanstack/react-query';
import { Ban, Check, Gauge, Landmark, Pencil } from 'lucide-react';
import { useState } from 'react';

import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  Field,
  LoadingBlock,
  Modal,
  Notice,
  PageHeader,
  Pagination,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { dateTime, money, percentRaw, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import type { Loan, Page } from '../../types/api';

/**
 * Loan approval queue.
 *
 * The model has already scored every application; this screen exists for the
 * cases it routed to manual review, plus the ability to override it. Overrides
 * are recorded explicitly (`manual_override`) so the audit trail distinguishes a
 * model decision from a human one — important when a portfolio is later analysed
 * for which decisions the model actually made.
 */
export default function AdminLoanQueue() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [selected, setSelected] = useState<Loan | null>(null);
  const [approvedAmount, setApprovedAmount] = useState('');
  const [interestRate, setInterestRate] = useState('');
  const [note, setNote] = useState('');
  const [override, setOverride] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const params: Record<string, unknown> = { page, page_size: 15 };
  if (statusFilter) params.status = statusFilter;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.admin.loanQueue(params),
    queryFn: () => get<Page<Loan>>('/admin/loans/queue', { params }),
  });

  const decide = useMutation({
    mutationFn: ({ loan, decision }: { loan: Loan; decision: 'approve' | 'reject' }) =>
      post<Loan>(`/admin/loans/queue/${loan.id}/decide`, {
        decision,
        approved_amount: decision === 'approve' && approvedAmount ? approvedAmount : null,
        interest_rate: decision === 'approve' && interestRate ? Number(interestRate) : null,
        note: note.trim() || null,
        override_model: override,
      }),
    onSuccess: async () => {
      setSelected(null);
      setNote('');
      setApprovedAmount('');
      setInterestRate('');
      setOverride(false);
      setActionError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['admin', 'loans'] }),
        queryClient.invalidateQueries({ queryKey: qk.admin.stats }),
      ]);
    },
    onError: (err) => setActionError(errorMessage(err)),
  });

  function openReview(loan: Loan) {
    setSelected(loan);
    // Pre-fill with the model's own recommendation so an unchanged approval
    // keeps the model's pricing rather than silently zeroing it.
    setApprovedAmount(loan.approved_amount ?? loan.requested_amount);
    setInterestRate(loan.interest_rate ? String(loan.interest_rate) : '');
    setNote('');
    setOverride(false);
    setActionError(null);
  }

  return (
    <div>
      <PageHeader
        title="Loan approval queue"
        subtitle="Applications the credit model routed to manual review."
      />

      {actionError && (
        <div className="mb-5">
          <Notice tone="danger">{actionError}</Notice>
        </div>
      )}

      <div className="mb-5">
        <select
          className="input w-auto"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(1);
          }}
          aria-label="Filter by status"
        >
          <option value="">Pending review</option>
          <option value="submitted">Submitted</option>
          <option value="under_review">Under review</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="disbursed">Disbursed</option>
        </select>
      </div>

      {isLoading ? (
        <LoadingBlock rows={5} label="Loading loan queue" />
      ) : error ? (
        <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
      ) : data?.items.length ? (
        <Card className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">Loan applications awaiting decision</caption>
              <thead className="table-header">
                <tr>
                  <th scope="col" className="px-4 py-3">Application</th>
                  <th scope="col" className="px-4 py-3">Type</th>
                  <th scope="col" className="px-4 py-3 text-right">Requested</th>
                  <th scope="col" className="hidden px-4 py-3 text-right sm:table-cell">Tenure</th>
                  <th scope="col" className="hidden px-4 py-3 text-right md:table-cell">Model rate</th>
                  <th scope="col" className="px-4 py-3">Status</th>
                  <th scope="col" className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {data.items.map((loan) => (
                  <tr key={loan.id} className="transition hover:bg-surface-raised/60">
                    <td className="px-4 py-3">
                      <p className="font-mono text-xs text-primary">{loan.application_ref}</p>
                      <p className="text-[11px] text-muted">{dateTime(loan.created_at)}</p>
                    </td>
                    <td className="px-4 py-3 text-primary">{titleCase(loan.loan_type)}</td>
                    <td className="tnum px-4 py-3 text-right text-primary">
                      {money(loan.requested_amount)}
                    </td>
                    <td className="tnum hidden px-4 py-3 text-right text-muted sm:table-cell">
                      {loan.tenure_months} mo
                    </td>
                    <td className="tnum hidden px-4 py-3 text-right text-muted md:table-cell">
                      {loan.interest_rate ? percentRaw(loan.interest_rate) : '—'}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1.5">
                        <StatusBadge status={loan.status} />
                        {loan.manual_override && <Badge tone="warning">Override</Badge>}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        className="btn-ghost px-2.5 py-1 text-xs"
                        onClick={() => openReview(loan)}
                      >
                        <Pencil className="h-3.5 w-3.5" aria-hidden />
                        Decide
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Pagination
            page={data.page}
            totalPages={data.total_pages}
            total={data.total}
            pageSize={data.page_size}
            onPageChange={setPage}
          />
        </Card>
      ) : (
        <EmptyState
          icon={<Landmark className="h-10 w-10" aria-hidden />}
          title="No applications pending"
          description="Applications routed to manual review will appear here."
        />
      )}

      {/* ------------------------------ decide ------------------------------ */}
      <Modal
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={`Application ${selected?.application_ref ?? ''}`}
        wide
        footer={
          selected && !['disbursed', 'closed'].includes(selected.status) ? (
            <>
              <button
                type="button"
                className="btn-secondary px-4 py-2"
                onClick={() => setSelected(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-ghost px-4 py-2 text-alert"
                onClick={() => selected && decide.mutate({ loan: selected, decision: 'reject' })}
                disabled={decide.isPending}
              >
                <Ban className="h-4 w-4" aria-hidden />
                Reject
              </button>
              <button
                type="button"
                className="btn-primary px-4 py-2"
                onClick={() => selected && decide.mutate({ loan: selected, decision: 'approve' })}
                disabled={decide.isPending}
              >
                <Check className="h-4 w-4" aria-hidden />
                Approve
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
                <p className="text-[11px] tracking-wide text-muted uppercase">Requested</p>
                <p className="tnum mt-0.5 font-bold text-primary">{money(selected.requested_amount)}</p>
              </div>
              <div className="rounded-lg border border-line bg-ink/50 p-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">Tenure</p>
                <p className="tnum mt-0.5 font-bold text-primary">{selected.tenure_months} mo</p>
              </div>
              <div className="rounded-lg border border-line bg-ink/50 p-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">Model rate</p>
                <p className="tnum mt-0.5 font-bold text-primary">
                  {selected.interest_rate ? percentRaw(selected.interest_rate) : '—'}
                </p>
              </div>
            </div>

            <dl className="divide-y divide-line text-sm">
              {[
                ['Type', titleCase(selected.loan_type)],
                ...(selected.purpose ? [['Purpose', selected.purpose]] : []),
                ...(selected.emi_amount ? [['Model EMI', money(selected.emi_amount)]] : []),
                ...(selected.total_payable ? [['Total payable', money(selected.total_payable)]] : []),
                ...(selected.processing_fee ? [['Processing fee', money(selected.processing_fee)]] : []),
                ['Decision source', titleCase(selected.decision_source)],
                ['Applied', dateTime(selected.created_at)],
              ].map(([label, value], i) => (
                <div key={i} className="flex items-center justify-between gap-3 py-2.5">
                  <dt className="text-muted">{label}</dt>
                  <dd className="tnum text-right text-primary">{value}</dd>
                </div>
              ))}
            </dl>

            {selected.decision_reason && (
              <Notice tone="info" title="Model reasoning">
                {selected.decision_reason}
              </Notice>
            )}

            {!['disbursed', 'closed'].includes(selected.status) && (
              <div className="space-y-4 border-t border-line pt-4">
                <p className="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-muted uppercase">
                  <Gauge className="h-3.5 w-3.5" aria-hidden />
                  Adjust the offer
                </p>

                <div className="grid gap-4 sm:grid-cols-2">
                  <Field
                    label="Approved amount"
                    htmlFor="approvedAmount"
                    hint="Cannot exceed the requested amount"
                  >
                    <input
                      id="approvedAmount"
                      inputMode="decimal"
                      className="input tnum"
                      value={approvedAmount}
                      onChange={(e) => setApprovedAmount(e.target.value.replace(/[^\d.]/g, ''))}
                    />
                  </Field>
                  <Field label="Interest rate (%)" htmlFor="interestRate">
                    <input
                      id="interestRate"
                      inputMode="decimal"
                      className="input tnum"
                      value={interestRate}
                      onChange={(e) => setInterestRate(e.target.value.replace(/[^\d.]/g, ''))}
                    />
                  </Field>
                </div>

                <Field label="Decision note" htmlFor="loanNote" hint="Shown to the applicant">
                  <textarea
                    id="loanNote"
                    rows={3}
                    className="input resize-none"
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    maxLength={1000}
                    placeholder="Approved at a reduced amount given the debt-to-income ratio."
                  />
                </Field>

                <label className="flex cursor-pointer items-start gap-2.5 text-sm text-primary">
                  <input
                    type="checkbox"
                    className="mt-0.5 h-4 w-4 accent-warning"
                    checked={override}
                    onChange={(e) => setOverride(e.target.checked)}
                  />
                  <span>
                    Mark as a model override
                    <span className="block text-xs text-muted">
                      Flags this decision as diverging from the model's recommendation, so later
                      portfolio analysis can separate human calls from model calls.
                    </span>
                  </span>
                </label>

                <p className="text-xs text-faint">
                  EMI and total payable are recalculated server-side from the amount and rate you
                  set, so a manual change cannot leave stale pricing on the record.
                </p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
