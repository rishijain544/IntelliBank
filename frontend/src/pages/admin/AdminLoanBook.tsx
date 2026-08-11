import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { CalendarClock, Mail, PiggyBank, TriangleAlert } from 'lucide-react';
import { useState } from 'react';

import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  Notice,
  PageHeader,
  Pagination,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { dateShort, money, percentRaw, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import type { LoanBookRow, MessageResponse, Page } from '../../types/api';

/**
 * Active loan portfolio.
 *
 * Distinct from the approval queue: that screen decides applications, this one
 * tracks repayment on loans already disbursed. `next_due_date` and
 * `days_overdue` are computed server-side from the schedule origin and the paid
 * counter, so this component only renders them — recomputing them here would
 * risk a client and server that disagree about what "overdue" means.
 */

/** Buckets mirror how a collections team actually triages, not even intervals. */
function overdueTone(days: number): 'neutral' | 'warning' | 'danger' {
  if (days <= 0) return 'neutral';
  if (days < 30) return 'warning';
  return 'danger';
}

function overdueLabel(days: number): string {
  if (days <= 0) return 'Current';
  return `${days} ${days === 1 ? 'day' : 'days'} overdue`;
}

export default function AdminLoanBook() {
  const [page, setPage] = useState(1);
  const [overdueOnly, setOverdueOnly] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [sent, setSent] = useState<string | null>(null);
  const [remindingId, setRemindingId] = useState<number | null>(null);

  const params: Record<string, unknown> = { page, page_size: 15 };
  if (overdueOnly) params.overdue_only = true;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.admin.loanBook(params),
    queryFn: () => get<Page<LoanBookRow>>('/admin/loans/book', { params }),
  });

  const remind = useMutation({
    mutationFn: (loan: LoanBookRow) =>
      post<MessageResponse>(`/admin/loans/${loan.id}/remind`, {}),
    onMutate: (loan) => {
      setRemindingId(loan.id);
      setActionError(null);
      setSent(null);
    },
    onSuccess: async (result) => {
      setSent(result.message);
      // The server may reject a reminder it considers no longer overdue, so
      // refresh the rows rather than trusting what is already on screen.
      await queryClient.invalidateQueries({ queryKey: ['admin', 'loans'] });
    },
    onError: (err) => setActionError(errorMessage(err)),
    onSettled: () => setRemindingId(null),
  });

  const rows = data?.items ?? [];
  const overdueCount = rows.filter((r) => r.days_overdue > 0).length;

  return (
    <div>
      <PageHeader
        title="Loan book"
        subtitle="Disbursed loans and their repayment position, most overdue first."
      />

      {actionError && (
        <div className="mb-5">
          <Notice tone="danger">{actionError}</Notice>
        </div>
      )}
      {sent && (
        <div className="mb-5">
          <Notice tone="success" title="Reminder sent">
            {sent}
          </Notice>
        </div>
      )}

      {/* ------------------------------ filters ------------------------------ */}
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <select
          className="input w-auto"
          value={overdueOnly ? 'overdue' : 'all'}
          onChange={(e) => {
            setOverdueOnly(e.target.value === 'overdue');
            setPage(1);
          }}
          aria-label="Filter the loan book"
        >
          <option value="all">All disbursed loans</option>
          <option value="overdue">Overdue only</option>
        </select>

        {!isLoading && !error && (
          <p className="text-xs text-muted">
            <span className="tnum font-semibold text-primary">{data?.total ?? 0}</span> loans
            {overdueCount > 0 && (
              <>
                {' · '}
                <span className="tnum font-semibold text-alert">{overdueCount}</span> overdue on
                this page
              </>
            )}
          </p>
        )}
      </div>

      {isLoading ? (
        <LoadingBlock rows={5} label="Loading loan book" />
      ) : error ? (
        <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
      ) : data && rows.length ? (
        <Card className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">Disbursed loans and repayment position</caption>
              <thead className="table-header">
                <tr>
                  <th scope="col" className="px-4 py-3">Borrower</th>
                  <th scope="col" className="px-4 py-3">Loan</th>
                  <th scope="col" className="px-4 py-3 text-right">EMI</th>
                  <th scope="col" className="hidden px-4 py-3 text-right md:table-cell">
                    Outstanding
                  </th>
                  <th scope="col" className="hidden px-4 py-3 text-right sm:table-cell">
                    Progress
                  </th>
                  <th scope="col" className="px-4 py-3">Next due</th>
                  <th scope="col" className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {rows.map((loan) => {
                  const tone = overdueTone(loan.days_overdue);
                  return (
                    <tr key={loan.id} className="transition hover:bg-surface-raised/60">
                      <td className="px-4 py-3">
                        <p className="font-medium text-primary">{loan.borrower_name}</p>
                        <p className="text-[11px] text-muted">{loan.borrower_email}</p>
                      </td>
                      <td className="px-4 py-3">
                        <p className="font-mono text-xs text-primary">{loan.application_ref}</p>
                        <p className="text-[11px] text-muted">
                          {titleCase(loan.loan_type)}
                          {loan.interest_rate ? ` · ${percentRaw(loan.interest_rate)}` : ''}
                        </p>
                      </td>
                      <td className="tnum px-4 py-3 text-right text-primary">
                        {money(loan.emi_amount)}
                      </td>
                      <td className="tnum hidden px-4 py-3 text-right text-primary md:table-cell">
                        {money(loan.outstanding_principal)}
                      </td>
                      <td className="tnum hidden px-4 py-3 text-right text-muted sm:table-cell">
                        {loan.emis_paid}/{loan.tenure_months}
                        {loan.emis_missed > 0 && (
                          <span className="ml-1.5 text-alert">({loan.emis_missed} missed)</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-col items-start gap-1">
                          <span
                            className={clsx(
                              'tnum text-xs',
                              tone === 'danger' ? 'text-alert' : 'text-primary',
                            )}
                          >
                            {loan.next_due_date ? dateShort(loan.next_due_date) : 'Not scheduled'}
                          </span>
                          {loan.days_overdue > 0 ? (
                            <Badge tone={tone}>
                              <TriangleAlert className="h-3 w-3" aria-hidden />
                              {overdueLabel(loan.days_overdue)}
                            </Badge>
                          ) : (
                            <Badge tone="neutral">Current</Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        {loan.days_overdue > 0 ? (
                          <button
                            type="button"
                            className="btn-ghost px-2.5 py-1 text-xs"
                            onClick={() => remind.mutate(loan)}
                            disabled={remind.isPending}
                            aria-label={`Send payment reminder for ${loan.application_ref}`}
                          >
                            <Mail className="h-3.5 w-3.5" aria-hidden />
                            {remindingId === loan.id ? 'Sending…' : 'Remind'}
                          </button>
                        ) : (
                          <span className="text-[11px] text-faint">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
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
          icon={
            overdueOnly ? (
              <CalendarClock className="h-10 w-10" aria-hidden />
            ) : (
              <PiggyBank className="h-10 w-10" aria-hidden />
            )
          }
          title={overdueOnly ? 'Nothing overdue' : 'No disbursed loans'}
          description={
            overdueOnly
              ? 'Every active loan is current on its EMI schedule.'
              : 'Approved applications appear here once they are disbursed.'
          }
        />
      )}

      <p className="mt-4 text-xs text-faint">
        Reminder emails are simulated: the message is composed and logged server-side, and the
        borrower always receives the matching in-app notification.
      </p>
    </div>
  );
}
