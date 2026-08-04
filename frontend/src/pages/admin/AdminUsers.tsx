import { useMutation, useQuery } from '@tanstack/react-query';
import { Ban, Check, Search, ShieldCheck, UserCog, X } from 'lucide-react';
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
import { errorMessage, get, patch } from '../../lib/api';
import { dateTime, money, num, timeAgo, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import { useAuth } from '../../store/auth';
import type { Page, User, UserSummary } from '../../types/api';

interface UserDetail {
  user: User;
  accounts: { id: number; account_number: string; account_type: string; status: string; balance: number }[];
  total_balance: number;
  txn_count: number;
  open_fraud_alerts: number;
  loans: number;
  latest_credit_score: {
    score: number;
    risk_band: string;
    probability_of_default: number;
    created_at: string;
  } | null;
}

export default function AdminUsers() {
  const currentAdmin = useAuth((s) => s.user);

  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [searchDraft, setSearchDraft] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [kycFilter, setKycFilter] = useState('');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reason, setReason] = useState('');

  const params: Record<string, unknown> = { page, page_size: 20 };
  if (search) params.search = search;
  if (statusFilter) params.status = statusFilter;
  if (kycFilter) params.kyc_status = kycFilter;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.admin.users(params),
    queryFn: () => get<Page<UserSummary>>('/admin/users', { params }),
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: qk.admin.user(selectedId ?? 0),
    queryFn: () => get<UserDetail>(`/admin/users/${selectedId}`),
    enabled: selectedId !== null,
  });

  async function refreshAll() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin', 'users'] }),
      queryClient.invalidateQueries({ queryKey: qk.admin.stats }),
    ]);
  }

  const setStatus = useMutation({
    mutationFn: ({ userId, status }: { userId: number; status: string }) =>
      patch<User>(`/admin/users/${userId}/status`, { status, reason: reason.trim() || null }),
    onSuccess: async () => {
      setReason('');
      setActionError(null);
      await refreshAll();
    },
    onError: (err) => setActionError(errorMessage(err)),
  });

  const decideKyc = useMutation({
    mutationFn: ({ userId, decision }: { userId: number; decision: 'verify' | 'reject' }) =>
      patch<User>(`/admin/users/${userId}/kyc`, { decision, reason: reason.trim() || null }),
    onSuccess: async () => {
      setReason('');
      setActionError(null);
      await refreshAll();
    },
    onError: (err) => setActionError(errorMessage(err)),
  });

  return (
    <div>
      <PageHeader
        title="User management"
        subtitle="Search customers, review KYC and freeze or reactivate accounts."
      />

      {actionError && (
        <div className="mb-5">
          <Notice tone="danger">{actionError}</Notice>
        </div>
      )}

      {/* ------------------------------ filters ------------------------------ */}
      <div className="mb-5 flex flex-wrap gap-2">
        <form
          className="relative min-w-56 flex-1"
          onSubmit={(e) => {
            e.preventDefault();
            setSearch(searchDraft);
            setPage(1);
          }}
        >
          <Search className="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            type="search"
            className="input pl-9"
            placeholder="Search by name, email or phone…"
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            aria-label="Search users"
          />
        </form>

        <select
          className="input w-auto"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(1);
          }}
          aria-label="Filter by status"
        >
          <option value="">Any status</option>
          <option value="active">Active</option>
          <option value="pending">Pending</option>
          <option value="frozen">Frozen</option>
          <option value="suspended">Suspended</option>
        </select>

        <select
          className="input w-auto"
          value={kycFilter}
          onChange={(e) => {
            setKycFilter(e.target.value);
            setPage(1);
          }}
          aria-label="Filter by KYC status"
        >
          <option value="">Any KYC</option>
          <option value="verified">Verified</option>
          <option value="submitted">Submitted</option>
          <option value="not_started">Not started</option>
          <option value="rejected">Rejected</option>
        </select>

        {(search || statusFilter || kycFilter) && (
          <button
            type="button"
            className="btn-ghost px-3 py-2.5"
            onClick={() => {
              setSearch('');
              setSearchDraft('');
              setStatusFilter('');
              setKycFilter('');
              setPage(1);
            }}
          >
            <X className="h-4 w-4" aria-hidden />
            Clear
          </button>
        )}
      </div>

      {/* ------------------------------- table ------------------------------- */}
      <Card className="p-0">
        {isLoading ? (
          <div className="p-5">
            <LoadingBlock rows={8} label="Loading users" />
          </div>
        ) : error ? (
          <div className="p-5">
            <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
          </div>
        ) : !data?.items.length ? (
          <div className="p-5">
            <EmptyState title="No users found" description="Try adjusting your search or filters." />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Registered users</caption>
                <thead className="table-header">
                  <tr>
                    <th scope="col" className="px-4 py-3">User</th>
                    <th scope="col" className="hidden px-4 py-3 md:table-cell">Role</th>
                    <th scope="col" className="px-4 py-3">Status</th>
                    <th scope="col" className="hidden px-4 py-3 sm:table-cell">KYC</th>
                    <th scope="col" className="hidden px-4 py-3 lg:table-cell">Last sign-in</th>
                    <th scope="col" className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.items.map((row) => (
                    <tr key={row.id} className="transition hover:bg-surface-raised/60">
                      <td className="px-4 py-3">
                        <p className="font-medium text-primary">{row.full_name}</p>
                        <p className="text-xs text-muted">{row.email}</p>
                      </td>
                      <td className="hidden px-4 py-3 md:table-cell">
                        {row.role === 'admin' ? (
                          <Badge tone="warning">Admin</Badge>
                        ) : (
                          <span className="text-muted">Customer</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={row.status} />
                      </td>
                      <td className="hidden px-4 py-3 sm:table-cell">
                        <StatusBadge status={row.kyc_status} />
                      </td>
                      <td className="tnum hidden px-4 py-3 text-xs text-muted lg:table-cell">
                        {row.last_login_at ? timeAgo(row.last_login_at) : 'Never'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          className="btn-ghost px-2.5 py-1 text-xs"
                          onClick={() => {
                            setSelectedId(row.id);
                            setReason('');
                            setActionError(null);
                          }}
                        >
                          <UserCog className="h-3.5 w-3.5" aria-hidden />
                          Manage
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
          </>
        )}
      </Card>

      {/* ------------------------------- detail ------------------------------- */}
      <Modal
        open={selectedId !== null}
        onClose={() => setSelectedId(null)}
        title="Manage user"
        wide
        footer={
          <button type="button" className="btn-secondary px-4 py-2" onClick={() => setSelectedId(null)}>
            Close
          </button>
        }
      >
        {detailLoading ? (
          <LoadingBlock rows={4} />
        ) : detail ? (
          <div className="space-y-5">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-semibold text-primary">{detail.user.full_name}</h3>
                <StatusBadge status={detail.user.status} />
                <StatusBadge status={detail.user.kyc_status} />
                {detail.user.role === 'admin' && <Badge tone="warning">Administrator</Badge>}
              </div>
              <p className="mt-1 text-sm text-muted">{detail.user.email}</p>
              <p className="tnum text-xs text-muted">Joined {dateTime(detail.user.created_at)}</p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ['Balance', money(detail.total_balance)],
                ['Transactions', num(detail.txn_count)],
                ['Open alerts', num(detail.open_fraud_alerts)],
                ['Loans', num(detail.loans)],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-line bg-ink/40 p-3">
                  <p className="text-[11px] tracking-wide text-muted uppercase">{label}</p>
                  <p className="tnum mt-0.5 font-semibold text-primary">{value}</p>
                </div>
              ))}
            </div>

            {detail.latest_credit_score && (
              <div className="rounded-lg border border-line bg-ink/40 p-3.5">
                <p className="mb-1.5 text-xs font-semibold tracking-wide text-muted uppercase">
                  Latest credit score
                </p>
                <div className="flex items-center gap-3">
                  <span className="tnum text-2xl font-bold text-primary">
                    {detail.latest_credit_score.score}
                  </span>
                  <Badge tone="info">Band {detail.latest_credit_score.risk_band}</Badge>
                  <span className="tnum text-xs text-muted">
                    scored {timeAgo(detail.latest_credit_score.created_at)}
                  </span>
                </div>
              </div>
            )}

            {detail.accounts.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">
                  Accounts
                </p>
                <ul className="divide-y divide-line">
                  {detail.accounts.map((account) => (
                    <li key={account.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                      <div>
                        <p className="font-mono text-xs text-primary">{account.account_number}</p>
                        <p className="text-xs text-muted">{titleCase(account.account_type)}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <StatusBadge status={account.status} />
                        <span className="tnum font-medium text-primary">{money(account.balance)}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* ---------------------------- actions ---------------------------- */}
            <div className="space-y-3 border-t border-line pt-4">
              {actionError && <Notice tone="danger">{actionError}</Notice>}

              {detail.user.id === currentAdmin?.id ? (
                <Notice tone="info">
                  This is your own account. Status changes to your own profile are blocked to prevent
                  locking yourself out.
                </Notice>
              ) : (
                <>
                  <Field
                    label="Reason"
                    htmlFor="actionReason"
                    hint="Recorded in the audit trail and shown to the customer"
                  >
                    <input
                      id="actionReason"
                      className="input"
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      maxLength={500}
                      placeholder="Suspicious activity reported"
                    />
                  </Field>

                  <div className="flex flex-wrap gap-2">
                    {detail.user.status === 'active' ? (
                      <button
                        type="button"
                        className="btn-danger px-3.5 py-2 text-xs"
                        onClick={() => setStatus.mutate({ userId: detail.user.id, status: 'frozen' })}
                        disabled={setStatus.isPending}
                      >
                        <Ban className="h-3.5 w-3.5" aria-hidden />
                        Freeze account
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="btn-secondary px-3.5 py-2 text-xs"
                        onClick={() => setStatus.mutate({ userId: detail.user.id, status: 'active' })}
                        disabled={setStatus.isPending}
                      >
                        <Check className="h-3.5 w-3.5" aria-hidden />
                        Reactivate
                      </button>
                    )}

                    {detail.user.kyc_status !== 'verified' && (
                      <button
                        type="button"
                        className="btn-secondary px-3.5 py-2 text-xs"
                        onClick={() => decideKyc.mutate({ userId: detail.user.id, decision: 'verify' })}
                        disabled={decideKyc.isPending}
                      >
                        <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
                        Approve KYC
                      </button>
                    )}

                    {detail.user.kyc_status !== 'rejected' && (
                      <button
                        type="button"
                        className="btn-ghost px-3.5 py-2 text-xs"
                        onClick={() => decideKyc.mutate({ userId: detail.user.id, decision: 'reject' })}
                        disabled={decideKyc.isPending}
                      >
                        <X className="h-3.5 w-3.5" aria-hidden />
                        Reject KYC
                      </button>
                    )}
                  </div>

                  <p className="text-xs text-faint">
                    Freezing a customer also freezes their accounts, and takes effect on their next
                    request rather than at token expiry.
                  </p>
                </>
              )}
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
