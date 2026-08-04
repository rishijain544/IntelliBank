import { useQuery } from '@tanstack/react-query';
import {
  Download,
  FileText,
  Filter,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import { useState } from 'react';

import {
  Badge,
  Card,
  EmptyState,
  ErrorBlock,
  Field,
  LoadingBlock,
  Modal,
  PageHeader,
  Pagination,
  StatusBadge,
} from '../../components/ui';
import { downloadFile, errorMessage, get } from '../../lib/api';
import {
  categoryLabel,
  dateTime,
  money,
  moneySigned,
  percent,
  titleCase,
} from '../../lib/format';
import { qk } from '../../lib/query';
import type { Account, Page, Transaction } from '../../types/api';

const CATEGORIES = [
  'groceries', 'dining', 'transport', 'shopping', 'utilities', 'entertainment',
  'healthcare', 'education', 'travel', 'rent', 'investment', 'cash', 'transfer', 'other',
];

const STATUSES = ['completed', 'pending', 'held', 'blocked', 'failed', 'reversed'];

interface Filters {
  account_id?: string;
  status?: string;
  category?: string;
  min_amount?: string;
  max_amount?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
  flagged_only?: boolean;
}

const EMPTY_FILTERS: Filters = {};

export default function Transactions() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [draft, setDraft] = useState<Filters>(EMPTY_FILTERS);
  const [showFilters, setShowFilters] = useState(false);
  const [detail, setDetail] = useState<Transaction | null>(null);
  const [downloading, setDownloading] = useState<'csv' | 'pdf' | null>(null);

  const { data: accounts } = useQuery({
    queryKey: qk.accounts,
    queryFn: () => get<Account[]>('/accounts'),
    staleTime: 5 * 60_000,
  });

  // Strip empty values so the query key stays stable and the URL stays clean.
  const params: Record<string, unknown> = { page, page_size: 20 };
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '' && value !== false) params[key] = value;
  });

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: qk.transactions(params),
    queryFn: () => get<Page<Transaction>>('/transactions', { params }),
  });

  function applyFilters() {
    setFilters(draft);
    setPage(1);
    setShowFilters(false);
  }

  function clearFilters() {
    setDraft(EMPTY_FILTERS);
    setFilters(EMPTY_FILTERS);
    setPage(1);
    setShowFilters(false);
  }

  const activeCount = Object.values(filters).filter((v) => v !== undefined && v !== '' && v !== false).length;

  async function handleExport(kind: 'csv' | 'pdf') {
    setDownloading(kind);
    try {
      const exportParams: Record<string, unknown> = {};
      if (filters.account_id) exportParams.account_id = filters.account_id;
      if (filters.date_from) exportParams.date_from = filters.date_from;
      if (filters.date_to) exportParams.date_to = filters.date_to;
      if (kind === 'csv' && filters.category) exportParams.category = filters.category;

      await downloadFile(
        `/transactions/export/${kind}`,
        kind === 'csv' ? 'transactions.csv' : 'statement.pdf',
        { params: exportParams },
      );
    } catch {
      // The download helper throws on network failure; the button simply resets.
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div>
      <PageHeader
        title="Transactions"
        subtitle="Search, filter and export your full transaction history."
        action={
          <div className="flex gap-2">
            <button
              type="button"
              className="btn-secondary px-3.5 py-2.5"
              onClick={() => void handleExport('csv')}
              disabled={downloading !== null}
            >
              <Download className="h-4 w-4" aria-hidden />
              {downloading === 'csv' ? 'Preparing…' : 'CSV'}
            </button>
            <button
              type="button"
              className="btn-secondary px-3.5 py-2.5"
              onClick={() => void handleExport('pdf')}
              disabled={downloading !== null}
            >
              <FileText className="h-4 w-4" aria-hidden />
              {downloading === 'pdf' ? 'Preparing…' : 'Statement'}
            </button>
          </div>
        }
      />

      {/* --------------------------- search + filter --------------------------- */}
      <div className="mb-5 flex flex-wrap gap-2">
        <form
          className="relative min-w-56 flex-1"
          onSubmit={(e) => {
            e.preventDefault();
            setFilters((f) => ({ ...f, search: draft.search }));
            setPage(1);
          }}
        >
          <Search className="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            type="search"
            className="input pl-9"
            placeholder="Search description, merchant or reference…"
            value={draft.search ?? ''}
            onChange={(e) => setDraft((d) => ({ ...d, search: e.target.value }))}
            aria-label="Search transactions"
          />
        </form>

        <button
          type="button"
          className="btn-secondary px-3.5 py-2.5"
          onClick={() => setShowFilters(true)}
        >
          <Filter className="h-4 w-4" aria-hidden />
          Filters
          {activeCount > 0 && <Badge tone="info">{activeCount}</Badge>}
        </button>

        {activeCount > 0 && (
          <button type="button" className="btn-ghost px-3 py-2.5" onClick={clearFilters}>
            <X className="h-4 w-4" aria-hidden />
            Clear
          </button>
        )}
      </div>

      {/* ------------------------------- table ------------------------------- */}
      <Card className="p-0">
        {isLoading ? (
          <div className="p-5">
            <LoadingBlock rows={8} label="Loading transactions" />
          </div>
        ) : error ? (
          <div className="p-5">
            <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />
          </div>
        ) : !data?.items.length ? (
          <div className="p-5">
            <EmptyState
              title="No transactions found"
              description={activeCount > 0 ? 'Try widening or clearing your filters.' : 'Transactions will appear here once you start using your account.'}
              action={
                activeCount > 0 ? (
                  <button type="button" className="btn-secondary px-4 py-2" onClick={clearFilters}>
                    Clear filters
                  </button>
                ) : undefined
              }
            />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Transaction history</caption>
                <thead className="table-header">
                  <tr>
                    <th scope="col" className="px-4 py-3">Date</th>
                    <th scope="col" className="px-4 py-3">Description</th>
                    <th scope="col" className="hidden px-4 py-3 md:table-cell">Category</th>
                    <th scope="col" className="hidden px-4 py-3 lg:table-cell">Status</th>
                    <th scope="col" className="px-4 py-3 text-right">Amount</th>
                    <th scope="col" className="hidden px-4 py-3 text-right sm:table-cell">Balance</th>
                  </tr>
                </thead>
                <tbody className={`divide-y divide-line ${isFetching ? 'opacity-60' : ''}`}>
                  {data.items.map((txn) => {
                    const amount = Number.parseFloat(txn.signed_amount);
                    return (
                      <tr
                        key={txn.id}
                        className="cursor-pointer transition hover:bg-surface-raised/60"
                        onClick={() => setDetail(txn)}
                      >
                        <td className="px-4 py-3 whitespace-nowrap text-xs text-muted">
                          {dateTime(txn.occurred_at)}
                        </td>
                        <td className="max-w-64 px-4 py-3">
                          <p className="truncate font-medium text-primary">
                            {txn.description ?? txn.merchant_name ?? titleCase(txn.merchant_category)}
                          </p>
                          <p className="font-mono text-[11px] text-muted">{txn.reference}</p>
                        </td>
                        <td className="hidden px-4 py-3 text-muted md:table-cell">
                          {categoryLabel(txn.merchant_category)}
                        </td>
                        <td className="hidden px-4 py-3 lg:table-cell">
                          <div className="flex items-center gap-1.5">
                            <StatusBadge status={txn.status} />
                            {txn.is_flagged && <Badge tone="warning">Flagged</Badge>}
                          </div>
                        </td>
                        <td
                          className={`tnum px-4 py-3 text-right font-semibold whitespace-nowrap ${
                            amount >= 0 ? 'text-positive' : 'text-primary'
                          }`}
                        >
                          {moneySigned(txn.signed_amount)}
                        </td>
                        <td className="tnum hidden px-4 py-3 text-right text-muted sm:table-cell">
                          {money(txn.balance_after)}
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
          </>
        )}
      </Card>

      {/* ------------------------------ filters ------------------------------ */}
      <Modal
        open={showFilters}
        onClose={() => setShowFilters(false)}
        title="Filter transactions"
        wide
        footer={
          <>
            <button type="button" className="btn-ghost px-4 py-2" onClick={clearFilters}>
              Clear all
            </button>
            <button type="button" className="btn-primary px-4 py-2" onClick={applyFilters}>
              Apply filters
            </button>
          </>
        }
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Account" htmlFor="fAccount">
            <select
              id="fAccount"
              className="input"
              value={draft.account_id ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, account_id: e.target.value || undefined }))}
            >
              <option value="">All accounts</option>
              {accounts?.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.nickname ?? titleCase(a.account_type)} · {a.account_number.slice(-4)}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Status" htmlFor="fStatus">
            <select
              id="fStatus"
              className="input"
              value={draft.status ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, status: e.target.value || undefined }))}
            >
              <option value="">Any status</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>{titleCase(s)}</option>
              ))}
            </select>
          </Field>

          <Field label="Category" htmlFor="fCategory">
            <select
              id="fCategory"
              className="input"
              value={draft.category ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, category: e.target.value || undefined }))}
            >
              <option value="">All categories</option>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{categoryLabel(c)}</option>
              ))}
            </select>
          </Field>

          <Field label="Minimum amount" htmlFor="fMin">
            <input
              id="fMin"
              inputMode="decimal"
              className="input tnum"
              value={draft.min_amount ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, min_amount: e.target.value.replace(/[^\d.]/g, '') || undefined }))}
              placeholder="0"
            />
          </Field>

          <Field label="Maximum amount" htmlFor="fMax">
            <input
              id="fMax"
              inputMode="decimal"
              className="input tnum"
              value={draft.max_amount ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, max_amount: e.target.value.replace(/[^\d.]/g, '') || undefined }))}
              placeholder="No limit"
            />
          </Field>

          <Field label="From date" htmlFor="fFrom">
            <input
              id="fFrom"
              type="date"
              className="input"
              value={draft.date_from ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, date_from: e.target.value || undefined }))}
            />
          </Field>

          <Field label="To date" htmlFor="fTo">
            <input
              id="fTo"
              type="date"
              className="input"
              value={draft.date_to ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, date_to: e.target.value || undefined }))}
            />
          </Field>

          <div className="flex items-end">
            <label className="flex cursor-pointer items-center gap-2.5 text-sm text-primary">
              <input
                type="checkbox"
                className="h-4 w-4 accent-gold"
                checked={draft.flagged_only ?? false}
                onChange={(e) => setDraft((d) => ({ ...d, flagged_only: e.target.checked || undefined }))}
              />
              Only show flagged transactions
            </label>
          </div>
        </div>
      </Modal>

      {/* ------------------------------- detail ------------------------------- */}
      <Modal
        open={detail !== null}
        onClose={() => setDetail(null)}
        title="Transaction detail"
        footer={
          <button type="button" className="btn-secondary px-4 py-2" onClick={() => setDetail(null)}>
            Close
          </button>
        }
      >
        {detail && (
          <div className="space-y-4">
            <div className="text-center">
              <p
                className={`tnum text-3xl font-bold ${
                  Number.parseFloat(detail.signed_amount) >= 0 ? 'text-positive' : 'text-primary'
                }`}
              >
                {moneySigned(detail.signed_amount)}
              </p>
              <p className="mt-1 text-sm text-muted">
                {detail.description ?? detail.merchant_name ?? titleCase(detail.merchant_category)}
              </p>
              <div className="mt-2 flex items-center justify-center gap-1.5">
                <StatusBadge status={detail.status} />
                {detail.is_flagged && <Badge tone="warning">Flagged for review</Badge>}
              </div>
            </div>

            <dl className="divide-y divide-line text-sm">
              {[
                ['Reference', <span className="font-mono text-xs">{detail.reference}</span>],
                ['Date', dateTime(detail.occurred_at)],
                ['Type', titleCase(detail.txn_type)],
                ['Channel', detail.channel.toUpperCase()],
                ['Category', categoryLabel(detail.merchant_category)],
                ...(detail.merchant_name ? [['Merchant', detail.merchant_name]] : []),
                ...(detail.counterparty_name ? [['Counterparty', detail.counterparty_name]] : []),
                ...(detail.counterparty_account_number
                  ? [['To account', <span className="font-mono text-xs">{detail.counterparty_account_number}</span>]]
                  : []),
                ...(Number.parseFloat(detail.fee) > 0 ? [['Fee', money(detail.fee)]] : []),
                ...(detail.balance_after ? [['Balance after', money(detail.balance_after)]] : []),
                ...(detail.location_city ? [['Location', detail.location_city]] : []),
              ].map(([label, value], i) => (
                <div key={i} className="flex items-center justify-between gap-3 py-2.5">
                  <dt className="text-muted">{label}</dt>
                  <dd className="text-right text-primary">{value}</dd>
                </div>
              ))}
            </dl>

            {/* Model output is shown to the customer for transparency. */}
            {(detail.fraud_score !== null || detail.anomaly_score !== null) && (
              <div className="rounded-lg border border-line bg-ink/50 p-3.5">
                <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-intelligence uppercase">
                  <Sparkles className="h-3 w-3" aria-hidden />
                  Risk assessment
                </p>
                <div className="space-y-1.5 text-xs">
                  {detail.fraud_score !== null && (
                    <div className="flex justify-between">
                      <span className="text-muted">Fraud risk score</span>
                      <span className="tnum text-primary">{percent(detail.fraud_score, 2)}</span>
                    </div>
                  )}
                  {detail.anomaly_score !== null && (
                    <div className="flex justify-between">
                      <span className="text-muted">Unusualness score</span>
                      <span className="tnum text-primary">{percent(detail.anomaly_score, 2)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
