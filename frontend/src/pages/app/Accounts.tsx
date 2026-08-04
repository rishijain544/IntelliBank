import { useMutation, useQuery } from '@tanstack/react-query';
import { Copy, Plus, Snowflake, Wallet } from 'lucide-react';
import { useState } from 'react';

import {
  Card,
  ErrorBlock,
  Field,
  LoadingBlock,
  Modal,
  Notice,
  PageHeader,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { money, percentRaw, titleCase } from '../../lib/format';
import { invalidateAfterMoneyMovement, qk, queryClient } from '../../lib/query';
import { canBank, useAuth } from '../../store/auth';
import type { Account } from '../../types/api';

const ACCOUNT_TYPES = [
  { value: 'savings', label: 'Savings', note: '3.5% interest · ₹500 minimum balance' },
  { value: 'current', label: 'Current', note: 'No interest · ₹25,000 overdraft' },
  { value: 'salary', label: 'Salary', note: '3.0% interest · ₹5,000 overdraft' },
  { value: 'fixed_deposit', label: 'Fixed deposit', note: '7.1% interest · locked term' },
] as const;

export default function Accounts() {
  const user = useAuth((s) => s.user);
  const allowed = canBank(user);

  const [openNew, setOpenNew] = useState(false);
  const [openFund, setOpenFund] = useState<Account | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: qk.accounts,
    queryFn: () => get<Account[]>('/accounts'),
  });

  const [accountType, setAccountType] = useState<string>('savings');
  const [nickname, setNickname] = useState('');
  const [initialDeposit, setInitialDeposit] = useState('');
  const [fundAmount, setFundAmount] = useState('');
  const [formError, setFormError] = useState<string | null>(null);

  const createAccount = useMutation({
    mutationFn: () =>
      post<Account>('/accounts', {
        account_type: accountType,
        nickname: nickname.trim() || null,
        initial_deposit: initialDeposit || '0',
      }),
    onSuccess: async () => {
      setOpenNew(false);
      setNickname('');
      setInitialDeposit('');
      setFormError(null);
      await queryClient.invalidateQueries({ queryKey: qk.accounts });
      await queryClient.invalidateQueries({ queryKey: qk.dashboard });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const fundAccount = useMutation({
    mutationFn: (account: Account) =>
      post(`/accounts/${account.id}/deposit`, {
        account_id: account.id,
        amount: fundAmount,
        description: 'Simulated deposit',
      }),
    onSuccess: async () => {
      setOpenFund(null);
      setFundAmount('');
      setFormError(null);
      await invalidateAfterMoneyMovement();
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  async function copyAccountNumber(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(value);
      window.setTimeout(() => setCopied(null), 1800);
    } catch {
      // Clipboard access can be denied; failing silently is fine here since the
      // number is visible on screen anyway.
    }
  }

  if (isLoading) return <LoadingBlock rows={4} label="Loading accounts" />;
  if (error) return <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />;

  return (
    <div>
      <PageHeader
        title="Accounts"
        subtitle="Open accounts, check balances and add simulated funds."
        action={
          <button
            type="button"
            className="btn-primary px-4 py-2.5"
            onClick={() => setOpenNew(true)}
            disabled={!allowed}
          >
            <Plus className="h-4 w-4" aria-hidden />
            Open account
          </button>
        }
      />

      {!allowed && (
        <div className="mb-6">
          <Notice tone="warning" title="Account unavailable">
            Your account is currently restricted. Contact support for assistance.
          </Notice>
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {data?.map((account) => {
          const held = Number.parseFloat(account.hold_amount);
          return (
            <Card key={account.id} className="flex flex-col">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <h2 className="truncate font-semibold text-primary">
                      {account.nickname ?? titleCase(account.account_type)}
                    </h2>
                    {account.is_primary && (
                      <span className="badge bg-gold/15 text-gold-bright">Primary</span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-muted">{titleCase(account.account_type)}</p>
                </div>
                <StatusBadge status={account.status} />
              </div>

              <div className="mt-4">
                <p className="text-xs text-muted">Available balance</p>
                <p className="tnum text-2xl font-bold text-primary">
                  {money(account.available_balance)}
                </p>
                {held > 0 && (
                  <p className="tnum mt-1 text-xs text-warning">
                    <Snowflake className="mr-1 inline h-3 w-3" aria-hidden />
                    {money(account.hold_amount)} held pending security review
                  </p>
                )}
              </div>

              <dl className="mt-4 space-y-2 border-t border-line pt-4 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <dt className="text-muted">Account number</dt>
                  <dd className="flex items-center gap-1.5">
                    <span className="font-mono text-primary">{account.account_number}</span>
                    <button
                      type="button"
                      onClick={() => void copyAccountNumber(account.account_number)}
                      className="rounded p-1 text-muted transition hover:text-primary"
                      aria-label="Copy account number"
                    >
                      <Copy className="h-3 w-3" aria-hidden />
                    </button>
                  </dd>
                </div>
                {copied === account.account_number && (
                  <p className="text-right text-positive">Copied</p>
                )}
                <div className="flex justify-between">
                  <dt className="text-muted">IFSC</dt>
                  <dd className="font-mono text-primary">{account.ifsc_code}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted">Interest rate</dt>
                  <dd className="tnum text-primary">{percentRaw(account.interest_rate)}</dd>
                </div>
                {Number.parseFloat(account.overdraft_limit) > 0 && (
                  <div className="flex justify-between">
                    <dt className="text-muted">Overdraft</dt>
                    <dd className="tnum text-primary">{money(account.overdraft_limit)}</dd>
                  </div>
                )}
              </dl>

              <button
                type="button"
                className="btn-secondary mt-4 w-full py-2 text-xs"
                onClick={() => {
                  setOpenFund(account);
                  setFormError(null);
                }}
                disabled={!allowed || account.status !== 'active'}
              >
                <Wallet className="h-3.5 w-3.5" aria-hidden />
                Add simulated funds
              </button>
            </Card>
          );
        })}
      </div>

      {/* --------------------------- new account --------------------------- */}
      <Modal
        open={openNew}
        onClose={() => setOpenNew(false)}
        title="Open a new account"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setOpenNew(false)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary px-4 py-2"
              onClick={() => createAccount.mutate()}
              disabled={createAccount.isPending}
            >
              {createAccount.isPending ? 'Opening…' : 'Open account'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {formError && <Notice tone="danger">{formError}</Notice>}

          <Field label="Account type" htmlFor="accountType" required>
            <div className="space-y-2">
              {ACCOUNT_TYPES.map((type) => (
                <label
                  key={type.value}
                  className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition ${
                    accountType === type.value
                      ? 'border-gold bg-gold/5'
                      : 'border-line hover:border-line-strong'
                  }`}
                >
                  <input
                    type="radio"
                    name="accountType"
                    value={type.value}
                    checked={accountType === type.value}
                    onChange={(e) => setAccountType(e.target.value)}
                    className="mt-1 accent-gold"
                  />
                  <span>
                    <span className="block text-sm font-medium text-primary">{type.label}</span>
                    <span className="block text-xs text-muted">{type.note}</span>
                  </span>
                </label>
              ))}
            </div>
          </Field>

          <Field label="Nickname" htmlFor="nickname" hint="Optional — helps you identify it later">
            <input
              id="nickname"
              className="input"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder="Holiday fund"
              maxLength={80}
            />
          </Field>

          <Field
            label="Initial deposit (INR)"
            htmlFor="initialDeposit"
            hint="Simulated funds. Savings accounts must keep a ₹500 minimum balance."
          >
            <input
              id="initialDeposit"
              inputMode="decimal"
              className="input tnum"
              value={initialDeposit}
              onChange={(e) => setInitialDeposit(e.target.value.replace(/[^\d.]/g, ''))}
              placeholder="50000"
            />
          </Field>
        </div>
      </Modal>

      {/* ----------------------------- add funds ----------------------------- */}
      <Modal
        open={openFund !== null}
        onClose={() => setOpenFund(null)}
        title="Add simulated funds"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setOpenFund(null)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary px-4 py-2"
              onClick={() => openFund && fundAccount.mutate(openFund)}
              disabled={fundAccount.isPending || !fundAmount}
            >
              {fundAccount.isPending ? 'Adding…' : 'Add funds'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {formError && <Notice tone="danger">{formError}</Notice>}
          <Notice tone="info">
            This simulates an incoming credit such as a salary payment, so you have a balance to
            test transfers and loans with. No real money is involved.
          </Notice>
          {openFund && (
            <p className="text-sm text-muted">
              Depositing into{' '}
              <span className="font-medium text-primary">
                {openFund.nickname ?? titleCase(openFund.account_type)}
              </span>{' '}
              <span className="font-mono text-xs text-muted">({openFund.account_number})</span>
            </p>
          )}
          <Field label="Amount (INR)" htmlFor="fundAmount" required>
            <input
              id="fundAmount"
              inputMode="decimal"
              autoFocus
              className="input tnum"
              value={fundAmount}
              onChange={(e) => setFundAmount(e.target.value.replace(/[^\d.]/g, ''))}
              placeholder="25000"
            />
          </Field>
          <div className="flex gap-2">
            {['10000', '50000', '100000'].map((preset) => (
              <button
                key={preset}
                type="button"
                className="tnum btn-secondary flex-1 py-1.5 text-xs"
                onClick={() => setFundAmount(preset)}
              >
                {money(preset)}
              </button>
            ))}
          </div>
        </div>
      </Modal>
    </div>
  );
}
