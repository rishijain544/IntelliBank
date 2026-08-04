import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  ArrowRight,
  Ban,
  CheckCircle2,
  Clock,
  Plus,
  ShieldAlert,
  Sparkles,
  Trash2,
  Users,
  Zap,
} from 'lucide-react';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import {
  Badge,
  Card,
  EmptyState,
  Field,
  LoadingBlock,
  Modal,
  Notice,
  PageHeader,
  SectionHeading,
} from '../../components/ui';
import { del, errorMessage, get, post } from '../../lib/api';
import { latency, money, percent, titleCase } from '../../lib/format';
import { invalidateAfterMoneyMovement, qk, queryClient } from '../../lib/query';
import { canBank, isKycVerified, useAuth } from '../../store/auth';
import type { Account, Beneficiary, MessageResponse, TransferResponse } from '../../types/api';

type Mode = 'internal' | 'external';

/**
 * Result panel shown after a transfer.
 *
 * This is where the fraud model becomes visible to the customer: the score, the
 * decision, and the specific factors behind it. Showing the reasoning rather
 * than a bare "declined" is the difference between a trustworthy system and an
 * opaque one.
 */
function TransferResult({ result, onDismiss }: { result: TransferResponse; onDismiss: () => void }) {
  const { fraud, transaction, message } = result;

  const tone =
    fraud.action === 'block' ? 'danger' : fraud.action === 'review' ? 'warning' : 'success';
  const Icon = fraud.action === 'block' ? Ban : fraud.action === 'review' ? Clock : CheckCircle2;

  return (
    <Card
      className={clsx(
        'border-2',
        tone === 'danger' && 'border-alert/40 bg-alert/5',
        tone === 'warning' && 'border-warning/40 bg-warning/5',
        tone === 'success' && 'border-positive/40 bg-positive/5',
      )}
    >
      <div className="flex items-start gap-4">
        <span
          className={clsx(
            'grid h-11 w-11 shrink-0 place-items-center rounded-xl',
            tone === 'danger' && 'bg-alert/15 text-alert',
            tone === 'warning' && 'bg-warning/15 text-warning',
            tone === 'success' && 'bg-positive/15 text-positive',
          )}
        >
          <Icon className="h-5.5 w-5.5" aria-hidden />
        </span>

        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-primary">
            {fraud.action === 'block'
              ? 'Transfer blocked'
              : fraud.action === 'review'
                ? 'Held for review'
                : 'Transfer completed'}
          </h3>
          <p className="mt-1 text-sm text-primary">{message}</p>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-line bg-ink/50 p-3">
              <p className="text-[11px] tracking-wide text-intelligence uppercase">Risk score</p>
              <p className="tnum mt-0.5 text-lg font-bold text-primary">
                {percent(fraud.risk_score, 2)}
              </p>
            </div>
            <div className="rounded-lg border border-line bg-ink/50 p-3">
              <p className="text-[11px] tracking-wide text-muted uppercase">Decision</p>
              <p className="mt-0.5 text-lg font-bold text-primary capitalize">{fraud.action}</p>
            </div>
            <div className="rounded-lg border border-line bg-ink/50 p-3">
              <p className="text-[11px] tracking-wide text-muted uppercase">Scored in</p>
              <p className="tnum mt-0.5 text-lg font-bold text-primary">{latency(fraud.latency_ms)}</p>
            </div>
          </div>

          {fraud.reasons.length > 0 && (
            <div className="mt-4">
              <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-intelligence uppercase">
                <Sparkles className="h-3 w-3" aria-hidden />
                Why
              </p>
              <ul className="space-y-1">
                {fraud.reasons.slice(0, 4).map((reason) => (
                  <li key={reason} className="flex gap-2 text-xs text-muted">
                    <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-faint" aria-hidden />
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {fraud.top_factors.length > 0 && (
            <details className="mt-4">
              {/* SHAP contributions from the fraud model. */}
              <summary className="cursor-pointer text-xs font-medium text-intelligence hover:opacity-80">
                Model factor breakdown
              </summary>
              <ul className="mt-2 space-y-1.5">
                {fraud.top_factors.slice(0, 5).map((factor) => (
                  <li key={factor.feature} className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-muted">{factor.label}</span>
                    <Badge tone={factor.direction === 'increases risk' ? 'danger' : 'success'}>
                      {factor.direction === 'increases risk' ? '↑ risk' : '↓ risk'}
                    </Badge>
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-[11px] text-faint">
                Contributions from {fraud.model_name} v{fraud.model_version}
                {!fraud.model_available && ' (rules fallback — model unavailable)'}
              </p>
            </details>
          )}

          <div className="mt-5 flex flex-wrap gap-2">
            <button type="button" className="btn-secondary px-4 py-2 text-xs" onClick={onDismiss}>
              Make another transfer
            </button>
            {fraud.is_flagged && (
              <Link to="/app/fraud-center" className="btn-ghost px-4 py-2 text-xs">
                <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                Open Security Center
              </Link>
            )}
            <Link to="/app/transactions" className="btn-ghost px-4 py-2 text-xs">
              View transaction {transaction.reference}
            </Link>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default function Transfer() {
  const user = useAuth((s) => s.user);
  const allowed = canBank(user);
  const kycVerified = isKycVerified(user);

  const [mode, setMode] = useState<Mode>('internal');
  const [fromAccount, setFromAccount] = useState('');
  const [toAccountNumber, setToAccountNumber] = useState('');
  const [beneficiaryId, setBeneficiaryId] = useState('');
  const [amount, setAmount] = useState('');
  const [channel, setChannel] = useState('imps');
  const [description, setDescription] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [result, setResult] = useState<TransferResponse | null>(null);

  const [addOpen, setAddOpen] = useState(false);
  const [benName, setBenName] = useState('');
  const [benAccount, setBenAccount] = useState('');
  const [benIfsc, setBenIfsc] = useState('');
  const [benBank, setBenBank] = useState('');

  const { data: accounts, isLoading: accountsLoading } = useQuery({
    queryKey: qk.accounts,
    queryFn: () => get<Account[]>('/accounts'),
  });

  const { data: beneficiaries } = useQuery({
    queryKey: qk.beneficiaries,
    queryFn: () => get<Beneficiary[]>('/beneficiaries'),
  });

  const usableAccounts = accounts?.filter((a) => a.status === 'active') ?? [];
  const selectedAccount = usableAccounts.find((a) => String(a.id) === fromAccount);

  const transfer = useMutation({
    mutationFn: async () => {
      if (mode === 'internal') {
        return post<TransferResponse>('/transfers/internal', {
          from_account_id: Number(fromAccount),
          to_account_number: toAccountNumber.trim(),
          amount,
          description: description.trim() || null,
        });
      }
      return post<TransferResponse>('/transfers/external', {
        from_account_id: Number(fromAccount),
        beneficiary_id: Number(beneficiaryId),
        amount,
        channel,
        description: description.trim() || null,
      });
    },
    onSuccess: async (data) => {
      setResult(data);
      setAmount('');
      setDescription('');
      setFormError(null);
      await invalidateAfterMoneyMovement();
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const addBeneficiary = useMutation({
    mutationFn: () =>
      post<Beneficiary>('/beneficiaries', {
        name: benName.trim(),
        account_number: benAccount.trim(),
        ifsc_code: benIfsc.trim().toUpperCase(),
        bank_name: benBank.trim() || 'IntelliBank',
      }),
    onSuccess: async () => {
      setAddOpen(false);
      setBenName('');
      setBenAccount('');
      setBenIfsc('');
      setBenBank('');
      await queryClient.invalidateQueries({ queryKey: qk.beneficiaries });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const removeBeneficiary = useMutation({
    mutationFn: (id: number) => del<MessageResponse>(`/beneficiaries/${id}`),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: qk.beneficiaries });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const canSubmit =
    fromAccount &&
    amount &&
    Number.parseFloat(amount) > 0 &&
    (mode === 'internal'
      ? allowed && toAccountNumber.length >= 6
      : kycVerified && beneficiaryId);

  if (accountsLoading) return <LoadingBlock rows={5} label="Loading transfer form" />;

  return (
    <div>
      <PageHeader
        title="Send money"
        subtitle="Every transfer is scored by our fraud model before the money moves."
      />

      {/* Only external transfers require verified identity, matching how banks
          gate interbank payments more strictly than on-us transfers. */}
      {mode === 'external' && !kycVerified && (
        <div className="mb-6">
          <Notice tone="warning" title="Verification required for external transfers">
            Transfers to other banks need KYC verification. You can still send money to
            IntelliBank accounts without it.{' '}
            <Link to="/app/settings" className="font-semibold underline">
              Verify now
            </Link>
          </Notice>
        </div>
      )}

      {result && (
        <div className="mb-6">
          <TransferResult result={result} onDismiss={() => setResult(null)} />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <Card>
          {/* Mode switch */}
          <div className="mb-5 grid grid-cols-2 gap-2 rounded-lg border border-line p-1">
            {(['internal', 'external'] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setMode(value);
                  setFormError(null);
                }}
                className={clsx(
                  'rounded-md px-3 py-2 text-sm font-medium transition',
                  mode === value ? 'bg-gold text-ink' : 'text-muted hover:text-primary',
                )}
              >
                {value === 'internal' ? 'To IntelliBank' : 'To another bank'}
              </button>
            ))}
          </div>

          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              setFormError(null);
              transfer.mutate();
            }}
          >
            {formError && <Notice tone="danger">{formError}</Notice>}

            <Field label="From account" htmlFor="fromAccount" required>
              <select
                id="fromAccount"
                className="input"
                value={fromAccount}
                onChange={(e) => setFromAccount(e.target.value)}
                required
              >
                <option value="">Select an account</option>
                {usableAccounts.map((account) => (
                  <option key={account.id} value={account.id}>
                    {account.nickname ?? titleCase(account.account_type)} ·{' '}
                    {account.account_number.slice(-4)} · {money(account.available_balance)}
                  </option>
                ))}
              </select>
            </Field>

            {selectedAccount && (
              <p className="-mt-2 text-xs text-muted">
                Available: <span className="tnum text-primary">{money(selectedAccount.available_balance)}</span>
              </p>
            )}

            {mode === 'internal' ? (
              <Field
                label="Recipient account number"
                htmlFor="toAccount"
                hint="Any IntelliBank account number, including your own"
                required
              >
                <input
                  id="toAccount"
                  className="input font-mono"
                  value={toAccountNumber}
                  onChange={(e) => setToAccountNumber(e.target.value.replace(/\D/g, ''))}
                  placeholder="50123456789012"
                  maxLength={20}
                  required
                />
              </Field>
            ) : (
              <>
                <Field label="Beneficiary" htmlFor="beneficiary" required>
                  <select
                    id="beneficiary"
                    className="input"
                    value={beneficiaryId}
                    onChange={(e) => setBeneficiaryId(e.target.value)}
                    required
                  >
                    <option value="">Select a beneficiary</option>
                    {beneficiaries?.map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name} · {b.bank_name} · {b.account_number.slice(-4)}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field label="Payment method" htmlFor="channel">
                  <div className="grid grid-cols-3 gap-2">
                    {(['imps', 'neft', 'upi'] as const).map((value) => (
                      <button
                        key={value}
                        type="button"
                        onClick={() => setChannel(value)}
                        className={clsx(
                          'rounded-lg border px-3 py-2 text-xs font-medium uppercase transition',
                          channel === value
                            ? 'border-gold bg-gold/10 text-gold-bright'
                            : 'border-line text-muted hover:border-line-strong',
                        )}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                </Field>
              </>
            )}

            <Field label="Amount (INR)" htmlFor="amount" required>
              <input
                id="amount"
                inputMode="decimal"
                className="input tnum text-lg"
                value={amount}
                onChange={(e) => setAmount(e.target.value.replace(/[^\d.]/g, ''))}
                placeholder="0.00"
                required
              />
            </Field>

            <Field label="Note" htmlFor="description" hint="Optional — shown on both statements">
              <input
                id="description"
                className="input"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                maxLength={255}
                placeholder="Rent for March"
              />
            </Field>

            <button type="submit" className="btn-primary w-full py-3" disabled={!canSubmit || transfer.isPending}>
              {transfer.isPending ? (
                <>
                  <Zap className="h-4 w-4 animate-pulse" aria-hidden />
                  Scoring and sending…
                </>
              ) : (
                <>
                  Send {amount ? money(amount) : 'money'}
                  <ArrowRight className="h-4 w-4" aria-hidden />
                </>
              )}
            </button>

            <p className="text-center text-xs text-faint">
              Daily transfer limit ₹5,00,000 · IMPS ₹5 fee · NEFT ₹2.50 fee · UPI free
            </p>
          </form>
        </Card>

        {/* --------------------------- beneficiaries --------------------------- */}
        <Card>
          <SectionHeading
            title="Beneficiaries"
            action={
              <button
                type="button"
                className="btn-secondary px-3 py-1.5 text-xs"
                onClick={() => setAddOpen(true)}
                disabled={!allowed}
              >
                <Plus className="h-3.5 w-3.5" aria-hidden />
                Add
              </button>
            }
          />

          {beneficiaries?.length ? (
            <ul className="space-y-2">
              {beneficiaries.map((b) => (
                <li
                  key={b.id}
                  className="flex items-center gap-3 rounded-lg border border-line bg-ink/40 p-3"
                >
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-surface-raised text-xs font-bold text-primary">
                    {b.name.slice(0, 2).toUpperCase()}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-primary">{b.name}</p>
                    <p className="font-mono text-[11px] text-muted">
                      {b.bank_name} · ••{b.account_number.slice(-4)}
                    </p>
                  </div>
                  {b.is_internal && <Badge tone="info">Internal</Badge>}
                  <button
                    type="button"
                    className="rounded p-1.5 text-muted transition hover:text-alert"
                    aria-label={`Remove ${b.name}`}
                    onClick={() => removeBeneficiary.mutate(b.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={<Users className="h-8 w-8" aria-hidden />}
              title="No beneficiaries yet"
              description="Save a recipient to send money to another bank."
            />
          )}
        </Card>
      </div>

      {/* --------------------------- add beneficiary --------------------------- */}
      <Modal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        title="Add a beneficiary"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setAddOpen(false)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary px-4 py-2"
              onClick={() => addBeneficiary.mutate()}
              disabled={addBeneficiary.isPending || !benName || !benAccount || !benIfsc}
            >
              {addBeneficiary.isPending ? 'Saving…' : 'Save beneficiary'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <Field label="Recipient name" htmlFor="benName" required>
            <input id="benName" className="input" value={benName} onChange={(e) => setBenName(e.target.value)} />
          </Field>
          <Field label="Account number" htmlFor="benAccount" required>
            <input
              id="benAccount"
              className="input font-mono"
              value={benAccount}
              onChange={(e) => setBenAccount(e.target.value.replace(/\s/g, ''))}
            />
          </Field>
          <Field
            label="IFSC code"
            htmlFor="benIfsc"
            hint="Use SMRT0000001 for a IntelliBank account"
            required
          >
            <input
              id="benIfsc"
              className="input font-mono uppercase"
              value={benIfsc}
              onChange={(e) => setBenIfsc(e.target.value.toUpperCase())}
              maxLength={15}
            />
          </Field>
          <Field label="Bank name" htmlFor="benBank">
            <input
              id="benBank"
              className="input"
              value={benBank}
              onChange={(e) => setBenBank(e.target.value)}
              placeholder="HDFC Bank"
            />
          </Field>
        </div>
      </Modal>
    </div>
  );
}
