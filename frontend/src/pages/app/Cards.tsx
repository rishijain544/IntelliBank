import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { CreditCard, Globe, Plus, Radio, Settings2, Snowflake, Trash2, Wifi } from 'lucide-react';
import { useState } from 'react';

import {
  Card as Panel,
  EmptyState,
  ErrorBlock,
  Field,
  LoadingBlock,
  Modal,
  Notice,
  PageHeader,
  StatusBadge,
} from '../../components/ui';
import { del, errorMessage, get, patch, post } from '../../lib/api';
import { money, titleCase } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import { canBank, useAuth } from '../../store/auth';
import type { Account, Card, MessageResponse } from '../../types/api';

/**
 * Virtual card management.
 *
 * The full card number is deliberately never available: the backend hashes it at
 * issuance and returns only the last four digits, mirroring how PCI-DSS scope is
 * limited in practice. So there is no "reveal number" affordance here — showing
 * one would be dishonest about what the system stores.
 */
function CardVisual({ card }: { card: Card }) {
  const isCredit = card.card_type === 'virtual_credit';
  return (
    <div
      className={clsx(
        'relative overflow-hidden rounded-xl p-5 text-white',
        // Gold, not teal: a payment card is a premium/trust surface, not AI
        // output. Credit cards get a deeper treatment than debit.
        isCredit
          ? 'bg-gradient-to-br from-gold-deep via-gold-deep to-surface'
          : 'bg-gradient-to-br from-gold-deep via-gold-shadow to-ink',
        card.status !== 'active' && 'opacity-55 saturate-50',
      )}
    >
      <div
        className="pointer-events-none absolute -top-16 -right-16 h-40 w-40 rounded-full bg-white/10"
        aria-hidden
      />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-[10px] tracking-widest uppercase opacity-70">
            {isCredit ? 'Virtual credit' : 'Virtual debit'}
          </p>
          <p className="mt-0.5 text-sm font-semibold">IntelliBank</p>
        </div>
        <span className="text-xs font-bold tracking-wider italic opacity-90">
          {card.card_network}
        </span>
      </div>

      <p className="tnum relative mt-7 font-mono text-lg tracking-[0.2em]">{card.masked_number}</p>

      <div className="relative mt-5 flex items-end justify-between text-xs">
        <div>
          <p className="text-[9px] tracking-wider uppercase opacity-60">Cardholder</p>
          <p className="mt-0.5 font-medium">{card.cardholder_name}</p>
        </div>
        <div className="text-right">
          <p className="text-[9px] tracking-wider uppercase opacity-60">Expires</p>
          <p className="tnum mt-0.5 font-medium">
            {String(card.expiry_month).padStart(2, '0')}/{String(card.expiry_year).slice(-2)}
          </p>
        </div>
      </div>

      {card.status === 'frozen' && (
        <div className="absolute inset-0 grid place-items-center bg-black/45">
          <span className="flex items-center gap-1.5 rounded-full bg-black/70 px-3 py-1.5 text-xs font-semibold">
            <Snowflake className="h-3.5 w-3.5" aria-hidden />
            Frozen
          </span>
        </div>
      )}
    </div>
  );
}

export default function Cards() {
  const user = useAuth((s) => s.user);
  const allowed = canBank(user);

  const [issueOpen, setIssueOpen] = useState(false);
  const [limitsFor, setLimitsFor] = useState<Card | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const [accountId, setAccountId] = useState('');
  const [cardType, setCardType] = useState('virtual_debit');

  // Limit editor state, seeded when the modal opens.
  const [daily, setDaily] = useState('');
  const [perTxn, setPerTxn] = useState('');
  const [monthly, setMonthly] = useState('');
  const [online, setOnline] = useState(true);
  const [international, setInternational] = useState(false);
  const [contactless, setContactless] = useState(true);
  const [atm, setAtm] = useState(true);

  const { data: cards, isLoading, error, refetch } = useQuery({
    queryKey: qk.cards,
    queryFn: () => get<Card[]>('/cards'),
  });

  const { data: accounts } = useQuery({
    queryKey: qk.accounts,
    queryFn: () => get<Account[]>('/accounts'),
    staleTime: 5 * 60_000,
  });

  const issue = useMutation({
    mutationFn: () =>
      post<Card>('/cards', {
        account_id: Number(accountId),
        card_type: cardType,
        daily_limit: '50000',
        per_txn_limit: '25000',
        monthly_limit: '200000',
      }),
    onSuccess: async () => {
      setIssueOpen(false);
      setFormError(null);
      await queryClient.invalidateQueries({ queryKey: qk.cards });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const toggleFreeze = useMutation({
    mutationFn: (card: Card) =>
      patch<Card>(`/cards/${card.id}/freeze`, {
        freeze: card.status === 'active',
        reason: card.status === 'active' ? 'Frozen by cardholder' : null,
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: qk.cards });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const updateLimits = useMutation({
    mutationFn: (card: Card) =>
      patch<Card>(`/cards/${card.id}/limits`, {
        daily_limit: daily,
        per_txn_limit: perTxn,
        monthly_limit: monthly,
        online_enabled: online,
        international_enabled: international,
        contactless_enabled: contactless,
        atm_enabled: atm,
      }),
    onSuccess: async () => {
      setLimitsFor(null);
      setFormError(null);
      await queryClient.invalidateQueries({ queryKey: qk.cards });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const cancelCard = useMutation({
    mutationFn: (card: Card) => del<MessageResponse>(`/cards/${card.id}`),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: qk.cards });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  function openLimits(card: Card) {
    setLimitsFor(card);
    setDaily(card.daily_limit);
    setPerTxn(card.per_txn_limit);
    setMonthly(card.monthly_limit);
    setOnline(card.online_enabled);
    setInternational(card.international_enabled);
    setContactless(card.contactless_enabled);
    setAtm(card.atm_enabled);
    setFormError(null);
  }

  if (isLoading) return <LoadingBlock rows={4} label="Loading cards" />;
  if (error) return <ErrorBlock message={errorMessage(error)} onRetry={() => void refetch()} />;

  const activeAccounts = accounts?.filter((a) => a.status === 'active') ?? [];

  return (
    <div>
      <PageHeader
        title="Cards"
        subtitle="Issue virtual cards, freeze them instantly and control spending limits."
        action={
          <button
            type="button"
            className="btn-primary px-4 py-2.5"
            onClick={() => {
              setIssueOpen(true);
              setAccountId(String(activeAccounts[0]?.id ?? ''));
              setFormError(null);
            }}
            disabled={!allowed || activeAccounts.length === 0}
          >
            <Plus className="h-4 w-4" aria-hidden />
            Issue card
          </button>
        }
      />

      {formError && (
        <div className="mb-5">
          <Notice tone="danger">{formError}</Notice>
        </div>
      )}

      {!allowed && (
        <div className="mb-6">
          <Notice tone="warning" title="Verification required">
            Complete KYC verification to issue cards.
          </Notice>
        </div>
      )}

      {cards?.length ? (
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {cards.map((card) => (
            <Panel key={card.id} className="flex flex-col gap-4">
              <CardVisual card={card} />

              <div className="flex items-center justify-between">
                <StatusBadge status={card.status} />
                <span className="text-xs text-muted">{titleCase(card.card_type)}</span>
              </div>

              <dl className="space-y-1.5 border-t border-line pt-3 text-xs">
                <div className="flex justify-between">
                  <dt className="text-muted">Per transaction</dt>
                  <dd className="tnum text-primary">{money(card.per_txn_limit)}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted">Daily limit</dt>
                  <dd className="tnum text-primary">{money(card.daily_limit)}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted">Monthly limit</dt>
                  <dd className="tnum text-primary">{money(card.monthly_limit)}</dd>
                </div>
              </dl>

              <div className="flex flex-wrap gap-1.5">
                {[
                  { on: card.online_enabled, icon: Wifi, label: 'Online' },
                  { on: card.international_enabled, icon: Globe, label: 'International' },
                  { on: card.contactless_enabled, icon: Radio, label: 'Contactless' },
                  { on: card.atm_enabled, icon: CreditCard, label: 'ATM' },
                ].map(({ on, icon: Icon, label }) => (
                  <span
                    key={label}
                    className={clsx(
                      'badge',
                      on ? 'bg-positive/12 text-positive' : 'bg-surface-raised/70 text-muted',
                    )}
                    title={`${label} ${on ? 'enabled' : 'disabled'}`}
                  >
                    <Icon className="h-3 w-3" aria-hidden />
                    {label}
                  </span>
                ))}
              </div>

              {card.freeze_reason && (
                <p className="text-xs text-warning">Reason: {card.freeze_reason}</p>
              )}

              {card.status !== 'cancelled' && (
                <div className="mt-auto flex gap-2 border-t border-line pt-3">
                  <button
                    type="button"
                    className="btn-secondary flex-1 py-1.5 text-xs"
                    onClick={() => toggleFreeze.mutate(card)}
                    disabled={toggleFreeze.isPending}
                  >
                    <Snowflake className="h-3.5 w-3.5" aria-hidden />
                    {card.status === 'active' ? 'Freeze' : 'Unfreeze'}
                  </button>
                  <button
                    type="button"
                    className="btn-secondary flex-1 py-1.5 text-xs"
                    onClick={() => openLimits(card)}
                  >
                    <Settings2 className="h-3.5 w-3.5" aria-hidden />
                    Limits
                  </button>
                  <button
                    type="button"
                    className="btn-ghost p-1.5"
                    aria-label="Cancel card"
                    onClick={() => {
                      if (window.confirm('Cancel this card permanently? This cannot be undone.')) {
                        cancelCard.mutate(card);
                      }
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden />
                  </button>
                </div>
              )}
            </Panel>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<CreditCard className="h-9 w-9" aria-hidden />}
          title="No cards yet"
          description="Issue a virtual card to start making simulated card payments."
          action={
            allowed && activeAccounts.length > 0 ? (
              <button
                type="button"
                className="btn-primary px-4 py-2"
                onClick={() => {
                  setIssueOpen(true);
                  setAccountId(String(activeAccounts[0]?.id ?? ''));
                }}
              >
                <Plus className="h-4 w-4" aria-hidden />
                Issue your first card
              </button>
            ) : undefined
          }
        />
      )}

      {/* ------------------------------ issue ------------------------------ */}
      <Modal
        open={issueOpen}
        onClose={() => setIssueOpen(false)}
        title="Issue a virtual card"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setIssueOpen(false)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary px-4 py-2"
              onClick={() => issue.mutate()}
              disabled={issue.isPending || !accountId}
            >
              {issue.isPending ? 'Issuing…' : 'Issue card'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {formError && <Notice tone="danger">{formError}</Notice>}

          <Notice tone="info">
            The full card number is hashed on the server and never stored or displayed — only the
            last four digits are retained, the same way real card data is scoped.
          </Notice>

          <Field label="Linked account" htmlFor="cardAccount" required>
            <select
              id="cardAccount"
              className="input"
              value={accountId}
              onChange={(e) => setAccountId(e.target.value)}
            >
              {activeAccounts.map((account) => (
                <option key={account.id} value={account.id}>
                  {account.nickname ?? titleCase(account.account_type)} ·{' '}
                  <span className="tnum">{account.account_number.slice(-4)}</span>
                </option>
              ))}
            </select>
          </Field>

          <Field label="Card type" htmlFor="cardType" required>
            <select id="cardType" className="input" value={cardType} onChange={(e) => setCardType(e.target.value)}>
              <option value="virtual_debit">Virtual debit</option>
              <option value="virtual_credit">Virtual credit</option>
            </select>
          </Field>

          <p className="text-xs text-muted">
            Default limits: ₹25,000 per transaction, ₹50,000 daily, ₹2,00,000 monthly. Adjustable
            after issuance.
          </p>
        </div>
      </Modal>

      {/* ------------------------------ limits ------------------------------ */}
      <Modal
        open={limitsFor !== null}
        onClose={() => setLimitsFor(null)}
        title="Card limits and controls"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setLimitsFor(null)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary px-4 py-2"
              onClick={() => limitsFor && updateLimits.mutate(limitsFor)}
              disabled={updateLimits.isPending}
            >
              {updateLimits.isPending ? 'Saving…' : 'Save changes'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          {formError && <Notice tone="danger">{formError}</Notice>}

          <Field
            label="Per-transaction limit"
            htmlFor="perTxn"
            hint="Must not exceed the daily limit"
            required
          >
            <input
              id="perTxn"
              inputMode="decimal"
              className="input tnum"
              value={perTxn}
              onChange={(e) => setPerTxn(e.target.value.replace(/[^\d.]/g, ''))}
            />
          </Field>

          <Field label="Daily limit" htmlFor="daily" hint="Must not exceed the monthly limit" required>
            <input
              id="daily"
              inputMode="decimal"
              className="input tnum"
              value={daily}
              onChange={(e) => setDaily(e.target.value.replace(/[^\d.]/g, ''))}
            />
          </Field>

          <Field label="Monthly limit" htmlFor="monthly" required>
            <input
              id="monthly"
              inputMode="decimal"
              className="input tnum"
              value={monthly}
              onChange={(e) => setMonthly(e.target.value.replace(/[^\d.]/g, ''))}
            />
          </Field>

          <fieldset className="space-y-2.5 border-t border-line pt-4">
            <legend className="mb-1 text-xs font-semibold tracking-wide text-muted uppercase">
              Where this card works
            </legend>
            {[
              { checked: online, set: setOnline, label: 'Online payments', icon: Wifi },
              { checked: international, set: setInternational, label: 'International payments', icon: Globe },
              { checked: contactless, set: setContactless, label: 'Contactless (tap to pay)', icon: Radio },
              { checked: atm, set: setAtm, label: 'ATM withdrawals', icon: CreditCard },
            ].map(({ checked, set, label, icon: Icon }) => (
              <label key={label} className="flex cursor-pointer items-center gap-3 text-sm text-primary">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-gold"
                  checked={checked}
                  onChange={(e) => set(e.target.checked)}
                />
                <Icon className="h-4 w-4 text-muted" aria-hidden />
                {label}
              </label>
            ))}
          </fieldset>
        </div>
      </Modal>
    </div>
  );
}
