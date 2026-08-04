import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  BadgeCheck,
  Calculator,
  Gauge,
  Landmark,
  Loader2,
  Sparkles,
  ThumbsDown,
  TrendingDown,
  TrendingUp,
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
  RiskBandBadge,
  SectionHeading,
  StatusBadge,
} from '../../components/ui';
import { errorMessage, get, post } from '../../lib/api';
import { dateShort, latency, money, percent, percentRaw, scoreLabel, titleCase } from '../../lib/format';
import { invalidateAfterMoneyMovement, qk, queryClient } from '../../lib/query';
import { canBank, isKycVerified, useAuth } from '../../store/auth';
import type { Account, CreditScore, Loan, LoanApplicationResult, Page } from '../../types/api';

const LOAN_TYPES = [
  { value: 'personal', label: 'Personal', tenures: [12, 24, 36, 48, 60] },
  { value: 'auto', label: 'Vehicle', tenures: [36, 48, 60, 84] },
  { value: 'home', label: 'Home', tenures: [120, 180, 240, 300] },
  { value: 'education', label: 'Education', tenures: [36, 60, 84, 120] },
  { value: 'business', label: 'Business', tenures: [12, 24, 36, 60] },
] as const;

/**
 * Credit decision panel.
 *
 * Shows the score, band, price and — importantly — the factors that drove it.
 * A lending decision the applicant cannot interrogate is a bad lending decision,
 * so the SHAP contributions are surfaced rather than hidden behind an API.
 */
function CreditResult({ credit, title }: { credit: CreditScore; title?: string }) {
  const decisionTone =
    credit.decision === 'approve' ? 'success' : credit.decision === 'review' ? 'warning' : 'danger';
  const DecisionIcon =
    credit.decision === 'approve' ? BadgeCheck : credit.decision === 'review' ? Gauge : ThumbsDown;

  // Position on the 300–900 scale, for the gauge.
  const scorePct = ((credit.score - 300) / 600) * 100;

  return (
    <Card
      className={clsx(
        'border-2',
        decisionTone === 'success' && 'border-positive/40 bg-positive/5',
        decisionTone === 'warning' && 'border-warning/40 bg-warning/5',
        decisionTone === 'danger' && 'border-alert/40 bg-alert/5',
      )}
    >
      {title && <p className="mb-4 text-xs font-semibold tracking-wide text-muted uppercase">{title}</p>}

      <div className="flex flex-wrap items-start justify-between gap-5">
        <div>
          <p className="text-xs text-muted">Credit score</p>
          <p className="tnum text-4xl font-bold text-primary">{credit.score}</p>
          <p className="mt-0.5 text-sm text-muted">{scoreLabel(credit.score)}</p>
          <div className="mt-3 flex items-center gap-2">
            <RiskBandBadge band={credit.risk_band} />
            <Badge tone={decisionTone}>
              <DecisionIcon className="h-3 w-3" aria-hidden />
              {titleCase(credit.decision)}
            </Badge>
          </div>
        </div>

        <div className="min-w-40 flex-1">
          {/* 300–900 scale visualisation */}
          <div className="mb-1 flex justify-between text-[10px] text-faint">
            <span>300</span>
            <span>900</span>
          </div>
          <div className="relative h-2.5 overflow-hidden rounded-full bg-gradient-to-r from-alert via-warning to-positive">
            <div
              className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded-full bg-white shadow-lg"
              style={{ left: `calc(${Math.min(Math.max(scorePct, 1), 99)}% - 2px)` }}
              aria-hidden
            />
          </div>
          <p className="mt-2 text-xs text-muted">
            Default probability{' '}
            <span className="tnum font-medium text-primary">
              {percent(credit.probability_of_default, 1)}
            </span>
          </p>
        </div>
      </div>

      {credit.decision !== 'reject' && (
        <dl className="mt-5 grid gap-3 border-t border-line pt-4 sm:grid-cols-4">
          {[
            ['Offer amount', money(credit.approved_amount)],
            ['Interest rate', percentRaw(credit.suggested_rate)],
            ['Monthly EMI', money(credit.emi_amount)],
            ['Total payable', money(credit.total_payable)],
          ].map(([label, value]) => (
            <div key={label}>
              <dt className="text-[11px] tracking-wide text-muted uppercase">{label}</dt>
              <dd className="tnum mt-0.5 font-semibold text-primary">{value}</dd>
            </div>
          ))}
        </dl>
      )}

      {credit.reasons.length > 0 && (
        <div className="mt-4 border-t border-line pt-4">
          <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold tracking-wide text-intelligence uppercase">
            <Sparkles className="h-3 w-3" aria-hidden />
            Decision notes
          </p>
          <ul className="space-y-1">
            {credit.reasons.map((reason) => (
              <li key={reason} className="flex gap-2 text-xs text-muted">
                <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-faint" aria-hidden />
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {credit.top_factors.length > 0 && (
        <details className="mt-4 border-t border-line pt-4">
          {/* Model reasoning, so it carries the intelligence token. */}
          <summary className="cursor-pointer text-xs font-medium text-intelligence hover:opacity-80">
            What influenced this score
          </summary>
          <ul className="mt-3 space-y-2">
            {credit.top_factors.map((factor) => {
              const raises = factor.direction === 'increases risk';
              return (
                <li key={factor.feature} className="flex items-center justify-between gap-3">
                  <span className="text-xs text-muted">{factor.label}</span>
                  <span
                    className={clsx(
                      'flex items-center gap-1 text-xs font-medium',
                      raises ? 'text-alert' : 'text-positive',
                    )}
                  >
                    {raises ? (
                      <TrendingUp className="h-3 w-3" aria-hidden />
                    ) : (
                      <TrendingDown className="h-3 w-3" aria-hidden />
                    )}
                    {raises ? 'Raises risk' : 'Lowers risk'}
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="tnum mt-3 text-[11px] text-faint">
            {credit.model_name} v{credit.model_version} · scored in {latency(credit.latency_ms)}
            {!credit.model_available && ' · heuristic fallback (model unavailable)'}
          </p>
        </details>
      )}
    </Card>
  );
}

export default function Loans() {
  const user = useAuth((s) => s.user);
  const allowed = canBank(user);
  const kycVerified = isKycVerified(user);

  const [loanType, setLoanType] = useState<string>('personal');
  const [amount, setAmount] = useState('300000');
  const [tenure, setTenure] = useState(36);
  const [purpose, setPurpose] = useState('');
  const [disbursementAccount, setDisbursementAccount] = useState('');
  const [quote, setQuote] = useState<CreditScore | null>(null);
  const [applied, setApplied] = useState<LoanApplicationResult | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [detailLoan, setDetailLoan] = useState<Loan | null>(null);

  const { data: loans, isLoading } = useQuery({
    queryKey: qk.loans(1),
    queryFn: () => get<Page<Loan>>('/loans', { params: { page: 1, page_size: 20 } }),
  });

  const { data: accounts } = useQuery({
    queryKey: qk.accounts,
    queryFn: () => get<Account[]>('/accounts'),
    staleTime: 5 * 60_000,
  });

  const availableTenures =
    LOAN_TYPES.find((t) => t.value === loanType)?.tenures ?? [12, 24, 36, 48, 60];

  const checkEligibility = useMutation({
    mutationFn: () =>
      post<CreditScore>('/loans/eligibility', {
        loan_type: loanType,
        amount,
        tenure_months: tenure,
      }),
    onSuccess: (data) => {
      setQuote(data);
      setApplied(null);
      setFormError(null);
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const apply = useMutation({
    mutationFn: () =>
      post<LoanApplicationResult>('/loans/apply', {
        loan_type: loanType,
        amount,
        tenure_months: tenure,
        purpose: purpose.trim() || null,
        disbursement_account_id: disbursementAccount ? Number(disbursementAccount) : null,
      }),
    onSuccess: async (data) => {
      setApplied(data);
      setQuote(null);
      setFormError(null);
      await queryClient.invalidateQueries({ queryKey: ['loans'] });
      await queryClient.invalidateQueries({ queryKey: qk.dashboard });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  const acceptLoan = useMutation({
    mutationFn: (loan: Loan) => post<Loan>(`/loans/${loan.id}/accept`),
    onSuccess: async () => {
      setDetailLoan(null);
      await invalidateAfterMoneyMovement();
      await queryClient.invalidateQueries({ queryKey: ['loans'] });
    },
    onError: (err) => setFormError(errorMessage(err)),
  });

  return (
    <div>
      <PageHeader
        title="Loans"
        subtitle="Get a live credit decision, priced by our calibrated scoring model."
      />

      {/* Eligibility checks are open to everyone; only submitting a binding
          application requires verified identity. */}
      {!kycVerified && (
        <div className="mb-6">
          <Notice tone="info" title="Check your eligibility without verifying">
            You can get a full credit decision below right now. KYC verification is only
            needed to submit a formal application.{' '}
            <Link to="/app/settings" className="font-semibold underline">
              Verify now
            </Link>
          </Notice>
        </div>
      )}

      {formError && (
        <div className="mb-5">
          <Notice tone="danger">{formError}</Notice>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        {/* ---------------------------- calculator ---------------------------- */}
        <Card>
          <SectionHeading
            title="Check your eligibility"
            subtitle="A quote does not create an application or affect your record."
          />

          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              checkEligibility.mutate();
            }}
          >
            <Field label="Loan type" htmlFor="loanType" required>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {LOAN_TYPES.map((type) => (
                  <button
                    key={type.value}
                    type="button"
                    onClick={() => {
                      setLoanType(type.value);
                      // Keep tenure valid for the newly selected product.
                      if (!type.tenures.includes(tenure as never)) setTenure(type.tenures[0]);
                    }}
                    className={clsx(
                      'rounded-lg border px-3 py-2 text-xs font-medium transition',
                      loanType === type.value
                        ? 'border-gold bg-gold/10 text-gold-bright'
                        : 'border-line text-muted hover:border-line-strong',
                    )}
                  >
                    {type.label}
                  </button>
                ))}
              </div>
            </Field>

            <Field label="Amount (INR)" htmlFor="loanAmount" required>
              <input
                id="loanAmount"
                inputMode="decimal"
                className="input tnum text-lg"
                value={amount}
                onChange={(e) => setAmount(e.target.value.replace(/[^\d.]/g, ''))}
                required
              />
            </Field>

            <Field label="Tenure" htmlFor="tenure" required>
              <div className="flex flex-wrap gap-2">
                {availableTenures.map((months) => (
                  <button
                    key={months}
                    type="button"
                    onClick={() => setTenure(months)}
                    className={clsx(
                      'rounded-lg border px-3 py-1.5 text-xs font-medium transition',
                      tenure === months
                        ? 'border-gold bg-gold/10 text-gold-bright'
                        : 'border-line text-muted hover:border-line-strong',
                    )}
                  >
                    {months >= 12 ? `${months / 12} yr` : `${months} mo`}
                  </button>
                ))}
              </div>
            </Field>

            <button
              type="submit"
              className="btn-secondary w-full py-2.5"
              disabled={!allowed || checkEligibility.isPending || !amount}
            >
              {checkEligibility.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <Calculator className="h-4 w-4" aria-hidden />
              )}
              {checkEligibility.isPending ? 'Scoring…' : 'Check eligibility'}
            </button>
          </form>

          {quote && (
            <form
              className="mt-5 space-y-4 border-t border-line pt-5"
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                apply.mutate();
              }}
            >
              <Field label="Purpose" htmlFor="purpose" hint="Optional">
                <input
                  id="purpose"
                  className="input"
                  value={purpose}
                  onChange={(e) => setPurpose(e.target.value)}
                  placeholder="Home renovation"
                  maxLength={255}
                />
              </Field>

              <Field label="Disburse to" htmlFor="disburseTo">
                <select
                  id="disburseTo"
                  className="input"
                  value={disbursementAccount}
                  onChange={(e) => setDisbursementAccount(e.target.value)}
                >
                  <option value="">Primary account</option>
                  {accounts
                    ?.filter((a) => a.status === 'active')
                    .map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.nickname ?? titleCase(a.account_type)} · {a.account_number.slice(-4)}
                      </option>
                    ))}
                </select>
              </Field>

              {/* Submitting is the binding step, so this one stays KYC-gated. */}
              <button
                type="submit"
                className="btn-primary w-full py-2.5"
                disabled={apply.isPending || !kycVerified}
              >
                {apply.isPending ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
                {apply.isPending ? 'Submitting…' : 'Submit application'}
              </button>
              {!kycVerified && (
                <p className="text-center text-xs text-muted">
                  Verify your identity to submit this application.
                </p>
              )}
            </form>
          )}
        </Card>

        {/* ------------------------------ result ------------------------------ */}
        <div className="space-y-6">
          {applied ? (
            <>
              <Notice tone={applied.loan.status === 'approved' ? 'success' : 'info'} title={`Application ${applied.loan.application_ref}`}>
                {applied.loan.status === 'approved'
                  ? 'Approved. Accept the offer below to receive the funds.'
                  : applied.loan.status === 'under_review'
                    ? 'Your application is queued for manual review by our credit team.'
                    : 'We were unable to approve this application.'}
              </Notice>
              <CreditResult credit={applied.credit} title="Decision" />
            </>
          ) : quote ? (
            <CreditResult credit={quote} title="Indicative quote" />
          ) : (
            <Card>
              <EmptyState
                icon={<Gauge className="h-9 w-9" aria-hidden />}
                title="No quote yet"
                description="Choose an amount and tenure, then check your eligibility to see a live decision with the reasoning behind it."
              />
            </Card>
          )}
        </div>
      </div>

      {/* ---------------------------- applications ---------------------------- */}
      <div className="mt-8">
        <SectionHeading title="Your applications" />
        {isLoading ? (
          <LoadingBlock rows={3} />
        ) : loans?.items.length ? (
          <Card className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Loan applications</caption>
                <thead className="table-header">
                  <tr>
                    <th scope="col" className="px-4 py-3">Reference</th>
                    <th scope="col" className="px-4 py-3">Type</th>
                    <th scope="col" className="px-4 py-3 text-right">Amount</th>
                    <th scope="col" className="hidden px-4 py-3 text-right sm:table-cell">EMI</th>
                    <th scope="col" className="hidden px-4 py-3 text-right md:table-cell">Rate</th>
                    <th scope="col" className="px-4 py-3">Status</th>
                    <th scope="col" className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {loans.items.map((loan) => (
                    <tr key={loan.id} className="transition hover:bg-surface-raised/60">
                      <td className="px-4 py-3">
                        <p className="font-mono text-xs text-primary">{loan.application_ref}</p>
                        <p className="text-[11px] text-muted">{dateShort(loan.created_at)}</p>
                      </td>
                      <td className="px-4 py-3 text-primary">{titleCase(loan.loan_type)}</td>
                      <td className="tnum px-4 py-3 text-right text-primary">
                        {money(loan.approved_amount ?? loan.requested_amount)}
                      </td>
                      <td className="tnum hidden px-4 py-3 text-right text-muted sm:table-cell">
                        {money(loan.emi_amount)}
                      </td>
                      <td className="tnum hidden px-4 py-3 text-right text-muted md:table-cell">
                        {loan.interest_rate ? percentRaw(loan.interest_rate) : '—'}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={loan.status} />
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          className="btn-ghost px-2.5 py-1 text-xs"
                          onClick={() => {
                            setDetailLoan(loan);
                            setFormError(null);
                          }}
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ) : (
          <EmptyState
            icon={<Landmark className="h-9 w-9" aria-hidden />}
            title="No applications yet"
            description="Check your eligibility above to get started."
          />
        )}
      </div>

      {/* ------------------------------ detail ------------------------------ */}
      <Modal
        open={detailLoan !== null}
        onClose={() => setDetailLoan(null)}
        title={`Loan ${detailLoan?.application_ref ?? ''}`}
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setDetailLoan(null)}>
              Close
            </button>
            {detailLoan?.status === 'approved' && (
              <button
                type="button"
                className="btn-primary px-4 py-2"
                onClick={() => detailLoan && acceptLoan.mutate(detailLoan)}
                disabled={acceptLoan.isPending}
              >
                {acceptLoan.isPending ? 'Disbursing…' : 'Accept and disburse'}
              </button>
            )}
          </>
        }
      >
        {detailLoan && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <StatusBadge status={detailLoan.status} />
              <span className="text-xs text-muted">{dateShort(detailLoan.created_at)}</span>
            </div>

            <dl className="divide-y divide-line text-sm">
              {[
                ['Type', titleCase(detailLoan.loan_type)],
                ['Requested', money(detailLoan.requested_amount)],
                ...(detailLoan.approved_amount ? [['Approved', money(detailLoan.approved_amount)]] : []),
                ['Tenure', `${detailLoan.tenure_months} months`],
                ...(detailLoan.interest_rate ? [['Interest rate', percentRaw(detailLoan.interest_rate)]] : []),
                ...(detailLoan.emi_amount ? [['Monthly EMI', money(detailLoan.emi_amount)]] : []),
                ...(detailLoan.total_payable ? [['Total payable', money(detailLoan.total_payable)]] : []),
                ...(detailLoan.processing_fee ? [['Processing fee', money(detailLoan.processing_fee)]] : []),
                ...(detailLoan.purpose ? [['Purpose', detailLoan.purpose]] : []),
                ['Decision by', titleCase(detailLoan.decision_source)],
                ...(detailLoan.disbursed_at ? [['Disbursed', dateShort(detailLoan.disbursed_at)]] : []),
                ...(detailLoan.outstanding_principal
                  ? [['Outstanding', money(detailLoan.outstanding_principal)]]
                  : []),
                ...(detailLoan.emis_paid > 0 ? [['EMIs paid', String(detailLoan.emis_paid)]] : []),
                ...(detailLoan.emis_missed > 0
                  ? [['EMIs missed', <span className="text-alert">{detailLoan.emis_missed}</span>]]
                  : []),
              ].map(([label, value], i) => (
                <div key={i} className="flex items-center justify-between gap-3 py-2.5">
                  <dt className="text-muted">{label}</dt>
                  <dd className="tnum text-right text-primary">{value}</dd>
                </div>
              ))}
            </dl>

            {detailLoan.decision_reason && (
              <Notice tone="info" title="Decision reason">
                {detailLoan.decision_reason}
              </Notice>
            )}

            {detailLoan.manual_override && (
              <Notice tone="warning" title="Manually reviewed">
                This decision was adjusted by our credit team, overriding the model's
                recommendation.
              </Notice>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
