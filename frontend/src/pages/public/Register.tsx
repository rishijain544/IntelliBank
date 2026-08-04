/**
 * Multi-step registration with simulated KYC.
 *
 * Two real API calls happen here: `/auth/register` creates the account, then
 * `/auth/kyc` verifies it. The account exists after step 1, so the KYC steps run
 * against an authenticated session — meaning a user who abandons midway can
 * finish later from Settings rather than losing the registration.
 *
 * The ID fields accept PAN/Aadhaar *formats* only. Values are hashed server-side
 * and never stored in the clear; the copy says so explicitly because asking for
 * government IDs without explanation is not acceptable, even in a demo.
 */
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Check, FileText, Loader2, ShieldCheck, Upload, UserPlus } from 'lucide-react';
import { clsx } from 'clsx';

import { Card, Field, Notice } from '../../components/ui';
import { errorMessage, fieldErrors, post } from '../../lib/api';
import { useAuth } from '../../store/auth';
import type { User } from '../../types/api';

const STEPS = ['Account', 'Identity', 'Financial', 'Done'] as const;

const EMPLOYMENT = [
  'salaried',
  'self_employed',
  'government',
  'contract',
  'gig',
  'student',
  'unemployed',
  'retired',
] as const;

export default function Register() {
  const navigate = useNavigate();
  const login = useAuth((s) => s.login);
  const setUser = useAuth((s) => s.setUser);

  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Step 1
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [dob, setDob] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');

  // Step 2
  const [pan, setPan] = useState('');
  const [aadhaar, setAadhaar] = useState('');
  const [documentType, setDocumentType] = useState('passport');
  const [documentName, setDocumentName] = useState('');
  const [address1, setAddress1] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [postal, setPostal] = useState('');

  // Step 3
  const [income, setIncome] = useState('');
  const [employment, setEmployment] = useState<string>('salaried');
  const [employmentYears, setEmploymentYears] = useState('3');
  const [dependents, setDependents] = useState('0');
  const [housing, setHousing] = useState('rent');

  function validateAccountStep(): boolean {
    const next: Record<string, string> = {};
    if (fullName.trim().length < 2) next.full_name = 'Enter your full name';
    if (!/^\S+@\S+\.\S+$/.test(email)) next.email = 'Enter a valid email address';
    if (password.length < 10) next.password = 'Use at least 10 characters';
    else if (!/[a-z]/.test(password)) next.password = 'Include a lowercase letter';
    else if (!/[A-Z]/.test(password)) next.password = 'Include an uppercase letter';
    else if (!/\d/.test(password)) next.password = 'Include a digit';
    else if (!/[^A-Za-z0-9]/.test(password)) next.password = 'Include a symbol';
    if (password !== confirm) next.confirm = 'Passwords do not match';
    if (dob) {
      const age = (Date.now() - new Date(dob).getTime()) / 31_557_600_000;
      if (age < 18) next.date_of_birth = 'You must be at least 18';
    }
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  function validateIdentityStep(): boolean {
    const next: Record<string, string> = {};
    if (!/^[A-Za-z]{5}\d{4}[A-Za-z]$/.test(pan)) next.pan = 'Format must be AAAAA9999A';
    if (!/^\d{12}$/.test(aadhaar.replace(/\s/g, ''))) next.aadhaar = 'Must be 12 digits';
    if (!documentName.trim()) next.document_name = 'Attach a document';
    if (address1.trim().length < 3) next.address_line1 = 'Enter your street address';
    if (city.trim().length < 2) next.city = 'Enter your city';
    if (state.trim().length < 2) next.state = 'Enter your state';
    if (postal.trim().length < 4) next.postal_code = 'Enter a valid postal code';
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  /** Step 1 → creates the account and signs the user in. */
  async function submitAccount() {
    if (!validateAccountStep()) return;
    setBusy(true);
    setError(null);
    try {
      await post<User>('/auth/register', {
        email,
        password,
        full_name: fullName.trim(),
        phone: phone || null,
        date_of_birth: dob || null,
      });
      // Sign in immediately so the KYC call is authenticated.
      await login({ email, password });
      setStep(1);
    } catch (err) {
      setError(errorMessage(err));
      setErrors(fieldErrors(err));
    } finally {
      setBusy(false);
    }
  }

  /** Step 3 → submits KYC, which activates the account and opens a savings account. */
  async function submitKyc() {
    setBusy(true);
    setError(null);
    try {
      const user = await post<User>('/auth/kyc', {
        pan: pan.toUpperCase(),
        aadhaar: aadhaar.replace(/\s/g, ''),
        document_type: documentType,
        document_name: documentName,
        address_line1: address1,
        city,
        state,
        postal_code: postal,
        annual_income: income || '600000',
        employment_status: employment,
        employment_years: Number(employmentYears) || 0,
        dependents: Number(dependents) || 0,
        housing_status: housing,
      });
      setUser(user);
      setStep(3);
    } catch (err) {
      setError(errorMessage(err));
      setErrors(fieldErrors(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl px-4 py-12">
      <div className="mb-7 text-center">
        <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-xl bg-gold/15 text-gold">
          <UserPlus className="h-6 w-6" aria-hidden />
        </span>
        <h1 className="text-2xl font-bold text-primary">Open a IntelliBank account</h1>
        <p className="mt-1.5 text-sm text-muted">
          Already registered?{' '}
          <Link to="/login" className="font-medium text-gold hover:text-gold-bright">
            Sign in
          </Link>
        </p>
      </div>

      {/* Stepper */}
      <ol className="mb-6 flex items-center gap-2" aria-label="Progress">
        {STEPS.map((label, index) => (
          <li key={label} className="flex flex-1 items-center gap-2">
            <span
              className={clsx(
                'grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-bold transition',
                index < step
                  ? 'bg-positive text-ink'
                  : index === step
                    ? 'bg-gold text-ink'
                    : 'bg-surface-raised text-muted',
              )}
              aria-current={index === step ? 'step' : undefined}
            >
              {index < step ? <Check className="h-3.5 w-3.5" aria-hidden /> : index + 1}
            </span>
            <span className={clsx('hidden text-xs sm:block', index <= step ? 'text-primary' : 'text-faint')}>
              {label}
            </span>
            {index < STEPS.length - 1 && (
              <span className={clsx('h-px flex-1', index < step ? 'bg-positive/50' : 'bg-surface-raised')} />
            )}
          </li>
        ))}
      </ol>

      <Card>
        {error && (
          <div className="mb-4">
            <Notice tone="danger">{error}</Notice>
          </div>
        )}

        {/* ---------------------------- step 1 ---------------------------- */}
        {step === 0 && (
          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              void submitAccount();
            }}
          >
            <Field label="Full name" htmlFor="fullName" error={errors.full_name} required>
              <input
                id="fullName"
                className={clsx('input', errors.full_name && 'input-error')}
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
              />
            </Field>

            <Field label="Email address" htmlFor="email" error={errors.email} required>
              <input
                id="email"
                type="email"
                className={clsx('input', errors.email && 'input-error')}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Phone" htmlFor="phone" error={errors.phone} hint="Optional">
                <input
                  id="phone"
                  className={clsx('input', errors.phone && 'input-error')}
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="+91…"
                  autoComplete="tel"
                />
              </Field>
              <Field label="Date of birth" htmlFor="dob" error={errors.date_of_birth} hint="Optional">
                <input
                  id="dob"
                  type="date"
                  className={clsx('input', errors.date_of_birth && 'input-error')}
                  value={dob}
                  onChange={(e) => setDob(e.target.value)}
                />
              </Field>
            </div>

            <Field
              label="Password"
              htmlFor="password"
              error={errors.password}
              hint="At least 10 characters with upper, lower, digit and symbol"
              required
            >
              <input
                id="password"
                type="password"
                className={clsx('input', errors.password && 'input-error')}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
              />
            </Field>

            <Field label="Confirm password" htmlFor="confirm" error={errors.confirm} required>
              <input
                id="confirm"
                type="password"
                className={clsx('input', errors.confirm && 'input-error')}
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
              />
            </Field>

            <button type="submit" className="btn-primary w-full py-2.5" disabled={busy}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
              Continue
              {!busy && <ArrowRight className="h-4 w-4" aria-hidden />}
            </button>
          </form>
        )}

        {/* ---------------------------- step 2 ---------------------------- */}
        {step === 1 && (
          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              if (validateIdentityStep()) setStep(2);
            }}
          >
            <Notice tone="info" title="Simulated verification">
              These fields validate <em>format</em> only. Values are hashed before storage and never
              kept in readable form. Use fictional numbers — never a real government ID.
            </Notice>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="PAN-style ID" htmlFor="pan" error={errors.pan} hint="e.g. ABCDE1234F" required>
                <input
                  id="pan"
                  className={clsx('input font-mono uppercase', errors.pan && 'input-error')}
                  maxLength={10}
                  value={pan}
                  onChange={(e) => setPan(e.target.value.toUpperCase())}
                />
              </Field>
              <Field label="Aadhaar-style ID" htmlFor="aadhaar" error={errors.aadhaar} hint="12 digits" required>
                <input
                  id="aadhaar"
                  inputMode="numeric"
                  className={clsx('input font-mono', errors.aadhaar && 'input-error')}
                  maxLength={12}
                  value={aadhaar}
                  onChange={(e) => setAadhaar(e.target.value.replace(/\D/g, ''))}
                />
              </Field>
            </div>

            <Field label="Document type" htmlFor="documentType" required>
              <select
                id="documentType"
                className="input"
                value={documentType}
                onChange={(e) => setDocumentType(e.target.value)}
              >
                <option value="passport">Passport</option>
                <option value="driving_licence">Driving licence</option>
                <option value="voter_id">Voter ID</option>
                <option value="aadhaar_card">Aadhaar card</option>
              </select>
            </Field>

            <Field
              label="Document upload"
              htmlFor="document"
              error={errors.document_name}
              hint="Nothing is uploaded — only the filename is recorded"
              required
            >
              <label
                htmlFor="document"
                className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-line-strong px-3.5 py-3 text-sm text-muted transition hover:border-gold/60 hover:text-primary"
              >
                {documentName ? (
                  <>
                    <FileText className="h-4 w-4 text-positive" aria-hidden />
                    <span className="truncate text-primary">{documentName}</span>
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" aria-hidden />
                    Choose a file
                  </>
                )}
                <input
                  id="document"
                  type="file"
                  className="sr-only"
                  accept="image/*,.pdf"
                  onChange={(e) => setDocumentName(e.target.files?.[0]?.name ?? '')}
                />
              </label>
            </Field>

            <Field label="Street address" htmlFor="address1" error={errors.address_line1} required>
              <input
                id="address1"
                className={clsx('input', errors.address_line1 && 'input-error')}
                value={address1}
                onChange={(e) => setAddress1(e.target.value)}
                autoComplete="address-line1"
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="City" htmlFor="city" error={errors.city} required>
                <input
                  id="city"
                  className={clsx('input', errors.city && 'input-error')}
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                />
              </Field>
              <Field label="State" htmlFor="state" error={errors.state} required>
                <input
                  id="state"
                  className={clsx('input', errors.state && 'input-error')}
                  value={state}
                  onChange={(e) => setState(e.target.value)}
                />
              </Field>
              <Field label="Postal code" htmlFor="postal" error={errors.postal_code} required>
                <input
                  id="postal"
                  className={clsx('input', errors.postal_code && 'input-error')}
                  value={postal}
                  onChange={(e) => setPostal(e.target.value)}
                />
              </Field>
            </div>

            <div className="flex gap-3">
              <button type="button" className="btn-secondary flex-1 py-2.5" onClick={() => setStep(0)}>
                <ArrowLeft className="h-4 w-4" aria-hidden />
                Back
              </button>
              <button type="submit" className="btn-primary flex-1 py-2.5">
                Continue
                <ArrowRight className="h-4 w-4" aria-hidden />
              </button>
            </div>
          </form>
        )}

        {/* ---------------------------- step 3 ---------------------------- */}
        {step === 2 && (
          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              void submitKyc();
            }}
          >
            <Notice tone="info" title="Why we ask">
              These figures feed the credit-scoring model. They determine your eligibility and
              interest rate if you later apply for a loan.
            </Notice>

            <Field label="Annual income (INR)" htmlFor="income" error={errors.annual_income} required>
              <input
                id="income"
                inputMode="numeric"
                className={clsx('input tnum', errors.annual_income && 'input-error')}
                value={income}
                onChange={(e) => setIncome(e.target.value.replace(/[^\d.]/g, ''))}
                placeholder="900000"
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Employment status" htmlFor="employment" required>
                <select
                  id="employment"
                  className="input"
                  value={employment}
                  onChange={(e) => setEmployment(e.target.value)}
                >
                  {EMPLOYMENT.map((value) => (
                    <option key={value} value={value}>
                      {value.replace('_', ' ').replace(/^\w/, (c) => c.toUpperCase())}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Years in employment" htmlFor="employmentYears">
                <input
                  id="employmentYears"
                  inputMode="decimal"
                  className="input tnum"
                  value={employmentYears}
                  onChange={(e) => setEmploymentYears(e.target.value.replace(/[^\d.]/g, ''))}
                />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Dependents" htmlFor="dependents">
                <input
                  id="dependents"
                  inputMode="numeric"
                  className="input tnum"
                  value={dependents}
                  onChange={(e) => setDependents(e.target.value.replace(/\D/g, ''))}
                />
              </Field>
              <Field label="Housing" htmlFor="housing">
                <select id="housing" className="input" value={housing} onChange={(e) => setHousing(e.target.value)}>
                  <option value="rent">Rented</option>
                  <option value="own">Owned</option>
                  <option value="mortgage">Mortgaged</option>
                  <option value="family">Family owned</option>
                </select>
              </Field>
            </div>

            <div className="flex gap-3">
              <button type="button" className="btn-secondary flex-1 py-2.5" onClick={() => setStep(1)} disabled={busy}>
                <ArrowLeft className="h-4 w-4" aria-hidden />
                Back
              </button>
              <button type="submit" className="btn-primary flex-1 py-2.5" disabled={busy}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <ShieldCheck className="h-4 w-4" aria-hidden />}
                {busy ? 'Verifying…' : 'Complete verification'}
              </button>
            </div>
          </form>
        )}

        {/* ---------------------------- step 4 ---------------------------- */}
        {step === 3 && (
          <div className="py-6 text-center">
            <span className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-full bg-positive/15 text-positive">
              <Check className="h-7 w-7" aria-hidden />
            </span>
            <h2 className="text-xl font-bold text-primary">You are verified</h2>
            <p className="mx-auto mt-2 max-w-sm text-sm text-muted">
              Your savings account is open and ready. Fund it from the Accounts page, then try a
              transfer to watch the fraud model score it in real time.
            </p>
            <button type="button" className="btn-primary mt-7 px-6 py-2.5" onClick={() => navigate('/app')}>
              Go to dashboard
              <ArrowRight className="h-4 w-4" aria-hidden />
            </button>
          </div>
        )}
      </Card>
    </div>
  );
}
