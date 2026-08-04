import { useQuery } from '@tanstack/react-query';
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  Fingerprint,
  Gauge,
  Lock,
  ShieldCheck,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { Link } from 'react-router-dom';

import { Badge, Card } from '../../components/ui';
import { get } from '../../lib/api';
import { qk } from '../../lib/query';
import type { ModelStatusEntry } from '../../types/api';

/**
 * Marketing landing page.
 *
 * The headline metrics are fetched live from `/ml/status` rather than hardcoded,
 * so the numbers on the page always match the artifacts actually loaded by the
 * running backend. If the API is unreachable the section degrades to the
 * documented training figures instead of showing broken values.
 */
const FALLBACK = {
  fraudRecall: '93.4%',
  fraudPrecision: '85.0%',
  creditAuc: '0.785',
  anomalyAuc: '0.854',
  latency: '< 4 ms',
};

function useHeadlineMetrics() {
  const { data } = useQuery({
    queryKey: qk.mlStatus,
    queryFn: () => get<Record<string, ModelStatusEntry>>('/ml/status'),
    staleTime: 5 * 60_000,
    retry: false,
  });

  if (!data) return FALLBACK;

  const fraud = data.fraud?.metrics as Record<string, number> | undefined;
  const credit = data.credit?.metrics as Record<string, number> | undefined;
  const anomaly = data.anomaly?.metrics as Record<string, number> | undefined;
  const p95 = data.fraud?.latency_benchmark?.p95_ms;

  return {
    fraudRecall: fraud?.recall ? `${(fraud.recall * 100).toFixed(1)}%` : FALLBACK.fraudRecall,
    fraudPrecision: fraud?.precision
      ? `${(fraud.precision * 100).toFixed(1)}%`
      : FALLBACK.fraudPrecision,
    creditAuc: credit?.roc_auc ? credit.roc_auc.toFixed(3) : FALLBACK.creditAuc,
    anomalyAuc: anomaly?.roc_auc ? anomaly.roc_auc.toFixed(3) : FALLBACK.anomalyAuc,
    latency: p95 ? `${p95 < 10 ? p95.toFixed(1) : Math.round(p95)} ms` : FALLBACK.latency,
  };
}

export default function Landing() {
  const metrics = useHeadlineMetrics();

  return (
    <div>
      {/* ------------------------------- hero ------------------------------- */}
      <section className="relative overflow-hidden">
        {/* Decorative gradient; hidden from assistive tech. */}
        <div
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_50%_0%,rgba(14,165,233,0.16),transparent)]"
          aria-hidden
        />
        <div className="relative mx-auto max-w-6xl px-4 py-20 text-center sm:py-28">
          <Badge tone="info" className="mb-5">
            <Brain className="h-3.5 w-3.5" aria-hidden />
            Three production ML models, wired into real banking logic
          </Badge>

          <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight text-primary sm:text-6xl">
            Banking that catches fraud{' '}
            <span className="bg-gradient-to-r from-gold to-gold-bright bg-clip-text text-transparent">
              before the money moves
            </span>
          </h1>

          {/* Brand tagline, sits between the headline and the supporting copy. */}
          <p className="mx-auto mt-5 max-w-2xl text-xl font-medium text-primary sm:text-2xl">
            AI-Powered Banking for Everyone.
          </p>

          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Every transfer is scored in milliseconds. Loans are priced by a calibrated credit model.
            Unusual spending is surfaced before you notice it yourself.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <Link to="/register" className="btn-primary px-6 py-3 text-base">
              Open an account
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Link>
            <Link to="/features" className="btn-secondary px-6 py-3 text-base">
              How the models work
            </Link>
          </div>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-x-7 gap-y-3 text-xs text-muted">
            <span className="flex items-center gap-1.5">
              <Lock className="h-3.5 w-3.5" aria-hidden /> bcrypt + JWT rotation
            </span>
            <span className="flex items-center gap-1.5">
              <Fingerprint className="h-3.5 w-3.5" aria-hidden /> TOTP two-factor
            </span>
            <span className="flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5" aria-hidden /> {metrics.latency} p95 scoring
            </span>
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5" aria-hidden /> Full audit trail
            </span>
          </div>
        </div>
      </section>

      {/* ----------------------------- metrics ----------------------------- */}
      <section className="mx-auto max-w-6xl px-4 pb-20">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { label: 'Fraud recall', value: metrics.fraudRecall, note: `at ${metrics.fraudPrecision} precision` },
            { label: 'Credit ROC-AUC', value: metrics.creditAuc, note: 'calibrated, ECE 0.008' },
            { label: 'Anomaly ROC-AUC', value: metrics.anomalyAuc, note: 'unsupervised detection' },
            { label: 'Scoring latency', value: metrics.latency, note: 'p95, 200 ms budget' },
          ].map((m) => (
            <Card key={m.label} className="text-center">
              <p className="text-xs font-medium tracking-wide text-muted uppercase">{m.label}</p>
              <p className="tnum mt-2 text-3xl font-bold text-primary">{m.value}</p>
              <p className="mt-1 text-xs text-muted">{m.note}</p>
            </Card>
          ))}
        </div>
        <p className="mt-3 text-center text-xs text-faint">
          Measured on held-out data with user-grouped splits. Trained on calibrated synthetic
          transactions — see the project README for the full methodology and its limitations.
        </p>
      </section>

      {/* ---------------------------- the models ---------------------------- */}
      <section className="border-t border-line bg-surface/30">
        <div className="mx-auto max-w-6xl px-4 py-20">
          <div className="mx-auto mb-12 max-w-2xl text-center">
            <h2 className="text-3xl font-bold text-primary">Three models, three different jobs</h2>
            <p className="mt-3 text-muted">
              Not one model doing everything badly — a supervised classifier, a calibrated
              regressor and an unsupervised detector, each chosen for the problem it fits.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {[
              {
                icon: ShieldCheck,
                tone: 'text-alert bg-alert/10',
                title: 'Fraud detection',
                model: 'XGBoost · supervised',
                body: 'Scores every transfer against your own history — amount deviation, velocity, device and location novelty, declined attempts. High-risk transfers are held before the money leaves.',
                points: ['0.4% positive class handled with SMOTE', 'Explains why it flagged, not just a score', 'Never auto-blocks a brand-new account'],
              },
              {
                icon: TrendingUp,
                tone: 'text-positive bg-positive/10',
                title: 'Credit scoring',
                model: 'XGBoost + isotonic calibration',
                body: 'Prices loans from your real account behaviour rather than a form. Produces a 300–900 score, a risk band and an interest rate you can see the reasoning behind.',
                points: ['Calibrated so the rate is honest', 'Monotone constraints on risk drivers', 'Affordability capped, not just score-gated'],
              },
              {
                icon: Zap,
                tone: 'text-intelligence bg-intelligence/10',
                title: 'Anomaly detection',
                model: 'Isolation Forest · unsupervised',
                body: 'Learns what normal looks like for you specifically, then tells you in plain language when something drifts — "3.2x your usual dining spend this week".',
                points: ['Learns per-user baselines', 'Never blocks money, only informs', 'Stable alert rate by design'],
              },
            ].map((m) => (
              <Card key={m.title} className="flex flex-col">
                <span className={`mb-4 grid h-11 w-11 place-items-center rounded-xl ${m.tone}`}>
                  <m.icon className="h-5.5 w-5.5" aria-hidden />
                </span>
                <h3 className="text-lg font-semibold text-primary">{m.title}</h3>
                <p className="mt-0.5 font-mono text-xs text-muted">{m.model}</p>
                <p className="mt-3 flex-1 text-sm text-muted">{m.body}</p>
                <ul className="mt-4 space-y-2 border-t border-line pt-4">
                  {m.points.map((p) => (
                    <li key={p} className="flex gap-2 text-xs text-muted">
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-positive" aria-hidden />
                      {p}
                    </li>
                  ))}
                </ul>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------- cta ------------------------------- */}
      <section className="mx-auto max-w-4xl px-4 py-20 text-center">
        <h2 className="text-3xl font-bold text-primary">See it work end to end</h2>
        <p className="mx-auto mt-3 max-w-xl text-muted">
          Open a simulated account, move some money and watch the fraud model score it live — or
          sign in with the demo credentials to explore a pre-populated account.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link to="/register" className="btn-primary px-6 py-3 text-base">
            Create an account
          </Link>
          <Link to="/login" className="btn-secondary px-6 py-3 text-base">
            Use demo credentials
          </Link>
        </div>
      </section>
    </div>
  );
}
