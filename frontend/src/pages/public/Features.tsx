import { useQuery } from '@tanstack/react-query';
import { Activity, Brain, Database, GitBranch, Lock, ShieldCheck, TrendingUp, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

import { Badge, Card } from '../../components/ui';
import { get } from '../../lib/api';
import { qk } from '../../lib/query';
import type { ModelStatusEntry } from '../../types/api';

/**
 * Non-technical explanation of the three models, with live metrics.
 *
 * Written for a visitor who has not read the README: what each model does, why
 * it exists, and what the numbers mean — without assuming ML vocabulary.
 */
function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line py-2 last:border-0">
      <span className="text-xs text-muted">{label}</span>
      <span className="tnum text-sm font-semibold text-primary">{value}</span>
    </div>
  );
}

export default function Features() {
  const { data } = useQuery({
    queryKey: qk.mlStatus,
    queryFn: () => get<Record<string, ModelStatusEntry>>('/ml/status'),
    staleTime: 5 * 60_000,
    retry: false,
  });

  const fraud = data?.fraud?.metrics as Record<string, number> | undefined;
  const credit = data?.credit?.metrics as Record<string, number> | undefined;
  const anomaly = data?.anomaly?.metrics as Record<string, number> | undefined;

  const pct = (v: number | undefined, fallback: string) =>
    v !== undefined ? `${(v * 100).toFixed(1)}%` : fallback;
  const dec = (v: number | undefined, fallback: string) =>
    v !== undefined ? v.toFixed(3) : fallback;

  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="mx-auto mb-14 max-w-2xl text-center">
        <Badge tone="info" className="mb-4">
          <Brain className="h-3.5 w-3.5" aria-hidden />
          How it works
        </Badge>
        <h1 className="text-3xl font-bold text-primary sm:text-4xl">
          Machine learning doing actual banking work
        </h1>
        <p className="mt-4 text-muted">
          Three models run inside the transaction flow — not as a demo bolted on the side. Here is
          what each one does and how well it does it.
        </p>
      </div>

      {/* ------------------------------ fraud ------------------------------ */}
      <section className="mb-12">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Card>
            <span className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-alert/10 text-alert">
              <ShieldCheck className="h-5.5 w-5.5" aria-hidden />
            </span>
            <h2 className="text-xl font-semibold text-primary">1. Fraud detection</h2>
            <p className="mt-1 font-mono text-xs text-muted">
              XGBoost gradient-boosted trees · supervised classification
            </p>

            <p className="mt-4 text-sm text-muted">
              Every transfer is scored before the money moves. The model compares the transaction
              against <strong className="text-primary">your own history</strong> — not a generic
              rule like "flag anything over ₹50,000". A ₹80,000 transfer is unremarkable if you do
              that monthly, and highly suspicious if your largest ever transfer was ₹5,000.
            </p>

            <p className="mt-3 text-sm text-muted">
              It weighs 24 signals: how far the amount deviates from your norm, how many
              transactions happened in the last hour, whether the device and city are ones you have
              used before, whether recent attempts were declined, and the time of day.
            </p>

            <div className="mt-5 rounded-lg border border-line bg-ink/50 p-4">
              <p className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">
                The hard part
              </p>
              <p className="text-sm text-muted">
                Roughly 4 in 1,000 transactions are fraudulent. A model that simply answers "not
                fraud" every time would be 99.6% accurate and completely useless. So accuracy is
                ignored entirely — the model is tuned to catch as much fraud as possible while
                keeping false alarms low enough that a review queue stays manageable.
              </p>
            </div>
          </Card>

          <Card>
            <p className="mb-3 text-xs font-semibold tracking-wide text-muted uppercase">
              Measured performance
            </p>
            <MetricRow label="Fraud caught (recall)" value={pct(fraud?.recall, '93.4%')} />
            <MetricRow label="Alerts that are real (precision)" value={pct(fraud?.precision, '85.0%')} />
            <MetricRow label="PR-AUC" value={dec(fraud?.pr_auc, '0.965')} />
            <MetricRow label="ROC-AUC" value={dec(fraud?.roc_auc, '0.999')} />
            <MetricRow
              label="Scoring time (p95)"
              value={
                data?.fraud?.latency_benchmark?.p95_ms
                  ? `${data.fraud.latency_benchmark.p95_ms.toFixed(1)} ms`
                  : '3.7 ms'
              }
            />
            <p className="mt-4 text-xs text-faint">
              Out of every 100 fraudulent transactions, roughly 93 are caught. Of the alerts raised,
              about 85 in 100 are genuine fraud rather than false alarms.
            </p>
          </Card>
        </div>
      </section>

      {/* ------------------------------ credit ------------------------------ */}
      <section className="mb-12">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Card>
            <span className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-positive/10 text-positive">
              <TrendingUp className="h-5.5 w-5.5" aria-hidden />
            </span>
            <h2 className="text-xl font-semibold text-primary">2. Credit scoring</h2>
            <p className="mt-1 font-mono text-xs text-muted">
              XGBoost + isotonic calibration · monotone constraints
            </p>

            <p className="mt-4 text-sm text-muted">
              When you apply for a loan, the decision is based on how you actually bank — average
              balance, how volatile it is, whether you spend more than you earn, existing
              commitments, repayment history — rather than only what you type into a form. You
              cannot inflate a transaction record.
            </p>

            <p className="mt-3 text-sm text-muted">
              The output is a 300–900 score, a risk band from A to E, and an interest rate. Crucially
              you also see <strong className="text-primary">which factors drove the decision</strong>{' '}
              and in which direction.
            </p>

            <div className="mt-5 rounded-lg border border-line bg-ink/50 p-4">
              <p className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">
                Why calibration matters
              </p>
              <p className="text-sm text-muted">
                The predicted probability becomes an interest rate, so it has to be truthful, not
                merely well-ordered. If the model says "12% chance of default", then about 12 in 100
                such applicants should actually default. Calibration reduced this error by 96%,
                which is the difference between fair pricing and systematically overcharging.
              </p>
            </div>
          </Card>

          <Card>
            <p className="mb-3 text-xs font-semibold tracking-wide text-muted uppercase">
              Measured performance
            </p>
            <MetricRow label="ROC-AUC" value={dec(credit?.roc_auc, '0.785')} />
            <MetricRow label="Gini coefficient" value={dec(credit?.gini, '0.571')} />
            <MetricRow label="KS statistic" value={dec(credit?.ks_statistic, '0.431')} />
            <MetricRow label="Calibration error" value={dec(credit?.ece, '0.008')} />
            <p className="mt-4 text-xs text-faint">
              A Gini of 0.57 is in the range banks consider strong for consumer lending. Perfect
              separation is not achievable — and any model claiming it on real credit data is
              almost certainly leaking information from the future.
            </p>
          </Card>
        </div>
      </section>

      {/* ------------------------------ anomaly ------------------------------ */}
      <section className="mb-14">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Card>
            <span className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-intelligence/10 text-intelligence">
              <Zap className="h-5.5 w-5.5" aria-hidden />
            </span>
            <h2 className="text-xl font-semibold text-primary">3. Spending anomaly detection</h2>
            <p className="mt-1 font-mono text-xs text-muted">
              Isolation Forest · unsupervised learning
            </p>

            <p className="mt-4 text-sm text-muted">
              This one is not about theft. It answers a softer question:{' '}
              <em className="text-primary">does this look like you?</em> It learns your personal
              rhythm — which categories you spend on, roughly how much, at what times — and flags
              genuine departures from it.
            </p>

            <p className="mt-3 text-sm text-muted">
              You get plain language, not a score: <em>"You spent about 3.2x your usual weekly
              amount on dining."</em> No money is ever blocked by this model. It only informs.
            </p>

            <div className="mt-5 rounded-lg border border-line bg-ink/50 p-4">
              <p className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">
                Learning without labels
              </p>
              <p className="text-sm text-muted">
                Nobody labels their own spending as "unusual", so there is no answer key to learn
                from. This model instead isolates points that sit far from everything else — the
                statistical outliers. It is the unsupervised counterpart to the fraud model, which
                learns from labelled examples.
              </p>
            </div>
          </Card>

          <Card>
            <p className="mb-3 text-xs font-semibold tracking-wide text-muted uppercase">
              Measured performance
            </p>
            <MetricRow label="ROC-AUC" value={dec(anomaly?.roc_auc, '0.854')} />
            <MetricRow label="PR-AUC" value={dec(anomaly?.pr_auc, '0.362')} />
            <MetricRow label="Alert rate" value="~3% of transactions" />
            <p className="mt-4 text-xs text-faint">
              The alert rate is anchored deliberately: a fixed share of the most unusual activity is
              surfaced, so a quiet week does not flood you with notifications.
            </p>
          </Card>
        </div>
      </section>

      {/* --------------------------- engineering --------------------------- */}
      <section className="mb-14">
        <h2 className="mb-6 text-center text-2xl font-bold text-primary">
          The engineering behind the models
        </h2>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {[
            {
              icon: GitBranch,
              title: 'No train/serve skew',
              body: 'Feature definitions live in one module. Training and live scoring call the same functions, so the model cannot silently receive different inputs in production than it learned from.',
            },
            {
              icon: Database,
              title: 'No data leakage',
              body: 'Train/test splits are grouped by customer, so no one appears on both sides. Thresholds are tuned on a validation set; the test set is scored exactly once.',
            },
            {
              icon: Activity,
              title: 'Drift monitoring',
              body: 'Live score distributions are compared against the training baseline with PSI, so a model quietly degrading in production is visible rather than invisible.',
            },
            {
              icon: Brain,
              title: 'Explainable decisions',
              body: 'Every fraud flag and credit decision ships with SHAP contributions naming the factors that drove it — no unexplainable verdicts.',
            },
            {
              icon: Lock,
              title: 'Fails safe',
              body: 'If a model artifact is missing, scoring degrades to deterministic rules rather than failing. Money movement never depends on a model file loading.',
            },
            {
              icon: ShieldCheck,
              title: 'Human in the loop',
              body: 'Analyst verdicts on flagged transactions become the labelled ground truth for the next retraining run, closing the feedback loop.',
            },
          ].map((item) => (
            <Card key={item.title}>
              <item.icon className="mb-3 h-5 w-5 text-gold" aria-hidden />
              <h3 className="font-semibold text-primary">{item.title}</h3>
              <p className="mt-2 text-sm text-muted">{item.body}</p>
            </Card>
          ))}
        </div>
      </section>

      <div className="rounded-xl border border-line bg-surface/50 p-8 text-center">
        <h2 className="text-xl font-bold text-primary">Try it yourself</h2>
        <p className="mx-auto mt-2 max-w-md text-sm text-muted">
          Sign in with a demo account and make a transfer — the fraud score, decision and reasoning
          appear immediately.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          <Link to="/login" className="btn-primary px-5 py-2.5">
            Use a demo account
          </Link>
          <Link to="/register" className="btn-secondary px-5 py-2.5">
            Open your own
          </Link>
        </div>
      </div>
    </div>
  );
}
