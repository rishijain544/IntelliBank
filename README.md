<div align="center">

# 🏦 IntelliBank — AI-Powered Banking for Everyone

**A full-stack banking platform where three machine-learning models are wired into real banking logic — not bolted on as a demo.**
Fraud scoring gates money movement, credit scoring prices loans, and anomaly detection drives customer insights.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-intelli--bank.vercel.app-1E90FF?style=for-the-badge&logo=vercel&logoColor=white)](https://intelli-bank.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#architecture)
[![Frontend](https://img.shields.io/badge/Frontend-React%2019-61DAFB?style=for-the-badge&logo=react&logoColor=black)](#architecture)

**🔗 [Live Demo → intelli-bank.vercel.app](https://intelli-bank.vercel.app/)**

</div>

> ⚠️ **Educational / portfolio project.** This is a **simulated** banking system. It holds no real
> money, is not connected to any payment network, is not a licensed financial institution, and
> must not be used to store real personal or financial data. All accounts, balances, card numbers
> and identity documents are synthetic.

---

## 📸 Screenshots

<!--
  Drop your images into a `docs/screenshots/` folder in the repo root and keep these filenames
  (or update the paths below to match whatever you use). GitHub and most markdown renderers will
  pick them up automatically once they exist in the repo.
-->

<table>
  <tr>
    <td width="50%" align="center">
      <b>Landing Page</b><br/>
      <img src="docs/screenshots/landing.png" alt="IntelliBank landing page with live model metrics" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>Customer Dashboard</b><br/>
      <img src="docs/screenshots/dashboard.png" alt="Customer dashboard with accounts overview" width="100%"/>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Transfer — Live Fraud Score</b><br/>
      <img src="docs/screenshots/transfer-fraud-score.png" alt="Transfer page showing live fraud score and contributing factors" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>Loans — Live Credit Decision</b><br/>
      <img src="docs/screenshots/loans-credit-decision.png" alt="Loan application with live credit decision and explanation" width="100%"/>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Admin — Fraud Review Queue</b><br/>
      <img src="docs/screenshots/admin-fraud-queue.png" alt="Admin fraud review queue" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>Admin — Model Analytics & PSI Drift</b><br/>
      <img src="docs/screenshots/admin-drift.png" alt="Admin model analytics with PSI drift monitoring" width="100%"/>
    </td>
  </tr>
</table>

*Screenshots pending — add your own images to `docs/screenshots/` using the filenames above, or swap in your own paths.*

---

## Table of Contents

- [Why this project is different](#why-this-project-is-different)
- [Model results](#model-results)
- [Architecture](#architecture)
- [Security](#security)
- [Getting started](#getting-started)
- [Demo credentials](#demo-credentials)
- [Commands](#commands)
- [API](#api)
- [Pages](#pages)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

---

## Why this project is different

Most "ML + web app" projects call `model.predict()` in a route handler and stop there. The
engineering problems that actually matter in a production ML system are addressed here explicitly:

| Problem | How it is handled |
|---|---|
| **Train/serve skew** | Feature definitions live in one module (`app/ml/schema.py`) with a single set of pure builders (`app/ml/features.py`). The offline simulator and the live SQL path both call the *same* functions, so a training matrix cannot silently diverge from a serving vector. |
| **Class imbalance** | Fraud is 0.4% positive. Accuracy is meaningless, so the model is selected on PR-AUC and the operating threshold is chosen from the precision–recall curve under a precision floor. SMOTE is implemented in-repo (`app/ml/sampling.py`) and compared empirically against class weighting and Borderline-SMOTE. |
| **Data leakage** | Splits are grouped by `user_id`, so no customer's behavioural baseline appears on both sides of the train/test boundary. Thresholds are tuned on a validation fold; the test fold is scored exactly once. |
| **Cold start** | The model treats "new device + new city + new merchant" as near-certain fraud, because every simulated training user has warm-up history. Real customers do not. Accounts below a history threshold are damped toward policy rules and are never auto-blocked. |
| **Calibration** | A credit PD becomes an interest rate, so ranking alone is insufficient. Isotonic calibration is compared against the raw model and selected on Expected Calibration Error, not just AUC. |
| **Explainability** | TreeSHAP contributions are returned with every fraud and credit decision, so the UI can state *why* something was flagged instead of showing a bare probability. |
| **Monotonicity** | A scorecard cannot claim that more prior defaults reduce risk. XGBoost monotone constraints enforce the direction of effect for features where the sign is not negotiable. |
| **Drift monitoring** | PSI is computed against the training score distribution, with the per-bin training shares persisted in the artifact — necessary because zero-inflated fraud scores break the usual "quantile bins are uniform" assumption. PSI is suppressed below 50 live scores, where it would be pure noise. |
| **Retraining loop** | Admin verdicts in the fraud review queue write `Transaction.is_fraud_label`, the ground truth for the next training run. Ambiguous cases are dismissed with a `NULL` label rather than becoming noisy training data. |
| **Model unavailability** | Money movement never depends on a model file. If an artifact is missing, fraud scoring degrades to a deterministic rule engine and reports `model_available: false`. |

---

## Model results

Measured on held-out data with grouped splits (no user appears in both train and test).
Reproduce with `python manage.py train`; written to `backend/ml_artifacts/training_summary.json`.

### 1. Fraud detection — XGBoost, supervised, 0.38% positive rate

| Metric | Value |
|---|---|
| **Recall** | **93.4%** |
| **Precision** | **85.0%** |
| PR-AUC | 0.965 |
| ROC-AUC | 0.9995 |
| KS statistic | 0.975 |
| **p95 inference latency** | **3.7 ms** (budget: 200 ms) |
| Dataset | 661,669 transactions, 2,522 fraud |

Selected strategy: **SMOTE at 0.30 target ratio**, chosen at train time by comparing class
weighting (val PR-AUC 0.946), SMOTE (0.951) and Borderline-SMOTE (0.950). Resampling is applied to
training folds only — validation and test keep the true 0.4% prior, because tuning against a
balanced validation set optimises a distribution that never occurs in production.

> **On the realism of these numbers.** An earlier version of the generator produced ROC-AUC 1.00
> and 99.8% recall. That was a *data* problem, not a modelling win: the synthetic fraud episodes
> were near-linearly separable because foreign country, unknown device, 3am timing and round
> amounts always co-occurred. The generator was rewritten so ~49% of fraud is "stealthy" (known
> device, home city, ordinary amount) and ~3.7% of legitimate activity looks superficially
> suspicious (holiday spending abroad, a new phone, a late-night order). The figures above come
> from that harder distribution.

**Features (24):** amount deviation from the user's own history, velocity windows (1h/24h),
device/location/merchant novelty, channel risk, declined-attempt count, cyclically-encoded
time-of-day, category rarity, balance drain ratio.

### 2. Credit scoring — XGBoost + isotonic calibration, monotone constraints

| Metric | Value |
|---|---|
| ROC-AUC | 0.785 |
| **Gini** | **0.571** |
| KS statistic | 0.431 |
| **Expected Calibration Error** | **0.008** (from 0.212 uncalibrated) |
| p95 latency | 16.8 ms |
| Dataset | 30,000 applicants, 17.9% default rate |

Calibration is the headline here, not AUC: the predicted default probability becomes an interest
rate, so a model that ranks well but reports inflated probabilities would overprice every loan.
Isotonic regression cut ECE by **96%** with negligible ranking loss.

A logistic-regression scorecard is trained as a genuine baseline (val ROC-AUC 0.799 vs 0.792 for
monotone XGBoost) — if the linear model had won outright, it would have been the defensible choice.
Monotone constraints prevent the model from ever learning that *more* prior defaults reduce risk.

**Output:** 300–900 score, risk band A–E, suggested rate, max eligible amount. Eligibility is
capped by both a band-specific income multiple *and* residual EMI affordability, so a high score
alone cannot approve an unaffordable loan.

### 3. Anomaly detection — Isolation Forest, unsupervised

| Metric | Value |
|---|---|
| ROC-AUC vs held-out labels | 0.854 |
| PR-AUC | 0.362 |
| Flag rate | 2.9% (target: 3%) |
| p95 latency | 131 ms |
| Dataset | 319,431 transactions, 3.0% anomalous |

Fitted on **unlabelled** data; labels exist only for evaluation. Scores are mapped to a 0–1
"unusualness" scale via the training distribution, and the alert threshold is anchored to a
training percentile so the production alert rate stays stable rather than flooding users during a
quiet week.

Output is a plain-language nudge — *"You spent about 3.2x your usual weekly amount on dining"* —
because a raw score is not actionable. **No money is ever blocked on an anomaly score.**

---

## Architecture

```
backend/
  app/
    core/          config, DB engine (SQLite/Postgres), security, cache, rate limiting
    models/        14 SQLAlchemy tables
    ml/
      schema.py    feature contracts (single source of truth)
      features.py  pure feature builders, shared by training and serving
      sampling.py  SMOTE / Borderline-SMOTE (in-repo)
      datasets.py  synthetic generators + Kaggle adapters
      metrics.py   PR-curve thresholds, calibration, PSI, latency percentiles
      registry.py  artifact bundling (estimator + features + threshold + baseline)
      train_*.py   three trainers
      inference.py runtime scoring, SHAP explanations, cold-start guard
    services/
      banking.py      accounts, ledger, transfer engine (fraud-gated)
      ml_features.py  SQL aggregates -> the shared feature contract
      exports.py      CSV / PDF statements
      notifications.py
    api/routes/    auth, profile, banking, loans, risk, admin
    main.py        app assembly, error mapping, security headers
    seed.py        demo data with realistic history
  tests/           78 tests

frontend/
  src/
    lib/
      api.ts       axios client with de-duplicated token refresh
      query.ts     TanStack Query client + centralised query keys
      format.ts    money/date formatting (Decimal strings parsed only at render)
      charts.ts    typed Recharts formatters
    store/auth.ts  Zustand auth store, bridged to the axios layer
    components/    shared UI primitives
    layouts/       public / customer / admin shells
    pages/
      public/      landing, features, login, register+KYC, contact
      app/         dashboard, accounts, transactions, transfer, cards,
                   loans, fraud center, insights, settings
      admin/       overview, users, fraud queue, loan queue, analytics
```

**Backend:** FastAPI · SQLAlchemy 2 · Pydantic v2 · XGBoost · scikit-learn · PyJWT · bcrypt ·
SQLite (Postgres-ready) · optional Redis

**Frontend:** React 19 · TypeScript 5.9 · Vite 7 · Tailwind CSS v4 · TanStack Query v5 ·
Recharts 3 · Zustand 5 · React Router 7

### Design notes

- **Money is `Decimal`, never `float`.** A custom `Money` type enforces 2dp on both backends, and
  the frontend keeps amounts as strings until the moment they are formatted for display.
- **The ledger is append-only.** A fraud reversal posts a compensating credit rather than mutating
  the original row.
- **Transfers are atomic.** Debit, credit, alert and notification commit in one transaction, so a
  mid-flight failure cannot debit without crediting.
- **Held funds are tracked separately.** A transfer under review debits the balance and increments
  `hold_amount`, so the money is neither spendable nor lost.
- **Audit rows commit with their action**, never independently.
- **Token refresh is de-duplicated.** Concurrent 401s share one refresh promise; firing several
  would replay a rotated refresh token and trip the server's theft detection, logging the user out
  of every device.

---

## Security

- bcrypt at 12 rounds, with HMAC pre-hashing so long passphrases are not silently truncated at
  bcrypt's 72-byte limit
- JWT access + refresh tokens with **rotation and reuse detection** — replaying a rotated refresh
  token revokes the entire token family and signs out every device
- Role-based access control; account status is re-checked per request, so freezing a user takes
  effect immediately rather than at token expiry
- TOTP two-factor authentication; disabling it requires a valid code
- Rate limiting on login, registration and transfers
- Login failures return a uniform message, preventing account enumeration
- Government ID numbers are salted-hashed; only masked forms are stored
- Card numbers are hashed; only the last four digits are retained
- Account lockout after repeated failures
- Full audit trail of privileged actions
- Security headers on every response; HSTS only when TLS is configured

---

## Getting started

**Prerequisites:** Python 3.11+ (developed on 3.14). Node 18+ for the frontend.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

cp .env.example .env             # optional; defaults work out of the box

python manage.py bootstrap       # trains models + seeds a populated demo database
python manage.py serve           # http://127.0.0.1:8000/docs
```

`bootstrap` takes several minutes: it simulates ~650k transactions, trains three models and seeds
demo data. To move faster while developing:

```bash
python -m app.ml.train --all --quick   # smaller datasets, ~1 minute
```

Then start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

The Vite dev server proxies `/api` to the backend, so the app runs same-origin and CORS is not
load-bearing in development.

> 🌐 Prefer not to set anything up? Try the hosted version at
> **[intelli-bank.vercel.app](https://intelli-bank.vercel.app/)**.

---

## Demo credentials

| Role | Email | Password |
|---|---|---|
| Admin | `admin@intellibank.dev` | `Admin@Pass123` |
| Customer (affluent, has alerts) | `priya@intellibank.dev` | `Demo@Pass123` |
| Customer (average) | `arjun@intellibank.dev` | `Demo@Pass123` |
| Customer (thin file) | `kavya@intellibank.dev` | `Demo@Pass123` |
| Customer (volatile) | `rohan@intellibank.dev` | `Demo@Pass123` |

---

## Commands

```bash
# backend
python manage.py check      # environment + artifact report
python manage.py train      # retrain all models
python manage.py reset      # rebuild schema and reseed
python manage.py test       # run 78 tests
python manage.py serve 8000

# frontend
npm run dev                 # dev server with API proxy
npm run typecheck           # tsc -b (checks both tsconfig projects)
npm run build               # typecheck + production bundle
```

### PostgreSQL (optional)

```bash
docker compose up -d postgres
pip install "psycopg[binary]==3.2.10"
# set in .env:
DATABASE_URL=postgresql+psycopg://smartbank:smartbank@localhost:5432/smartbank
```

> **Note on internal names.** The product was renamed from SmartBank to IntelliBank, but three
> classes of identifier deliberately still read `smartbank`, because renaming them would break
> running installations rather than improve anything:
>
> | Identifier | Where | Why it was left alone |
> |---|---|---|
> | `smartbank.db` | `config.py`, `.env.example` | Renaming orphans any existing local database file. |
> | `smartbank` role / DB / container names | `docker-compose.yml` | Renaming orphans the existing Postgres volume, losing its data. |
> | `smartbank-prehash-v1` | `core/security.py` | This pepper is HMAC'd into every stored password hash. Changing it would invalidate every password and lock out all users. |
>
> None of these are user-visible. To rename the database or Postgres role on a fresh install,
> change them together in `config.py`, `.env`, and `docker-compose.yml`, then re-run
> `python manage.py reset`. The password pepper should not be changed on an existing install
> under any circumstances.

### Real Kaggle datasets (optional)

The models train on calibrated synthetic data by default, so the project is reproducible with no
downloads. To train on the real datasets, place them in `data/raw/` and pass `--kaggle`:

- `creditcard.csv` — Credit Card Fraud Detection (mlg-ulb)
- `german_credit.csv` — Statlog German Credit Data

```bash
python -m app.ml.train --all --kaggle
```

**Caveat, stated plainly:** `creditcard.csv` ships PCA-anonymised components (`V1..V28`) that
cannot be mapped to named banking signals. It therefore trains a **separate benchmark model** in
its own feature space; the model served by the API keeps the interpretable 24-feature schema,
because the Fraud Center has to explain its decisions. German Credit maps onto the credit schema
with documented approximations (1990s Deutsche Mark amounts, no transaction history).

---

## API

61 endpoints; interactive docs at `/docs`.

| Area | Endpoints |
|---|---|
| Auth | register, KYC, login, refresh, logout, logout-all, change password, 2FA setup/enable/disable |
| Accounts | list, open, detail, deposit |
| Transactions | filtered + paginated history, detail, CSV export, PDF statement |
| Transfers | internal, external (NEFT/IMPS/UPI simulated) — both fraud-gated |
| Beneficiaries | list, add, remove |
| Cards | issue, freeze/unfreeze, update limits, cancel |
| Loans | **live eligibility scoring**, apply, detail, accept/disburse, cancel |
| Risk | dashboard, insights, fraud alerts, confirm/dispute, security summary |
| Notifications | list, mark read, mark all read |
| Admin | stats, user management, KYC decisions, **fraud review queue**, loan approval queue, model performance + PSI drift, analytics, audit trail |

---

## Pages

19 pages across three shells.

**Public** — landing (live model metrics from `/ml/status`), features (non-technical explanation of
all three models), login (with demo account shortcuts), multi-step registration with simulated KYC,
support.

**Customer** — dashboard, accounts, transactions (filter/search/CSV/PDF), transfer (shows the live
fraud score, decision and contributing factors), cards, loans (live credit decision with
explanation), Security Center (confirm/dispute flagged transactions), insights (spending charts and
anomaly nudges), settings (2FA, password, devices, notification preferences).

**Admin** — platform overview, user management with KYC decisions, fraud review queue (writes the
retraining label), loan approval queue with model override, model analytics with PSI drift
monitoring and artifact hot-reload.

---

## Testing

```bash
cd backend && python manage.py test
```

78 tests covering ledger invariants (money is conserved across transfers), authorisation
boundaries, refresh-token theft detection, cold-start fraud behaviour, credit ranking
monotonicity, limit-hierarchy enforcement, and latency budgets.

Several tests exist specifically as regression guards for bugs found during development:
first-transaction auto-blocking, refresh-reuse detection being unreachable, and export routes
being shadowed by a dynamic path segment.

Frontend correctness is enforced by `npm run typecheck` in strict mode with
`noUnusedLocals`/`noUnusedParameters`, which caught a handful of real issues during the build —
including UI fields the API did not actually return.

### A note on the drift indicator

On seeded data the admin panel reports the fraud model as `drifting` (PSI ≈ 1.8). That is the
monitor working, not a defect: the seeder deliberately injects fraud episodes, so the live score
distribution genuinely differs from the training baseline. It is a useful demonstration of exactly
what the drift panel is for.

---

## Limitations

Stated honestly, because a resume project that overclaims is worse than one that scopes itself:

- Training data is **synthetic by default**. The generators are calibrated to realistic rates
  (0.4% fraud, ~18% loan default) with deliberate class overlap, but they are not real
  transactions, and metrics on simulated data are an upper bound on real-world performance.
- KYC and payment rails are simulated. No document verification, no NEFT/IMPS network.
- Email/SMS delivery is recorded as intent, not sent.
- The in-memory rate limiter is per-process; multi-worker deployments need `REDIS_URL`.
- Retraining is triggered manually. Labels accumulate through the admin queue, but there is no
  scheduled pipeline.
- No secrets manager, no HTTPS termination, no container image for the app itself.
- The frontend has no component test suite. Type safety and the backend's 78 tests cover the
  contract between them, but UI behaviour is verified manually.
- Tokens are stored in `localStorage`, which is XSS-readable. Production would use httpOnly
  cookies; this trade-off was taken to keep the demo self-contained.

---

## License

MIT — for educational use. Not for handling real financial data.

<div align="center">

**[⬆ Back to top](#-intellibank--ai-powered-banking-for-everyone)** · **[🔗 Live Demo](https://intelli-bank.vercel.app/)**

</div>
