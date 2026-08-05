# Deployment Guide

This guide walks through deploying IntelliBank: backend on Render with managed Postgres, frontend on Vercel.

## Architecture

```
User Browser
    ↓
Vercel (Frontend SPA)
    ↓ CORS
Render (FastAPI Backend)
    ↓
Render Managed Postgres
```

The frontend and backend are deployed separately because Vercel's serverless Python runtime has a 250 MB bundle limit, and this app's ML dependencies measure 534 MB before any application code.

---

## Prerequisites

1. **GitHub repo pushed** — https://github.com/rishijain544/IntelliBank
2. **Render account** — https://render.com (free tier sufficient)
3. **Vercel account** — https://vercel.com (free tier sufficient)
4. **Gemini API key** (optional) — https://aistudio.google.com/app/apikey
   - Without one the assistant degrades to a rules-based router; the rest of the app works normally

---

## Step 1: Deploy Backend (Render)

### 1.1 Create a New Blueprint Instance

1. Go to https://dashboard.render.com/blueprints
2. Click **New Blueprint Instance**
3. Connect your GitHub account if not already connected
4. Select the `rishijain544/IntelliBank` repository
5. Render reads `render.yaml` and provisions:
   - One **web service** (`intellibank-api`)
   - One **Postgres database** (`intellibank-db`)

### 1.2 Set Required Environment Variables

Before deploying, click on the `intellibank-api` service in the blueprint screen and add:

| Variable | Value | Notes |
|---|---|---|
| `JWT_SECRET` | *(generate one)* | **Required.** Run: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `CORS_ORIGINS` | *(leave empty for now)* | Will be set after Step 2 once you have the Vercel URL |
| `GEMINI_API_KEY` | *(your key or leave empty)* | Optional. Assistant works without it. |

`DATABASE_URL` is auto-filled from the Postgres service.

### 1.3 Deploy

Click **Apply** and wait ~5-8 minutes for:
- Dependency installation (534 MB of packages)
- Model artifacts verified (3 models already committed)
- Schema initialization on Postgres
- First health check

Once live, note the service URL — it will look like:
```
https://intellibank-api-xxxx.onrender.com
```

### 1.4 Verify Backend is Up

Visit `https://intellibank-api-xxxx.onrender.com/docs` — you should see the OpenAPI documentation with 63 endpoints.

The service sleeps after 15 minutes of inactivity on the free tier. First request after sleep takes ~50s (cold start).

---

## Step 2: Deploy Frontend (Vercel)

### 2.1 Import the Repo

1. Go to https://vercel.com/new
2. Import `rishijain544/IntelliBank`
3. **Root Directory**: leave as `.` (Vercel auto-detects `frontend/` from `vercel.json`)
4. **Framework Preset**: Vite
5. **Build Command**: `npm run build`
6. **Output Directory**: `dist`

### 2.2 Set Environment Variable

Before deploying, add one environment variable:

| Variable | Value |
|---|---|
| `VITE_API_URL` | `https://intellibank-api-xxxx.onrender.com/api/v1` |

Replace `xxxx` with your actual Render service URL from Step 1.3.

> **The `/api/v1` suffix is required.** The axios client appends only the endpoint
> path (for example `/auth/login`) to this value. Omitting the suffix produces
> 404s on every request, with no error at build time.

### 2.3 Deploy

Click **Deploy** and wait ~2 minutes.

Once live, Vercel assigns a URL like:
```
https://intellibank-something.vercel.app
```

---

## Step 3: Wire CORS

The backend must allow requests from the frontend's origin.

1. Go back to your Render dashboard → `intellibank-api` → Environment
2. Set **`CORS_ORIGINS`** to your Vercel URL:
   ```
   https://intellibank-something.vercel.app
   ```
   (No trailing slash, exact match required)
3. Save — Render redeploys automatically (~30s)

---

## Step 4: Seed Demo Data (Optional but Recommended)

The database starts empty. To populate it with 5 demo users and realistic transaction history:

1. Go to Render dashboard → `intellibank-api` → Shell
2. Run:
   ```bash
   python manage.py seed --reset
   ```
   Takes ~5-10 seconds. You'll see:
   ```
   SEED COMPLETE
     users                 5
     accounts              6
     transactions      1,398
     fraud_alerts          2
     cards                 5
     loans                 2
   ```

**Demo credentials:**
- Customer: `priya@intellibank.dev` / `Demo@Pass123`
- Admin: `admin@intellibank.dev` / `Admin@Pass123`

---

## Step 5: Verify End-to-End

1. Open your Vercel URL
2. Click **Sign In**
3. Enter `priya@intellibank.dev` / `Demo@Pass123`
4. You should land on the Dashboard showing:
   - Account balances
   - Recent transactions
   - ML-powered fraud scores visible on the Fraud Center page
5. Try a transfer — fraud detection runs live, <200ms

---

## Post-Deployment Checklist

### Security

- [x] `JWT_SECRET` is unique and **not** the dev default
- [x] `CORS_ORIGINS` lists only your Vercel URL, never `*`
- [x] `COOKIE_SECURE=true` (already set in `render.yaml`)
- [ ] Rotate the Gemini API key you used during local dev (treat it as exposed)

### Monitoring

**Render Free Tier Limits:**
- 512 MB RAM — the app uses ~350 MB per worker, so 1 worker is configured
- 750 hours/month — enough for constant uptime, but the service sleeps after inactivity
- Database: 90-day expiration, 1 GB storage

**Watch for:**
- Memory spikes during cold start (all 3 models load at once)
- First request after sleep: ~50s response time
- Postgres free tier expires in 90 days — upgrade to Starter ($7/mo) or accept re-seeding

### Costs

**Free Configuration (what this guide uses):**
- Render Web Service: $0
- Render Postgres: $0 (90-day limit)
- Vercel Hosting: $0
- Total: **$0/month**

**Production-Ready Upgrade:**
- Render Web Service Starter: $7/mo (always-on, no cold start)
- Render Postgres Starter: $7/mo (10 GB, persistent)
- Vercel Pro: $20/mo (custom domain, analytics)
- Total: ~$34/mo

---

## Troubleshooting

### Backend won't start

**Symptom:** Render build succeeds but service fails health check

**Check:**
1. Render logs → look for `FATAL CONFIG:` lines
2. Common causes:
   - `JWT_SECRET` is still the dev default → service refuses to boot
   - `CORS_ORIGINS` is `*` → blocked by security checks
3. Fix the env var and Render redeploys automatically

### Frontend loads but API calls fail

**Check:**
1. Browser DevTools → Network tab → look for CORS errors
2. If you see `Access-Control-Allow-Origin` errors:
   - Verify `CORS_ORIGINS` on Render matches your Vercel URL exactly
   - No trailing slash, case-sensitive
3. If Render service is asleep (15min inactivity), first request takes ~50s

### Database connection errors

**Symptom:** `could not translate host name` or `connection refused`

**Fix:**
- Render's internal `DATABASE_URL` should auto-populate
- If you're seeing SQLite errors, check that `DATABASE_URL` is actually set in Render Environment

### Models not loading

**Symptom:** Dashboard shows "Risk scoring will fall back to rules"

**Check:**
1. Render logs → startup should show:
   ```
   Model fraud_xgb: loaded
   Model credit_xgb: loaded
   Model anomaly_iforest: loaded
   ```
2. If all show `NOT TRAINED`:
   - The 3 `.joblib` files and their `*_metrics.json` sidecars should be in the repo
   - Check GitHub → `backend/ml_artifacts/` → should have 7 files
   - If missing, re-push from local or run `python manage.py train --all` in Render Shell

### Assistant not responding

**Symptom:** Chat widget sends a message but gets no reply

**Cause:** `GEMINI_API_KEY` is unset or invalid

**Fix:**
- Set a valid key in Render Environment, or
- Accept the fallback: assistant still works, but responses are rule-based templates rather than LLM-generated

---

## Environment Variables Reference

### Backend (Render)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `APP_ENV` | No | `development` | Set to `production` (already in `render.yaml`) |
| `APP_NAME` | No | `IntelliBank` | Branding |
| `DATABASE_URL` | Yes | *(from Render)* | Auto-populated by blueprint |
| `JWT_SECRET` | **Yes** | *(generate)* | Must be 32+ bytes, unique per deploy |
| `CORS_ORIGINS` | **Yes** | *(empty)* | Your Vercel URL, comma-separated if multiple |
| `COOKIE_SECURE` | No | `true` | Leave as `true` behind Render's TLS |
| `REDIS_URL` | No | *(empty)* | Optional. In-memory fallback works for 1 worker. |
| `GEMINI_API_KEY` | No | *(empty)* | Optional. Assistant degrades gracefully without it. |
| `ASSISTANT_ENABLED` | No | `true` | Set to `false` to disable the assistant entirely |
| `DEBUG` | No | `false` | Never `true` in production |

### Frontend (Vercel)

| Variable | Required | Notes |
|---|---|---|
| `VITE_API_URL` | **Yes** | Render backend URL INCLUDING the /api/v1 suffix, no trailing slash |

---

## Updating the Deployment

### Code Changes

1. Push to `main` branch on GitHub
2. Render and Vercel auto-deploy (both watch the repo)
3. Render build: ~5 min, Vercel build: ~2 min

### Retrain Models

If you change the ML code or want to regenerate with different seeds:

```bash
# In Render Shell
python manage.py train --all
```

Takes ~10 minutes. The new `.joblib` files are written to the ephemeral filesystem and lost on restart unless you commit them back to the repo.

### Schema Changes

For SQLAlchemy model changes:

```bash
# In Render Shell
python manage.py reset    # drops all tables, recreates schema, seeds demo data
```

Destroys existing data. For zero-downtime migrations use Alembic (not included).

---

## Limits and Disclaimers

**This is a portfolio/educational project.** It:
- Holds no real money
- Is not connected to any payment network
- Is not a licensed financial institution
- Must not store real personal or financial data

**ML Model Accuracy:**
- Fraud detection: 93.4% recall, but trained on synthetic data
- Credit scoring: calibrated on simulated applicants, not real credit bureau data
- Models drift if the production data distribution differs from training

**Free Tier Constraints:**
- Render: 512 MB RAM limits to 1 worker; scales vertically but not horizontally on free tier
- Postgres expires in 90 days on free tier
- Both services sleep with inactivity

For a live production banking system you would need:
- Real KYC/AML compliance
- PCI-DSS certification for card data
- Multi-region deployment
- Row-level security and encryption at rest
- Alembic migrations for zero-downtime schema changes
- Comprehensive audit trail (this app has one, but it's not immutable)

---

## Support

- **Backend issues:** Check Render logs first — explicit error messages at startup
- **Frontend issues:** Browser DevTools → Console tab for React errors
- **ML questions:** See `backend/README.md` for model architecture and metrics
- **Repo:** https://github.com/rishijain544/IntelliBank

Render's free tier is sufficient for a portfolio demo that will be viewed by recruiters and shown in interviews. Expect ~50s cold-start latency after 15min idle.
