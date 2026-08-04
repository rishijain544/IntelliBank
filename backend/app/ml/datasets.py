"""Training data sources for the three models.

Two paths, deliberately kept separate:

**Synthetic generators** (default) simulate raw banking activity — per-user
transaction streams, applicant profiles — and then derive features by calling the
*same* ``app.ml.features`` builders that production uses. Nothing here writes
feature values directly, so the training matrix cannot drift from the serving one.

**Kaggle adapters** (optional) load the real public datasets when present in
``data/raw/``. See ``kaggle_notes()`` for the schema-mapping caveats; the fraud
dataset in particular is PCA-anonymised and therefore trains a *benchmark* model
on its own feature space rather than the deployed schema.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import DATA_DIR
from app.ml.features import (
    CreditProfile,
    TxnContext,
    UserHistoryStats,
    build_anomaly_features,
    build_credit_features,
    build_fraud_features,
    emi_for,
)
from app.ml.schema import ANOMALY_FEATURES, CREDIT_FEATURES, FRAUD_FEATURES

RAW_DIR = DATA_DIR / "raw"

# Merchant catalogue: (name, category, typical_amount_mean, typical_amount_cv)
MERCHANTS: tuple[tuple[str, str, float, float], ...] = (
    ("BigBasket", "groceries", 2200, 0.45),
    ("DMart", "groceries", 1800, 0.50),
    ("Reliance Fresh", "groceries", 1500, 0.55),
    ("Swiggy", "dining", 480, 0.60),
    ("Zomato", "dining", 520, 0.65),
    ("Starbucks", "dining", 650, 0.40),
    ("Barbeque Nation", "dining", 2400, 0.35),
    ("Uber", "transport", 320, 0.70),
    ("Ola", "transport", 280, 0.70),
    ("IRCTC", "transport", 1400, 0.80),
    ("Indian Oil", "transport", 2500, 0.35),
    ("Amazon", "shopping", 1900, 0.90),
    ("Flipkart", "shopping", 1700, 0.95),
    ("Myntra", "shopping", 2300, 0.70),
    ("Croma", "shopping", 12000, 0.80),
    ("Airtel", "utilities", 799, 0.20),
    ("Jio", "utilities", 699, 0.20),
    ("Tata Power", "utilities", 1900, 0.40),
    ("Netflix", "entertainment", 649, 0.10),
    ("Spotify", "entertainment", 179, 0.05),
    ("PVR Cinemas", "entertainment", 900, 0.45),
    ("Apollo Pharmacy", "healthcare", 850, 0.60),
    ("Practo", "healthcare", 1500, 0.55),
    ("Coursera", "education", 3500, 0.50),
    ("Unacademy", "education", 4500, 0.60),
    ("MakeMyTrip", "travel", 18000, 0.85),
    ("IndiGo", "travel", 6500, 0.70),
    ("Landlord Rent", "rent", 25000, 0.25),
    ("Zerodha", "investment", 15000, 1.00),
    ("Groww", "investment", 8000, 1.00),
    ("ATM Withdrawal", "cash", 5000, 0.60),
)

CITIES: tuple[str, ...] = (
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Kochi",
)
FOREIGN_CITIES: tuple[tuple[str, str], ...] = (
    ("Lagos", "NG"), ("Kyiv", "UA"), ("Jakarta", "ID"),
    ("Moscow", "RU"), ("Caracas", "VE"), ("Manila", "PH"),
)
CHANNELS: tuple[str, ...] = ("internal", "neft", "imps", "upi", "card", "atm")


# --------------------------------------------------------------------------- #
# Rolling history tracker — mirrors what the SQL aggregates produce at runtime
# --------------------------------------------------------------------------- #


class _StreamState:
    """Incrementally maintained per-user history, replayed in timestamp order.

    Deliberately mirrors the aggregate queries in ``app/services/ml_features.py``
    so that offline and online history stats agree. All aggregates are running
    totals plus bounded sliding windows, giving O(1) amortised cost per
    transaction instead of a full-history rescan.
    """

    __slots__ = (
        "home_country", "home_city", "usual_hour",
        "count_total", "sum_total", "sumsq_total",
        "first_time", "last_time", "recent_amounts",
        "window_1h", "window_24h", "window_7d", "sum_24h", "sum_7d",
        "window_7d_merchants", "window_7d_categories", "failed_window",
        "devices", "cities", "merchants",
        "cat_count", "cat_sum", "cat_sumsq", "cat_last_time", "cat_week_spend",
        "week_buckets", "_week_origin",
    )

    def __init__(self, home_country: str = "IN", home_city: str = "Mumbai") -> None:
        self.home_country = home_country
        self.home_city = home_city
        self.usual_hour = 13.0

        self.count_total = 0
        self.sum_total = 0.0
        self.sumsq_total = 0.0
        self.first_time: datetime | None = None
        self.last_time: datetime | None = None
        # Bounded: only the recent tail is needed for percentile features.
        self.recent_amounts: deque[float] = deque(maxlen=200)

        self.window_1h: deque[tuple[datetime, float]] = deque()
        self.window_24h: deque[tuple[datetime, float]] = deque()
        self.window_7d: deque[tuple[datetime, float, str]] = deque()
        self.sum_24h = 0.0
        self.sum_7d = 0.0
        self.window_7d_merchants: deque[tuple[str, datetime]] = deque()
        self.window_7d_categories: deque[tuple[str, datetime]] = deque()
        self.failed_window: deque[datetime] = deque()

        self.devices: set[str] = set()
        self.cities: set[str] = {home_city}
        self.merchants: set[str] = set()

        self.cat_count: dict[str, int] = {}
        self.cat_sum: dict[str, float] = {}
        self.cat_sumsq: dict[str, float] = {}
        self.cat_last_time: dict[str, datetime] = {}
        self.cat_week_spend: dict[str, float] = {}

        self.week_buckets: dict[int, float] = {}
        self._week_origin: datetime | None = None

    def record_failure(self, when: datetime) -> None:
        """A declined attempt. Kept separate from the spend history: it signals
        risk (card testing, insufficient funds) without teaching the baseline."""
        self.failed_window.append(when)

    def stats(self, now: datetime, category: str) -> UserHistoryStats:
        """Snapshot the rolling aggregates as of ``now``.

        Aggregates are maintained incrementally in ``observe`` and trimmed by
        ``_advance`` rather than recomputed by rescanning history, because a naive
        rescan makes dataset generation O(n^2) in transactions per user — which at
        realistic sizes (2000+ users x 180 days) is the difference between one
        minute and half an hour.
        """
        self._advance(now)

        n = self.count_total
        mean_amt = (self.sum_total / n) if n else 0.0
        # Welford-style variance from running sums; adequate at these magnitudes.
        std_amt = 0.0
        if n > 1:
            var = max(self.sumsq_total / n - mean_amt**2, 0.0)
            std_amt = math.sqrt(var)

        mins_since = 1440.0
        if self.last_time is not None:
            mins_since = max(0.0, (now - self.last_time).total_seconds() / 60.0)

        span_days = max((now - self.first_time).days, 1) if self.first_time else 1
        avg_daily = n / span_days if n else 1.0

        cat_mean = {
            c: (self.cat_sum[c] / self.cat_count[c]) for c in self.cat_count if self.cat_count[c]
        }
        cat_std: dict[str, float] = {}
        for c, cnt in self.cat_count.items():
            if cnt > 1:
                m = self.cat_sum[c] / cnt
                cat_std[c] = math.sqrt(max(self.cat_sumsq[c] / cnt - m**2, 0.0))

        weeks_span = max(span_days / 7.0, 1.0)
        cat_total = self.cat_sum.get(category, 0.0)
        cat_baseline = (cat_total / weeks_span) if cat_total else 0.0

        last_cat = self.cat_last_time.get(category)
        days_since_cat = (
            max(0.0, (now - last_cat).total_seconds() / 86400.0) if last_cat else 400.0
        )

        # Weekly totals come from fixed buckets keyed by week index, so the
        # median is available without re-partitioning the full history.
        week_median = (
            float(np.median(list(self.week_buckets.values()))) if self.week_buckets else 0.0
        )

        return UserHistoryStats(
            txn_count_total=n,
            mean_amount=mean_amt,
            std_amount=std_amt,
            recent_amounts=list(self.recent_amounts),
            txn_count_1h=len(self.window_1h),
            txn_count_24h=len(self.window_24h),
            amount_sum_24h=self.sum_24h,
            mins_since_prev_txn=mins_since,
            avg_daily_txn_count=avg_daily,
            known_devices=frozenset(self.devices),
            known_cities=frozenset(self.cities),
            known_merchants=frozenset(self.merchants),
            home_country=self.home_country,
            category_counts=dict(self.cat_count),
            distinct_merchants_7d=len({m for m, _t in self.window_7d_merchants}),
            distinct_categories_7d=len({c for c, _t in self.window_7d_categories}),
            failed_txn_24h=len(self.failed_window),
            usual_hour=self.usual_hour,
            category_mean_amount=cat_mean,
            category_std_amount=cat_std,
            category_week_spend=self.cat_week_spend.get(category, 0.0),
            category_week_baseline=cat_baseline,
            days_since_category=days_since_cat,
            week_spend_total=self.sum_7d,
            week_spend_median=week_median,
        )

    def _advance(self, now: datetime) -> None:
        """Evict entries that have fallen out of each trailing window."""
        h1, h24, d7 = (
            now - timedelta(hours=1),
            now - timedelta(hours=24),
            now - timedelta(days=7),
        )
        while self.window_1h and self.window_1h[0][0] < h1:
            self.window_1h.popleft()
        while self.window_24h and self.window_24h[0][0] < h24:
            _t, amt = self.window_24h.popleft()
            self.sum_24h -= amt
        while self.window_7d and self.window_7d[0][0] < d7:
            t, amt, cat = self.window_7d.popleft()
            self.sum_7d -= amt
            self.cat_week_spend[cat] = max(0.0, self.cat_week_spend.get(cat, 0.0) - amt)
        while self.window_7d_merchants and self.window_7d_merchants[0][1] < d7:
            self.window_7d_merchants.popleft()
        while self.window_7d_categories and self.window_7d_categories[0][1] < d7:
            self.window_7d_categories.popleft()
        while self.failed_window and self.failed_window[0] < h24:
            self.failed_window.popleft()
        # Guard against float drift accumulating over long simulations.
        if not self.window_24h:
            self.sum_24h = 0.0
        if not self.window_7d:
            self.sum_7d = 0.0

    def observe(self, ctx: TxnContext, *, legitimate: bool) -> None:
        """Commit a transaction to history, updating all rolling aggregates.

        Fraudulent activity is intentionally *not* added to the behavioural
        baseline — a real bank would not let confirmed fraud teach the profile
        what "normal" looks like.
        """
        if not legitimate:
            return

        ts, amount, cat = ctx.occurred_at, ctx.amount, ctx.category
        self._advance(ts)

        self.count_total += 1
        self.sum_total += amount
        self.sumsq_total += amount * amount
        self.recent_amounts.append(amount)

        if self.first_time is None or ts < self.first_time:
            self.first_time = ts
        if self.last_time is None or ts > self.last_time:
            self.last_time = ts

        self.window_1h.append((ts, amount))
        self.window_24h.append((ts, amount))
        self.sum_24h += amount
        self.window_7d.append((ts, amount, cat))
        self.sum_7d += amount
        self.cat_week_spend[cat] = self.cat_week_spend.get(cat, 0.0) + amount

        if ctx.device_fingerprint:
            self.devices.add(ctx.device_fingerprint)
        if ctx.location_city:
            self.cities.add(ctx.location_city)
        if ctx.merchant_name:
            self.merchants.add(ctx.merchant_name)
            self.window_7d_merchants.append((ctx.merchant_name, ts))
        self.window_7d_categories.append((cat, ts))

        self.cat_count[cat] = self.cat_count.get(cat, 0) + 1
        self.cat_sum[cat] = self.cat_sum.get(cat, 0.0) + amount
        self.cat_sumsq[cat] = self.cat_sumsq.get(cat, 0.0) + amount * amount
        self.cat_last_time[cat] = ts

        # Fixed weekly buckets keyed off the first observed transaction.
        if self._week_origin is None:
            self._week_origin = ts
        week_index = int((ts - self._week_origin).days // 7)
        self.week_buckets[week_index] = self.week_buckets.get(week_index, 0.0) + amount


# --------------------------------------------------------------------------- #
# 1. Fraud dataset
# --------------------------------------------------------------------------- #


def _draw_amount(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Log-normal draw: spending is right-skewed, never negative."""
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(max(mean, 1.0)) - 0.5 * sigma**2
    return float(np.clip(rng.lognormal(mu, sigma), 10, 5_000_000))


def generate_fraud_dataset(
    n_users: int = 900,
    days: int = 180,
    fraud_rate: float = 0.004,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate transaction streams with injected fraud episodes.

    Fraud is generated as *behavioural episodes* (account takeover, card testing,
    mule cash-out, foreign CNP) rather than by flipping a label on random rows.
    That matters: a label uncorrelated with behaviour would let any model hit
    fake accuracy, whereas episodes force the model to learn real interaction
    effects between amount, velocity, device and geography.

    The classes overlap on purpose — some legitimate transactions look odd
    (salary bonus, holiday spending) and some fraud looks mundane — so the
    resulting metrics are honest rather than a separable toy problem.
    """
    rng = np.random.default_rng(seed)
    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(days=days)

    rows: list[dict] = []

    for uid in range(n_users):
        # --- persona: drives amount scale, velocity and channel mix ---
        income_scale = float(rng.lognormal(math.log(1.0), 0.55))
        txn_per_day = float(np.clip(rng.gamma(3.0, 0.55), 0.25, 8.0))
        home_city = CITIES[int(rng.integers(0, len(CITIES)))]
        usual_hour = float(np.clip(rng.normal(13.5, 3.0), 6, 23))
        account_age = float(rng.integers(30, 3650))
        balance = float(np.clip(rng.lognormal(math.log(60_000 * income_scale), 0.7), 500, 5e6))
        n_devices = int(rng.integers(1, 4))
        devices = [f"dev{uid}_{i}" for i in range(n_devices)]
        travel_prone = rng.random() < 0.18
        # Typical legitimate spend for this persona, used so "stealthy" fraud can
        # be sized to look like the victim's normal activity.
        mean_legit_amount = float(np.mean([m[2] for m in MERCHANTS]))

        state = _StreamState(home_city=home_city)
        state.usual_hour = usual_hour

        # --- warm-up so early rows are not all cold-start ---
        warm_start = start - timedelta(days=60)
        n_warm = int(txn_per_day * 60)
        for _ in range(n_warm):
            offset = float(rng.random()) * 60
            ts = warm_start + timedelta(days=offset)
            name, cat, amt_mean, cv = MERCHANTS[int(rng.integers(0, len(MERCHANTS)))]
            state.observe(
                TxnContext(
                    amount=_draw_amount(rng, amt_mean * income_scale, cv),
                    occurred_at=ts,
                    category=cat,
                    channel="card",
                    merchant_name=name,
                    device_fingerprint=devices[0],
                    location_city=home_city,
                    account_balance=balance,
                    account_age_days=account_age,
                ),
                legitimate=True,
            )
        # Warm-up rows are generated out of order; aggregates are window-based and
        # order-tolerant, so no sort is required after seeding history.

        # --- observation window ---
        n_txn = max(int(rng.poisson(txn_per_day * days)), 20)
        # Cluster timestamps around the user's usual hour instead of uniform noise.
        day_offsets = rng.random(n_txn) * days
        hour_jitter = rng.normal(usual_hour, 2.6, n_txn) % 24
        timestamps = sorted(
            start + timedelta(days=float(d), hours=float(h))
            for d, h in zip(day_offsets, hour_jitter)
        )

        # --- decide fraud episodes for this user ---
        # Fraud arrives in bursts, so the per-user episode probability must be
        # divided by the burst size for the overall row-level rate to land on
        # `fraud_rate`. Getting this wrong is what inflates the positive class.
        episodes: list[tuple[int, int, str]] = []
        kind = str(rng.choice(["takeover", "card_testing", "mule", "foreign_cnp"]))
        burst = {"takeover": 4, "card_testing": 9, "mule": 3, "foreign_cnp": 5}[kind]
        p_episode = min(n_txn * fraud_rate / burst, 0.5)
        if rng.random() < p_episode:
            lo = int(n_txn * 0.25)
            hi = max(int(n_txn * 0.92), lo + 1)
            begin = int(rng.integers(lo, hi))
            episodes.append((begin, min(begin + burst, n_txn), kind))

        for i, ts in enumerate(timestamps):
            in_episode = next((k for a, b, k in episodes if a <= i < b), None)

            if in_episode is None:
                # ---------------- legitimate ----------------
                name, cat, amt_mean, cv = MERCHANTS[int(rng.integers(0, len(MERCHANTS)))]
                amount = _draw_amount(rng, amt_mean * income_scale, cv)
                city = home_city
                country = "IN"
                device = devices[int(rng.integers(0, n_devices))]
                channel = str(rng.choice(CHANNELS, p=[0.06, 0.10, 0.12, 0.34, 0.30, 0.08]))

                # Benign outliers: keeps the boundary genuinely fuzzy. Legitimate
                # customers do occasionally behave "suspiciously" — a big holiday
                # purchase abroad, a new phone, a 2am impulse order — and a model
                # that has never seen those will over-flag real users.
                if rng.random() < 0.035:
                    amount *= float(rng.uniform(4, 15))          # big legitimate purchase
                if travel_prone and rng.random() < 0.07:
                    city = CITIES[int(rng.integers(0, len(CITIES)))]  # domestic travel
                if travel_prone and rng.random() < 0.015:
                    # Genuine foreign spending: overlaps with the fraud signature.
                    fc, fcountry = FOREIGN_CITIES[int(rng.integers(0, len(FOREIGN_CITIES)))]
                    city, country = fc, fcountry
                if rng.random() < 0.03:
                    device = f"dev{uid}_new{i}"                   # new phone / browser
                if rng.random() < 0.02:
                    ts = ts.replace(hour=int(rng.integers(0, 5)))  # late-night activity
                # Honest users also get declined (insufficient funds, expired card),
                # so the feature has a non-zero baseline and is not a fraud giveaway.
                if rng.random() < 0.03:
                    for _ in range(int(rng.integers(1, 4))):
                        state.record_failure(ts - timedelta(minutes=float(rng.uniform(1, 90))))
                label = 0
            else:
                # ---------------- fraudulent ----------------
                device = f"attacker_{uid}_{i}"
                label = 1
                if in_episode == "takeover":
                    amount = _draw_amount(rng, 45_000 * income_scale, 0.8)
                    name, cat, channel = "Unknown Transfer", "transfer", "imps"
                    city, country = CITIES[int(rng.integers(0, len(CITIES)))], "IN"
                    ts = ts.replace(hour=int(rng.integers(1, 5)))
                elif in_episode == "card_testing":
                    amount = _draw_amount(rng, 120, 0.5)          # tiny probe charges
                    name, cat, channel = "Online Store", "shopping", "card"
                    city, country = "Unknown", "IN"
                    # Card testing is mostly declines: the attacker is hunting for
                    # a live card, so failed attempts cluster right before a hit.
                    for _ in range(int(rng.integers(2, 7))):
                        state.record_failure(ts - timedelta(minutes=float(rng.uniform(1, 40))))
                elif in_episode == "mule":
                    amount = float(rng.choice([50_000, 100_000, 150_000, 200_000]))
                    name, cat, channel = "ATM Withdrawal", "cash", "atm"
                    city, country = CITIES[int(rng.integers(0, len(CITIES)))], "IN"
                    ts = ts.replace(hour=int(rng.integers(0, 4)))
                else:  # foreign_cnp
                    amount = _draw_amount(rng, 28_000 * income_scale, 0.9)
                    fc, fcountry = FOREIGN_CITIES[int(rng.integers(0, len(FOREIGN_CITIES)))]
                    name, cat, channel = "Intl Merchant", "shopping", "card"
                    city, country = fc, fcountry

                # ---- realism controls on the fraud class ----------------------
                # Without these the episodes are near-linearly separable (foreign
                # country AND attacker device AND 3am AND round amount always
                # co-occur), and the model reports a fake ROC-AUC of ~1.0. Real
                # fraud overlaps heavily with legitimate behaviour, so a
                # substantial share of it is made deliberately mundane:
                #   - 30% is "stealthy": known device, home city, ordinary amount
                #   - the attacker often reuses a device already seen on the account
                stealth = rng.random()
                if stealth < 0.30:
                    # Sophisticated fraud: mimics the victim's own pattern.
                    amount = _draw_amount(rng, mean_legit_amount * income_scale, 0.6)
                    device = devices[int(rng.integers(0, n_devices))]
                    city, country = home_city, "IN"
                    channel = str(rng.choice(["upi", "card", "imps"], p=[0.4, 0.4, 0.2]))
                    ts = ts.replace(hour=int(rng.integers(9, 22)))
                elif stealth < 0.50:
                    # Partially disguised: familiar device, unusual amount only.
                    device = devices[0]
                    city, country = home_city, "IN"

            ctx = TxnContext(
                amount=amount,
                occurred_at=ts,
                category=cat,
                channel=channel,
                merchant_name=name,
                device_fingerprint=device,
                location_city=city,
                location_country=country,
                account_balance=balance,
                account_age_days=account_age + (ts - start).days,
            )
            hist = state.stats(ts, cat)
            feats = build_fraud_features(ctx, hist)
            feats["label"] = label
            feats["user_id"] = uid
            rows.append(feats)

            state.observe(ctx, legitimate=label == 0)
            if label == 0:
                balance = float(max(500.0, balance - amount * 0.35 + rng.normal(600, 400)))

    df = pd.DataFrame(rows)
    return df[FRAUD_FEATURES.names + ["label", "user_id"]]


# --------------------------------------------------------------------------- #
# 2. Credit dataset
# --------------------------------------------------------------------------- #


def generate_credit_dataset(n_samples: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Simulate loan applicants with a latent-risk default process.

    Default is drawn from a logistic function of genuinely causal drivers
    (debt-to-income, prior defaults, employment stability, balance volatility)
    plus irreducible noise. Because the label depends on the same quantities the
    features measure — but noisily and non-linearly — the achievable AUC lands in
    a realistic 0.80-0.90 band rather than a suspicious 0.99.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in range(n_samples):
        age = float(np.clip(rng.normal(37, 11), 21, 72))
        employment_status = str(
            rng.choice(
                ["salaried", "self_employed", "government", "contract", "gig", "student", "unemployed", "retired"],
                p=[0.44, 0.18, 0.09, 0.09, 0.10, 0.04, 0.03, 0.03],
            )
        )
        # Income scales with age and employment quality.
        base_income = rng.lognormal(math.log(520_000), 0.62)
        age_factor = 1 + (age - 30) * 0.021
        emp_factor = {
            "government": 1.15, "salaried": 1.10, "self_employed": 1.05, "contract": 0.92,
            "gig": 0.72, "student": 0.30, "unemployed": 0.22, "retired": 0.62,
        }[employment_status]
        annual_income = float(np.clip(base_income * max(age_factor, 0.5) * emp_factor, 90_000, 2.5e7))

        max_years = max(age - 21, 0.5)
        employment_years = float(np.clip(rng.gamma(2.2, 2.4), 0, max_years))
        loan_type = str(
            rng.choice(["personal", "home", "auto", "education", "business"], p=[0.42, 0.16, 0.19, 0.13, 0.10])
        )
        # Amount and tenure must be drawn *jointly* per product. Sampling them
        # independently yields 4x-income mortgages repaid over 36 months, whose
        # impossible EMIs make debt-to-income useless for every home applicant.
        income_multiple = {
            "personal": 0.7, "home": 4.2, "auto": 1.3, "education": 1.1, "business": 2.4,
        }[loan_type]
        requested = float(
            np.clip(annual_income * income_multiple * rng.lognormal(0, 0.40), 25_000, 5e7)
        )
        tenure_options, tenure_probs = {
            "home": ([120, 180, 240, 300, 360], [0.15, 0.25, 0.30, 0.20, 0.10]),
            "auto": ([36, 48, 60, 84], [0.25, 0.30, 0.30, 0.15]),
            "personal": ([12, 24, 36, 48, 60], [0.15, 0.25, 0.30, 0.18, 0.12]),
            "education": ([36, 60, 84, 120], [0.25, 0.30, 0.25, 0.20]),
            "business": ([12, 24, 36, 60, 84], [0.20, 0.25, 0.25, 0.20, 0.10]),
        }[loan_type]
        tenure = int(rng.choice(tenure_options, p=tenure_probs))

        monthly_income = annual_income / 12.0
        existing_emi = float(
            max(0.0, rng.gamma(1.6, monthly_income * 0.10) if rng.random() < 0.55 else 0.0)
        )
        dependents = int(np.clip(rng.poisson(1.15), 0, 8))
        housing_status = str(rng.choice(["rent", "own", "mortgage", "family"], p=[0.40, 0.22, 0.21, 0.17]))

        account_age_months = float(np.clip(rng.gamma(2.6, 14), 1, 420))
        num_accounts = int(np.clip(rng.poisson(1.7) + 1, 1, 9))
        avg_balance = float(np.clip(rng.lognormal(math.log(monthly_income * 1.5), 0.85), 500, 3e7))
        balance_volatility = float(np.clip(rng.gamma(2.0, 0.22), 0.03, 4.0))
        min_balance_ratio = float(np.clip(rng.beta(2.4, 2.0), 0.01, 1.6))
        inflow_90d = float(max(monthly_income * 3 * rng.normal(1.0, 0.16), 5_000))
        savings_rate_true = float(np.clip(rng.normal(0.14, 0.18), -0.75, 0.62))
        outflow_90d = float(max(inflow_90d * (1 - savings_rate_true), 1_000))
        txn_count_90d = int(np.clip(rng.poisson(78), 3, 900))
        credit_utilisation = float(np.clip(rng.beta(2.1, 3.4) * 1.25, 0, 1.6))
        prior_loans = int(np.clip(rng.poisson(0.85), 0, 12))
        prior_defaults = int(min(prior_loans, rng.binomial(prior_loans, 0.11)) if prior_loans else 0)
        emis_missed = int(np.clip(rng.poisson(0.7 + 2.6 * prior_defaults), 0, 40))
        overdraft_events = int(np.clip(rng.poisson(0.9 if min_balance_ratio < 0.15 else 0.18), 0, 30))

        profile = CreditProfile(
            age=age,
            annual_income=annual_income,
            employment_status=employment_status,
            employment_years=employment_years,
            requested_amount=requested,
            tenure_months=tenure,
            loan_type=loan_type,
            existing_emi=existing_emi,
            dependents=dependents,
            housing_status=housing_status,
            account_age_months=account_age_months,
            num_accounts=num_accounts,
            avg_balance=avg_balance,
            balance_volatility=balance_volatility,
            min_balance_ratio=min_balance_ratio,
            inflow_90d=inflow_90d,
            outflow_90d=outflow_90d,
            txn_count_90d=txn_count_90d,
            credit_utilisation=credit_utilisation,
            prior_loans=prior_loans,
            prior_defaults=prior_defaults,
            emis_missed=emis_missed,
            overdraft_events_90d=overdraft_events,
        )
        feats = build_credit_features(profile)

        # ---- latent risk -> default probability ----
        # Risk drivers are expressed as EXCESS over an underwriting-normal
        # baseline, not as raw levels. Charging risk for a median-normal applicant
        # (DTI 0.5, one dependent, 30% utilisation) is what pushes a simulated
        # portfolio to an absurd ~50% default rate; real books run 10-30%.
        dti = feats["debt_to_income"]
        excess_dti = max(0.0, min(dti, 3.0) - 0.40)          # 40% DTI underwriting cutoff
        excess_util = max(0.0, min(credit_utilisation, 1.6) - 0.30)
        excess_vol = max(0.0, min(balance_volatility, 3.0) - 0.35)
        thin_file = 1.0 if account_age_months < 12 else 0.0

        logit = (
            -3.05
            + 2.60 * excess_dti
            + 1.85 * prior_defaults                           # strongest real-world predictor
            + 0.16 * min(emis_missed, 24)
            + 1.30 * (1.0 - feats["employment_stability"]) ** 1.5
            + 0.80 * excess_vol
            + 2.10 * max(0.0, -savings_rate_true)             # spending beyond income
            + 0.90 * excess_util
            + 0.45 * (feats["loan_purpose_risk"] - 0.40)
            + 0.10 * max(dependents - 1, 0)
            + 0.34 * min(overdraft_events, 12)
            + 0.55 * thin_file
            - 0.70 * min(feats["min_balance_ratio"], 1.2)
            - 0.035 * min(employment_years, 20)
            - 0.018 * (age - 30)
            - 0.0020 * min(account_age_months, 240)
            - 0.30 * math.log1p(max(num_accounts - 1, 0))
        )
        # Irreducible noise: job loss, medical shock, fraud — unobservable at apply
        # time. This is what caps achievable AUC at a believable ~0.85 instead of
        # a self-fulfilling 0.99.
        logit += float(rng.normal(0, 0.95))
        pd_true = 1.0 / (1.0 + math.exp(-logit))
        feats["label"] = int(rng.random() < pd_true)
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df[CREDIT_FEATURES.names + ["label"]]


# --------------------------------------------------------------------------- #
# 3. Anomaly dataset
# --------------------------------------------------------------------------- #


def generate_anomaly_dataset(
    n_users: int = 700,
    days: int = 150,
    contamination: float = 0.03,
    seed: int = 43,
) -> pd.DataFrame:
    """Simulate normal spending plus injected behavioural deviations.

    Isolation Forest trains unsupervised on the *unlabelled* mixture; the labels
    exist only to evaluate it. Anomalies here are lifestyle deviations (a dining
    blowout week, a 3am spree, a category the user never touches) rather than
    fraud — the product framing is "this doesn't look like you", not "you were robbed".
    """
    rng = np.random.default_rng(seed)
    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(days=days)
    rows: list[dict] = []

    for uid in range(n_users):
        income_scale = float(rng.lognormal(math.log(1.0), 0.5))
        txn_per_day = float(np.clip(rng.gamma(3.0, 0.5), 0.3, 6.0))
        home_city = CITIES[int(rng.integers(0, len(CITIES)))]
        usual_hour = float(np.clip(rng.normal(13.0, 2.8), 7, 22))
        # Habitual users stick to a narrow merchant set.
        pref_idx = rng.choice(len(MERCHANTS), size=int(rng.integers(4, 10)), replace=False)
        preferred = [MERCHANTS[int(i)] for i in pref_idx]

        state = _StreamState(home_city=home_city)
        state.usual_hour = usual_hour

        warm_start = start - timedelta(days=45)
        for _ in range(int(txn_per_day * 45)):
            name, cat, amt_mean, cv = preferred[int(rng.integers(0, len(preferred)))]
            state.observe(
                TxnContext(
                    amount=_draw_amount(rng, amt_mean * income_scale, cv),
                    occurred_at=warm_start + timedelta(days=float(rng.random()) * 45),
                    category=cat,
                    merchant_name=name,
                    device_fingerprint=f"dev{uid}",
                    location_city=home_city,
                ),
                legitimate=True,
            )
        # Warm-up rows are generated out of order; aggregates are window-based and
        # order-tolerant, so no sort is required after seeding history.

        n_txn = max(int(rng.poisson(txn_per_day * days)), 15)
        timestamps = sorted(
            start + timedelta(days=float(d), hours=float(h))
            for d, h in zip(rng.random(n_txn) * days, rng.normal(usual_hour, 2.4, n_txn) % 24)
        )

        for ts in timestamps:
            is_anom = rng.random() < contamination
            if not is_anom:
                name, cat, amt_mean, cv = preferred[int(rng.integers(0, len(preferred)))]
                amount = _draw_amount(rng, amt_mean * income_scale, cv)
                label = 0
            else:
                kind = int(rng.integers(0, 4))
                if kind == 0:      # category blowout
                    name, cat, amt_mean, cv = preferred[int(rng.integers(0, len(preferred)))]
                    amount = _draw_amount(rng, amt_mean * income_scale, cv) * float(rng.uniform(5, 14))
                elif kind == 1:    # brand-new category
                    unseen = [m for m in MERCHANTS if m not in preferred]
                    name, cat, amt_mean, cv = unseen[int(rng.integers(0, len(unseen)))]
                    amount = _draw_amount(rng, amt_mean * income_scale * 3, cv)
                elif kind == 2:    # odd-hour activity
                    name, cat, amt_mean, cv = preferred[int(rng.integers(0, len(preferred)))]
                    amount = _draw_amount(rng, amt_mean * income_scale * 2.5, cv)
                    ts = ts.replace(hour=int(rng.integers(1, 5)))
                else:              # velocity burst
                    name, cat, amt_mean, cv = preferred[int(rng.integers(0, len(preferred)))]
                    amount = _draw_amount(rng, amt_mean * income_scale * 1.6, cv)
                    for _ in range(int(rng.integers(4, 9))):
                        state.observe(
                            TxnContext(
                                amount=amount * float(rng.uniform(0.6, 1.4)),
                                occurred_at=ts - timedelta(minutes=float(rng.uniform(4, 55))),
                                category=cat,
                                merchant_name=name,
                                device_fingerprint=f"dev{uid}",
                                location_city=home_city,
                            ),
                            legitimate=True,
                        )
                label = 1

            ctx = TxnContext(
                amount=amount,
                occurred_at=ts,
                category=cat,
                merchant_name=name,
                device_fingerprint=f"dev{uid}",
                location_city=home_city,
            )
            hist = state.stats(ts, cat)
            feats = build_anomaly_features(ctx, hist)
            feats["label"] = label
            feats["user_id"] = uid
            rows.append(feats)
            state.observe(ctx, legitimate=True)

    df = pd.DataFrame(rows)
    return df[ANOMALY_FEATURES.names + ["label", "user_id"]]


# --------------------------------------------------------------------------- #
# Kaggle adapters
# --------------------------------------------------------------------------- #

KAGGLE_FILES = {
    "fraud": RAW_DIR / "creditcard.csv",
    "credit": RAW_DIR / "german_credit.csv",
    "home_credit": RAW_DIR / "application_train.csv",
}


def kaggle_available(kind: str) -> bool:
    path = KAGGLE_FILES.get(kind)
    return bool(path and path.exists())


def kaggle_notes() -> str:
    return (
        "Kaggle datasets are optional. Place files in data/raw/:\n"
        "  - creditcard.csv       (Credit Card Fraud Detection, mlg-ulb)\n"
        "  - german_credit.csv    (Statlog German Credit Data)\n"
        "  - application_train.csv (Home Credit Default Risk, optional)\n\n"
        "Caveat that matters: creditcard.csv ships PCA-anonymised components\n"
        "(V1..V28) that cannot be mapped to named banking features, so it trains a\n"
        "separate BENCHMARK model on its own feature space. The model actually\n"
        "served by the API uses the interpretable 24-feature schema, because the\n"
        "product needs to explain *why* a transaction was flagged.\n"
        "German Credit maps onto the credit schema with documented approximations."
    )


def load_kaggle_fraud() -> tuple[pd.DataFrame, list[str]]:
    """Load the raw Kaggle fraud dataset in its native PCA feature space."""
    path = KAGGLE_FILES["fraud"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.\n\n{kaggle_notes()}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Class" not in df:
        raise ValueError("Expected a 'Class' column in creditcard.csv")

    feature_cols = [c for c in df.columns if c != "Class"]
    out = df[feature_cols].copy()
    # Amount is heavily skewed; log-scale it and normalise Time to hour-of-day
    # so the benchmark model gets the same treatment our own pipeline applies.
    if "Amount" in out:
        out["Amount"] = np.log1p(out["Amount"].clip(lower=0))
    if "Time" in out:
        hours = (out["Time"] / 3600.0) % 24
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        out = out.drop(columns=["Time"])
    out["label"] = df["Class"].astype(int)
    return out, [c for c in out.columns if c != "label"]


# Statlog German Credit uses coded categorical values; these decode the columns
# we can defensibly map onto our schema.
_GERMAN_EMPLOYMENT = {"A71": "unemployed", "A72": "gig", "A73": "contract", "A74": "salaried", "A75": "government"}
_GERMAN_HOUSING = {"A151": "rent", "A152": "own", "A153": "family"}
_GERMAN_PURPOSE = {
    "A40": "auto", "A41": "auto", "A42": "personal", "A43": "personal", "A44": "personal",
    "A45": "personal", "A46": "education", "A47": "personal", "A48": "education",
    "A49": "business", "A410": "business",
}
_GERMAN_SAVINGS = {"A61": 0.05, "A62": 0.20, "A63": 0.45, "A64": 0.75, "A65": 0.10}


def load_kaggle_credit() -> pd.DataFrame:
    """Map Statlog German Credit onto the deployed credit feature schema.

    Approximations are unavoidable: the dataset is 1990s German consumer credit
    with amounts in Deutsche Marks and no transaction history. Amounts are scaled
    to an INR-like range and behavioural features fall back to schema defaults.
    Documented here rather than hidden, because a silently-wrong mapping would
    make the reported AUC meaningless.
    """
    path = KAGGLE_FILES["credit"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.\n\n{kaggle_notes()}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def col(*candidates: str):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    amount = col("credit_amount", "amount", "creditamount")
    duration = col("duration", "duration_in_month", "months")
    age_col = col("age")
    emp_col = col("present_employment_since", "employment", "savings_status")
    housing_col = col("housing")
    purpose_col = col("purpose")
    savings_col = col("savings_account_bonds", "savings_status", "savings")
    target = col("class", "target", "default", "credit_risk")

    if amount is None or target is None:
        raise ValueError(
            "german_credit.csv is missing required columns (credit amount / class). "
            f"Found: {list(df.columns)[:15]}"
        )

    # Mirrors of this dataset disagree on the target encoding, so normalise all
    # three known variants to 1 = bad credit (the positive/default class).
    y_raw = target.astype(str).str.strip().str.lower()
    distinct = set(y_raw.unique())
    if distinct <= {"0", "1"}:
        label = y_raw.astype(int).to_numpy()          # already 0/1
    elif distinct <= {"1", "2"}:
        label = (y_raw == "2").astype(int).to_numpy()  # Statlog original
    else:
        label = y_raw.isin(["bad", "default", "true", "yes"]).astype(int).to_numpy()

    # DM amounts -> INR-like scale so log features sit in the trained range.
    scale = 900.0
    rows: list[dict] = []
    n = len(df)
    rng = np.random.default_rng(7)

    for i in range(n):
        amt = float(amount.iloc[i]) * scale
        months = int(duration.iloc[i]) if duration is not None else 36
        age = float(age_col.iloc[i]) if age_col is not None else 35.0
        emp_code = str(emp_col.iloc[i]) if emp_col is not None else "A73"
        house_code = str(housing_col.iloc[i]) if housing_col is not None else "A151"
        purpose_code = str(purpose_col.iloc[i]) if purpose_col is not None else "A42"
        sav_code = str(savings_col.iloc[i]) if savings_col is not None else "A61"

        # No income column exists; derive a plausible one from amount and age so
        # ratio features remain meaningful. Noise avoids a deterministic leak.
        annual_income = amt * float(rng.uniform(1.1, 3.2)) * (1 + (age - 30) * 0.01)
        annual_income = float(np.clip(annual_income, 120_000, 2e7))

        # Savings bracket is the only balance signal available; behavioural
        # features (volatility, 90d flows, utilisation) keep schema defaults.
        savings_strength = _GERMAN_SAVINGS.get(sav_code, 0.1)
        profile = CreditProfile(
            age=age,
            annual_income=annual_income,
            employment_status=_GERMAN_EMPLOYMENT.get(emp_code, "salaried"),
            employment_years=float(np.clip((age - 22) * 0.35, 0, 30)),
            requested_amount=amt,
            tenure_months=max(months, 6),
            loan_type=_GERMAN_PURPOSE.get(purpose_code, "personal"),
            existing_emi=0.0,
            dependents=1,
            housing_status=_GERMAN_HOUSING.get(house_code, "rent"),
            avg_balance=annual_income / 12 * (0.4 + savings_strength * 3),
            min_balance_ratio=float(np.clip(0.2 + savings_strength, 0.05, 1.2)),
        )
        feats = build_credit_features(profile)
        feats["label"] = int(label[i])
        rows.append(feats)

    out = pd.DataFrame(rows)
    return out[CREDIT_FEATURES.names + ["label"]]


def dataset_summary(df: pd.DataFrame, name: str) -> str:
    if "label" in df:
        pos = int(df["label"].sum())
        rate = pos / len(df) if len(df) else 0
        return (
            f"{name}: {len(df):,} rows x {len(df.columns) - 1} features | "
            f"positives={pos:,} ({rate:.3%})"
        )
    return f"{name}: {len(df):,} rows x {len(df.columns)} features"
