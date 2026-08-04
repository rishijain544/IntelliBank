/**
 * API response types mirroring the backend Pydantic schemas.
 *
 * Kept hand-written rather than generated from OpenAPI: the surface is small
 * enough that a generator adds build complexity without buying much, and this
 * way the types document only what the UI actually consumes.
 *
 * Money arrives as a string, never a number. The backend uses Decimal, and
 * parsing currency into a JS float would reintroduce exactly the rounding drift
 * the backend avoids. Convert only at the formatting boundary.
 */

export type UserRole = 'customer' | 'admin';
export type UserStatus = 'pending' | 'active' | 'frozen' | 'suspended';
export type KycStatus = 'not_started' | 'submitted' | 'verified' | 'rejected';

export interface User {
  id: number;
  email: string;
  full_name: string;
  phone: string | null;
  role: UserRole;
  status: UserStatus;
  kyc_status: KycStatus;
  two_factor_enabled: boolean;
  city: string | null;
  country: string;
  annual_income: string | null;
  employment_status: string | null;
  pan_masked: string | null;
  created_at: string;
  last_login_at: string | null;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface Account {
  id: number;
  account_number: string;
  ifsc_code: string;
  nickname: string | null;
  account_type: 'savings' | 'current' | 'fixed_deposit' | 'salary';
  status: 'active' | 'frozen' | 'closed' | 'dormant';
  currency: string;
  balance: string;
  hold_amount: string;
  available_balance: string;
  overdraft_limit: string;
  interest_rate: number;
  is_primary: boolean;
  opened_on: string | null;
  created_at: string;
}

export interface Transaction {
  id: number;
  reference: string;
  account_id: number;
  txn_type: string;
  channel: string;
  status: 'pending' | 'completed' | 'failed' | 'reversed' | 'blocked' | 'held';
  amount: string;
  fee: string;
  signed_amount: string;
  balance_after: string | null;
  currency: string;
  description: string | null;
  merchant_name: string | null;
  merchant_category: string;
  counterparty_name: string | null;
  counterparty_account_number: string | null;
  location_city: string | null;
  location_country: string | null;
  is_foreign: boolean;
  fraud_score: number | null;
  anomaly_score: number | null;
  is_flagged: boolean;
  occurred_at: string;
  created_at: string;
}

/** A single SHAP-style contribution returned with every model decision. */
export interface TopFactor {
  feature: string;
  label: string;
  value: number;
  contribution: number;
  direction: 'increases risk' | 'reduces risk';
}

export interface FraudAssessment {
  risk_score: number;
  action: 'allow' | 'review' | 'block';
  severity: 'low' | 'medium' | 'high' | 'critical';
  is_flagged: boolean;
  auto_blocked: boolean;
  reasons: string[];
  triggered_rules: string[];
  top_factors: TopFactor[];
  model_name: string;
  model_version: string;
  model_available: boolean;
  decision_source: string;
  latency_ms: number;
}

export interface TransferResponse {
  transaction: Transaction;
  status: string;
  message: string;
  fraud: FraudAssessment;
}

export interface Beneficiary {
  id: number;
  name: string;
  nickname: string | null;
  account_number: string;
  ifsc_code: string;
  bank_name: string;
  is_internal: boolean;
  is_verified: boolean;
  transfer_limit: string | null;
  last_used_at: string | null;
  usage_count: number;
  created_at: string;
}

export interface Card {
  id: number;
  account_id: number;
  card_last4: string;
  masked_number: string;
  card_network: string;
  card_type: 'virtual_debit' | 'virtual_credit';
  status: 'active' | 'frozen' | 'expired' | 'cancelled';
  cardholder_name: string;
  expiry_month: number;
  expiry_year: number;
  daily_limit: string;
  per_txn_limit: string;
  monthly_limit: string;
  credit_limit: string | null;
  online_enabled: boolean;
  international_enabled: boolean;
  contactless_enabled: boolean;
  atm_enabled: boolean;
  freeze_reason: string | null;
  created_at: string;
}

export interface CreditScore {
  score: number;
  probability_of_default: number;
  risk_band: 'A' | 'B' | 'C' | 'D' | 'E';
  decision: 'approve' | 'review' | 'reject';
  suggested_rate: number;
  max_eligible_amount: string;
  approved_amount: string;
  emi_amount: string;
  total_payable: string;
  processing_fee: string;
  reasons: string[];
  top_factors: TopFactor[];
  model_name: string;
  model_version: string;
  model_available: boolean;
  latency_ms: number;
}

export interface Loan {
  id: number;
  application_ref: string;
  loan_type: 'personal' | 'home' | 'auto' | 'education' | 'business';
  status:
    | 'draft'
    | 'submitted'
    | 'under_review'
    | 'approved'
    | 'rejected'
    | 'disbursed'
    | 'closed'
    | 'defaulted';
  requested_amount: string;
  approved_amount: string | null;
  tenure_months: number;
  purpose: string | null;
  interest_rate: number | null;
  emi_amount: string | null;
  total_payable: string | null;
  processing_fee: string | null;
  decision_source: string;
  decision_reason: string | null;
  decided_at: string | null;
  manual_override: boolean;
  disbursed_at: string | null;
  outstanding_principal: string | null;
  emis_paid: number;
  emis_missed: number;
  created_at: string;
}

export interface LoanApplicationResult {
  loan: Loan;
  credit: CreditScore;
}

export interface FraudAlert {
  id: number;
  alert_ref: string;
  transaction_id: number;
  risk_score: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status:
    | 'open'
    | 'confirmed_fraud'
    | 'disputed'
    | 'resolved_legit'
    | 'resolved_fraud'
    | 'dismissed';
  decision_source: string;
  auto_blocked: boolean;
  reasons: string[] | null;
  top_factors: TopFactor[] | null;
  triggered_rules: string[] | null;
  model_name: string;
  model_version: string;
  inference_latency_ms: number | null;
  customer_response: string | null;
  customer_note: string | null;
  review_note: string | null;
  final_label: boolean | null;
  created_at: string;
  transaction: Transaction | null;
}

export interface AnomalyAlert {
  id: number;
  transaction_id: number | null;
  anomaly_score: number;
  severity: string;
  anomaly_type: string;
  title: string;
  message: string;
  category: string | null;
  baseline_value: number | null;
  observed_value: number | null;
  deviation_ratio: number | null;
  model_name: string;
  model_version: string;
  acknowledged: boolean;
  created_at: string;
}

export interface CategoryBreakdown {
  category: string;
  total: string;
  count: number;
  percentage: number;
  avg_amount: string;
}

export interface MonthlyTrend {
  month: string;
  inflow: string;
  outflow: string;
  net: string;
  txn_count: number;
}

export interface DailySpend {
  date: string;
  amount: string;
  count: number;
}

export interface DashboardData {
  total_balance: string;
  accounts: Account[];
  recent_transactions: Transaction[];
  spend_last_30d: string;
  received_last_30d: string;
  open_fraud_alerts: number;
  unread_notifications: number;
  active_loans: number;
  category_breakdown: CategoryBreakdown[];
  daily_spend: DailySpend[];
  latest_credit_score: number | null;
}

export interface InsightsData {
  period_days: number;
  total_spent: string;
  total_received: string;
  net_change: string;
  txn_count: number;
  avg_transaction: string;
  largest_transaction: string;
  category_breakdown: CategoryBreakdown[];
  monthly_trends: MonthlyTrend[];
  daily_spend: DailySpend[];
  top_merchants: { merchant: string; total: number; count: number }[];
  anomaly_alerts: AnomalyAlert[];
}

export interface Notification {
  id: number;
  notif_type: string;
  severity: string;
  title: string;
  body: string;
  action_url: string | null;
  meta: Record<string, unknown> | null;
  is_read: boolean;
  created_at: string;
}

export interface AdminStats {
  total_users: number;
  active_users: number;
  pending_kyc: number;
  frozen_users: number;
  total_accounts: number;
  total_balance: string;
  txn_count_today: number;
  txn_volume_today: string;
  txn_count_30d: number;
  txn_volume_30d: string;
  fraud_alerts_open: number;
  fraud_alerts_total: number;
  fraud_confirmed: number;
  blocked_transactions: number;
  loans_pending: number;
  loans_approved: number;
  loans_disbursed_value: string;
  model_status: Record<string, ModelStatusEntry>;
}

export interface ModelStatusEntry {
  loaded: boolean;
  name: string;
  version: string | null;
  trained_at: string | null;
  threshold: number | null;
  n_features: number | null;
  metrics: Record<string, unknown>;
  latency_benchmark: Record<string, number>;
}

export interface ModelPerformance {
  model_name: string;
  model_version: string | null;
  loaded: boolean;
  training_metrics: Record<string, number | null>;
  training_latency: Record<string, number>;
  live_inference_count: number;
  live_flagged_count: number;
  live_labelled_count: number;
  live_mean_score: number | null;
  live_p95_score: number | null;
  live_mean_latency_ms: number | null;
  live_p95_latency_ms: number | null;
  realised_precision: number | null;
  realised_recall: number | null;
  psi: number | null;
  drift_status: 'stable' | 'watch' | 'drifting' | null;
  score_histogram: { bin: string; count: number }[];
}

export interface UserSummary {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  status: UserStatus;
  kyc_status: KycStatus;
  created_at: string;
  last_login_at: string | null;
}

export interface AuditLogEntry {
  id: number;
  actor_id: number | null;
  actor_email: string | null;
  actor_role: string | null;
  action: string;
  entity_type: string | null;
  entity_id: string | null;
  summary: string | null;
  ip_address: string | null;
  success: boolean;
  created_at: string;
}

export interface AnalyticsData {
  period_days: number;
  daily_volume: { date: string; volume: number; count: number; flagged: number }[];
  category_distribution: { category: string; total: number; count: number }[];
  credit_band_distribution: { band: string; count: number }[];
  fraud_severity_distribution: { severity: string; count: number }[];
  loan_status_distribution: { status: string; count: number }[];
}

/** Generic pagination envelope used by every list endpoint. */
export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface MessageResponse {
  message: string;
  detail: string | null;
}

/** Flattened validation errors from the backend's RequestValidationError handler. */
export interface ApiErrorBody {
  detail: string;
  fields?: Record<string, string>;
}

/* -------------------------------------------------------------------------- */
/* AI assistant                                                               */
/* -------------------------------------------------------------------------- */

export interface AssistantTurn {
  role: 'user' | 'assistant';
  content: string;
}

/** A tool the model invoked, surfaced so the answer can be audited. */
export interface AssistantToolCall {
  name: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  duration_ms: number;
}

export interface AssistantChatResponse {
  message: string;
  tool_calls: AssistantToolCall[];
  /** "gemini" when the LLM wrote the prose, "fallback" when the rule router did. */
  engine: string;
  model: string | null;
  latency_ms: number;
  degraded_reason: string | null;
}

export interface AssistantStatus {
  enabled: boolean;
  engine: string;
  model: string | null;
  api_key_configured: boolean;
  tools: string[];
  capabilities: string;
}
