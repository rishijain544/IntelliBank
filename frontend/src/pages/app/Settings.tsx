import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Check,
  KeyRound,
  Loader2,
  Monitor,
  ShieldCheck,
  Smartphone,
  Trash2,
} from 'lucide-react';
import { useState } from 'react';

import {
  Badge,
  Card,
  Field,
  LoadingBlock,
  Modal,
  Notice,
  PageHeader,
  SectionHeading,
  StatusBadge,
} from '../../components/ui';
import { del, errorMessage, get, patch, post } from '../../lib/api';
import { dateTime, timeAgo } from '../../lib/format';
import { qk, queryClient } from '../../lib/query';
import { useAuth } from '../../store/auth';
import type { MessageResponse, User } from '../../types/api';

interface Device {
  id: number;
  user_agent: string | null;
  ip_address: string | null;
  trusted: boolean;
  login_count: number;
  last_seen_at: string | null;
  created_at: string;
}

interface TwoFactorSetup {
  secret: string;
  provisioning_uri: string;
  detail: string;
}

export default function Settings() {
  const { user, setUser } = useAuth();

  const [banner, setBanner] = useState<{ tone: 'success' | 'danger'; text: string } | null>(null);

  // Profile
  const [fullName, setFullName] = useState(user?.full_name ?? '');
  const [phone, setPhone] = useState(user?.phone ?? '');
  const [city, setCity] = useState(user?.city ?? '');

  // Password
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // 2FA
  const [setupData, setSetupData] = useState<TwoFactorSetup | null>(null);
  const [totpCode, setTotpCode] = useState('');
  const [disableOpen, setDisableOpen] = useState(false);

  const { data: devices, isLoading: devicesLoading } = useQuery({
    queryKey: qk.devices,
    queryFn: () => get<Device[]>('/profile/devices'),
  });

  function notify(tone: 'success' | 'danger', text: string) {
    setBanner({ tone, text });
    window.setTimeout(() => setBanner(null), 5000);
  }

  const updateProfile = useMutation({
    mutationFn: () =>
      patch<User>('/profile', {
        full_name: fullName.trim(),
        phone: phone.trim() || null,
        city: city.trim() || null,
      }),
    onSuccess: (data) => {
      setUser(data);
      notify('success', 'Profile updated.');
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const updatePrefs = useMutation({
    mutationFn: (patchBody: Record<string, boolean>) => patch<User>('/profile/notifications', patchBody),
    onSuccess: (data) => {
      setUser(data);
      notify('success', 'Notification preferences saved.');
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const changePassword = useMutation({
    mutationFn: () =>
      post<MessageResponse>('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      }),
    onSuccess: (data) => {
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      notify('success', data.detail ?? data.message);
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const startTwoFactor = useMutation({
    mutationFn: () => post<TwoFactorSetup>('/auth/2fa/setup'),
    onSuccess: (data) => setSetupData(data),
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const enableTwoFactor = useMutation({
    mutationFn: () => post<MessageResponse>('/auth/2fa/enable', { code: totpCode }),
    onSuccess: async () => {
      setSetupData(null);
      setTotpCode('');
      notify('success', 'Two-factor authentication enabled.');
      const refreshed = await get<User>('/auth/me');
      setUser(refreshed);
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const disableTwoFactor = useMutation({
    mutationFn: () => post<MessageResponse>('/auth/2fa/disable', { code: totpCode }),
    onSuccess: async () => {
      setDisableOpen(false);
      setTotpCode('');
      notify('success', 'Two-factor authentication disabled.');
      const refreshed = await get<User>('/auth/me');
      setUser(refreshed);
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const forgetDevice = useMutation({
    mutationFn: (id: number) => del<MessageResponse>(`/profile/devices/${id}`),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: qk.devices });
      notify('success', 'Device removed.');
    },
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const signOutEverywhere = useMutation({
    mutationFn: () => post<MessageResponse>('/auth/logout-all'),
    onSuccess: (data) => notify('success', data.message),
    onError: (err) => notify('danger', errorMessage(err)),
  });

  const passwordMismatch = confirmPassword.length > 0 && newPassword !== confirmPassword;

  return (
    <div>
      <PageHeader title="Settings" subtitle="Manage your profile, security and notifications." />

      {banner && (
        <div className="mb-6">
          <Notice tone={banner.tone}>{banner.text}</Notice>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ------------------------------ profile ------------------------------ */}
        <Card>
          <SectionHeading title="Profile" />
          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              updateProfile.mutate();
            }}
          >
            <Field label="Full name" htmlFor="settingsName">
              <input
                id="settingsName"
                className="input"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </Field>
            <Field label="Phone" htmlFor="settingsPhone">
              <input
                id="settingsPhone"
                className="input"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
              />
            </Field>
            <Field label="City" htmlFor="settingsCity">
              <input
                id="settingsCity"
                className="input"
                value={city}
                onChange={(e) => setCity(e.target.value)}
              />
            </Field>

            <dl className="space-y-2 border-t border-line pt-4 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted">Email</dt>
                <dd className="text-primary">{user?.email}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Account status</dt>
                <dd>{user && <StatusBadge status={user.status} />}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">KYC</dt>
                <dd>{user && <StatusBadge status={user.kyc_status} />}</dd>
              </div>
              {user?.pan_masked && (
                <div className="flex justify-between">
                  <dt className="text-muted">PAN on file</dt>
                  <dd className="font-mono text-xs text-muted">{user.pan_masked}</dd>
                </div>
              )}
            </dl>

            <button type="submit" className="btn-primary w-full py-2.5" disabled={updateProfile.isPending}>
              {updateProfile.isPending ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
              Save profile
            </button>
          </form>
        </Card>

        {/* ------------------------------ password ------------------------------ */}
        <Card>
          <SectionHeading
            title="Password"
            subtitle="Changing your password signs out every other device."
          />
          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              if (!passwordMismatch) changePassword.mutate();
            }}
          >
            <Field label="Current password" htmlFor="currentPassword" required>
              <input
                id="currentPassword"
                type="password"
                className="input"
                autoComplete="current-password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
              />
            </Field>
            <Field
              label="New password"
              htmlFor="newPassword"
              hint="At least 10 characters with upper, lower, digit and symbol"
              required
            >
              <input
                id="newPassword"
                type="password"
                className="input"
                autoComplete="new-password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </Field>
            <Field
              label="Confirm new password"
              htmlFor="confirmPassword"
              error={passwordMismatch ? 'Passwords do not match' : undefined}
              required
            >
              <input
                id="confirmPassword"
                type="password"
                className="input"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </Field>

            <button
              type="submit"
              className="btn-primary w-full py-2.5"
              disabled={changePassword.isPending || passwordMismatch || !currentPassword || !newPassword}
            >
              {changePassword.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <KeyRound className="h-4 w-4" aria-hidden />
              )}
              Change password
            </button>
          </form>
        </Card>

        {/* -------------------------------- 2FA -------------------------------- */}
        <Card>
          <SectionHeading
            title="Two-factor authentication"
            action={user?.two_factor_enabled ? <Badge tone="success">Enabled</Badge> : <Badge tone="warning">Off</Badge>}
          />

          {user?.two_factor_enabled ? (
            <div className="space-y-4">
              <Notice tone="success" title="Your account has a second factor">
                Sign-in requires a code from your authenticator app in addition to your password.
              </Notice>
              <button
                type="button"
                className="btn-secondary w-full py-2.5"
                onClick={() => {
                  setDisableOpen(true);
                  setTotpCode('');
                }}
              >
                Disable two-factor authentication
              </button>
            </div>
          ) : setupData ? (
            <div className="space-y-4">
              <Notice tone="info" title="Add this secret to your authenticator">
                Enter the key below in Google Authenticator, Authy or 1Password, then confirm with a
                generated code.
              </Notice>

              <div className="rounded-lg border border-line bg-ink/60 p-3.5">
                <p className="mb-1 text-xs text-muted">Setup key</p>
                <p className="font-mono text-sm break-all text-gold-bright">{setupData.secret}</p>
              </div>

              <Field label="Verification code" htmlFor="enableTotp" required>
                <input
                  id="enableTotp"
                  inputMode="numeric"
                  maxLength={6}
                  className="input tnum text-center text-lg tracking-[0.4em]"
                  value={totpCode}
                  onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
                  placeholder="000000"
                  autoFocus
                />
              </Field>

              <div className="flex gap-2">
                <button
                  type="button"
                  className="btn-secondary flex-1 py-2.5"
                  onClick={() => {
                    setSetupData(null);
                    setTotpCode('');
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn-primary flex-1 py-2.5"
                  onClick={() => enableTwoFactor.mutate()}
                  disabled={enableTwoFactor.isPending || totpCode.length !== 6}
                >
                  {enableTwoFactor.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                  ) : (
                    <Check className="h-4 w-4" aria-hidden />
                  )}
                  Enable
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-muted">
                Protect your account with a time-based code from an authenticator app. This is the
                single most effective defence against a stolen password.
              </p>
              <button
                type="button"
                className="btn-primary w-full py-2.5"
                onClick={() => startTwoFactor.mutate()}
                disabled={startTwoFactor.isPending}
              >
                {startTwoFactor.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                ) : (
                  <ShieldCheck className="h-4 w-4" aria-hidden />
                )}
                Set up two-factor authentication
              </button>
            </div>
          )}
        </Card>

        {/* --------------------------- notifications --------------------------- */}
        <Card>
          <SectionHeading title="Notifications" />
          <div className="space-y-3">
            {[
              { key: 'notify_large_txn', label: 'Large transactions', desc: 'Alert me above ₹50,000' },
              { key: 'notify_login', label: 'New device sign-in', desc: 'Tell me when a new device is used' },
              { key: 'notify_email', label: 'Email delivery', desc: 'Queue notifications for email (simulated)' },
              { key: 'notify_sms', label: 'SMS delivery', desc: 'Queue notifications for SMS (simulated)' },
              { key: 'notify_marketing', label: 'Product updates', desc: 'Occasional non-essential messages' },
            ].map((pref) => {
              const checked = Boolean(user?.[pref.key as keyof User]);
              return (
                <label
                  key={pref.key}
                  className="flex cursor-pointer items-start gap-3 rounded-lg border border-line p-3 transition hover:border-line-strong"
                >
                  <input
                    type="checkbox"
                    className="mt-0.5 h-4 w-4 accent-gold"
                    checked={checked}
                    onChange={(e) => updatePrefs.mutate({ [pref.key]: e.target.checked })}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium text-primary">{pref.label}</span>
                    <span className="block text-xs text-muted">{pref.desc}</span>
                  </span>
                </label>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-faint">
            Security notifications cannot be turned off — you will always be told about password
            changes and fraud alerts.
          </p>
        </Card>

        {/* ------------------------------ devices ------------------------------ */}
        <Card className="lg:col-span-2">
          <SectionHeading
            title="Signed-in devices"
            subtitle="Devices that have accessed your account"
            action={
              <button
                type="button"
                className="btn-secondary px-3.5 py-2 text-xs"
                onClick={() => signOutEverywhere.mutate()}
                disabled={signOutEverywhere.isPending}
              >
                Sign out everywhere
              </button>
            }
          />

          {devicesLoading ? (
            <LoadingBlock rows={2} />
          ) : devices?.length ? (
            <ul className="divide-y divide-line">
              {devices.map((device) => {
                const isMobile = /mobile|android|iphone/i.test(device.user_agent ?? '');
                return (
                  <li key={device.id} className="flex items-center gap-3 py-3">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-surface-raised text-muted">
                      {isMobile ? (
                        <Smartphone className="h-4 w-4" aria-hidden />
                      ) : (
                        <Monitor className="h-4 w-4" aria-hidden />
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm text-primary">
                        {device.user_agent?.slice(0, 68) ?? 'Unknown device'}
                      </p>
                      <p className="text-xs text-muted">
                        {device.ip_address ?? 'Unknown IP'} · {device.login_count} sign-in
                        {device.login_count === 1 ? '' : 's'} ·{' '}
                        {device.last_seen_at ? timeAgo(device.last_seen_at) : dateTime(device.created_at)}
                      </p>
                    </div>
                    {device.trusted && <Badge tone="success">Trusted</Badge>}
                    <button
                      type="button"
                      className="rounded p-1.5 text-muted transition hover:text-alert"
                      aria-label="Forget this device"
                      onClick={() => forgetDevice.mutate(device.id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden />
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="py-6 text-center text-sm text-muted">No devices recorded yet.</p>
          )}
        </Card>
      </div>

      {/* --------------------------- disable 2FA --------------------------- */}
      <Modal
        open={disableOpen}
        onClose={() => setDisableOpen(false)}
        title="Disable two-factor authentication"
        footer={
          <>
            <button type="button" className="btn-secondary px-4 py-2" onClick={() => setDisableOpen(false)}>
              Keep it enabled
            </button>
            <button
              type="button"
              className="btn-danger px-4 py-2"
              onClick={() => disableTwoFactor.mutate()}
              disabled={disableTwoFactor.isPending || totpCode.length !== 6}
            >
              {disableTwoFactor.isPending ? 'Disabling…' : 'Disable'}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <Notice tone="warning">
            Disabling two-factor authentication makes your account significantly easier to
            compromise if your password is ever leaked.
          </Notice>
          <Field
            label="Current authentication code"
            htmlFor="disableTotp"
            hint="Required — a stolen session alone cannot remove your second factor"
            required
          >
            <input
              id="disableTotp"
              inputMode="numeric"
              maxLength={6}
              className="input tnum text-center text-lg tracking-[0.4em]"
              value={totpCode}
              onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
              placeholder="000000"
              autoFocus
            />
          </Field>
        </div>
      </Modal>
    </div>
  );
}
