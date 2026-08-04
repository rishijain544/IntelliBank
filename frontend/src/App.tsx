/**
 * Route table and access guards.
 *
 * Routing is the only place role logic lives on the client. The server re-checks
 * role and account status on every request, so these guards are purely about not
 * rendering a shell the user cannot use — they are not a security boundary.
 */
import { lazy, Suspense, useEffect } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';

import { Spinner } from './components/ui';
import AdminLayout from './layouts/AdminLayout';
import AppLayout from './layouts/AppLayout';
import PublicLayout from './layouts/PublicLayout';
import { useAuth } from './store/auth';

// Public pages are eager: they are the first paint and are small.
import Landing from './pages/public/Landing';
import Login from './pages/public/Login';
import Register from './pages/public/Register';

// Everything behind auth is lazy, so the landing page does not download the
// charting library or the admin console.
const Features = lazy(() => import('./pages/public/Features'));
const Contact = lazy(() => import('./pages/public/Contact'));

const Dashboard = lazy(() => import('./pages/app/Dashboard'));
const Accounts = lazy(() => import('./pages/app/Accounts'));
const Transactions = lazy(() => import('./pages/app/Transactions'));
const Transfer = lazy(() => import('./pages/app/Transfer'));
const Cards = lazy(() => import('./pages/app/Cards'));
const Loans = lazy(() => import('./pages/app/Loans'));
const FraudCenter = lazy(() => import('./pages/app/FraudCenter'));
const Insights = lazy(() => import('./pages/app/Insights'));
const Settings = lazy(() => import('./pages/app/Settings'));

const AdminDashboard = lazy(() => import('./pages/admin/AdminDashboard'));
const AdminUsers = lazy(() => import('./pages/admin/AdminUsers'));
const AdminFraudQueue = lazy(() => import('./pages/admin/AdminFraudQueue'));
const AdminLoanQueue = lazy(() => import('./pages/admin/AdminLoanQueue'));
const AdminAnalytics = lazy(() => import('./pages/admin/AdminAnalytics'));

function FullPageSpinner() {
  return (
    <div className="flex min-h-screen items-center justify-center" role="status" aria-live="polite">
      <Spinner className="h-7 w-7 text-gold" />
      <span className="sr-only">Loading</span>
    </div>
  );
}

function RequireAuth({ children, adminOnly = false }: { children: React.ReactNode; adminOnly?: boolean }) {
  const { user, initialising } = useAuth();
  const location = useLocation();

  // Wait for the session check rather than bouncing an authenticated user to
  // /login on a hard refresh.
  if (initialising) return <FullPageSpinner />;

  if (!user) {
    // Remember the attempted URL so login can return the user to it.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (adminOnly && user.role !== 'admin') {
    return <Navigate to="/app" replace />;
  }
  return <>{children}</>;
}

/** Signed-in users have no reason to see the marketing or auth pages. */
function RedirectIfAuthed({ children }: { children: React.ReactNode }) {
  const { user, initialising } = useAuth();
  if (initialising) return <FullPageSpinner />;
  if (user) return <Navigate to={user.role === 'admin' ? '/admin' : '/app'} replace />;
  return <>{children}</>;
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  const bootstrap = useAuth((s) => s.bootstrap);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  return (
    <>
      <ScrollToTop />
      <Suspense fallback={<FullPageSpinner />}>
        <Routes>
          {/* ---------------- public ---------------- */}
          <Route element={<PublicLayout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/features" element={<Features />} />
            <Route path="/contact" element={<Contact />} />
            <Route
              path="/login"
              element={
                <RedirectIfAuthed>
                  <Login />
                </RedirectIfAuthed>
              }
            />
            <Route
              path="/register"
              element={
                <RedirectIfAuthed>
                  <Register />
                </RedirectIfAuthed>
              }
            />
          </Route>

          {/* ---------------- customer ---------------- */}
          <Route
            path="/app"
            element={
              <RequireAuth>
                <AppLayout />
              </RequireAuth>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="accounts" element={<Accounts />} />
            <Route path="transactions" element={<Transactions />} />
            <Route path="transfer" element={<Transfer />} />
            <Route path="cards" element={<Cards />} />
            <Route path="loans" element={<Loans />} />
            <Route path="fraud-center" element={<FraudCenter />} />
            <Route path="insights" element={<Insights />} />
            <Route path="settings" element={<Settings />} />
          </Route>

          {/* ---------------- admin ---------------- */}
          <Route
            path="/admin"
            element={
              <RequireAuth adminOnly>
                <AdminLayout />
              </RequireAuth>
            }
          >
            <Route index element={<AdminDashboard />} />
            <Route path="users" element={<AdminUsers />} />
            <Route path="fraud" element={<AdminFraudQueue />} />
            <Route path="loans" element={<AdminLoanQueue />} />
            <Route path="analytics" element={<AdminAnalytics />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </>
  );
}
