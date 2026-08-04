import { useState } from 'react';
import { AlertTriangle, BookOpen, Github, Mail, MessageSquare, Send, ShieldCheck } from 'lucide-react';

import { Card, Field, Notice } from '../../components/ui';

/**
 * Support / contact page.
 *
 * The form is intentionally non-functional and says so: wiring a real inbox to a
 * portfolio project invites spam and implies a support commitment that does not
 * exist. Pretending it sends would be worse than being honest about it.
 */
export default function Contact() {
  const [submitted, setSubmitted] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [topic, setTopic] = useState('general');
  const [message, setMessage] = useState('');

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setSubmitted(true);
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-16">
      <div className="mx-auto mb-12 max-w-2xl text-center">
        <h1 className="text-3xl font-bold text-primary sm:text-4xl">Support</h1>
        <p className="mt-3 text-muted">
          This is a portfolio project rather than a live bank, so here is what that means in
          practice.
        </p>
      </div>

      <div className="mb-10">
        <Notice tone="warning" title="No real support desk exists">
          IntelliBank is an educational demonstration. There is no operations team, no real money and
          no customer service line. The form below is a UI mock and does not send anything. Never
          submit real personal or financial details.
        </Notice>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_1.2fr]">
        <div className="space-y-4">
          {[
            {
              icon: BookOpen,
              title: 'Technical documentation',
              body: 'Architecture, model methodology, measured metrics and the full list of known limitations are in the project README.',
            },
            {
              icon: MessageSquare,
              title: 'Interactive API reference',
              body: 'Every endpoint is browsable and callable at /docs while the backend is running.',
            },
            {
              icon: Github,
              title: 'Source code',
              body: 'The backend, ML pipeline and this frontend are all readable end to end — including the tests that guard the tricky parts.',
            },
            {
              icon: ShieldCheck,
              title: 'Security questions',
              body: 'Password hashing, token rotation and reuse detection, rate limiting and audit logging are documented in the README security section.',
            },
          ].map((item) => (
            <Card key={item.title} className="flex gap-4">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-gold/10 text-gold">
                <item.icon className="h-5 w-5" aria-hidden />
              </span>
              <div>
                <h2 className="font-semibold text-primary">{item.title}</h2>
                <p className="mt-1 text-sm text-muted">{item.body}</p>
              </div>
            </Card>
          ))}
        </div>

        <Card>
          {submitted ? (
            <div className="py-10 text-center">
              <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-full bg-warning/15 text-warning">
                <AlertTriangle className="h-6 w-6" aria-hidden />
              </span>
              <h2 className="text-lg font-semibold text-primary">Nothing was sent</h2>
              <p className="mx-auto mt-2 max-w-xs text-sm text-muted">
                As noted above, this form is a mock. Your message was not transmitted or stored
                anywhere — the interaction exists to demonstrate the UI only.
              </p>
              <button
                type="button"
                className="btn-secondary mt-6 px-5 py-2"
                onClick={() => {
                  setSubmitted(false);
                  setMessage('');
                }}
              >
                Back to the form
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <h2 className="font-semibold text-primary">Send a message (demo only)</h2>

              <Field label="Name" htmlFor="name">
                <input
                  id="name"
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="off"
                />
              </Field>

              <Field label="Email" htmlFor="contactEmail" hint="Use a placeholder address">
                <input
                  id="contactEmail"
                  type="email"
                  className="input"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="off"
                  placeholder="you@example.com"
                />
              </Field>

              <Field label="Topic" htmlFor="topic">
                <select id="topic" className="input" value={topic} onChange={(e) => setTopic(e.target.value)}>
                  <option value="general">General question</option>
                  <option value="models">About the ML models</option>
                  <option value="architecture">Architecture and code</option>
                  <option value="security">Security design</option>
                </select>
              </Field>

              <Field label="Message" htmlFor="message">
                <textarea
                  id="message"
                  rows={5}
                  className="input resize-none"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="What would you like to know?"
                />
              </Field>

              <button type="submit" className="btn-primary w-full py-2.5">
                <Send className="h-4 w-4" aria-hidden />
                Send message
              </button>

              <p className="flex items-center justify-center gap-1.5 text-xs text-faint">
                <Mail className="h-3 w-3" aria-hidden />
                This form does not transmit or store data
              </p>
            </form>
          )}
        </Card>
      </div>
    </div>
  );
}
