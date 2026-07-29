import { useState } from "react";
import CustomerForm from "./CustomerForm";
import ResultPanel from "./ResultPanel";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-bg-deep px-4 py-10 md:py-16">
      <div className="max-w-5xl mx-auto">
        <header className="mb-10 text-center">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-signal-safe mb-2">
            Retention diagnostics
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary">
            Customer Churn Signal
          </h1>
          <p className="text-text-muted text-sm mt-2 max-w-md mx-auto">
            Enter a customer's account details to read their churn risk and the
            factors driving it.
          </p>
        </header>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-bg-panel border border-border-hair rounded-xl p-6">
            <CustomerForm
              onResult={(r) => { setResult(r); setError(null); }}
              onError={(e) => { setError(e); if (e) setResult(null); }}
              onLoadingChange={setLoading}
            />
          </div>

          <div className="bg-bg-panel border border-border-hair rounded-xl p-6 min-h-[520px]">
            <ResultPanel result={result} loading={loading} error={error} />
          </div>
        </div>
      </div>
    </div>
  );
}