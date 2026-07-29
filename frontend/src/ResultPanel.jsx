// The circumference math below draws a 270-degree arc (a gauge with a gap
// at the bottom, like an analog speedometer) using pathLength=100 so the
// dasharray percentages are easy to reason about: the full track is 75 of
// a possible 100 units (270/360 = 0.75), and the filled portion scales with
// the churn probability.
const ARC_FRACTION = 0.75;

function riskColor(prob) {
  if (prob < 0.34) return "var(--color-signal-safe)";
  if (prob < 0.67) return "var(--color-signal-warn)";
  return "var(--color-signal-risk)";
}

function riskLabel(prob) {
  if (prob < 0.34) return "Low risk";
  if (prob < 0.67) return "Moderate risk";
  return "High risk";
}

function Gauge({ probability }) {
  const color = riskColor(probability);
  const filled = probability * ARC_FRACTION * 100;
  const track = ARC_FRACTION * 100;

  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg viewBox="0 0 100 100" className="w-full h-full rotate-[135deg]">
        <circle
          cx="50" cy="50" r="42"
          pathLength="100"
          fill="none"
          stroke="var(--color-bg-panel-soft)"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${track} 100`}
        />
        <circle
          cx="50" cy="50" r="42"
          pathLength="100"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filled} 100`}
          style={{ transition: "stroke-dasharray 0.6s ease, stroke 0.6s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-4xl font-medium" style={{ color }}>
          {(probability * 100).toFixed(1)}
          <span className="text-lg">%</span>
        </span>
        <span className="text-xs uppercase tracking-wider text-text-muted mt-1">
          churn probability
        </span>
      </div>
    </div>
  );
}

function ReasonRow({ reason }) {
  const isIncrease = reason.direction === "increases";
  return (
    <li className="flex items-start gap-3 py-3 border-b border-border-hair last:border-0">
      <span
        className="font-mono text-sm mt-0.5 shrink-0 w-5 text-center"
        style={{ color: isIncrease ? "var(--color-signal-risk)" : "var(--color-signal-safe)" }}
        aria-hidden="true"
      >
        {isIncrease ? "▲" : "▼"}
      </span>
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-text-primary">{reason.feature}</span>
        <span className="text-xs text-text-muted">{reason.description}</span>
      </div>
    </li>
  );
}

export default function ResultPanel({ result, loading, error }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-text-muted">
        <div className="w-10 h-10 border-2 border-border-hair border-t-signal-safe rounded-full animate-spin" />
        <span className="text-sm font-mono">Reading signal…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-2 h-full justify-center">
        <span className="font-display text-sm font-semibold" style={{ color: "var(--color-signal-risk)" }}>
          Signal lost
        </span>
        <p className="text-sm text-text-muted">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
        <div className="w-16 h-16 rounded-full border-2 border-dashed border-border-hair" />
        <p className="text-sm text-text-muted max-w-56">
          Fill in the customer details and run a prediction to see their churn signal.
        </p>
      </div>
    );
  }

  const color = riskColor(result.churn_probability);

  return (
    <div className="flex flex-col gap-6">
      <Gauge probability={result.churn_probability} />

      <div className="text-center">
        <span
          className="inline-block font-mono text-xs uppercase tracking-wider px-3 py-1 rounded-full"
          style={{ color, border: `1px solid ${color}` }}
        >
          {riskLabel(result.churn_probability)} · predicted churn: {result.churn_prediction}
        </span>
      </div>

      <div>
        <h3 className="font-display text-sm font-semibold text-text-primary mb-1">
          Top signal drivers
        </h3>
        <p className="text-xs text-text-muted mb-2">
          The features that most influenced this specific prediction.
        </p>
        <ul>
          {result.top_reasons.map((reason, i) => (
            <ReasonRow key={i} reason={reason} />
          ))}
        </ul>
      </div>
    </div>
  );
}