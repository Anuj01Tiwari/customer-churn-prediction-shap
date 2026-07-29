import { useState } from "react";

// Default values -- a "typical" mid-tenure customer to start from.
// Using the e
const initialForm = {
  "Gender": "Female",
  "Senior Citizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Tenure Months": 12,
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "Fiber optic",
  "Online Security": "No",
  "Online Backup": "No",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 70,
  "Total Charges": 840,
};

// Small reusable field wrapper so every row gets the same label + spacing
function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
        {label}
      </span>
      {children}
    </label>
  );
}

const selectClasses =
  "bg-bg-panel-soft border border-border-hair rounded-md px-3 py-2 text-sm " +
  "text-text-primary focus:border-signal-safe transition-colors disabled:opacity-40 disabled:cursor-not-allowed";

const numberClasses =
  "bg-bg-panel-soft border border-border-hair rounded-md px-3 py-2 text-sm font-mono " +
  "text-text-primary focus:border-signal-safe transition-colors";

function YesNoSelect({ value, onChange, disabled = false, extraOption = null }) {
  return (
    <select
      className={selectClasses}
      value={value}
      onChange={onChange}
      disabled={disabled}
    >
      <option value="Yes">Yes</option>
      <option value="No">No</option>
      {extraOption && <option value={extraOption}>{extraOption}</option>}
    </select>
  );
}

export default function CustomerForm({ onResult, onError, onLoadingChange }) {
  const [form, setForm] = useState(initialForm);
  const [submitting, setSubmitting] = useState(false);

  const hasPhone = form["Phone Service"] === "Yes";
  const hasInternet = form["Internet Service"] !== "No";

  function update(key, value) {
    setForm((prev) => {
      const next = { ...prev, [key]: value };

      // Domain logic learned back in Stage 1 cleaning: these fields only make
      // sense conditionally. Keep the payload consistent with what the model
      // was trained on instead of letting the UI drift into contradictions.
      if (key === "Phone Service" && value === "No") {
        next["Multiple Lines"] = "No phone service";
      }
      if (key === "Internet Service" && value === "No") {
        next["Online Security"] = "No internet service";
        next["Online Backup"] = "No internet service";
        next["Device Protection"] = "No internet service";
        next["Tech Support"] = "No internet service";
        next["Streaming TV"] = "No internet service";
        next["Streaming Movies"] = "No internet service";
      }
      // Coming back out of "No", reset dependents to a sane default instead
      // of leaving the placeholder value selected.
      if (key === "Phone Service" && value === "Yes" && prev["Multiple Lines"] === "No phone service") {
        next["Multiple Lines"] = "No";
      }
      if (key === "Internet Service" && value !== "No" && prev["Internet Service"] === "No") {
        ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
          .forEach((f) => { next[f] = "No"; });
      }
      return next;
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    onLoadingChange(true);
    onError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(
          detail?.detail
            ? typeof detail.detail === "string"
              ? detail.detail
              : "Check the highlighted fields and try again."
            : `Server responded with ${response.status}`
        );
      }

      const data = await response.json();
      onResult(data);
    } catch (err) {
      onError(
        err.message === "Failed to fetch"
          ? "Can't reach the prediction server. Make sure the FastAPI backend is running on port 8000."
          : err.message
      );
    } finally {
      setSubmitting(false);
      onLoadingChange(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      {/* Section: Profile */}
      <section className="flex flex-col gap-3">
        <h2 className="font-display text-sm font-semibold text-text-primary tracking-wide">
          Customer profile
        </h2>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Gender">
            <select className={selectClasses} value={form["Gender"]} onChange={(e) => update("Gender", e.target.value)}>
              <option value="Female">Female</option>
              <option value="Male">Male</option>
            </select>
          </Field>
          <Field label="Senior citizen">
            <YesNoSelect value={form["Senior Citizen"]} onChange={(e) => update("Senior Citizen", e.target.value)} />
          </Field>
          <Field label="Partner">
            <YesNoSelect value={form["Partner"]} onChange={(e) => update("Partner", e.target.value)} />
          </Field>
          <Field label="Dependents">
            <YesNoSelect value={form["Dependents"]} onChange={(e) => update("Dependents", e.target.value)} />
          </Field>
        </div>
      </section>

      {/* Section: Account */}
      <section className="flex flex-col gap-3">
        <h2 className="font-display text-sm font-semibold text-text-primary tracking-wide">
          Account
        </h2>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Tenure (months)">
            <input
              type="number" min={0} max={100}
              className={numberClasses}
              value={form["Tenure Months"]}
              onChange={(e) => update("Tenure Months", Number(e.target.value))}
            />
          </Field>
          <Field label="Contract">
            <select className={selectClasses} value={form["Contract"]} onChange={(e) => update("Contract", e.target.value)}>
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </Field>
          <Field label="Paperless billing">
            <YesNoSelect value={form["Paperless Billing"]} onChange={(e) => update("Paperless Billing", e.target.value)} />
          </Field>
          <Field label="Payment method">
            <select className={selectClasses} value={form["Payment Method"]} onChange={(e) => update("Payment Method", e.target.value)}>
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </Field>
          <Field label="Monthly charges ($)">
            <input
              type="number" min={0} step="0.01"
              className={numberClasses}
              value={form["Monthly Charges"]}
              onChange={(e) => update("Monthly Charges", Number(e.target.value))}
            />
          </Field>
          <Field label="Total charges ($)">
            <input
              type="number" min={0} step="0.01"
              className={numberClasses}
              value={form["Total Charges"]}
              onChange={(e) => update("Total Charges", Number(e.target.value))}
            />
          </Field>
        </div>
      </section>

      {/* Section: Services */}
      <section className="flex flex-col gap-3">
        <h2 className="font-display text-sm font-semibold text-text-primary tracking-wide">
          Services
        </h2>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Phone service">
            <YesNoSelect value={form["Phone Service"]} onChange={(e) => update("Phone Service", e.target.value)} />
          </Field>
          <Field label="Multiple lines">
            <select
              className={selectClasses}
              value={form["Multiple Lines"]}
              onChange={(e) => update("Multiple Lines", e.target.value)}
              disabled={!hasPhone}
            >
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No phone service">No phone service</option>
            </select>
          </Field>
          <Field label="Internet service">
            <select className={selectClasses} value={form["Internet Service"]} onChange={(e) => update("Internet Service", e.target.value)}>
              <option value="DSL">DSL</option>
              <option value="Fiber optic">Fiber optic</option>
              <option value="No">No</option>
            </select>
          </Field>
          {[
            ["Online Security", "Online security"],
            ["Online Backup", "Online backup"],
            ["Device Protection", "Device protection"],
            ["Tech Support", "Tech support"],
            ["Streaming TV", "Streaming TV"],
            ["Streaming Movies", "Streaming movies"],
          ].map(([key, label]) => (
            <Field key={key} label={label}>
              <select
                className={selectClasses}
                value={form[key]}
                onChange={(e) => update(key, e.target.value)}
                disabled={!hasInternet}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </Field>
          ))}
        </div>
      </section>

      <button
        type="submit"
        disabled={submitting}
        className="mt-2 font-display font-semibold text-sm rounded-md py-3
                   bg-signal-safe text-bg-deep hover:brightness-110
                   disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {submitting ? "Reading signal…" : "Run prediction"}
      </button>
    </form>
  );
}