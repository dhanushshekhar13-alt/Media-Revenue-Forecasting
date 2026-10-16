import { useEffect, useState } from "react";
import { API_URL, api, type DatasetSummary } from "./api";
import { ErrorBox, Loading } from "./components/ui";
import { title } from "./format";
import { Benchmark } from "./pages/Benchmark";
import { Data } from "./pages/Data";
import { Forecast } from "./pages/Forecast";
import { Optimizer } from "./pages/Optimizer";
import { Overview } from "./pages/Overview";

const TABS = ["overview", "forecast", "optimizer", "benchmark", "data"] as const;
type Tab = (typeof TABS)[number];
const TAB_LABELS: Record<Tab, string> = {
  overview: "Overview",
  forecast: "Forecast",
  optimizer: "Budget optimizer",
  benchmark: "Model benchmark",
  data: "Data",
};

function initialTab(): Tab {
  const h = window.location.hash.replace("#", "") as Tab;
  return TABS.includes(h) ? h : "overview";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [datasetId, setDatasetId] = useState("synthetic");
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    api.datasets().then(setDatasets).catch((e: Error) => setError(e.message));
  };
  useEffect(load, []);
  useEffect(() => {
    window.location.hash = tab;
  }, [tab]);
  useEffect(() => {
    const onHash = () => setTab(initialTab());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const current = datasets?.find((d) => d.dataset_id === datasetId);
  const channels = current?.channels ?? [];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="var(--accent)" />
            <path d="M6 22 L12 15 L17 19 L26 9" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="brand-name">Media Revenue Forecasting</div>
            <div className="muted small">Probabilistic P10 / P50 / P90 across {channels.length ? channels.map(title).join(", ") : "Google, Meta, Bing"}</div>
          </div>
        </div>
        {datasets && (
          <label className="dataset-picker">
            <span className="muted small">Dataset</span>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.name ?? (d.dataset_id === "synthetic" ? "Synthetic demo" : d.dataset_id)}
                </option>
              ))}
            </select>
          </label>
        )}
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
            {TAB_LABELS[t]}
          </button>
        ))}
      </nav>
      <main>
        {error && (
          <ErrorBox error={`Could not reach the API at ${API_URL} (${error}). It may be waking up; free Render instances sleep when idle.`} onRetry={load} />
        )}
        {!datasets && !error && <Loading label="Connecting to API" />}
        {datasets && (
          <>
            {tab === "overview" && <Overview datasetId={datasetId} />}
            {tab === "forecast" && <Forecast datasetId={datasetId} channels={channels} />}
            {tab === "optimizer" && <Optimizer datasetId={datasetId} channels={channels} />}
            {tab === "benchmark" && <Benchmark datasetId={datasetId} />}
            {tab === "data" && (
              <Data
                datasets={datasets}
                onUploaded={(d) => {
                  setDatasets([...datasets, d]);
                  setDatasetId(d.dataset_id);
                  setTab("forecast");
                }}
              />
            )}
          </>
        )}
      </main>
      <footer className="muted small">
        API: <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">{API_URL}/docs</a>
      </footer>
    </div>
  );
}
