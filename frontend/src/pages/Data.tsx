import { useState } from "react";
import { api, type DatasetSummary } from "../api";
import { Card, ErrorBox } from "../components/ui";
import { money, num, title } from "../format";

export function Data({ datasets, onUploaded }: { datasets: DatasetSummary[]; onUploaded: (d: DatasetSummary) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const upload = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onUploaded(await api.upload(file));
      setFile(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <Card title="Bring your own data" subtitle="Upload a daily CSV and every page (forecast, optimiser, model details) switches to it.">
        <p className="small">
          Required columns: <code>date</code>, <code>channel</code>, <code>spend</code>, <code>revenue</code> (one row per channel per day; extra
          columns are ignored). Each channel needs at least 450 days of history. Missing days are interpolated and the common date range across
          channels is used.
        </p>
        <pre className="code">{`date,channel,spend,revenue
2024-01-01,google,11843.20,35110.92
2024-01-01,meta,7420.55,16012.40
2024-01-01,bing,2310.00,7342.18`}</pre>
        <div className="controls">
          <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <button className="btn" disabled={!file || busy} onClick={upload}>
            {busy ? "Uploading…" : "Upload"}
          </button>
        </div>
        {error && <ErrorBox error={error} />}
        <p className="muted small">Uploaded datasets live in the API's memory and are cleared when the server restarts.</p>
      </Card>

      <Card title="Datasets on this server">
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Range</th>
                <th className="r">Rows</th>
                {["Spend", "Revenue", "ROAS"].map((h) => (
                  <th key={h} className="r">{h}</th>
                ))}
                <th>Channels</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const spend = Object.values(d.totals).reduce((a, t) => a + t.spend, 0);
                const rev = Object.values(d.totals).reduce((a, t) => a + t.revenue, 0);
                return (
                  <tr key={d.dataset_id}>
                    <td>{d.name ?? (d.dataset_id === "synthetic" ? "Synthetic demo" : d.dataset_id)}</td>
                    <td>
                      {d.start} – {d.end}
                    </td>
                    <td className="r">{d.rows.toLocaleString()}</td>
                    <td className="r">{money(spend)}</td>
                    <td className="r">{money(rev)}</td>
                    <td className="r">{num(rev / spend)}</td>
                    <td>{d.channels.map(title).join(", ")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
