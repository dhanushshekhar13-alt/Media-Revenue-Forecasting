import { useEffect, useMemo, useState } from "react";
import { api, type ForecastResponse, type ModelName } from "../api";
import { FanChart } from "../components/FanChart";
import { Card, ErrorBox, Loading, Stat, useAsync } from "../components/ui";
import { channelColor, downloadCsv, money, num, title } from "../format";

const MODEL_OPTIONS: Array<{ value: ModelName; label: string }> = [
  { value: "roas_quantile", label: "ROAS quantile (recommended)" },
  { value: "prophet_spend", label: "Prophet + spend regressor" },
  { value: "prophet", label: "Prophet (univariate baseline)" },
  { value: "arima", label: "SARIMA (baseline)" },
];

export function Forecast({ datasetId, channels }: { datasetId: string; channels: string[] }) {
  const [model, setModel] = useState<ModelName>("roas_quantile");
  const [horizon, setHorizon] = useState(28);
  const [mult, setMult] = useState<Record<string, number>>({});
  const [debounced, setDebounced] = useState(mult);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(mult), 350);
    return () => clearTimeout(t);
  }, [mult]);

  const fc = useAsync<ForecastResponse>(
    () => api.forecast({ dataset_id: datasetId, model, horizon, spend_multipliers: debounced }),
    [datasetId, model, horizon, JSON.stringify(debounced)],
  );
  const hist = useAsync(() => api.history(datasetId, "D"), [datasetId]);

  const actuals = useMemo(() => {
    const recs = (hist.data?.records ?? []).slice(-56);
    const series: Record<string, Array<{ date: string; actual: number }>> = { total: [] };
    for (const r of recs) {
      series.total.push({ date: r.date as string, actual: r.total_revenue as number });
      for (const ch of channels) (series[ch] ??= []).push({ date: r.date as string, actual: r[`${ch}_revenue`] as number });
    }
    return series;
  }, [hist.data, channels]);

  const exportCsv = () => {
    if (!fc.data) return;
    const rows = Object.entries(fc.data.channels).flatMap(([ch, rs]) =>
      rs.map((r) => ({ channel: ch, date: r.date, spend: r.spend, p10: r.p10, p50: r.p50, p90: r.p90 })),
    );
    rows.push(...fc.data.total.map((r) => ({ channel: "total", date: r.date, spend: r.spend, p10: r.p10, p50: r.p50, p90: r.p90 })));
    downloadCsv(`forecast_${model}_${horizon}d.csv`, rows);
  };

  const d = fc.data;
  const totalSpend = d ? Object.values(d.planned_spend).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="stack">
      <Card title="Scenario" subtitle="Forecasts assume the media plan is known. Scale each channel's business-as-usual plan to test what-ifs.">
        <div className="controls">
          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
              {MODEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Horizon: {horizon} days</span>
            <input type="range" min={7} max={90} step={7} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
          </label>
          {channels.map((ch, i) => (
            <label className="field" key={ch}>
              <span>
                <span className="dot" style={{ background: channelColor(ch, i) }} />
                {title(ch)} spend: {Math.round((mult[ch] ?? 1) * 100)}%
              </span>
              <input type="range" min={0.2} max={2.5} step={0.05} value={mult[ch] ?? 1} onChange={(e) => setMult({ ...mult, [ch]: Number(e.target.value) })} />
            </label>
          ))}
          <div className="field buttons">
            <button className="btn ghost" onClick={() => setMult({})}>
              Reset plan
            </button>
            <button className="btn" onClick={exportCsv} disabled={!d}>
              Export CSV
            </button>
          </div>
        </div>
      </Card>

      {fc.error && <ErrorBox error={fc.error} onRetry={fc.reload} />}
      {!d && !fc.error && <Loading label="Fitting models" />}
      {d && (
        <>
          <div className={`stats ${fc.loading ? "stale" : ""}`}>
            <Stat label={`P50 revenue, next ${d.horizon}d`} value={money(d.horizon_total.p50)} hint={`P10 ${money(d.horizon_total.p10)} · P90 ${money(d.horizon_total.p90)}`} />
            <Stat label="Planned spend" value={money(totalSpend)} />
            <Stat label="Expected ROAS (P50)" value={num(d.horizon_total.p50 / totalSpend)} />
            <Stat label="Training data through" value={d.train_end} hint={MODEL_OPTIONS.find((m) => m.value === d.model)?.label} />
          </div>
          <Card title="All channels" subtitle="Last 8 weeks of actuals, then P50 with the P10–P90 band">
            <div className={fc.loading ? "stale" : ""}>
              <FanChart forecast={d.total} actuals={actuals.total} color={channelColor("total")} height={340} />
            </div>
          </Card>
          <div className="grid-3">
            {Object.entries(d.channels).map(([ch, rows], i) => {
              const p50 = rows.reduce((a, r) => a + r.p50, 0);
              return (
                <Card key={ch} title={title(ch)} subtitle={`P50 ${money(p50)} on ${money(d.planned_spend[ch])} spend · ROAS ${num(p50 / d.planned_spend[ch])}`}>
                  <div className={fc.loading ? "stale" : ""}>
                    <FanChart forecast={rows} actuals={actuals[ch]?.slice(-28)} color={channelColor(ch, i)} height={220} compact />
                  </div>
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
