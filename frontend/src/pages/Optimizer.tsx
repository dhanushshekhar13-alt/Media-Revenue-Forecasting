import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Objective, type OptimizeResponse } from "../api";
import { Card, ErrorBox, Loading, Stat, useAsync } from "../components/ui";
import { channelColor, money, num, pct, title } from "../format";

function interpolate(points: Array<{ spend: number; p50: number }>, x: number): number {
  if (x <= points[0].spend) return (points[0].p50 * x) / points[0].spend;
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i].spend) {
      const a = points[i - 1], b = points[i];
      return a.p50 + ((b.p50 - a.p50) * (x - a.spend)) / (b.spend - a.spend);
    }
  }
  return points[points.length - 1].p50;
}

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 };

export function Optimizer({ datasetId, channels }: { datasetId: string; channels: string[] }) {
  const [horizon, setHorizon] = useState(28);
  const [budget, setBudget] = useState<string>("");
  const [objective, setObjective] = useState<Objective>("p50");
  const [trust, setTrust] = useState<string>("");
  const [bounds, setBounds] = useState<Record<string, { min?: string; max?: string }>>({});
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const curves = useAsync(() => api.curves(datasetId, horizon), [datasetId, horizon]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const b: Record<string, { min_spend?: number | null; max_spend?: number | null }> = {};
      for (const [ch, v] of Object.entries(bounds)) {
        if (v.min || v.max) b[ch] = { min_spend: v.min ? Number(v.min) : null, max_spend: v.max ? Number(v.max) : null };
      }
      const r = await api.optimize({
        dataset_id: datasetId,
        horizon,
        objective,
        total_budget: budget ? Number(budget) : undefined,
        bounds: b,
        ...(trust ? { max_change: Number(trust) / 100 } : {}),
      } as Parameters<typeof api.optimize>[0]);
      setResult(r);
      if (!budget) setBudget(String(Math.round(r.total_budget)));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    setResult(null);
    setBudget("");
    void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, horizon]);

  const allocData = result
    ? Object.entries(result.allocation).map(([ch, a]) => ({ channel: title(ch), Current: a.baseline_spend, Optimized: a.spend }))
    : [];

  return (
    <div className="stack">
      <Card title="Budget allocation" subtitle="Greedy marginal-ROAS allocation on each channel's fitted response curve. Risk-averse (P10) favours channels with tighter uncertainty.">
        <div className="controls">
          <label className="field">
            <span>Horizon budget ($)</span>
            <input type="number" min={0} value={budget} placeholder="current plan" onChange={(e) => setBudget(e.target.value)} />
          </label>
          <label className="field">
            <span>Horizon: {horizon} days</span>
            <input type="range" min={7} max={90} step={7} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>Objective</span>
            <select value={objective} onChange={(e) => setObjective(e.target.value as Objective)}>
              <option value="p50">Maximise P50 (expected)</option>
              <option value="p10">Maximise P10 (risk-averse)</option>
              <option value="p90">Maximise P90 (upside)</option>
            </select>
          </label>
          <label className="field">
            <span>Max change per channel (%)</span>
            <input type="number" min={1} max={100} value={trust} placeholder="no limit" onChange={(e) => setTrust(e.target.value)} />
          </label>
          {channels.map((ch) => (
            <div className="field" key={ch}>
              <span>{title(ch)} min / max ($)</span>
              <div className="pair">
                <input type="number" min={0} placeholder="min" value={bounds[ch]?.min ?? ""} onChange={(e) => setBounds({ ...bounds, [ch]: { ...bounds[ch], min: e.target.value } })} />
                <input type="number" min={0} placeholder="max" value={bounds[ch]?.max ?? ""} onChange={(e) => setBounds({ ...bounds, [ch]: { ...bounds[ch], max: e.target.value } })} />
              </div>
            </div>
          ))}
          <div className="field buttons">
            <button className="btn" onClick={run} disabled={running}>
              {running ? "Optimising…" : "Optimise"}
            </button>
          </div>
        </div>
      </Card>

      {error && <ErrorBox error={error} />}
      {!result && !error && <Loading label="Optimising" />}
      {result && (
        <>
          <div className="stats">
            <Stat label="Optimised revenue (P50)" value={money(result.optimized_forecast.p50)} hint={`P10 ${money(result.optimized_forecast.p10)} · P90 ${money(result.optimized_forecast.p90)}`} />
            <Stat label="Current mix revenue (P50)" value={money(result.baseline_forecast.p50)} hint={`same ${money(result.total_budget)} budget`} />
            <Stat label="Expected uplift" value={result.uplift_pct == null ? "–" : `${result.uplift_pct >= 0 ? "+" : ""}${result.uplift_pct.toFixed(2)}%`} hint={`objective ${objective.toUpperCase()}`} />
            <Stat label="Blended ROAS" value={num(result.expected_revenue / result.total_budget)} hint={`vs ${num(result.baseline_revenue / result.total_budget)} today`} />
          </div>
          <div className="grid-2">
            <Card title="Spend: current mix vs optimised">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={allocData} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="channel" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
                  <YAxis tickFormatter={(v) => money(v)} width={68} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => money(v)} cursor={{ fill: "var(--grid)" }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Current" fill="var(--muted-bar)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Optimized" fill="var(--accent)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Allocation detail" subtitle="At the optimum, marginal ROAS is equal across unconstrained channels">
              <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Channel</th>
                    <th className="r">Spend</th>
                    <th className="r">Share</th>
                    <th className="r">Revenue</th>
                    <th className="r">ROAS</th>
                    <th className="r">Marginal</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.allocation).map(([ch, a], i) => (
                    <tr key={ch}>
                      <td>
                        <span className="dot" style={{ background: channelColor(ch, i) }} />
                        {title(ch)}
                      </td>
                      <td className="r">
                        {money(a.spend)}
                        <div className="muted small">{a.spend >= a.baseline_spend ? "+" : ""}{pct(a.spend / a.baseline_spend - 1, 0)}</div>
                      </td>
                      <td className="r">{pct(a.share)}</td>
                      <td className="r">{money(a.expected_revenue)}</td>
                      <td className="r">{num(a.roas)}</td>
                      <td className="r">{num(a.marginal_roas)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </Card>
          </div>
        </>
      )}

      <Card title="Response curves" subtitle={`Horizon revenue vs horizon spend (P50 line, P10/P90 dashed). Dark dot = current plan, blue dot = optimised spend. ${horizon}-day horizon.`}>
        {curves.error && <ErrorBox error={curves.error} />}
        {!curves.data && !curves.error && <Loading />}
        {curves.data && (
          <div className="grid-3">
            {Object.entries(curves.data.curves).map(([ch, c], i) => {
              const plannedP50 = c.points.reduce((best, p) => (Math.abs(p.spend - c.planned_spend) < Math.abs(best.spend - c.planned_spend) ? p : best), c.points[0]);
              const opt = result?.allocation[ch];
              return (
                <div key={ch}>
                  <h4>
                    <span className="dot" style={{ background: channelColor(ch, i) }} />
                    {title(ch)} <span className="muted small">marginal ROAS at plan {num(c.marginal_roas_at_plan)}</span>
                  </h4>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={c.points} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="var(--grid)" vertical={false} />
                      <XAxis dataKey="spend" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => money(v)} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                      <YAxis tickFormatter={(v) => money(v)} width={56} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                      <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => `Spend ${money(Number(v))}`} formatter={(v: number, n: string) => [n === "roas_p50" ? num(v) : money(v), n.toUpperCase()]} />
                      <Line dataKey="p90" stroke={channelColor(ch, i)} strokeDasharray="3 3" dot={false} strokeWidth={1} isAnimationActive={false} />
                      <Line dataKey="p50" stroke={channelColor(ch, i)} dot={false} strokeWidth={2} isAnimationActive={false} />
                      <Line dataKey="p10" stroke={channelColor(ch, i)} strokeDasharray="3 3" dot={false} strokeWidth={1} isAnimationActive={false} />
                      <ReferenceDot x={plannedP50.spend} y={plannedP50.p50} r={5} fill="var(--fg)" stroke="var(--surface)" />
                      {opt && opt.spend <= c.points[c.points.length - 1].spend && (
                        <ReferenceDot x={opt.spend} y={interpolate(c.points, opt.spend)} r={5} fill="var(--accent)" stroke="var(--surface)" />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}
