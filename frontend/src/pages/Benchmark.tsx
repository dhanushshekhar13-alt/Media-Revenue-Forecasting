import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type ModelName } from "../api";
import { Card, ErrorBox, Loading, Stat, useAsync } from "../components/ui";
import { channelColor, num, pct, title } from "../format";

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 };
const ORDER: ModelName[] = ["roas_quantile", "prophet_spend", "prophet", "arima"];

export function Benchmark({ datasetId }: { datasetId: string }) {
  const rep = useAsync(() => api.benchmark(), []);
  const model = useAsync(() => api.roasModel(datasetId), [datasetId]);
  const cache = useAsync(() => api.cache(), []);

  if (rep.error) return <ErrorBox error={rep.error} onRetry={rep.reload} />;
  if (!rep.data) return <Loading />;
  const r = rep.data;
  const ours = r.accuracy.roas_quantile.metrics;
  const best = r.accuracy.prophet.metrics;
  const bars = ORDER.filter((m) => r.accuracy[m]).map((m) => ({
    model: r.accuracy[m].label.replace(" (ours)", ""),
    key: m,
    mape: r.accuracy[m].metrics.total.mape,
    channelMape: r.accuracy[m].metrics.channel_avg.mape,
    coverage: r.accuracy[m].metrics.channel_avg.coverage_p10_p90,
  }));

  return (
    <div className="stack">
      <div className="stats">
        <Stat label="Total revenue MAPE (ours)" value={pct(ours.total.mape, 2)} hint={`vs ${pct(best.total.mape, 2)} univariate Prophet`} />
        <Stat label="Channel-level MAPE (ours)" value={pct(ours.channel_avg.mape, 2)} hint={`P10–P90 coverage ${pct(ours.channel_avg.coverage_p10_p90)} (nominal 80%)`} />
        <Stat label="Cache speed-up" value={`${r.caching.speedup_warm}×`} hint={`${r.caching.per_request_ms_uncached} ms → ${r.caching.per_request_ms_warm} ms per scenario`} />
        <Stat label="Optimiser true uplift" value={`${r.optimizer.mean_true_uplift_pct >= 0 ? "+" : ""}${r.optimizer.mean_true_uplift_pct.toFixed(2)}%`} hint={`predicted ${r.optimizer.mean_predicted_uplift_pct.toFixed(2)}% · ${r.optimizer.origins.length} origins`} />
      </div>

      <Card title="Backtest results" subtitle={`Rolling origin: ${r.folds} folds × ${r.horizon}-day horizon on ${r.dataset.start} – ${r.dataset.end}. Generated ${r.generated_at}.`}>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th className="r">Total MAPE</th>
                <th className="r">Total coverage</th>
                <th className="r">Channel MAPE</th>
                <th className="r">Channel coverage</th>
                <th className="r">WAPE</th>
                <th className="r">Scaled pinball</th>
                <th className="r">Interval width</th>
                <th className="r">Fit (s)</th>
              </tr>
            </thead>
            <tbody>
              {ORDER.filter((m) => r.accuracy[m]).map((m) => {
                const a = r.accuracy[m];
                return (
                  <tr key={m} className={m === "roas_quantile" ? "highlight" : ""}>
                    <td>{a.label}</td>
                    <td className="r">{pct(a.metrics.total.mape, 2)}</td>
                    <td className="r">{pct(a.metrics.total.coverage_p10_p90)}</td>
                    <td className="r">{pct(a.metrics.channel_avg.mape, 2)}</td>
                    <td className="r">{pct(a.metrics.channel_avg.coverage_p10_p90)}</td>
                    <td className="r">{pct(a.metrics.channel_avg.wape, 2)}</td>
                    <td className="r">{num(a.metrics.channel_avg.scaled_pinball, 4)}</td>
                    <td className="r">{pct(a.metrics.channel_avg.interval_width_pct, 0)}</td>
                    <td className="r">{num(a.avg_fit_forecast_seconds, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid-2">
        <Card title="Channel-level MAPE by model" subtitle="Lower is better">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }}>
              <CartesianGrid stroke="var(--grid)" horizontal={false} />
              <XAxis type="number" tickFormatter={(v) => pct(v, 0)} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <YAxis type="category" dataKey="model" width={170} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => pct(v, 2)} cursor={{ fill: "var(--grid)" }} />
              <Bar dataKey="channelMape" name="MAPE" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell key={b.key} fill={b.key === "roas_quantile" ? "var(--accent)" : "var(--muted-bar)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="P10–P90 coverage by model" subtitle="Closer to the dashed 80% line is better calibrated">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }}>
              <CartesianGrid stroke="var(--grid)" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <YAxis type="category" dataKey="model" width={170} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => pct(v, 1)} cursor={{ fill: "var(--grid)" }} />
              <ReferenceLine x={0.8} stroke="var(--fg)" strokeDasharray="4 4" />
              <Bar dataKey="coverage" name="Coverage" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell key={b.key} fill={b.key === "roas_quantile" ? "var(--accent)" : "var(--muted-bar)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid-2">
        <Card title="ROAS model per channel" subtitle="Channel-specific hyper-parameters chosen by time-series validation">
          {model.error && <ErrorBox error={model.error} />}
          {!model.data && !model.error && <Loading />}
          {model.data && (
            <table className="table">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th className="r">Backtest MAPE</th>
                  <th className="r">Coverage</th>
                  <th className="r">Adstock decay</th>
                  <th className="r">Saturation β</th>
                  <th className="r">Residual σ</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(model.data).map(([ch, m], i) => (
                  <tr key={ch}>
                    <td>
                      <span className="dot" style={{ background: channelColor(ch, i) }} />
                      {title(ch)}
                    </td>
                    <td className="r">{pct(ours[ch]?.mape, 2)}</td>
                    <td className="r">{pct(ours[ch]?.coverage_p10_p90)}</td>
                    <td className="r">{num(m.adstock_decay, 2)}</td>
                    <td className="r">{num(m.coefficients.saturation, 3)}</td>
                    <td className="r">{num(m.residual_sd, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <Card title="Caching" subtitle={`${r.caching.scenario_requests} scenario forecasts on the same history`}>
          <table className="table">
            <tbody>
              <tr><td>Uncached (refit every request)</td><td className="r">{num(r.caching.uncached_seconds, 2)} s</td></tr>
              <tr><td>Cached, starting cold</td><td className="r">{num(r.caching.cached_seconds, 2)} s · {r.caching.speedup_cold_start}×</td></tr>
              <tr><td>Cached, warm (pre-fit at start-up)</td><td className="r">{num(r.caching.warm_seconds, 2)} s · {r.caching.speedup_warm}×</td></tr>
              <tr><td>Backtest re-run (4 folds)</td><td className="r">{num(r.caching.backtest_cold_seconds, 2)} s → {num(r.caching.backtest_warm_seconds, 2)} s · {r.caching.backtest_speedup}×</td></tr>
              {cache.data && (
                <tr><td>This server now</td><td className="r">{cache.data.entries} fits cached · hit rate {pct(cache.data.hit_rate, 0)} · {num(cache.data.seconds_saved, 1)} s saved</td></tr>
              )}
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Optimiser validation" subtitle="Because the data is synthetic, each recommended allocation can be scored under the true data-generating process at the same total budget.">
        <table className="table">
          <thead>
            <tr>
              <th>Origin</th>
              <th className="r">Predicted uplift</th>
              <th className="r">True uplift</th>
              {Object.keys(r.optimizer.origins[0]?.allocation_share ?? {}).map((ch) => (
                <th key={ch} className="r">{title(ch)} share</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {r.optimizer.origins.map((o) => (
              <tr key={o.origin}>
                <td>{o.origin}</td>
                <td className="r">{o.predicted_uplift_pct.toFixed(2)}%</td>
                <td className={`r ${o.true_uplift_pct >= 0 ? "pos" : "neg"}`}>{o.true_uplift_pct.toFixed(2)}%</td>
                {Object.values(o.allocation_share).map((s, i) => (
                  <td key={i} className="r">{pct(s)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
