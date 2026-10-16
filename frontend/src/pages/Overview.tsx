import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ErrorBox, Loading, Stat, useAsync } from "../components/ui";
import { channelColor, money, monthYear, num, title } from "../format";

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 };

export function Overview({ datasetId }: { datasetId: string }) {
  const kpis = useAsync(() => api.kpis(datasetId), [datasetId]);
  const hist = useAsync(() => api.history(datasetId, "W"), [datasetId]);

  if (kpis.error) return <ErrorBox error={kpis.error} onRetry={kpis.reload} />;
  if (!kpis.data) return <Loading />;
  const k = kpis.data;
  const delta = (a: number | null, b: number | null) => (a != null && b ? a / b - 1 : null);
  const channels = Object.keys(k.channels);
  const weekly = (hist.data?.records ?? []).slice(1, -1).map((r) => {
    const row: Record<string, number | string> = { date: r.date };
    for (const ch of channels) {
      row[ch] = r[`${ch}_revenue`] as number;
      row[`${ch}_roas`] = (r[`${ch}_revenue`] as number) / Math.max(r[`${ch}_spend`] as number, 1);
    }
    return row;
  });

  return (
    <div className="stack">
      <div className="stats">
        <Stat label="Revenue (last 28d)" value={money(k.total.revenue)} delta={delta(k.total.revenue, k.previous_total.revenue)} />
        <Stat label="Media spend (last 28d)" value={money(k.total.spend)} delta={delta(k.total.spend, k.previous_total.spend)} />
        <Stat label="Blended ROAS" value={num(k.total.roas)} delta={delta(k.total.roas, k.previous_total.roas)} />
        <Stat label="Data through" value={k.as_of} hint={`${channels.length} channels`} />
      </div>

      <Card title="Weekly revenue by channel" subtitle="Stacked; full history of the selected dataset">
        {hist.error ? (
          <ErrorBox error={hist.error} />
        ) : !hist.data ? (
          <Loading />
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={weekly} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="date" tickFormatter={monthYear} minTickGap={40} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <YAxis tickFormatter={(v) => money(v)} width={68} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={(d) => `Week of ${d}`} formatter={(v: number, n: string) => [money(v), title(n)]} />
              <Legend formatter={title} wrapperStyle={{ fontSize: 12 }} />
              {channels.map((ch, i) => (
                <Area key={ch} type="monotone" dataKey={ch} stackId="1" stroke={channelColor(ch, i)} fill={channelColor(ch, i)} fillOpacity={0.55} isAnimationActive={false} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </Card>

      <div className="grid-2">
        <Card title="Weekly ROAS" subtitle="Revenue ÷ spend. Efficiency is far steadier than revenue, which is why we forecast it.">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={weekly} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="date" tickFormatter={monthYear} minTickGap={40} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
              <YAxis width={40} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" domain={["auto", "auto"]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number, n: string) => [num(v), title(n.replace("_roas", ""))]} />
              {channels.map((ch, i) => (
                <Line key={ch} type="monotone" dataKey={`${ch}_roas`} stroke={channelColor(ch, i)} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Channel scorecard" subtitle="Last 28 days vs the 28 days before">
          <table className="table">
            <thead>
              <tr>
                <th>Channel</th>
                <th className="r">Spend</th>
                <th className="r">Revenue</th>
                <th className="r">ROAS</th>
                <th className="r">Δ ROAS</th>
              </tr>
            </thead>
            <tbody>
              {channels.map((ch, i) => {
                const c = k.channels[ch];
                const d = delta(c.current.roas, c.previous.roas);
                return (
                  <tr key={ch}>
                    <td>
                      <span className="dot" style={{ background: channelColor(ch, i) }} />
                      {title(ch)}
                    </td>
                    <td className="r">{money(c.current.spend)}</td>
                    <td className="r">{money(c.current.revenue)}</td>
                    <td className="r">{num(c.current.roas)}</td>
                    <td className={`r ${d != null && d >= 0 ? "pos" : "neg"}`}>{d == null ? "–" : `${(d * 100).toFixed(1)}%`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}
