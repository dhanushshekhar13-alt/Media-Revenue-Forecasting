import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ForecastRow } from "../api";
import { money, shortDate } from "../format";

export interface ActualPoint {
  date: string;
  actual: number;
}

interface Props {
  forecast: ForecastRow[];
  actuals?: ActualPoint[];
  color: string;
  height?: number;
  compact?: boolean;
}

/** P10–P90 band with the P50 line, preceded by recent actuals. */
export function FanChart({ forecast, actuals = [], color, height = 300, compact = false }: Props) {
  const data = [
    ...actuals.map((a) => ({ date: a.date, actual: a.actual })),
    ...forecast.map((f) => ({
      date: f.date,
      p50: f.p50,
      band: [f.p10, f.p90] as [number, number],
    })),
  ];
  // Anchor the forecast to the last actual so the lines join.
  if (actuals.length && forecast.length) {
    const last = data[actuals.length - 1] as Record<string, unknown>;
    last.p50 = actuals[actuals.length - 1].actual;
    last.band = [actuals[actuals.length - 1].actual, actuals[actuals.length - 1].actual];
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, left: compact ? 0 : 8, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={28} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
        <YAxis tickFormatter={(v) => money(v)} width={compact ? 56 : 68} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelFormatter={(d) => shortDate(String(d))}
          formatter={(v: unknown, name: string) => {
            if (Array.isArray(v)) return [`${money(v[0])} – ${money(v[1])}`, "P10–P90"];
            return [money(v as number), name];
          }}
        />
        {!compact && <Legend wrapperStyle={{ fontSize: 12 }} />}
        <Area type="monotone" dataKey="band" name="P10–P90" stroke="none" fill={color} fillOpacity={0.18} isAnimationActive={false} />
        <Line type="monotone" dataKey="actual" name="Actual" stroke="var(--fg)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="p50" name="P50 forecast" stroke={color} strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
