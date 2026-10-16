export const CHANNEL_COLORS: Record<string, string> = {
  google: "#2a5bd7",
  meta: "#d1495b",
  bing: "#0f9d8a",
  total: "#6b4fbb",
};

const fallback = ["#e08e0b", "#7a8b99", "#a05195", "#3d9970"];

export function channelColor(ch: string, i = 0): string {
  return CHANNEL_COLORS[ch] ?? fallback[i % fallback.length];
}

export function money(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "–";
  const abs = Math.abs(v);
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (abs >= 1e4) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "–";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  return v.toFixed(digits);
}

export function title(ch: string): string {
  return ch === "total" ? "All channels" : ch.charAt(0).toUpperCase() + ch.slice(1);
}

export function shortDate(d: string): string {
  const dt = new Date(d + "T00:00:00");
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function downloadCsv(filename: string, rows: Array<Record<string, unknown>>): void {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const body = [cols.join(","), ...rows.map((r) => cols.map((c) => String(r[c] ?? "")).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([body], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function monthYear(d: string): string {
  const dt = new Date(d + "T00:00:00");
  return dt.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
}
