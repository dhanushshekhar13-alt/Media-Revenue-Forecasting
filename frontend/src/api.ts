export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://localhost:8000";

export type ModelName = "roas_quantile" | "prophet" | "prophet_spend" | "arima";
export type Objective = "p10" | "p50" | "p90";

export interface DatasetSummary {
  dataset_id: string;
  rows: number;
  channels: string[];
  start: string;
  end: string;
  name?: string | null;
  totals: Record<string, { spend: number; revenue: number; roas: number | null }>;
}

export interface HistoryResponse {
  channels: string[];
  records: Array<Record<string, number | string>>;
}

export interface Agg { spend: number; revenue: number; roas: number | null }
export interface KPIs {
  as_of: string;
  window_days: number;
  total: Agg;
  previous_total: Agg;
  channels: Record<string, { current: Agg; previous: Agg }>;
}

export interface ForecastRow {
  date: string;
  spend: number;
  p10: number;
  p50: number;
  p90: number;
  mean: number;
  roas_p50?: number;
}

export interface ForecastResponse {
  model: ModelName;
  horizon: number;
  train_end: string;
  channels: Record<string, ForecastRow[]>;
  total: ForecastRow[];
  horizon_total: { p10: number; p50: number; p90: number };
  planned_spend: Record<string, number>;
}

export interface AllocationRow {
  spend: number;
  share: number;
  expected_revenue: number;
  roas: number | null;
  marginal_roas: number;
  baseline_spend: number;
  baseline_revenue: number;
}

export interface OptimizeResponse {
  objective_quantile: number;
  total_budget: number;
  planned_budget: number;
  allocation: Record<string, AllocationRow>;
  expected_revenue: number;
  baseline_revenue: number;
  uplift_pct: number | null;
  optimized_forecast: { p10: number; p50: number; p90: number };
  baseline_forecast: { p10: number; p50: number; p90: number };
}

export interface CurvePoint { spend: number; p10: number; p50: number; p90: number; roas_p50: number }
export interface CurvesResponse {
  curves: Record<string, { planned_spend: number; marginal_roas_at_plan: number; points: CurvePoint[] }>;
}

export interface Metric {
  mape: number;
  wape: number;
  coverage_p10_p90: number;
  scaled_pinball: number;
  interval_width_pct: number;
}

export interface BenchmarkModel {
  label: string;
  metrics: Record<string, Metric>;
  avg_fit_forecast_seconds: number;
  wall_seconds: number;
}

export interface BenchmarkReport {
  generated_at: string;
  folds: number;
  horizon: number;
  dataset: DatasetSummary;
  accuracy: Record<ModelName, BenchmarkModel>;
  caching: {
    scenario_requests: number;
    uncached_seconds: number;
    cached_seconds: number;
    warm_seconds: number;
    speedup_cold_start: number;
    speedup_warm: number;
    per_request_ms_uncached: number;
    per_request_ms_warm: number;
    backtest_cold_seconds: number;
    backtest_warm_seconds: number;
    backtest_speedup: number;
  };
  optimizer: {
    mean_predicted_uplift_pct: number;
    mean_true_uplift_pct: number;
    origins: Array<{ origin: string; predicted_uplift_pct: number; true_uplift_pct: number; allocation_share: Record<string, number> }>;
  };
}

export interface ChannelModelInfo {
  adstock_decay: number;
  half_saturation: number;
  ar_phi: number;
  residual_sd: number;
  coefficients: Record<string, number>;
}

export interface CacheStats {
  entries: number;
  hits: number;
  disk_hits: number;
  misses: number;
  hit_rate: number;
  compute_seconds: number;
  seconds_saved: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  datasets: () => request<DatasetSummary[]>("/api/datasets"),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DatasetSummary>("/api/datasets", { method: "POST", body: fd });
  },
  history: (dataset_id: string, freq: "D" | "W" | "MS" = "W") =>
    request<HistoryResponse>(`/api/history?dataset_id=${dataset_id}&freq=${freq}`),
  kpis: (dataset_id: string) => request<KPIs>(`/api/kpis?dataset_id=${dataset_id}`),
  forecast: (body: {
    dataset_id: string;
    model: ModelName;
    horizon: number;
    spend_multipliers?: Record<string, number>;
  }) => request<ForecastResponse>("/api/forecast", json(body)),
  optimize: (body: {
    dataset_id: string;
    total_budget?: number;
    horizon: number;
    objective: Objective;
    bounds?: Record<string, { min_spend?: number | null; max_spend?: number | null }>;
  }) => request<OptimizeResponse>("/api/optimize", json(body)),
  curves: (dataset_id: string, horizon: number) =>
    request<CurvesResponse>(`/api/response-curves?dataset_id=${dataset_id}&horizon=${horizon}`),
  benchmark: () => request<BenchmarkReport>("/api/benchmark"),
  roasModel: (dataset_id: string) => request<Record<string, ChannelModelInfo>>(`/api/models/roas?dataset_id=${dataset_id}`),
  cache: () => request<CacheStats>("/api/cache"),
};
