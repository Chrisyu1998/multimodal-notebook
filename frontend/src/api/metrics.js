const BASE = "/api/metrics";

export async function fetchTimeseries() {
  const res = await fetch(`${BASE}/timeseries`);
  if (!res.ok) throw new Error(`GET /metrics/timeseries → ${res.status}`);
  return res.json();
}

export async function fetchMetricsSummary() {
  const res = await fetch(`${BASE}/summary`);
  if (!res.ok) throw new Error(`GET /metrics/summary → ${res.status}`);
  return res.json();
}
