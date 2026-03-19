const BASE = "/api/eval";

export async function fetchLatestRun() {
  const res = await fetch(`${BASE}/latest`);
  if (!res.ok) throw new Error(`GET /eval/latest → ${res.status}`);
  return res.json();
}

export async function fetchRuns() {
  const res = await fetch(`${BASE}/runs`);
  if (!res.ok) throw new Error(`GET /eval/runs → ${res.status}`);
  return res.json();
}

export async function fetchRun(runId) {
  const res = await fetch(`${BASE}/runs/${runId}`);
  if (!res.ok) throw new Error(`GET /eval/runs/${runId} → ${res.status}`);
  return res.json();
}
